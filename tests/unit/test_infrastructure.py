"""Infrastructure contract tests for the optional ECS/Fargate deployment."""

from __future__ import annotations

from typing import Any, cast

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Template

from realtime_voice_agent.deployment.stack import RealtimeVoiceAgentStack

_ACCOUNT = "111122223333"
_REGION = "us-east-1"


@pytest.fixture
def template() -> Template:
    app = App()
    stack = RealtimeVoiceAgentStack(
        app,
        "TestStack",
        environment_name="test",
        env=Environment(account=_ACCOUNT, region=_REGION),
    )
    return Template.from_stack(stack)


def _only_properties(template: Template, resource_type: str) -> dict[str, Any]:
    resources = template.find_resources(resource_type)
    assert len(resources) == 1
    resource = cast(dict[str, Any], next(iter(resources.values())))
    return cast(dict[str, Any], resource["Properties"])


def test_task_is_bounded_read_only_python_service(template: Template) -> None:
    task = _only_properties(template, "AWS::ECS::TaskDefinition")
    assert task["Cpu"] == "512"
    assert task["Memory"] == "1024"
    assert task["NetworkMode"] == "awsvpc"
    assert task["RequiresCompatibilities"] == ["FARGATE"]
    assert task["RuntimePlatform"] == {
        "CpuArchitecture": "X86_64",
        "OperatingSystemFamily": "LINUX",
    }

    containers = cast(list[dict[str, Any]], task["ContainerDefinitions"])
    assert len(containers) == 1
    container = containers[0]
    assert container["ReadonlyRootFilesystem"] is True
    assert container["StopTimeout"] == 30
    assert container["PortMappings"] == [{"ContainerPort": 8000, "Protocol": "tcp"}]
    assert container["LogConfiguration"]["LogDriver"] == "awslogs"

    environment = {
        item["Name"]: item["Value"] for item in cast(list[dict[str, Any]], container["Environment"])
    }
    assert environment["TWILIO_VALIDATE_SIGNATURES"] == "true"
    assert environment["PUBLIC_BASE_URL"]["Fn::Join"][1][0] == "https://"
    assert environment["PUBLIC_MEDIA_WS_URL"]["Fn::Join"][1][0] == "wss://"
    assert environment["DEMO_MODE_ENABLED"] == "true"
    assert environment["DEMO_MAX_CALL_DURATION_SECONDS"] == "300"
    assert environment["DEMO_GLOBAL_CONCURRENCY_LIMIT"] == "2"
    assert environment["DEMO_PERSIST_TRANSCRIPTS"] == "false"
    assert "TWILIO_ACCOUNT_SID" not in environment
    assert "TWILIO_AUTH_TOKEN" not in environment
    assert {item["Name"] for item in container["Secrets"]} == {
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
    }


def test_cloudfront_is_the_only_forwarding_ingress(template: Template) -> None:
    listener = _only_properties(template, "AWS::ElasticLoadBalancingV2::Listener")
    assert listener["Port"] == 80
    assert listener["DefaultActions"] == [
        {
            "FixedResponseConfig": {
                "ContentType": "application/json",
                "MessageBody": '{"status":"forbidden"}',
                "StatusCode": "403",
            },
            "Type": "fixed-response",
        }
    ]

    rule = _only_properties(template, "AWS::ElasticLoadBalancingV2::ListenerRule")
    assert rule["Priority"] == 1
    assert rule["Conditions"][0]["HttpHeaderConfig"]["HttpHeaderName"] == "X-Origin-Verify"
    rule_secret = rule["Conditions"][0]["HttpHeaderConfig"]["Values"][0]

    distribution = _only_properties(template, "AWS::CloudFront::Distribution")["DistributionConfig"]
    behavior = distribution["DefaultCacheBehavior"]
    assert behavior["ViewerProtocolPolicy"] == "redirect-to-https"
    assert behavior["CachePolicyId"] == "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    assert behavior["OriginRequestPolicyId"] == "b689b0a8-53d0-40ab-baf2-68738e2966ac"
    assert "POST" in behavior["AllowedMethods"]
    origin = distribution["Origins"][0]
    assert origin["CustomOriginConfig"]["OriginProtocolPolicy"] == "http-only"
    assert origin["CustomOriginConfig"]["OriginKeepaliveTimeout"] == 60
    assert origin["OriginCustomHeaders"] == [
        {"HeaderName": "X-Origin-Verify", "HeaderValue": rule_secret}
    ]

    ingress_resources = template.find_resources("AWS::EC2::SecurityGroupIngress")
    ingress = [
        cast(dict[str, Any], resource)["Properties"] for resource in ingress_resources.values()
    ]
    task_ingress = [rule for rule in ingress if rule["FromPort"] == 8000]
    assert len(task_ingress) == 1
    assert "SourceSecurityGroupId" in task_ingress[0]
    assert "CidrIp" not in task_ingress[0]


def test_service_stays_single_task_for_process_global_demo_admission(template: Template) -> None:
    service = _only_properties(template, "AWS::ECS::Service")
    assert service["DesiredCount"] == 1
    assert service["LaunchType"] == "FARGATE"
    deployment = service["DeploymentConfiguration"]
    assert deployment["DeploymentCircuitBreaker"] == {"Enable": True, "Rollback": True}
    assert deployment["MaximumPercent"] == 100
    assert deployment["MinimumHealthyPercent"] == 0
    assert service["NetworkConfiguration"]["AwsvpcConfiguration"]["AssignPublicIp"] == "ENABLED"
    assert template.find_resources("AWS::ApplicationAutoScaling::ScalableTarget") == {}
    assert template.find_resources("AWS::ApplicationAutoScaling::ScalingPolicy") == {}


def test_task_role_is_limited_to_runtime_operations(template: Template) -> None:
    policies = template.find_resources("AWS::IAM::Policy")
    task_role_policy = next(
        cast(dict[str, Any], resource)["Properties"]
        for resource in policies.values()
        if any(
            "TaskDefinitionTaskRole" in str(role)
            for role in cast(dict[str, Any], resource)["Properties"]["Roles"]
        )
    )
    statements = cast(list[dict[str, Any]], task_role_policy["PolicyDocument"]["Statement"])
    actions = {
        action
        for statement in statements
        for action in (
            statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
        )
    }
    assert actions == {
        "bedrock:InvokeModel",
        "cloudwatch:PutMetricData",
        "dynamodb:DescribeTable",
        "dynamodb:DescribeTimeToLive",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    }
    wildcard_statements = [statement for statement in statements if statement["Resource"] == "*"]
    assert wildcard_statements == [
        {
            "Action": "cloudwatch:PutMetricData",
            "Condition": {
                "StringEquals": {"cloudwatch:namespace": "RealtimeVoiceAgent/VoiceAgent"}
            },
            "Effect": "Allow",
            "Resource": "*",
        }
    ]
    update_statement = next(
        statement
        for statement in statements
        if statement["Action"] == ["dynamodb:PutItem", "dynamodb:UpdateItem"]
    )
    assert "RealtimeVoiceAgentSessions" in str(update_statement["Resource"])
    assert "RealtimeVoiceAgentTranscriptTurns" not in str(update_statement["Resource"])
    application_logs = next(
        statement
        for statement in statements
        if statement["Action"] == ["logs:CreateLogStream", "logs:PutLogEvents"]
    )
    assert "log-stream:application-test" in str(application_logs["Resource"])
    assert "Fn::GetAtt" not in str(application_logs["Resource"])


def test_generated_secret_and_sensitive_log_retention_are_destroyed_with_stack(
    template: Template,
) -> None:
    secrets = template.find_resources("AWS::SecretsManager::Secret")
    assert len(secrets) == 1
    secret = cast(dict[str, Any], next(iter(secrets.values())))
    assert secret["DeletionPolicy"] == "Delete"
    assert secret["Properties"]["GenerateSecretString"] == {
        "ExcludePunctuation": True,
        "PasswordLength": 48,
    }

    log_group = cast(
        dict[str, Any],
        next(iter(template.find_resources("AWS::Logs::LogGroup").values())),
    )
    assert log_group["DeletionPolicy"] == "Delete"
    assert log_group["Properties"]["RetentionInDays"] == 7


def test_environment_name_is_bounded() -> None:
    app = App()
    with pytest.raises(ValueError, match="1-32"):
        RealtimeVoiceAgentStack(app, "EmptyEnvironment", environment_name="")
    with pytest.raises(ValueError, match="1-32"):
        RealtimeVoiceAgentStack(app, "LongEnvironment", environment_name="x" * 33)
    with pytest.raises(ValueError, match="lowercase alphanumeric"):
        RealtimeVoiceAgentStack(app, "UnsafeEnvironment", environment_name="../production")
