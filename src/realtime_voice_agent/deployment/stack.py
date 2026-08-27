"""Single-service ECS/Fargate deployment with CloudFront-managed TLS."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aws_cdk import (
    ArnFormat,
    CfnOutput,
    Duration,
    Environment,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_cloudfront as cloudfront,
)
from aws_cdk import (
    aws_cloudfront_origins as cloudfront_origins,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecr_assets as ecr_assets,
)
from aws_cdk import (
    aws_ecs as ecs,
)
from aws_cdk import (
    aws_elasticloadbalancingv2 as elbv2,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

_SERVICE_NAME = "realtime-voice-agent"
_MODEL_ID = "amazon.nova-2-sonic-v1:0"
_METRIC_NAMESPACE = "RealtimeVoiceAgent/VoiceAgent"
_CONTAINER_PORT = 8000


class RealtimeVoiceAgentStack(Stack):
    """Deploy one bounded voice-agent service and its public TLS ingress."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        environment_name: str = "deployment",
        twilio_secret_name: str | None = None,
        personas_table: str = "RealtimeVoiceAgentPersonas",
        sessions_table: str = "RealtimeVoiceAgentSessions",
        transcripts_table: str = "RealtimeVoiceAgentTranscriptTurns",
        env: Environment | dict[str, Any] | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(scope, construct_id, env=env, description=description)
        if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?", environment_name) is None:
            raise ValueError(
                "environment_name must contain 1-32 lowercase alphanumeric or hyphen characters"
            )
        secret_name = twilio_secret_name or f"{_SERVICE_NAME}/{environment_name}/twilio"

        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                )
            ],
        )
        cluster = ecs.Cluster(
            self, "Cluster", vpc=vpc, container_insights_v2=ecs.ContainerInsights.ENABLED
        )

        load_balancer_security_group = ec2.SecurityGroup(
            self,
            "LoadBalancerSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
            description="Public CloudFront origin listener",
        )
        load_balancer_security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(80),
            "CloudFront origin traffic; listener requires the generated origin header",
        )
        task_security_group = ec2.SecurityGroup(
            self,
            "TaskSecurityGroup",
            vpc=vpc,
            allow_all_outbound=True,
            description="Voice-agent tasks accept traffic only from the ALB",
        )
        task_security_group.add_ingress_rule(
            load_balancer_security_group,
            ec2.Port.tcp(_CONTAINER_PORT),
            "ALB to Uvicorn",
        )

        load_balancer = elbv2.ApplicationLoadBalancer(
            self,
            "LoadBalancer",
            vpc=vpc,
            internet_facing=True,
            security_group=load_balancer_security_group,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )
        load_balancer.set_attribute("idle_timeout.timeout_seconds", "1200")
        listener = load_balancer.add_listener(
            "HttpListener",
            port=80,
            protocol=elbv2.ApplicationProtocol.HTTP,
            default_action=elbv2.ListenerAction.fixed_response(
                status_code=403,
                content_type="application/json",
                message_body='{"status":"forbidden"}',
            ),
        )

        origin_secret = secretsmanager.Secret(
            self,
            "OriginVerificationSecret",
            description="Generated CloudFront-to-ALB origin verification value",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_punctuation=True,
                password_length=48,
            ),
        )
        origin_secret.apply_removal_policy(RemovalPolicy.DESTROY)
        origin_header_value = origin_secret.secret_value.unsafe_unwrap()

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=cloudfront_origins.HttpOrigin(
                    load_balancer.load_balancer_dns_name,
                    protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                    custom_headers={"X-Origin-Verify": origin_header_value},
                    connection_timeout=Duration.seconds(10),
                    read_timeout=Duration.seconds(60),
                    keepalive_timeout=Duration.seconds(60),
                ),
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                compress=False,
            ),
            enabled=True,
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
        )

        log_group = logs.LogGroup(
            self,
            "ApplicationLogGroup",
            log_group_name=f"/{_SERVICE_NAME}/{environment_name}",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )
        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDefinition",
            cpu=512,
            memory_limit_mib=1024,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )
        twilio_secret = secretsmanager.Secret.from_secret_name_v2(
            self,
            "TwilioSecret",
            secret_name,
        )
        public_base_url = f"https://{distribution.distribution_domain_name}"
        public_media_ws_url = f"wss://{distribution.distribution_domain_name}/media"
        container = task_definition.add_container(
            "Application",
            image=ecs.ContainerImage.from_asset(
                str(Path(__file__).resolve().parents[3]),
                platform=ecr_assets.Platform.LINUX_AMD64,
            ),
            command=[
                "uvicorn",
                "realtime_voice_agent.main:app",
                "--host",
                "0.0.0.0",  # noqa: S104 - the task must accept ALB traffic.
                "--port",
                str(_CONTAINER_PORT),
                "--no-access-log",
            ],
            environment={
                "APP_ENV": environment_name,
                "AWS_REGION": self.region,
                "CLOUDWATCH_ENABLED": "true",
                "CLOUDWATCH_LOG_GROUP": log_group.log_group_name,
                "CLOUDWATCH_LOG_STREAM": "application",
                "CLOUDWATCH_METRIC_NAMESPACE": _METRIC_NAMESPACE,
                "DEMO_MODE_ENABLED": "true",
                "DEMO_PERSONA_CHOICES": (
                    "1:care-coordinator,"
                    "2:financial-services-assistant,"
                    "3:travel-concierge,"
                    "4:history-guide"
                ),
                "DEMO_MAX_CALL_DURATION_SECONDS": "300",
                "DEMO_RATE_LIMIT_MAX_CALLS": "3",
                "DEMO_RATE_LIMIT_WINDOW_SECONDS": "3600",
                "DEMO_GLOBAL_CONCURRENCY_LIMIT": "2",
                "DEMO_BUDGET_MAX_CALLS": "50",
                "DEMO_BUDGET_WINDOW_SECONDS": "86400",
                "DEMO_RESERVATION_TTL_SECONDS": "30",
                "DEMO_PERSIST_TRANSCRIPTS": "false",
                "NOVA_MODEL_ID": _MODEL_ID,
                "PERSONAS_TABLE": personas_table,
                "PUBLIC_BASE_URL": public_base_url,
                "PUBLIC_MEDIA_WS_URL": public_media_ws_url,
                "SESSIONS_TABLE": sessions_table,
                "TRANSCRIPTS_TABLE": transcripts_table,
                "TWILIO_VALIDATE_SIGNATURES": "true",
            },
            secrets={
                "TWILIO_ACCOUNT_SID": ecs.Secret.from_secrets_manager(
                    twilio_secret,
                    "account_sid",
                ),
                "TWILIO_AUTH_TOKEN": ecs.Secret.from_secrets_manager(
                    twilio_secret,
                    "auth_token",
                ),
            },
            logging=ecs.LogDrivers.aws_logs(
                log_group=log_group,
                stream_prefix="ecs",
            ),
            readonly_root_filesystem=True,
            stop_timeout=Duration.seconds(30),
        )
        container.add_port_mappings(
            ecs.PortMapping(
                container_port=_CONTAINER_PORT,
                protocol=ecs.Protocol.TCP,
            )
        )

        service = ecs.FargateService(
            self,
            "Service",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=1,
            assign_public_ip=True,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            health_check_grace_period=Duration.seconds(60),
            min_healthy_percent=0,
            max_healthy_percent=100,
            security_groups=[task_security_group],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )
        target_group = elbv2.ApplicationTargetGroup(
            self,
            "TargetGroup",
            vpc=vpc,
            port=_CONTAINER_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            deregistration_delay=Duration.seconds(30),
            health_check=elbv2.HealthCheck(
                enabled=True,
                path="/health",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )
        service.attach_to_application_target_group(target_group)
        listener.add_action(
            "VerifiedOrigin",
            priority=1,
            conditions=[
                elbv2.ListenerCondition.http_header(
                    "X-Origin-Verify",
                    [origin_header_value],
                )
            ],
            action=elbv2.ListenerAction.forward([target_group]),
        )

        self._grant_runtime_permissions(
            task_definition=task_definition,
            environment_name=environment_name,
            log_group=log_group,
            personas_table=personas_table,
            sessions_table=sessions_table,
            transcripts_table=transcripts_table,
        )

        CfnOutput(self, "PublicBaseUrl", value=public_base_url)
        CfnOutput(self, "PublicMediaWebSocketUrl", value=public_media_ws_url)
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "ServiceName", value=service.service_name)
        CfnOutput(self, "TwilioSecretName", value=secret_name)

    def _grant_runtime_permissions(
        self,
        *,
        task_definition: ecs.FargateTaskDefinition,
        log_group: logs.LogGroup,
        environment_name: str,
        personas_table: str,
        sessions_table: str,
        transcripts_table: str,
    ) -> None:
        task_role = task_definition.task_role
        task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    self.format_arn(
                        service="bedrock",
                        account="",
                        resource="foundation-model",
                        resource_name=_MODEL_ID,
                        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
                    )
                ],
            )
        )
        task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["dynamodb:DescribeTable", "dynamodb:DescribeTimeToLive"],
                resources=[
                    self._table_arn(personas_table),
                    self._table_arn(sessions_table),
                    self._table_arn(transcripts_table),
                ],
            )
        )
        task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["dynamodb:GetItem"],
                resources=[self._table_arn(personas_table)],
            )
        )
        task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["dynamodb:PutItem", "dynamodb:UpdateItem"],
                resources=[self._table_arn(sessions_table)],
            )
        )
        task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["dynamodb:PutItem"],
                resources=[self._table_arn(transcripts_table)],
            )
        )
        task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": _METRIC_NAMESPACE}},
            )
        )
        task_role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[
                    self.format_arn(
                        service="logs",
                        resource="log-group",
                        resource_name=(
                            f"{log_group.log_group_name}:log-stream:application-{environment_name}"
                        ),
                        arn_format=ArnFormat.COLON_RESOURCE_NAME,
                    )
                ],
            )
        )

    def _table_arn(self, table_name: str) -> str:
        return self.format_arn(
            service="dynamodb",
            resource="table",
            resource_name=table_name,
            arn_format=ArnFormat.SLASH_RESOURCE_NAME,
        )
