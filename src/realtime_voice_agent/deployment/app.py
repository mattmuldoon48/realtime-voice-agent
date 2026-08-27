"""AWS CDK application entrypoint."""

from __future__ import annotations

import os

from aws_cdk import App, Environment, Tags

from realtime_voice_agent.deployment.stack import RealtimeVoiceAgentStack


def main() -> None:
    """Synthesize the configured deployment stack."""
    app = App()
    environment_name = str(app.node.try_get_context("environment") or "deployment")
    twilio_secret_name = app.node.try_get_context("twilioSecretName")
    stack = RealtimeVoiceAgentStack(
        app,
        "RealtimeVoiceAgentDeployment",
        environment_name=environment_name,
        twilio_secret_name=(str(twilio_secret_name) if twilio_secret_name is not None else None),
        env=Environment(
            account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
            region=os.environ.get("CDK_DEFAULT_REGION"),
        ),
        description="Optional real-time voice-agent ECS/Fargate deployment",
    )
    Tags.of(stack).add("Service", "realtime-voice-agent")
    Tags.of(stack).add("Environment", environment_name)
    app.synth()


if __name__ == "__main__":
    main()
