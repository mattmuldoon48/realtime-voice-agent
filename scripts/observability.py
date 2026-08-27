"""Local CloudWatch observability administration entrypoint."""

from __future__ import annotations

from realtime_voice_agent.observability.bootstrap import run

if __name__ == "__main__":
    raise SystemExit(run())
