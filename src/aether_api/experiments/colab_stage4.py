"""Stage 4: real-model smoke test for the aether_api Text API vertical slice.

Loads a real LLM backbone behind the same `InterleavedDecodeScheduler` proven
in stage2/3 (not the plain locked backend — using the raw backbone here
would silently lose the dual-stream lookahead the rest of the project
exists to prove), wires it into the actual FastAPI app via
`aether_api.http.app.create_app`, and drives one turn against a real
running uvicorn server over a real TCP socket.

A TestClient-based version was tried first and rejected: FastAPI's
in-process TestClient (httpx ASGITransport + a sync portal) drains the
whole SSE body before `iter_lines()` yields anything, so every event
reports the same arrival timestamp — it cannot show whether events are
actually streamed live or just batched at the end, which is the one thing
this stage exists to prove. A real socket does not have that problem.

This is the first time `aether_api` runs against a real model instead of
`ScriptedSharedBackend` fakes.
"""

import argparse
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from aether.experiments.colab_stage1 import environment_report, write_json
from aether.model.llm_backbone import LLMBackboneConfig, SharedLLMBackbone
from aether.model.llm_step_engine import LLMTokenStepEngine
from aether.model.step_scheduler import InterleavedDecodeScheduler
from aether.testing.fakes import FakeWeatherTool
from aether_api.auth import ApiKey, ApiKeyStore
from aether_api.http.app import create_app
from aether_api.turn_service import TurnService


def run_experiment(args: argparse.Namespace, output_dir: Path) -> int:
    import httpx
    import uvicorn

    report: Dict[str, Any] = {
        "experiment": "aether_api_stage4_real_model_http_smoke",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dtype": args.dtype,
        "request": args.request,
        "tool_latency_ms": args.tool_latency_ms,
        "environment": environment_report(),
        "status": "started",
    }
    write_json(output_dir / "report.json", report)

    backbone = SharedLLMBackbone(
        LLMBackboneConfig(
            model_path=args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            allow_download=args.allow_download,
            enable_thinking=False,
        )
    )
    try:
        load_started = time.monotonic_ns()
        backbone.load()
        report["model_load_ms"] = (time.monotonic_ns() - load_started) / 1_000_000

        engine = LLMTokenStepEngine(backbone)
        scheduler = InterleavedDecodeScheduler(
            engine, speaker_weight=args.speaker_weight, planner_weight=args.planner_weight
        )
        turn_service = TurnService(scheduler, FakeWeatherTool(latency_ms=args.tool_latency_ms))
        api_keys = ApiKeyStore({"dev-key": ApiKey(key="dev-key", owner="dev", max_concurrent_turns=1)})
        app = create_app(turn_service, api_keys)

        config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True, name="aether-api-stage4")
        server_thread.start()
        wait_started = time.monotonic()
        while not server.started:
            if time.monotonic() - wait_started > 30:
                raise RuntimeError("uvicorn server did not start within 30s")
            time.sleep(0.05)

        events: List[Dict[str, Any]] = []
        try:
            run_started = time.monotonic_ns()
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{args.port}/v1/turns",
                json={"message": args.request, "tools": ["weather"]},
                headers={"authorization": "Bearer dev-key"},
                timeout=120,
            ) as response:
                report["http_status"] = response.status_code
                report["content_type"] = response.headers.get("content-type")
                current_event = None
                for raw_line in response.iter_lines():
                    arrived_ms = (time.monotonic_ns() - run_started) / 1_000_000
                    if raw_line.startswith("event: "):
                        current_event = raw_line[len("event: "):].strip()
                    elif raw_line.startswith("data: "):
                        payload = json.loads(raw_line[len("data: "):])
                        events.append(
                            {"arrived_ms": arrived_ms, "type": current_event, "payload": payload}
                        )
        finally:
            server.should_exit = True
            server_thread.join(timeout=10)

        report["runtime_ms"] = (time.monotonic_ns() - run_started) / 1_000_000
        report["events"] = events
        report["event_types_in_order"] = [event["type"] for event in events]

        safe_delta = next((e for e in events if e["type"] == "response.safe_delta"), None)
        tool_completed = next((e for e in events if e["type"] == "tool.completed"), None)
        response_delta = next((e for e in events if e["type"] == "response.delta"), None)
        report["proof"] = {
            "http_200": report["http_status"] == 200,
            "streamed_progressively": (
                len(events) >= 2 and events[-1]["arrived_ms"] > events[0]["arrived_ms"] + 1
            ),
            "safe_delta_before_tool_completed": (
                safe_delta is not None
                and tool_completed is not None
                and safe_delta["arrived_ms"] < tool_completed["arrived_ms"]
            ),
            "response_delta_after_tool_completed": (
                response_delta is not None
                and tool_completed is not None
                and response_delta["arrived_ms"] > tool_completed["arrived_ms"]
            ),
            "ends_with_turn_completed": bool(events) and events[-1]["type"] == "turn.completed",
            "no_turn_failed": "turn.failed" not in [e["type"] for e in events],
        }
        report["status"] = "passed" if all(report["proof"].values()) else "failed"
        return 0 if report["status"] == "passed" else 1
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        (output_dir / "traceback.txt").write_text(report["traceback"], encoding="utf-8")
        return 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "report.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--request", default="Какая погода в Алматы и нужен ли зонт?")
    parser.add_argument("--tool-latency-ms", type=int, default=1500)
    parser.add_argument("--speaker-weight", type=int, default=3)
    parser.add_argument("--planner-weight", type=int, default=2)
    parser.add_argument("--device-map", default="auto")
    # float16, not "auto": Turing (T4) has no native bf16 tensor cores, and
    # many Qwen3 checkpoints default to bf16 - "auto" risks slow/unstable
    # compute there. A100 handles float16 fine too, so this stays the
    # default regardless of which GPU the notebook lands on.
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output-dir", default="artifacts/colab-stage4")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("pass --allow-download only in the target ML environment")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exit_code = run_experiment(args, output_dir)
    print(f"AETHER_STAGE4_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE4_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
