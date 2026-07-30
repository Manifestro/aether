"""Stage 2 interleaved LLM KV-cache experiment with diagnostic artifacts."""

import argparse
import asyncio
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aether.experiments.colab_stage1 import RecordingBackend, environment_report, write_json
from aether.model.llm_adapters import LLMPlannerAdapter, LLMSpeakerAdapter
from aether.model.llm_backbone import LLMBackboneConfig, SharedLLMBackbone
from aether.model.llm_step_engine import LLMTokenStepEngine
from aether.model.step_scheduler import InterleavedDecodeScheduler
from aether.runtime.dual_session import DualSessionRuntime
from aether.runtime.tool_executor import AllowlistToolExecutor
from aether.testing.fakes import FakeWeatherTool


def runtime_trace(result: Any) -> list:
    return [
        {
            "name": event.name,
            "timestamp_ms": event.timestamp_ns / 1_000_000,
            "absolute_timestamp_ns": result.timeline.origin_ns + event.timestamp_ns,
            "attributes": event.attributes,
        }
        for event in result.timeline.events
    ]


def scheduler_trace(scheduler: InterleavedDecodeScheduler) -> list:
    return [
        {
            "name": event.name,
            "timestamp_ms": event.timestamp_ns / 1_000_000,
            "absolute_timestamp_ns": event.absolute_timestamp_ns,
            "session_id": event.session_id,
            "role": event.role,
            "attributes": event.attributes,
        }
        for event in scheduler.trace
    ]


def first_absolute(events: list, name: str, role: Optional[str] = None) -> Optional[int]:
    for event in events:
        if event["name"] == name and (role is None or event.get("role") == role):
            return int(event["absolute_timestamp_ns"])
    return None


async def run_experiment(args: argparse.Namespace, output_dir: Path) -> int:
    report: Dict[str, Any] = {
        "experiment": "llm_stage2_interleaved_kv",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "request": args.request,
        "tool_latency_ms": args.tool_latency_ms,
        "scheduler_weights": {"speaker": args.speaker_weight, "planner": args.planner_weight},
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
    recorder = None
    try:
        load_started = time.monotonic_ns()
        backbone.load()
        report["model_load_ms"] = (time.monotonic_ns() - load_started) / 1_000_000

        engine = LLMTokenStepEngine(backbone)
        scheduler = InterleavedDecodeScheduler(
            engine,
            speaker_weight=args.speaker_weight,
            planner_weight=args.planner_weight,
        )
        recorder = RecordingBackend(scheduler)
        runtime = DualSessionRuntime(
            LLMPlannerAdapter(recorder, tools=["weather"]),
            LLMSpeakerAdapter(recorder),
            AllowlistToolExecutor(["weather"], FakeWeatherTool(latency_ms=args.tool_latency_ms)),
        )

        run_started = time.monotonic_ns()
        result = await runtime.run("colab-stage2-1", args.request)
        report["runtime_ms"] = (time.monotonic_ns() - run_started) / 1_000_000
        rt_trace = runtime_trace(result)
        dec_trace = scheduler_trace(scheduler)
        report["runtime_timeline"] = rt_trace
        report["decode_timeline"] = dec_trace

        speaker_first = first_absolute(dec_trace, "first_token", "speaker")
        planner_completed = first_absolute(dec_trace, "decode_completed", "planner")
        tool_completed = first_absolute(rt_trace, "tool_completed")
        report["proof"] = {
            "speaker_first_token_before_planner_complete": (
                speaker_first is not None
                and planner_completed is not None
                and speaker_first < planner_completed
            ),
            "speaker_first_token_before_tool_complete": (
                speaker_first is not None
                and tool_completed is not None
                and speaker_first < tool_completed
            ),
            "speaker_first_token_minus_tool_complete_ms": (
                None
                if speaker_first is None or tool_completed is None
                else (speaker_first - tool_completed) / 1_000_000
            ),
        }
        report["result_text"] = result.text
        report["semantic_events"] = [
            {
                "sequence": event.sequence,
                "kind": event.kind.value,
                "payload": dict(event.payload),
            }
            for event in result.semantic_events
        ]
        report["chunks"] = [
            {
                "chunk_id": chunk.chunk_id,
                "state": chunk.state.value,
                "dependencies": sorted(chunk.dependencies),
            }
            for chunk in result.chunks
        ]
        report["status"] = "passed"
        return 0
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        (output_dir / "traceback.txt").write_text(report["traceback"], encoding="utf-8")
        return 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["raw_generations"] = [] if recorder is None else recorder.generations
        write_json(output_dir / "report.json", report)
        with (output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as handle:
            for generation in report["raw_generations"]:
                handle.write(json.dumps(generation, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--request", default="Какая погода в Алматы и нужен ли зонт?")
    parser.add_argument("--tool-latency-ms", type=int, default=3000)
    parser.add_argument("--speaker-weight", type=int, default=3)
    parser.add_argument("--planner-weight", type=int, default=2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="artifacts/colab-stage2")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("pass --allow-download only in the target ML environment")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exit_code = asyncio.run(run_experiment(args, output_dir))
    print(f"AETHER_STAGE2_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE2_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
