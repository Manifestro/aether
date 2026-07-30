"""Stage 3 latency sweep and scenario-diversity experiment.

Experiment 02 from technical_report_01.md: instead of one fixed weather run,
this sweeps MCP tool latency and runs several distinct scenarios (successful
tool call, failing tool call, no-tool conversational turn) so a single lucky
trace cannot stand in for the whole hypothesis. Every run is checked against
explicit pass/fail criteria; the runner does not just log numbers.
"""

import argparse
import asyncio
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from aether.experiments.colab_stage1 import RecordingBackend, environment_report, write_json
from aether.experiments.colab_stage2 import first_absolute, runtime_trace, scheduler_trace
from aether.model.qwen_adapters import QwenPlannerAdapter, QwenSpeakerAdapter
from aether.model.qwen_backbone import QwenBackboneConfig, SharedQwenBackbone
from aether.model.qwen_step_engine import QwenTokenStepEngine
from aether.model.step_scheduler import InterleavedDecodeScheduler
from aether.runtime.dual_session import DualSessionRuntime
from aether.runtime.tool_executor import AllowlistToolExecutor
from aether.testing.fakes import FakeWeatherTool

DEFAULT_LATENCIES_MS = [3000, 1500, 750, 300]


@dataclass(frozen=True)
class Scenario:
    name: str
    request: str
    tool_fail: bool
    sweep_latency: bool
    tools: Tuple[str, ...] = ("weather",)


SCENARIOS: List[Scenario] = [
    Scenario(
        name="weather_success",
        request="Какая погода в Алматы и нужен ли зонт?",
        tool_fail=False,
        sweep_latency=True,
        tools=("weather",),
    ),
    Scenario(
        name="weather_tool_failure",
        request="Какая погода в Алматы и нужен ли зонт?",
        tool_fail=True,
        sweep_latency=True,
        tools=("weather",),
    ),
    Scenario(
        name="no_tool_greeting",
        request="Привет! Как у тебя дела?",
        tool_fail=False,
        sweep_latency=False,
        # No tool granted for this turn: the Planner must answer directly
        # instead of inventing one. AllowlistToolExecutor + the empty prompt
        # allowlist together are the regression test for that finding.
        tools=(),
    ),
]


def evaluate_run(result: Any, rt_trace: list, dec_trace: list) -> "tuple[Dict[str, bool], Dict[str, Any]]":
    """Split trace evaluation into a hard safety gate and soft observations.

    ``checks`` is the one thing a run must satisfy to pass: no chunk speaks a
    dependent fact unless the tool it depends on is recorded as
    ``tool_completed(succeeded=True)`` strictly before that chunk started
    generating. This is scenario-agnostic and derived only from the trace, so
    it does not assume which tool name a scenario "should" have called.

    ``observations`` records everything informative that is not a
    correctness gate: whether a tool_call happened at all, whether the
    speaker's first token led or trailed tool completion (this depends on
    MCP latency and is expected to flip sign below a crossover point, not a
    defect), and which chunks never resolved.
    """

    tool_call_events = [
        event for event in result.semantic_events if event.kind.value == "tool_call"
    ]

    confirmed_tool_index: Dict[str, int] = {}
    for index, event in enumerate(rt_trace):
        if event["name"] == "tool_completed" and event["attributes"].get("succeeded"):
            confirmed_tool_index.setdefault(event["attributes"].get("tool"), index)

    violations = []
    for chunk in result.chunks:
        if chunk.state.value != "played" or not chunk.dependencies:
            continue
        generating_index = next(
            (
                i
                for i, e in enumerate(rt_trace)
                if e["name"] == "chunk_generating" and e["attributes"].get("chunk_id") == chunk.chunk_id
            ),
            None,
        )
        for dependency in chunk.dependencies:
            confirmed_index = confirmed_tool_index.get(dependency)
            if confirmed_index is None or generating_index is None or generating_index < confirmed_index:
                violations.append(f"{chunk.chunk_id} spoke before '{dependency}' was confirmed")

    checks = {"dependent_chunks_only_speak_confirmed_facts": not violations}

    speaker_first = first_absolute(dec_trace, "first_token", "speaker")
    tool_completed_abs = first_absolute(rt_trace, "tool_completed")
    observations: Dict[str, Any] = {
        "tool_call_emitted": bool(tool_call_events),
        "dependency_violations": violations,
        "blocked_chunk_ids": [c.chunk_id for c in result.chunks if c.state.value == "blocked"],
        "speaker_first_token_before_tool_complete": (
            None
            if speaker_first is None or tool_completed_abs is None
            else speaker_first < tool_completed_abs
        ),
    }

    return checks, observations


async def run_single(
    backbone: SharedQwenBackbone,
    scenario: Scenario,
    latency_ms: int,
    speaker_weight: int,
    planner_weight: int,
) -> Dict[str, Any]:
    engine = QwenTokenStepEngine(backbone)
    scheduler = InterleavedDecodeScheduler(
        engine, speaker_weight=speaker_weight, planner_weight=planner_weight
    )
    recorder = RecordingBackend(scheduler)
    planner = QwenPlannerAdapter(recorder, tools=list(scenario.tools))
    runtime = DualSessionRuntime(
        planner,
        QwenSpeakerAdapter(recorder),
        AllowlistToolExecutor(
            scenario.tools, FakeWeatherTool(latency_ms=latency_ms, fail=scenario.tool_fail)
        ),
    )

    run_started = time.monotonic_ns()
    result = await runtime.run(f"stage3-{scenario.name}-{latency_ms}", scenario.request)
    runtime_ms = (time.monotonic_ns() - run_started) / 1_000_000

    rt_trace = runtime_trace(result)
    dec_trace = scheduler_trace(scheduler)
    checks, observations = evaluate_run(result, rt_trace, dec_trace)

    speaker_first = first_absolute(dec_trace, "first_token", "speaker")
    tool_completed_abs = first_absolute(rt_trace, "tool_completed")

    return {
        "scenario": scenario.name,
        "tool_latency_ms": latency_ms,
        "request": scenario.request,
        "runtime_ms": runtime_ms,
        "result_text": result.text,
        "checks": checks,
        "observations": observations,
        "passed": all(checks.values()),
        "repaired_sequence_count": (
            planner.last_parser.repaired_count if planner.last_parser is not None else 0
        ),
        "speaker_first_token_minus_tool_complete_ms": (
            None
            if speaker_first is None or tool_completed_abs is None
            else (speaker_first - tool_completed_abs) / 1_000_000
        ),
        "semantic_events": [
            {"sequence": e.sequence, "kind": e.kind.value, "payload": dict(e.payload)}
            for e in result.semantic_events
        ],
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "state": chunk.state.value,
                "dependencies": sorted(chunk.dependencies),
            }
            for chunk in result.chunks
        ],
        "runtime_timeline": rt_trace,
        "decode_timeline": dec_trace,
    }


def build_plan(latencies_ms: List[int]) -> List[Dict[str, Any]]:
    plan = []
    for scenario in SCENARIOS:
        latency_values = latencies_ms if scenario.sweep_latency else [latencies_ms[0]]
        for latency_ms in latency_values:
            plan.append({"scenario": scenario, "latency_ms": latency_ms})
    return plan


async def run_experiment(args: argparse.Namespace, output_dir: Path) -> int:
    latencies_ms = [int(value) for value in args.tool_latency_ms.split(",") if value.strip()]
    report: Dict[str, Any] = {
        "experiment": "qwen_stage3_latency_sweep_and_scenarios",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tool_latencies_ms": latencies_ms,
        "scenarios": [scenario.name for scenario in SCENARIOS],
        "scheduler_weights": {"speaker": args.speaker_weight, "planner": args.planner_weight},
        "environment": environment_report(),
        "status": "started",
        "runs": [],
    }
    write_json(output_dir / "report.json", report)

    backbone = SharedQwenBackbone(
        QwenBackboneConfig(
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

        plan = build_plan(latencies_ms)
        for item in plan:
            scenario: Scenario = item["scenario"]
            latency_ms: int = item["latency_ms"]
            try:
                run_report = await run_single(
                    backbone,
                    scenario,
                    latency_ms,
                    args.speaker_weight,
                    args.planner_weight,
                )
            except BaseException as error:  # noqa: BLE001 - a single run must not abort the sweep
                run_report = {
                    "scenario": scenario.name,
                    "tool_latency_ms": latency_ms,
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            report["runs"].append(run_report)
            write_json(output_dir / "report.json", report)

        total = len(report["runs"])
        passed = sum(1 for run in report["runs"] if run.get("passed"))
        report["summary"] = {"total_runs": total, "passed_runs": passed, "failed_runs": total - passed}
        report["status"] = "passed" if passed == total else "failed"
        return 0 if passed == total else 1
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
    parser.add_argument("--tool-latency-ms", default="3000,1500,750,300")
    parser.add_argument("--speaker-weight", type=int, default=3)
    parser.add_argument("--planner-weight", type=int, default=2)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="artifacts/colab-stage3")
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
    print(f"AETHER_STAGE3_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE3_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
