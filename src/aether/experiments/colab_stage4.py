"""Stage 4 — minimal Voice Head structural probe (Phase C, docs/plan.md).

This experiment does NOT test audio quality, prosody or emotion — that is
explicitly out of scope (see MinimalVoiceHead's docstring). It tests exactly
one architectural claim: that the existing commit horizon (``ChunkState``'s
transition table, already proven for text in Stage 1-3) generalizes to an
audio modality without being modified. A single Mimi codebook is predicted
by an untrained, from-scratch transformer conditioned on the Speaker's text;
the pass/fail criteria are trace-derived timing/cancellation facts, not
listening quality. Real Mimi weights are loaded (frozen, pretrained) so the
decode path is real; the Voice Head itself is intentionally not trained.
"""

import argparse
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from aether.audio.codec import MimiCodec, MimiCodecConfig
from aether.experiments.colab_stage1 import RecordingBackend, environment_report, write_json
from aether.experiments.colab_stage2 import runtime_trace
from aether.experiments.colab_stage3 import SCENARIOS, Scenario, evaluate_run
from aether.model.llm_adapters import LLMPlannerAdapter, LLMSpeakerAdapter
from aether.model.llm_backbone import LLMBackboneConfig, SharedLLMBackbone
from aether.model.llm_step_engine import LLMTokenStepEngine
from aether.model.step_scheduler import InterleavedDecodeScheduler
from aether.model.voice_head import MinimalVoiceHead, MinimalVoiceHeadConfig
from aether.runtime.dual_session import DualSessionRuntime
from aether.runtime.tool_executor import AllowlistToolExecutor
from aether.testing.fakes import FakeWeatherTool

# Only the two scenarios that exercise a distinct audio-path claim: a
# dependent chunk (must not commit audio before its tool result) and a
# dependency-free, no-tool chunk (audio path with no gating at all).
STAGE4_SCENARIO_NAMES = {"weather_success", "no_tool_greeting"}
STAGE4_SCENARIOS: List[Scenario] = [s for s in SCENARIOS if s.name in STAGE4_SCENARIO_NAMES]

DEFAULT_LATENCIES_MS = [3000, 1500, 750, 300]


def write_wav(path: Path, pcm: Any, sample_rate: int) -> None:
    import struct
    import wave

    clipped = pcm.clip(-1.0, 1.0)
    pcm_int16 = (clipped * 32767.0).astype("int16")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(pcm_int16)}h", *pcm_int16.tolist()))


def evaluate_audio(result: Any, rt_trace: list) -> "Tuple[Dict[str, bool], Dict[str, Any]]":
    """Audio-path analogue of colab_stage3.evaluate_run's checks/observations split."""

    confirmed_tool_index: Dict[str, int] = {}
    for index, event in enumerate(rt_trace):
        if event["name"] == "tool_completed" and event["attributes"].get("succeeded"):
            confirmed_tool_index.setdefault(event["attributes"].get("tool"), index)

    violations = []
    played_chunks = [chunk for chunk in result.chunks if chunk.state.value == "played"]
    for chunk in played_chunks:
        if not chunk.dependencies:
            continue
        audio_buffered_index = next(
            (
                i
                for i, e in enumerate(rt_trace)
                if e["name"] == "chunk_audio_buffered" and e["attributes"].get("chunk_id") == chunk.chunk_id
            ),
            None,
        )
        for dependency in chunk.dependencies:
            confirmed_index = confirmed_tool_index.get(dependency)
            if confirmed_index is None or audio_buffered_index is None or audio_buffered_index < confirmed_index:
                violations.append(f"{chunk.chunk_id} audio buffered before '{dependency}' was confirmed")

    missing_audio = [chunk.chunk_id for chunk in played_chunks if chunk.chunk_id not in result.audio]

    checks = {
        "dependent_chunk_audio_never_buffered_before_facts": not violations,
        "audio_generated_for_every_played_chunk": not missing_audio,
    }
    observations: Dict[str, Any] = {
        "audio_violations": violations,
        "chunks_missing_audio": missing_audio,
        "audio_token_counts": {
            chunk_id: len(audio_chunk.tokens) for chunk_id, audio_chunk in result.audio.items()
        },
    }
    return checks, observations


async def run_single(
    backbone: SharedLLMBackbone,
    voice_head: MinimalVoiceHead,
    mimi: MimiCodec,
    scenario: Scenario,
    latency_ms: int,
    speaker_weight: int,
    planner_weight: int,
    wav_dir: Path,
) -> Dict[str, Any]:
    engine = LLMTokenStepEngine(backbone)
    scheduler = InterleavedDecodeScheduler(
        engine, speaker_weight=speaker_weight, planner_weight=planner_weight
    )
    recorder = RecordingBackend(scheduler)
    planner = LLMPlannerAdapter(recorder, tools=list(scenario.tools))
    runtime = DualSessionRuntime(
        planner,
        LLMSpeakerAdapter(recorder),
        AllowlistToolExecutor(
            scenario.tools, FakeWeatherTool(latency_ms=latency_ms, fail=scenario.tool_fail)
        ),
        voice_head=voice_head,
    )

    run_id = f"stage4-{scenario.name}-{latency_ms}"
    run_started = time.monotonic_ns()
    result = await runtime.run(run_id, scenario.request)
    runtime_ms = (time.monotonic_ns() - run_started) / 1_000_000

    rt_trace = runtime_trace(result)
    text_checks, text_observations = evaluate_run(result, rt_trace, [])
    audio_checks, audio_observations = evaluate_audio(result, rt_trace)

    wav_paths: Dict[str, str] = {}
    for chunk_id, audio_chunk in result.audio.items():
        pcm = mimi.decode(audio_chunk.tokens)
        wav_path = wav_dir / f"{run_id}-{chunk_id}.wav"
        write_wav(wav_path, pcm, int(mimi.sample_rate))
        wav_paths[chunk_id] = str(wav_path)

    checks = {**text_checks, **audio_checks}
    return {
        "scenario": scenario.name,
        "tool_latency_ms": latency_ms,
        "request": scenario.request,
        "runtime_ms": runtime_ms,
        "result_text": result.text,
        "checks": checks,
        "passed": all(checks.values()),
        "observations": {**text_observations, **audio_observations},
        "wav_files": wav_paths,
        "chunks": [
            {"chunk_id": c.chunk_id, "state": c.state.value, "dependencies": sorted(c.dependencies)}
            for c in result.chunks
        ],
        "runtime_timeline": rt_trace,
    }


def build_plan(latencies_ms: List[int]) -> List[Dict[str, Any]]:
    plan = []
    for scenario in STAGE4_SCENARIOS:
        latency_values = latencies_ms if scenario.sweep_latency else [latencies_ms[0]]
        for latency_ms in latency_values:
            plan.append({"scenario": scenario, "latency_ms": latency_ms})
    return plan


async def run_experiment(args: argparse.Namespace, output_dir: Path) -> int:
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    latencies_ms = [int(value) for value in args.tool_latency_ms.split(",") if value.strip()]

    report: Dict[str, Any] = {
        "experiment": "voice_head_stage4_minimal_audio_probe",
        "scope_note": (
            "Structural probe only: does ChunkState's commit horizon generalize to "
            "audio without modification? Audio quality/prosody/emotion are explicitly "
            "out of scope; the Voice Head is untrained by design."
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tool_latencies_ms": latencies_ms,
        "scenarios": [s.name for s in STAGE4_SCENARIOS],
        "environment": environment_report(),
        "status": "started",
        "runs": [],
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
    voice_head = MinimalVoiceHead(
        MinimalVoiceHeadConfig(
            vocab_size=args.voice_head_vocab_size,
            max_audio_tokens=args.voice_head_max_tokens,
            device=args.voice_head_device,
        )
    )
    mimi = MimiCodec(
        MimiCodecConfig(device=args.mimi_device, num_codebooks=1, allow_download=args.allow_download)
    )

    try:
        load_started = time.monotonic_ns()
        backbone.load()
        voice_head.load()
        mimi.load()
        report["model_load_ms"] = (time.monotonic_ns() - load_started) / 1_000_000
        report["mimi_sample_rate"] = mimi.sample_rate
        report["mimi_frame_rate"] = mimi.frame_rate

        plan = build_plan(latencies_ms)
        for item in plan:
            scenario: Scenario = item["scenario"]
            latency_ms: int = item["latency_ms"]
            try:
                run_report = await run_single(
                    backbone,
                    voice_head,
                    mimi,
                    scenario,
                    latency_ms,
                    args.speaker_weight,
                    args.planner_weight,
                    wav_dir,
                )
            except BaseException as error:  # noqa: BLE001 - one run must not abort the sweep
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
    parser.add_argument("--voice-head-device", default="cpu")
    parser.add_argument("--voice-head-vocab-size", type=int, default=2048)
    parser.add_argument("--voice-head-max-tokens", type=int, default=32)
    parser.add_argument("--mimi-device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/colab-stage4")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("pass --allow-download only in the target ML environment")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import asyncio

    exit_code = asyncio.run(run_experiment(args, output_dir))
    print(f"AETHER_STAGE4_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE4_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
