"""Stage 6 -- structural probe: does a Planner "thought" hidden state,
injected as a soft prompt, measurably change Speaker's generation?

Reframing from Stage 5 (see docs/reports/technical_report_03.md and the
internal Stage 6 roadmap): Stage 5's hidden state was extracted from the
Speaker's OWN already-decided text -- circular, not "thought before
speech". This stage instead:

  1. Has the Planner write an extended internal "thought" (organizing what
     to say -- facts, structure, tone -- never shown to the user, and
     deliberately not the constrained JSONL grammar; that grammar's
     `speech_plan.goal` is a short safety-relevant label, not a thinking
     space).
  2. Extracts a hidden state for that thought
     (`SharedLLMBackbone.encode_hidden_state`).
  3. Projects it into a "soft prompt" (`ThoughtBridge`, untrained --
     randomly initialized, same structural-probe spirit as Stage 4's
     `MinimalVoiceHead`) and splices it into Speaker's input embeddings
     (`SharedLLMBackbone.generate_with_soft_prompt`).

Scope, deliberately: this asks only whether the injection channel has any
measurable effect on generation (deterministic given the same soft prompt;
changes output relative to no injection; changes output when a DIFFERENT
thought's soft prompt is swapped in, proving the channel carries
request-specific content, not generic noise). It does not ask whether the
conditioned response is *better* -- the bridge is untrained. That is
Stage 7 (training) and Stage 8 (evaluation), not this stage.
"""

import argparse
import asyncio
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from aether.experiments.colab_stage1 import environment_report, write_json
from aether.model.generation import GenerationRequest, GenerationSettings
from aether.model.llm_backbone import LLMBackboneConfig, SharedLLMBackbone
from aether.model.thought_bridge import ThoughtBridge, ThoughtBridgeConfig

_THOUGHT_SYSTEM_PROMPT = """You are the internal planning stage of a voice assistant.
Given the user's request, write a short internal plan for how to answer it:
key facts to include, structure, and tone. Do not write the final answer.
Do not use JSON. Do not address the user directly. This text is never
shown to the user."""

_SPEAKER_SYSTEM_PROMPT = """You are a voice assistant. Answer the user's
request directly, briefly, and naturally."""


@dataclass(frozen=True)
class Scenario:
    name: str
    request: str


SCENARIOS: List[Scenario] = [
    Scenario("weather", "What's the weather like in Almaty right now?"),
    Scenario("greeting", "Hello! How are you doing today?"),
]


async def generate_thought(
    backbone: SharedLLMBackbone, scenario_name: str, request: str, max_new_tokens: int
) -> str:
    messages = (
        {"role": "system", "content": _THOUGHT_SYSTEM_PROMPT},
        {"role": "user", "content": request},
    )
    generation = GenerationRequest(
        session_id=f"thought-{scenario_name}",
        role="planner",
        messages=messages,
        settings=GenerationSettings(max_new_tokens=max_new_tokens),
    )
    pieces = []
    async for piece in backbone.stream(generation):
        pieces.append(piece)
    return "".join(pieces).strip()


def speaker_messages(request: str) -> tuple:
    return (
        {"role": "system", "content": _SPEAKER_SYSTEM_PROMPT},
        {"role": "user", "content": request},
    )


async def run_experiment(args: argparse.Namespace, output_dir: Path) -> int:
    report: Dict[str, Any] = {
        "experiment": "planner_thought_soft_prompt_structural_probe",
        "scope_note": (
            "Structural probe only: does injecting a Planner thought's hidden state as a "
            "soft prompt measurably change Speaker's generation? The bridge is untrained "
            "(randomly initialized) -- this does not test whether the conditioned response "
            "is better, only whether the channel has any request-specific effect at all."
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "scenarios": [s.name for s in SCENARIOS],
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
        report["hidden_state_dim"] = backbone.hidden_size

        bridge = ThoughtBridge(
            ThoughtBridgeConfig(
                hidden_state_dim=backbone.hidden_size,
                embedding_dim=backbone.hidden_size,
                num_soft_tokens=args.num_soft_tokens,
                device=args.bridge_device,
            )
        )
        bridge.load()

        per_scenario: Dict[str, Any] = {}
        soft_prompts: Dict[str, Any] = {}
        for scenario in SCENARIOS:
            thought = await generate_thought(
                backbone, scenario.name, scenario.request, args.thought_max_tokens
            )
            hidden_state = backbone.encode_hidden_state(thought)
            soft_prompts[scenario.name] = bridge.project(hidden_state)
            per_scenario[scenario.name] = {"request": scenario.request, "thought": thought}
        write_json(output_dir / "report.json", report)

        scenario_names = [s.name for s in SCENARIOS]
        for index, scenario in enumerate(SCENARIOS):
            messages = speaker_messages(scenario.request)
            own_soft_prompt = soft_prompts[scenario.name]
            other_name = scenario_names[(index + 1) % len(scenario_names)]
            other_soft_prompt = soft_prompts[other_name]

            baseline = backbone.generate_with_soft_prompt(
                messages, soft_prompt_embeddings=None, max_new_tokens=args.speaker_max_tokens
            )
            conditioned = backbone.generate_with_soft_prompt(
                messages, soft_prompt_embeddings=own_soft_prompt, max_new_tokens=args.speaker_max_tokens
            )
            conditioned_repeat = backbone.generate_with_soft_prompt(
                messages, soft_prompt_embeddings=own_soft_prompt, max_new_tokens=args.speaker_max_tokens
            )
            cross_conditioned = backbone.generate_with_soft_prompt(
                messages, soft_prompt_embeddings=other_soft_prompt, max_new_tokens=args.speaker_max_tokens
            )

            checks = {
                "deterministic_given_same_soft_prompt": conditioned == conditioned_repeat,
                "soft_prompt_changes_output_vs_baseline": conditioned != baseline,
                "different_thought_changes_output": conditioned != cross_conditioned,
            }
            per_scenario[scenario.name].update(
                {
                    "baseline_response": baseline,
                    "conditioned_response": conditioned,
                    "conditioned_response_repeat": conditioned_repeat,
                    "cross_conditioned_response": cross_conditioned,
                    "cross_conditioned_from": other_name,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            write_json(output_dir / "report.json", report)

        report["results"] = per_scenario
        report["checks"] = {
            name: result["passed"] for name, result in per_scenario.items()
        }
        report["passed"] = all(report["checks"].values())
        report["status"] = "passed" if report["passed"] else "failed"
        return 0 if report["passed"] else 1
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
    parser.add_argument("--num-soft-tokens", type=int, default=1)
    parser.add_argument("--thought-max-tokens", type=int, default=150)
    parser.add_argument("--speaker-max-tokens", type=int, default=100)
    parser.add_argument("--bridge-device", default="cpu")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="artifacts/colab-stage6")
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
    print(f"AETHER_STAGE6_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"AETHER_STAGE6_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
