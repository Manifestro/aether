"""Discovery spike: can Moshi act as a teacher that gives us Mimi audio
tokens for an arbitrary piece of given English text?

This is deliberately NOT a proper experiment stage. Moshi's public
`moshi` package is built around a full-duplex, streaming *conversational*
loop (its demo client/server feed it live audio frames and read live audio
frames back) — not a "text in, audio tokens out" TTS call. Nobody on this
project has verified from inside this repository which, if any, of Moshi's
public entry points can be repurposed to produce audio tokens for a text we
choose, so guessing a specific call chain and wiring a whole distillation
trainer around it would risk building on a wrong assumption.

Instead this script:
  1. Loads Moshi's LM (real weights) defensively.
  2. Dumps its public surface (attributes, method signatures, docstrings)
     to the report, so the real shape of the API is known, not guessed.
  3. Tries a small number of plausible invocations, each independently
     guarded, and records exactly what happened for each — success,
     failure, and (on success) the shape of whatever came back.

Nothing here is a claim that a given call is "the" way to use Moshi as a
teacher. It is raw material for deciding the next step once the real
report comes back from Colab.
"""

import argparse
import inspect
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from aether.experiments.colab_stage1 import environment_report, write_json


def describe_object(obj: Any, max_members: int = 60) -> Dict[str, Any]:
    members: List[Dict[str, Any]] = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception as error:  # noqa: BLE001 - inspection must not crash the spike
            members.append({"name": name, "inspection_error": repr(error)})
            continue
        entry: Dict[str, Any] = {"name": name, "type": type(attr).__name__}
        if callable(attr):
            try:
                entry["signature"] = str(inspect.signature(attr))
            except (TypeError, ValueError):
                entry["signature"] = None
            entry["doc"] = inspect.getdoc(attr)
        members.append(entry)
        if len(members) >= max_members:
            break
    return {"type": type(obj).__name__, "module": type(obj).__module__, "members": members}


def try_attempt(name: str, fn: Any) -> Dict[str, Any]:
    started = time.monotonic_ns()
    try:
        result = fn()
        return {
            "attempt": name,
            "succeeded": True,
            "duration_ms": (time.monotonic_ns() - started) / 1_000_000,
            "result_repr": repr(result)[:2000],
            "result_type": type(result).__name__,
        }
    except BaseException as error:  # noqa: BLE001 - every attempt must be independently recorded
        return {
            "attempt": name,
            "succeeded": False,
            "duration_ms": (time.monotonic_ns() - started) / 1_000_000,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def run_spike(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "spike": "moshi_teacher_feasibility",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_report(),
        "text": args.text,
        "attempts": [],
    }

    try:
        from moshi.models import loaders
    except ImportError as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        return report

    report["loaders_module_surface"] = describe_object(loaders)

    checkpoint_attempt = try_attempt(
        "CheckpointInfo.from_hf_repo",
        lambda: loaders.CheckpointInfo.from_hf_repo(args.hf_repo),
    )
    report["attempts"].append(checkpoint_attempt)
    if not checkpoint_attempt["succeeded"]:
        report["status"] = "failed_at_checkpoint_info"
        return report

    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    report["checkpoint_info_surface"] = describe_object(checkpoint_info)

    mimi_attempt = try_attempt("checkpoint_info.get_mimi", lambda: checkpoint_info.get_mimi(device=args.device))
    report["attempts"].append(mimi_attempt)

    lm_attempt = try_attempt(
        "checkpoint_info.get_moshi_lm", lambda: checkpoint_info.get_moshi_lm(device=args.device)
    )
    report["attempts"].append(lm_attempt)
    if not lm_attempt["succeeded"]:
        report["status"] = "failed_at_lm_load"
        return report

    lm = checkpoint_info.get_moshi_lm(device=args.device)
    report["lm_surface"] = describe_object(lm)

    text_tokenizer_attempt = try_attempt(
        "checkpoint_info.get_text_tokenizer", lambda: checkpoint_info.get_text_tokenizer()
    )
    report["attempts"].append(text_tokenizer_attempt)

    lm_gen_holder: Dict[str, Any] = {}

    def _build_lm_gen() -> Any:
        from moshi.models import LMGen

        instance = LMGen(lm)
        lm_gen_holder["instance"] = instance
        return instance

    lm_gen_attempt = try_attempt("moshi.models.LMGen(lm)", _build_lm_gen)
    report["attempts"].append(lm_gen_attempt)
    if lm_gen_attempt["succeeded"]:
        report["lm_gen_surface"] = describe_object(lm_gen_holder["instance"])

    report["status"] = "completed"
    report["conclusion_note"] = (
        "This spike does not conclude Moshi can or cannot serve as a teacher for "
        "arbitrary text. It records what its real API surface looks like on the "
        "installed version. Read `lm_surface`/`lm_gen_surface`/`checkpoint_info_surface` "
        "for method names and signatures that plausibly accept a text prompt or a "
        "token-forcing argument, and treat those as the next thing to try by hand."
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text", default="The weather in Almaty is rainy and twenty four degrees.")
    parser.add_argument("--output-dir", default="artifacts/spike-moshi-teacher")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("pass --allow-download only in the target ML environment")
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_spike(args, output_dir)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "report.json", report)
    print(f"AETHER_SPIKE_STATUS={report.get('status')}")
    print(f"AETHER_SPIKE_REPORT={output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
