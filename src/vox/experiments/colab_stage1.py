"""Stage 1 Qwen smoke test with diagnostics suitable for a remote notebook.

This module never downloads a model unless the caller passes --allow-download.
It writes a report even when model loading, parsing or generation fails.
"""

import argparse
import asyncio
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

from vox.model.generation import GenerationRequest, TextGenerationBackend
from vox.model.qwen_adapters import QwenPlannerAdapter, QwenSpeakerAdapter
from vox.model.qwen_backbone import QwenBackboneConfig, SharedQwenBackbone
from vox.runtime.dual_session import DualSessionRuntime
from vox.testing.fakes import FakeWeatherTool


class RecordingBackend:
    """Transparent backend decorator that records raw model output and latency."""

    def __init__(self, backend: TextGenerationBackend) -> None:
        self.backend = backend
        self.generations: List[Dict[str, Any]] = []

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        started_ns = time.monotonic_ns()
        chunks: List[str] = []
        error = ""
        try:
            async for text in self.backend.stream(request):
                chunks.append(text)
                yield text
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            self.generations.append(
                {
                    "session_id": request.session_id,
                    "role": request.role,
                    "duration_ms": (time.monotonic_ns() - started_ns) / 1_000_000,
                    "settings": asdict(request.settings),
                    "messages": list(request.messages),
                    "chunks": chunks,
                    "text": "".join(chunks),
                    "error": error,
                }
            )


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def environment_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "accelerate", "safetensors", "peft")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [],
        }
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        report["cuda"] = cuda
    except Exception as error:
        report["cuda"] = {"available": False, "inspection_error": repr(error)}
    return report


def timeline_payload(result: Any) -> List[Dict[str, Any]]:
    return [
        {
            "name": event.name,
            "timestamp_ns": event.timestamp_ns,
            "timestamp_ms": event.timestamp_ns / 1_000_000,
            "attributes": event.attributes,
        }
        for event in result.timeline.events
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_experiment(args: argparse.Namespace, output_dir: Path) -> int:
    report: Dict[str, Any] = {
        "experiment": "qwen_stage1_weather_smoke",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "request": args.request,
        "tool_latency_ms": args.tool_latency_ms,
        "environment": environment_report(),
        "status": "started",
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
    recorder = RecordingBackend(backbone)

    try:
        load_started_ns = time.monotonic_ns()
        backbone.load()
        report["model_load_ms"] = (time.monotonic_ns() - load_started_ns) / 1_000_000

        runtime = DualSessionRuntime(
            QwenPlannerAdapter(recorder),
            QwenSpeakerAdapter(recorder),
            FakeWeatherTool(latency_ms=args.tool_latency_ms),
        )
        run_started_ns = time.monotonic_ns()
        result = await runtime.run("colab-weather-1", args.request)
        report["runtime_ms"] = (time.monotonic_ns() - run_started_ns) / 1_000_000
        report["status"] = "passed"
        report["result_text"] = result.text
        report["semantic_events"] = [
            {
                "turn_id": event.turn_id,
                "sequence": event.sequence,
                "kind": event.kind.value,
                "payload": dict(event.payload),
            }
            for event in result.semantic_events
        ]
        report["chunks"] = [
            {
                "chunk_id": chunk.chunk_id,
                "goal": chunk.goal,
                "dependencies": sorted(chunk.dependencies),
                "state": chunk.state.value,
                "turn_id": chunk.turn_id,
            }
            for chunk in result.chunks
        ]
        report["timeline"] = timeline_payload(result)
        report["session_request_counts"] = backbone.session_request_counts
        return 0
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        stack = traceback.format_exc()
        report["traceback"] = stack
        (output_dir / "traceback.txt").write_text(stack, encoding="utf-8")
        return 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["raw_generations"] = recorder.generations
        write_json(output_dir / "report.json", report)
        with (output_dir / "raw_generations.jsonl").open("w", encoding="utf-8") as handle:
            for generation in recorder.generations:
                handle.write(json.dumps(generation, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--request", default="Какая погода в Алматы и нужен ли зонт?")
    parser.add_argument("--tool-latency-ms", type=int, default=1500)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="artifacts/colab-stage1")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly permit Hugging Face model downloads in the target environment",
    )
    args = parser.parse_args()
    if not args.allow_download:
        parser.error(
            "model download is disabled by default; pass --allow-download only in the target ML environment"
        )
    return args


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exit_code = asyncio.run(run_experiment(args, output_dir))
    print(f"VOX_STAGE1_STATUS={'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"VOX_STAGE1_REPORT={output_dir / 'report.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

