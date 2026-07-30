"""Backbone and generation adapter contracts."""

from vox.model.generation import GenerationRequest, GenerationSettings, TextGenerationBackend
from vox.model.qwen_adapters import QwenPlannerAdapter, QwenSpeakerAdapter

__all__ = [
    "GenerationRequest",
    "GenerationSettings",
    "QwenPlannerAdapter",
    "QwenSpeakerAdapter",
    "TextGenerationBackend",
]

