from dataclasses import dataclass
from typing import AsyncIterator, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class GenerationRequest:
    """Provider-neutral request for one logical decoding session."""

    session_id: str
    role: str
    messages: Sequence[Mapping[str, str]]
    settings: GenerationSettings = GenerationSettings()

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if self.role not in {"planner", "speaker"}:
            raise ValueError(f"unsupported generation role: {self.role}")
        if not self.messages:
            raise ValueError("messages must not be empty")


class TextGenerationBackend(Protocol):
    """One shared model object capable of serving independent session ids."""

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        ...

