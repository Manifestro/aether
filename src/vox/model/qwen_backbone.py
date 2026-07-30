import asyncio
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

from vox.model.generation import GenerationRequest


@dataclass(frozen=True)
class QwenBackboneConfig:
    model_path: str
    device_map: str = "auto"
    dtype: str = "auto"
    allow_download: bool = False
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")


def _next_or_none(iterator: Any) -> Optional[str]:
    try:
        return next(iterator)
    except StopIteration:
        return None


class SharedQwenBackbone:
    """Lazily loaded Qwen model shared by independent logical sessions.

    The default configuration forbids network downloads. Importing this module,
    constructing the class and running unit tests do not import Transformers or
    touch model files. `load()` must be called explicitly in an ML environment.

    This first adapter serializes Hugging Face `generate` calls with a lock.
    True two-KV-cache concurrent decoding belongs to the next custom scheduler
    experiment; the public adapter contract will remain the same.
    """

    def __init__(self, config: QwenBackboneConfig) -> None:
        self.config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._generate_lock: Optional[asyncio.Lock] = None
        self._session_requests: Dict[str, int] = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def session_request_counts(self) -> Dict[str, int]:
        return dict(self._session_requests)

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Qwen backbone is not loaded")
        return self._tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Qwen backbone is not loaded")
        return self._model

    def load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "ML dependencies are not installed; install the 'ml' extra in the target environment"
            ) from error

        local_only = not self.config.allow_download
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=local_only,
        )
        dtype = self.config.dtype
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map=self.config.device_map,
            torch_dtype=torch_dtype,
            local_files_only=local_only,
        )
        self._model.eval()

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        if not self.loaded:
            raise RuntimeError("Qwen backbone is not loaded; call load() explicitly")
        if self._generate_lock is None:
            self._generate_lock = asyncio.Lock()
        self._session_requests[request.session_id] = (
            self._session_requests.get(request.session_id, 0) + 1
        )

        async with self._generate_lock:
            from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

            stop_requested = threading.Event()

            class StopWhenRequested(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    return stop_requested.is_set()

            prompt = self._tokenizer.apply_chat_template(
                list(request.messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            kwargs = dict(inputs)
            kwargs.update(
                streamer=streamer,
                max_new_tokens=request.settings.max_new_tokens,
                do_sample=request.settings.temperature > 0,
                stopping_criteria=StoppingCriteriaList([StopWhenRequested()]),
            )
            if request.settings.temperature > 0:
                kwargs.update(
                    temperature=request.settings.temperature,
                    top_p=request.settings.top_p,
                )

            worker = threading.Thread(target=self._model.generate, kwargs=kwargs, daemon=True)
            worker.start()
            iterator = iter(streamer)
            loop = asyncio.get_running_loop()
            try:
                while True:
                    piece = await loop.run_in_executor(None, _next_or_none, iterator)
                    if piece is None:
                        break
                    yield piece
            finally:
                stop_requested.set()
                worker.join()
