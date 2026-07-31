import asyncio
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Mapping, Optional, Sequence

from aether.model.generation import GenerationRequest


@dataclass(frozen=True)
class LLMBackboneConfig:
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


class SharedLLMBackbone:
    """Lazily loaded LLM shared by independent logical sessions.

    The default configuration forbids network downloads. Importing this module,
    constructing the class and running unit tests do not import Transformers or
    touch model files. `load()` must be called explicitly in an ML environment.

    This first adapter serializes Hugging Face `generate` calls with a lock.
    True two-KV-cache concurrent decoding belongs to the next custom scheduler
    experiment; the public adapter contract will remain the same.
    """

    def __init__(self, config: LLMBackboneConfig) -> None:
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
            raise RuntimeError("LLM backbone is not loaded")
        return self._tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("LLM backbone is not loaded")
        return self._model

    @property
    def hidden_size(self) -> int:
        if self._model is None:
            raise RuntimeError("LLM backbone is not loaded")
        return int(self._model.config.hidden_size)

    def encode_hidden_state(self, text: str) -> "list[float]":
        """Mean-pooled last-layer hidden state for ``text``, as a plain list.

        This is a plain forward pass with ``output_hidden_states=True`` —
        deliberately separate from ``stream()``'s threaded ``generate()``
        call, so this addition cannot affect the already-tested streaming
        decode path. It exists for Stage 5 (Level B hidden-state bridge):
        given the same text the Speaker produced, recover the internal
        state that produced it, to condition a Voice Head on state instead
        of the decoded string.
        """
        if not self.loaded:
            raise RuntimeError("LLM backbone is not loaded; call load() explicitly")
        import torch

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)
        pooled = last_hidden.mean(dim=1).squeeze(0)
        return pooled.float().cpu().tolist()

    def generate_with_soft_prompt(
        self,
        messages: Sequence[Mapping[str, str]],
        soft_prompt_embeddings: Optional[Any] = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Greedy-decodes a response with an optional soft prompt prepended.

        ``soft_prompt_embeddings`` (if given): a ``(num_soft_tokens,
        hidden_size)`` tensor — e.g. from
        ``aether.model.thought_bridge.ThoughtBridge.project(...)`` — spliced
        in as extra input embeddings *before* the tokenized prompt, instead
        of any additional token ids. ``None`` runs the plain baseline (no
        injection), for structural-probe comparison.

        This is a plain, synchronous, greedy generate call — deliberately
        separate from ``stream()``'s threaded streaming path, so this
        addition cannot affect the already-tested streaming decode logic.
        Stage 6 (docs/plan.md Level B): the point is to prove this channel
        changes generation at all, not to produce a good response — no
        training happens here, and none should be expected to matter yet.
        """
        if not self.loaded:
            raise RuntimeError("LLM backbone is not loaded; call load() explicitly")
        import torch

        prompt = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_embeddings = self._model.get_input_embeddings()(inputs["input_ids"])
        attention_mask = inputs["attention_mask"]

        if soft_prompt_embeddings is not None:
            soft = soft_prompt_embeddings.to(
                device=self._model.device, dtype=input_embeddings.dtype
            ).unsqueeze(0)
            input_embeddings = torch.cat([soft, input_embeddings], dim=1)
            soft_mask = torch.ones(
                (1, soft.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        with torch.no_grad():
            # With `inputs_embeds`, generate() returns only the newly
            # generated token ids (there is no token-id form of the prompt
            # to echo back), so no slicing is needed before decoding.
            output_ids = self._model.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
            raise RuntimeError("LLM backbone is not loaded; call load() explicitly")
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
