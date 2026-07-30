import asyncio
from dataclasses import dataclass, field
from typing import Any, List

from aether.model.generation import GenerationRequest
from aether.model.llm_backbone import SharedLLMBackbone
from aether.model.step_scheduler import DecodeStep


@dataclass
class LLMDecodeState:
    request: GenerationRequest
    input_ids: Any
    attention_mask: Any
    past_key_values: Any = None
    generated_token_ids: List[int] = field(default_factory=list)
    rendered_text: str = ""
    closed: bool = False


class LLMTokenStepEngine:
    """One-token LLM forward engine with a separate KV-cache per state."""

    def __init__(self, backbone: SharedLLMBackbone) -> None:
        if not backbone.loaded:
            raise RuntimeError("backbone must be explicitly loaded before creating step engine")
        self._backbone = backbone

    async def create(self, request: GenerationRequest) -> LLMDecodeState:
        tokenizer = self._backbone.tokenizer
        model = self._backbone.model
        prompt = tokenizer.apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._backbone.config.enable_thinking,
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        return LLMDecodeState(
            request=request,
            input_ids=encoded["input_ids"].to(model.device),
            attention_mask=encoded["attention_mask"].to(model.device),
        )

    async def step(self, state: LLMDecodeState) -> DecodeStep:
        if state.closed:
            raise RuntimeError("decode state is closed")
        return await asyncio.to_thread(self._step_sync, state)

    async def close(self, state: LLMDecodeState) -> None:
        state.closed = True
        state.past_key_values = None

    def _step_sync(self, state: LLMDecodeState) -> DecodeStep:
        import torch

        model = self._backbone.model
        tokenizer = self._backbone.tokenizer
        if state.past_key_values is None:
            step_input_ids = state.input_ids
            attention_mask = state.attention_mask
        else:
            step_input_ids = state.input_ids[:, -1:]
            total_length = state.attention_mask.shape[1] + 1
            attention_mask = state.attention_mask.new_ones((1, total_length))

        with torch.inference_mode():
            outputs = model(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        state.past_key_values = outputs.past_key_values
        state.generated_token_ids.append(next_token)
        state.input_ids = torch.tensor([[next_token]], device=model.device)
        state.attention_mask = attention_mask

        rendered = tokenizer.decode(state.generated_token_ids, skip_special_tokens=True)
        if rendered.startswith(state.rendered_text):
            delta = rendered[len(state.rendered_text) :]
        else:
            # Tokenizer normalization can occasionally rewrite the tail.
            common = 0
            for old, new in zip(state.rendered_text, rendered):
                if old != new:
                    break
                common += 1
            delta = rendered[common:]
        state.rendered_text = rendered

        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids or [])
        finished = (
            next_token in eos_ids
            or len(state.generated_token_ids) >= state.request.settings.max_new_tokens
        )
        return DecodeStep(text=delta, finished=finished, token_count=1)

