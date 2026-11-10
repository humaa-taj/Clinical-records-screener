"""Helpers shared by the decoders and benchmarks."""
import time
from dataclasses import dataclass, field

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from screener.data.prepare import load_split, render_prompt


@dataclass
class DecodeResult:
    tokens: list
    input_length: int
    prefill_seconds: float
    decode_seconds: float
    cache_lengths: list = field(default_factory=list)
    cache_bytes: list = field(default_factory=list)
    step_logits: list = field(default_factory=list)

    @property
    def total_seconds(self):
        return self.prefill_seconds + self.decode_seconds

    @property
    def decode_tokens_per_second(self):
        """Tokens made after prefill. The first new token comes from the prefill pass."""
        steps = len(self.tokens) - 1
        return steps / self.decode_seconds if steps > 0 and self.decode_seconds > 0 else 0.0

    @property
    def total_tokens_per_second(self):
        return len(self.tokens) / self.total_seconds if self.total_seconds > 0 else 0.0


def clock(device):
    """Read the time after queued GPU work has finished, so GPU time is not missed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def positions_from_mask(attention_mask):
    """Number the real tokens 0, 1, 2... Left padding has no effect on these positions."""
    return (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)


def load_for_inference(model_path, adapter=None, device="cuda", dtype=torch.float16):
    """Load a model and tokenizer, merging a LoRA adapter into the weights if one is given."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, local_files_only=True).to(device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tokenizer


def benchmark_prompt(tokenizer, length):
    """Exactly `length` tokens of real training-split prompts. Never touches the test split."""
    tokens = []
    for example in load_split("train"):
        tokens.extend(tokenizer(render_prompt(tokenizer, example), add_special_tokens=False)["input_ids"])
        if len(tokens) >= length:
            return tokens[:length]
    raise ValueError(f"not enough training text for {length} tokens")
