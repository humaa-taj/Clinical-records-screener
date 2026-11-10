"""Greedy prediction with fixed decoding settings, shared by every before/after comparison.

Qwen's bundled generation settings include sampling and a repetition penalty, so the settings are
stated explicitly here. A repetition penalty would also break equality with the custom decoders.
"""
from contextlib import nullcontext

import torch
from transformers import GenerationConfig

from screener.data.prepare import render_prompt
from screener.eval.verifier import verify_output


def eos_token_ids(model, tokenizer):
    ids = {tokenizer.eos_token_id}
    configured = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif configured:
        ids.update(configured)
    return sorted(i for i in ids if i is not None)


def greedy_config(model, tokenizer, max_new_tokens):
    return GenerationConfig(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_ids(model, tokenizer),
        use_cache=True,
    )


def decoding_settings(model, tokenizer, max_new_tokens):
    """The settings to publish alongside any result."""
    config = greedy_config(model, tokenizer, max_new_tokens)
    return {
        "mode": "greedy",
        "do_sample": config.do_sample,
        "num_beams": config.num_beams,
        "repetition_penalty": config.repetition_penalty,
        "max_new_tokens": config.max_new_tokens,
        "eos_token_id": config.eos_token_id,
    }


def _autocast(device):
    return torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def generate_texts(model, tokenizer, prompts, batch_size, max_new_tokens):
    """Generate one answer per prompt, returned in the input order."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    config = greedy_config(model, tokenizer, max_new_tokens)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [""] * len(prompts)
    try:
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            encoded = tokenizer(
                [prompts[i] for i in indices], padding=True, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            with _autocast(device):
                generated = model.generate(**encoded, generation_config=config)
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            for index, text in zip(indices, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
                outputs[index] = text
    finally:
        tokenizer.padding_side = previous_side
        model.train(was_training)
    return outputs


def predict_examples(model, tokenizer, examples, order, batch_size, max_new_tokens):
    """Generate and verify an answer for each example."""
    prompts = [render_prompt(tokenizer, example, order) for example in examples]
    texts = generate_texts(model, tokenizer, prompts, batch_size, max_new_tokens)
    return [verify_output(text, e.criterion_type, e.sentences) for text, e in zip(texts, examples)]
