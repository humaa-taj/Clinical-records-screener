"""Model access for the application. Screening uses this project's own decoders."""
import json
import os
from pathlib import Path

import torch

from app.pipeline import InputError, build_record
from screener.attention.reference import reference_attention
from screener.attention.tiled_online import TileStats, tiled_attention
from screener.data.prompts import build_messages
from screener.eval.predict import eos_token_ids
from screener.eval.verifier import verify_output
from screener.inference.cached_decoder import greedy_cached
from screener.inference.common import clock, load_for_inference
from screener.inference.speculative import greedy_speculative
from screener.inference.uncached_decoder import greedy_uncached

ROOT = Path(__file__).resolve().parents[1]
MAX_PROMPT_TOKENS = 2_048
MAX_NEW_TOKENS = 96
NOT_PINNED = "not pinned"


def _setting(name, default):
    return os.environ.get(name, default) or None


def pinned_versions(target_path, adapter_path):
    """Exact upstream versions of the local model folders, read from the project's lock files."""
    models = json.loads((ROOT / "resources.lock.json").read_text(encoding="utf-8"))["models"]
    adapters = json.loads((ROOT / "adapters.lock.json").read_text(encoding="utf-8"))
    target = next((f"{repo}@{entry['revision']}" for repo, entry in models.items() if Path(entry["path"]) == Path(target_path)), NOT_PINNED)
    adapter = adapters.get(Path(adapter_path).name) if adapter_path else None
    return {"target_revision": target, "adapter_revision": f"{adapter['repo']}@{adapter['revision']}" if adapter else NOT_PINNED}


class Engine:
    def __init__(self):
        self.state = "loading"
        self.error = None
        self.target = None
        self.tokenizer = None
        self.draft = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_path = _setting("CRS_TARGET", "models/qwen2.5-1.5b")
        self.draft_path = _setting("CRS_DRAFT", "models/qwen2.5-0.5b")
        self.target_adapter = _setting("CRS_ADAPTER", "models/crs-adapter")
        self.draft_adapter = _setting("CRS_DRAFT_ADAPTER", "models/crs-draft-adapter")
        self.order = _setting("CRS_ORDER", "label_first")
        self.draft_device = torch.device(_setting("CRS_DRAFT_DEVICE", "cpu"))

    def _adapter(self, relative):
        path = ROOT / relative if relative else None
        return path if path is not None and path.exists() else None

    def load(self):
        try:
            self.target, self.tokenizer = load_for_inference(
                ROOT / self.target_path, self._adapter(self.target_adapter), self.device, torch.float16
            )
            self.state = "ready"
        except Exception as error:  # Shown on screen in place of a server crash.
            self.state = "failed"
            self.error = f"The model could not be loaded: {error}"

    def _load_draft(self):
        if self.draft is None:
            dtype = torch.float16 if self.draft_device.type == "cuda" else torch.float32
            self.draft, _ = load_for_inference(ROOT / self.draft_path, self._adapter(self.draft_adapter), self.draft_device, dtype)
        return self.draft

    def info(self):
        adapter = self._adapter(self.target_adapter)
        return {
            "state": self.state,
            "error": self.error,
            "target_model": self.target_path,
            "adapter": str(adapter.relative_to(ROOT)) if adapter else None,
            "trained": adapter is not None,
            "output_order": self.order,
            "draft_model": self.draft_path,
            "device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
            "draft_device": str(self.draft_device),
        }

    def provenance(self):
        info = self.info()
        fields = {key: info[key] for key in ("target_model", "adapter", "trained", "output_order", "device")}
        return {**fields, **pinned_versions(self.target_path, info["adapter"])}

    def _encode(self, sentences, criterion_type, criterion):
        messages = build_messages(sentences, criterion_type, criterion, self.order)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        if ids.shape[1] > MAX_PROMPT_TOKENS:
            raise InputError(f"This note and criterion are {ids.shape[1]:,} tokens, over the {MAX_PROMPT_TOKENS:,} limit. Nothing was cut off. Shorten the note and try again.")
        return ids

    def _decode(self, ids, mode, max_new_tokens, eos, draft_length=4):
        if mode == "cached":
            return greedy_cached(self.target, ids, max_new_tokens=max_new_tokens, eos_ids=eos)
        if mode == "uncached":
            return greedy_uncached(self.target, ids, max_new_tokens=max_new_tokens, eos_ids=eos)
        return greedy_speculative(self.target, self._load_draft(), ids, max_new_tokens=max_new_tokens, eos_ids=eos, draft_length=draft_length)

    @staticmethod
    def _timing(result, mode):
        timing = {
            "mode": mode,
            "input_tokens": result.input_length,
            "output_tokens": len(result.tokens),
            "prefill_s": result.prefill_seconds,
            "decode_s": result.decode_seconds,
            "total_s": result.total_seconds,
            "tokens_per_s": result.total_tokens_per_second,
        }
        if hasattr(result, "rounds"):
            timing.update(rounds=result.rounds, drafted=result.drafted, accepted=result.accepted, acceptance_rate=result.acceptance_rate)
        return timing

    def screen_one(self, index, sentences, criterion_type, criterion, mode):
        ids = self._encode(sentences, criterion_type, criterion)
        result = self._decode(ids, mode, MAX_NEW_TOKENS, eos_token_ids(self.target, self.tokenizer))
        text = self.tokenizer.decode(result.tokens, skip_special_tokens=True)
        verified = verify_output(text, criterion_type, sentences)
        return build_record(index, criterion_type, criterion, verified, sentences, self._timing(result, mode))

    def compare_decoders(self, sentences, criterion_type, criterion, max_new_tokens, ignore_eos, draft_length):
        """Run every decoding mode on the same input, and report equality against the reference."""
        ids = self._encode(sentences, criterion_type, criterion)
        eos = [] if ignore_eos else eos_token_ids(self.target, self.tokenizer)
        rows, reference = [], None
        for mode in ("uncached", "cached", "speculative"):
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
            base = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
            result = self._decode(ids, mode, max_new_tokens, eos, draft_length)
            reference = reference if reference is not None else result.tokens
            row = {**self._timing(result, mode), "tokens_match_reference": result.tokens == reference}
            row["peak_gpu_extra_mb"] = (torch.cuda.max_memory_allocated(self.device) - base) / 1024**2 if self.device.type == "cuda" else None
            rows.append(row)
        return {"rows": rows, "draft_device": str(self.draft_device), "device": self.info()["device"], "text": self.tokenizer.decode(reference, skip_special_tokens=True)}

    def compare_attention(self, length, tile, heads=8, head_dim=64):
        """Run reference and tiled attention on the same random causal input."""
        generator = torch.Generator().manual_seed(length)
        make = lambda: torch.randn((1, heads, length, head_dim), generator=generator, dtype=torch.float32).to(self.device)
        query, key, value = make(), make(), make()
        mask = torch.ones(length, length, dtype=torch.bool, device=self.device).tril()[None, None]

        def measure(function):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            base = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
            start = clock(self.device)
            with torch.no_grad():
                output = function()
            seconds = clock(self.device) - start
            extra = (torch.cuda.max_memory_allocated(self.device) - base) / 1024**2 if self.device.type == "cuda" else None
            return output, seconds * 1000, extra

        stats = TileStats()
        reference, reference_ms, reference_mb = measure(lambda: reference_attention(query, key, value, mask))
        tiled, tiled_ms, tiled_mb = measure(lambda: tiled_attention(query, key, value, mask, tile, tile, stats=stats))
        difference = (reference - tiled).abs()
        return {
            "device": self.info()["device"], "length": length, "tile": tile, "heads": heads, "head_dim": head_dim,
            "reference_ms": reference_ms, "tiled_ms": tiled_ms, "reference_gpu_extra_mb": reference_mb, "tiled_gpu_extra_mb": tiled_mb,
            "max_abs_error": difference.max().item(), "mean_abs_error": difference.mean().item(),
            "tiles_computed": stats.computed, "tiles_skipped": stats.skipped,
        }
