"""LoRA fine-tuning with an explicit training loop. The test split is never read here.

Loss is computed only on the assistant's answer tokens. After every epoch the model generates real
greedy answers on validation data, and the best epoch by macro-F1 is kept as the adapter.

Usage: python -m screener.train.train_lora --run R1 [--limit 8 --epochs 3 --batch-size 2]
"""
import argparse
import csv
import gc
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from screener.data.prepare import load_split, render_prompt
from screener.data.prompts import build_target
from screener.eval.metrics import score_predictions, selection_key
from screener.eval.predict import decoding_settings, predict_examples

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "config" / "experiments.yaml"
DEFAULT_OUTPUT = ROOT / "runs"
HISTORY_FIELDS = ["epoch", "train_loss", "val_loss", "accuracy", "macro_f1", "invalid_rate",
                  "evidence_precision", "evidence_recall", "evidence_f1", "seconds"]


def load_experiments():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def resolve_config(run, overrides=None):
    data = load_experiments()
    return {**data["defaults"], **data["runs"].get(run, {}), **(overrides or {})}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_example(tokenizer, example, order):
    """Prompt tokens are masked with -100 so only the answer contributes to the loss."""
    prompt = render_prompt(tokenizer, example, order)
    target = build_target(example.label, example.evidence, order) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "label_name": example.label,
    }


def collate(features, pad_id):
    longest = max(len(f["input_ids"]) for f in features)
    batch = {
        "input_ids": [f["input_ids"] + [pad_id] * (longest - len(f["input_ids"])) for f in features],
        "labels": [f["labels"] + [-100] * (longest - len(f["labels"])) for f in features],
        "attention_mask": [f["attention_mask"] + [0] * (longest - len(f["attention_mask"])) for f in features],
    }
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def sampling_weights(labels, power):
    """Weight each example by (label count)^-power. Zero means uniform, one means fully balanced."""
    counts = Counter(labels)
    return [counts[label] ** (-power) for label in labels]


def make_train_loader(features, config, pad_id):
    generator = torch.Generator().manual_seed(config["seed"])
    collate_fn = lambda items: collate(items, pad_id)
    if config["oversample_power"] > 0:
        weights = sampling_weights([f["label_name"] for f in features], config["oversample_power"])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        return DataLoader(features, batch_size=config["batch_size"], sampler=sampler, collate_fn=collate_fn)
    return DataLoader(features, batch_size=config["batch_size"], shuffle=True, generator=generator, collate_fn=collate_fn)


def autocast(device):
    return torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def mean_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            loss = model(**batch, use_cache=False).loss
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def evaluate_generation(model, tokenizer, examples, config):
    verified = predict_examples(model, tokenizer, examples, config["order"], config["eval_batch_size"], config["max_new_tokens"])
    return score_predictions(examples, verified), verified


def write_predictions(path, examples, verified):
    with open(path, "w", encoding="utf-8") as handle:
        for example, result in zip(examples, verified):
            handle.write(json.dumps({
                "annotation_id": example.annotation_id,
                "gold_label": example.label,
                "pred_label": result.label,
                "valid": result.valid,
                "gold_evidence": list(example.evidence),
                "pred_evidence": list(result.evidence),
                "dropped_ids": [str(i) for i in result.dropped_ids],
                "errors": list(result.errors),
                "raw": result.raw,
            }) + "\n")


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def run_training(run, config, output_root=DEFAULT_OUTPUT, limit=None):
    """Train or baseline-evaluate one run, and write its logs and best adapter."""
    set_seed(config["seed"])
    output = Path(output_root) / run
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model_path = ROOT / config["model"]
    tokenizer = load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, local_files_only=True).to(device)

    train_examples = load_split("train")[:limit]
    val_examples = load_split("validation")[:limit]
    started = time.perf_counter()
    decoding = decoding_settings(model, tokenizer, config["max_new_tokens"])
    trainable, total = count_parameters(model)

    if not config["train"]:
        metrics, verified = evaluate_generation(model, tokenizer, val_examples, config)
        write_predictions(output / "predictions_validation.jsonl", val_examples, verified)
        best_epoch, epochs_run = 0, 0
    else:
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
        ))
        trainable, total = count_parameters(model)
        print(f"{run}: trainable {trainable:,} of {total:,} parameters", flush=True)

        order = config["order"]
        pad_id = tokenizer.pad_token_id
        train_features = [encode_example(tokenizer, e, order) for e in train_examples]
        val_features = [encode_example(tokenizer, e, order) for e in val_examples]
        train_loader = make_train_loader(train_features, config, pad_id)
        val_loader = DataLoader(val_features, batch_size=config["batch_size"], collate_fn=lambda items: collate(items, pad_id))

        accumulation = config["accumulation_steps"]
        steps_per_epoch = math.ceil(len(train_loader) / accumulation)
        total_steps = steps_per_epoch * config["epochs"]
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["learning_rate"], weight_decay=config["weight_decay"],
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, int(config["warmup_fraction"] * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        best_key, best_epoch, metrics, stale, epochs_run = None, 0, None, 0, 0
        with open(output / "history.csv", "w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for epoch in range(1, config["epochs"] + 1):
                epoch_start = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running = 0.0
                for step, batch in enumerate(train_loader, 1):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with autocast(device):
                        loss = model(**batch, use_cache=False).loss
                    running += loss.item()
                    scaler.scale(loss / accumulation).backward()
                    if step % accumulation == 0 or step == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                val_loss = mean_loss(model, val_loader, device)
                current, verified = evaluate_generation(model, tokenizer, val_examples, config)
                row = {"epoch": epoch, "train_loss": running / len(train_loader), "val_loss": val_loss,
                       "seconds": time.perf_counter() - epoch_start, **current}
                writer.writerow(row)
                log_file.flush()
                epochs_run = epoch
                print(f"{run} epoch {epoch}: train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
                      f"macro_f1={current['macro_f1']:.4f} evidence_f1={current['evidence_f1']:.4f} "
                      f"invalid={current['invalid_rate']:.3f}", flush=True)

                key = selection_key(current)
                if best_key is None or key > best_key:
                    best_key, best_epoch, metrics, stale = key, epoch, current, 0
                    model.save_pretrained(output / "adapter")
                    write_predictions(output / "predictions_validation.jsonl", val_examples, verified)
                else:
                    stale += 1
                    if stale >= config["patience"]:
                        print(f"{run}: early stop at epoch {epoch}", flush=True)
                        break

    result = {
        "run": run,
        "config": config,
        "decoding": decoding,
        "split": "validation",
        "num_train": len(train_examples),
        "num_validation": len(val_examples),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "metrics": metrics,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None, help="use only the first N examples, for smoke tests")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--eval-batch-size", type=int, dest="eval_batch_size")
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in ("epochs", "batch_size", "eval_batch_size") if getattr(args, k) is not None}
    result = run_training(args.run, resolve_config(args.run, overrides), args.output, args.limit)
    print(json.dumps({k: result[k] for k in ("best_epoch", "metrics", "peak_gpu_gb", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
