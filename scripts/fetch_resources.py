"""Download the project's resources and record exact versions in resources.lock.json.

Safe to re-run: finished files are skipped. Only data and safetensors weights are fetched,
and no downloaded code is executed.
"""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "resources.lock.json"

DATASET = ("ncbi/TrialGPT-Criterion-Annotations", ROOT / "data" / "raw")
MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": ROOT / "models" / "qwen2.5-1.5b",
    "Qwen/Qwen2.5-0.5B-Instruct": ROOT / "models" / "qwen2.5-0.5b",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ROOT / "models" / "smollm2-1.7b",
}
PAPERS = {
    "2307.15051": "trialgpt",
    "1706.03762": "attention-is-all-you-need",
    "2106.09685": "lora",
    "2305.14314": "qlora",
    "2211.17192": "speculative-decoding-leviathan",
    "2302.01318": "speculative-sampling-chen",
    "2205.14135": "flashattention",
    "1805.02867": "online-softmax",
    "2412.15115": "qwen2.5-report",
    "2505.09388": "qwen3-report",
    "2502.02737": "smollm2",
}
SKIP_PATTERNS = [".gitattributes", "onnx/*", "runs/*", "*.bin", "*.gguf", "*.onnx", "*.onnx_data", "*.pt", "*.pth", "*.py"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_papers(lock):
    folder = ROOT / "papers"
    folder.mkdir(exist_ok=True)
    for arxiv_id, name in PAPERS.items():
        target = folder / f"{arxiv_id}-{name}.pdf"
        if not (target.exists() and target.read_bytes()[:5] == b"%PDF-"):
            request = urllib.request.Request(
                f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "gen-ai-ass2-fetch/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
            if body[:5] != b"%PDF-":
                raise RuntimeError(f"{arxiv_id}: response is not a PDF")
            target.write_bytes(body)
            time.sleep(4)
        lock["papers"][arxiv_id] = {"file": target.relative_to(ROOT).as_posix(), "sha256": sha256(target)}
        print(f"paper ok: {target.name}", flush=True)


def fetch_hub(api, repo_id, folder, repo_type, lock_section):
    info = api.repo_info(repo_id, repo_type=repo_type)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        repo_type=repo_type,
        revision=info.sha,
        local_dir=folder,
        ignore_patterns=SKIP_PATTERNS,
    )
    lock_section[repo_id] = {"revision": info.sha, "path": folder.relative_to(ROOT).as_posix()}
    print(f"hub ok: {repo_id} @ {info.sha}", flush=True)


def fetch_nltk(lock):
    import nltk

    folder = ROOT / "data" / "nltk_data"
    folder.mkdir(parents=True, exist_ok=True)
    if not nltk.download("punkt_tab", download_dir=str(folder), quiet=True):
        raise RuntimeError("punkt_tab download failed")
    lock["nltk"] = {"package": "punkt_tab", "path": folder.relative_to(ROOT).as_posix()}
    print("nltk ok: punkt_tab", flush=True)


def fetch_models_from_lock():
    """Download only the models, at the revisions already recorded. Used on Kaggle."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for repo_id, entry in lock["models"].items():
        snapshot_download(
            repo_id, revision=entry["revision"], local_dir=ROOT / entry["path"], ignore_patterns=SKIP_PATTERNS
        )
        print(f"hub ok: {repo_id} @ {entry['revision']}", flush=True)


def main():
    if "--models-from-lock" in sys.argv:
        fetch_models_from_lock()
        return
    lock = {"dataset": {}, "models": {}, "papers": {}, "nltk": {}}
    api = HfApi()
    fetch_hub(api, DATASET[0], DATASET[1], "dataset", lock["dataset"])
    fetch_papers(lock)
    fetch_nltk(lock)
    for repo_id, folder in MODELS.items():
        fetch_hub(api, repo_id, folder, "model", lock["models"])
    parquet = DATASET[1] / "data" / "train-00000-of-00001.parquet"
    lock["dataset"][DATASET[0]]["parquet_sha256"] = sha256(parquet)
    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"wrote {LOCK.name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
