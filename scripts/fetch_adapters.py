"""Download the published LoRA adapters at the revisions in adapters.lock.json.

Usage: python scripts/fetch_adapters.py [--output models]
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "models"))
    args = parser.parse_args()
    lock = json.loads((ROOT / "adapters.lock.json").read_text(encoding="utf-8"))
    for folder, entry in lock.items():
        snapshot_download(entry["repo"], revision=entry["revision"], local_dir=Path(args.output) / folder)
        print(f"adapter ok: {entry['repo']} @ {entry['revision']}", flush=True)


if __name__ == "__main__":
    main()
