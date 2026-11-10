"""Create the assignment's fixed patient-grouped split and its published manifest.

The procedure is prescribed: sort unique patient IDs, shuffle with NumPy default_rng(42),
then assign the first 37 patients to train, the next 8 to validation and the last 8 to test.
The dataset's own `training` column is deliberately ignored.

Usage: python -m screener.data.split
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "raw" / "data" / "train-00000-of-00001.parquet"
MANIFEST = ROOT / "splits" / "manifest.json"
SEED = 42
SIZES = {"train": 37, "validation": 8, "test": 8}
ROW_KEY = ["patient_id", "trial_id", "annotation_id"]


def load_raw():
    return pd.read_parquet(PARQUET)


def assign_patients(patient_ids):
    """Return {split_name: [patient_id, ...]} using the prescribed procedure."""
    ordered = np.array(sorted(set(patient_ids)), dtype=object)
    if len(ordered) != sum(SIZES.values()):
        raise ValueError(f"expected {sum(SIZES.values())} patients, found {len(ordered)}")
    np.random.default_rng(SEED).shuffle(ordered)
    assignment = {}
    start = 0
    for name, size in SIZES.items():
        assignment[name] = [str(p) for p in ordered[start : start + size]]
        start += size
    return assignment


def blank_criterion_mask(frame):
    text = frame["criterion_text"]
    return text.isna() | (text.fillna("").str.strip() == "")


def row_ref(row):
    return {key: (row[key].item() if hasattr(row[key], "item") else row[key]) for key in ROW_KEY}


def build_manifest(frame):
    assignment = assign_patients(frame["patient_id"])
    blank = blank_criterion_mask(frame)
    splits = {}
    for name, patients in assignment.items():
        rows = frame[frame["patient_id"].isin(patients)]
        usable = rows[~blank.loc[rows.index]]
        labels = usable.groupby("criterion_type")["expert_eligibility"].value_counts()
        splits[name] = {
            "patients": patients,
            "num_patients": len(patients),
            "num_rows": int(len(rows)),
            "num_usable_rows": int(len(usable)),
            "rows_per_patient": {p: int((rows["patient_id"] == p).sum()) for p in patients},
            "label_counts": {f"{k[0]}/{k[1]}": int(v) for k, v in labels.items()},
        }
    return {
        "dataset": "ncbi/TrialGPT-Criterion-Annotations",
        "source_parquet_sha256": hashlib.sha256(PARQUET.read_bytes()).hexdigest(),
        "procedure": "sorted unique patient_id; numpy.random.default_rng(42).shuffle; first 37 train, next 8 validation, last 8 test",
        "seed": SEED,
        "total_rows": int(len(frame)),
        "total_patients": int(frame["patient_id"].nunique()),
        "row_key": ROW_KEY,
        "removed_rows": [
            {**row_ref(row), "reason": "blank criterion_text; removed from training and scoring in every split"}
            for _, row in frame[blank].iterrows()
        ],
        "splits": splits,
    }


def main():
    manifest = build_manifest(load_raw())
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, info in manifest["splits"].items():
        print(f"{name}: {info['num_patients']} patients")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
