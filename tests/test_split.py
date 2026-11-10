import json

import numpy as np
import pandas as pd

from screener.data.split import MANIFEST, PARQUET, SIZES, assign_patients, build_manifest


def raw():
    return pd.read_parquet(PARQUET, columns=["patient_id", "trial_id", "annotation_id", "criterion_type",
                                             "criterion_text", "expert_eligibility"])


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_procedure_matches_the_assignment_text():
    frame = raw()
    expected = np.array(sorted(frame["patient_id"].unique()), dtype=object)
    np.random.default_rng(42).shuffle(expected)
    got = assign_patients(frame["patient_id"])
    assert got["train"] == list(expected[:37])
    assert got["validation"] == list(expected[37:45])
    assert got["test"] == list(expected[45:])


def test_sizes_and_patient_disjointness():
    splits = manifest()["splits"]
    sets = {name: set(info["patients"]) for name, info in splits.items()}
    assert {name: len(s) for name, s in sets.items()} == SIZES
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert len(sets["train"] | sets["validation"] | sets["test"]) == 53


def test_every_row_is_in_exactly_one_split():
    frame = raw()
    data = manifest()
    owner = {p: name for name, info in data["splits"].items() for p in info["patients"]}
    assert frame["patient_id"].map(owner).notna().all()
    assert sum(info["num_rows"] for info in data["splits"].values()) == len(frame) == 1015
    for name, info in data["splits"].items():
        assert info["num_rows"] == int((frame["patient_id"].map(owner) == name).sum())


def test_manifest_is_reproducible():
    assert build_manifest(pd.read_parquet(PARQUET)) == manifest()


def test_only_the_blank_criterion_row_is_removed():
    removed = manifest()["removed_rows"]
    assert len(removed) == 1
    frame = raw()
    row = frame[frame["annotation_id"] == removed[0]["annotation_id"]].iloc[0]
    assert pd.isna(row["criterion_text"]) or not str(row["criterion_text"]).strip()
