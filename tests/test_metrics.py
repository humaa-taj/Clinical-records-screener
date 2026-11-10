import json

import pytest

from screener.data.prepare import Example
from screener.eval.evaluate import check_split_access, log_test_access, prediction_record
from screener.eval.metrics import (
    bootstrap_intervals,
    classification_report,
    evidence_report,
    evidence_row_scores,
    full_report,
    majority_baseline,
)
from screener.eval.verifier import INVALID, verify_output

INCLUSION = ["included", "not included", "not enough information", "not applicable"]
SENTENCES = {0: "First.", 1: "Second."}


def example(i, patient, kind, label, evidence=()):
    return Example(i, patient, "T", SENTENCES, kind, "criterion", label, tuple(evidence))


def test_classification_by_hand():
    gold = ["included", "included", "not included", "not enough information"]
    pred = ["included", "not included", "not included", INVALID]
    report = classification_report(gold, pred, INCLUSION)
    assert report["accuracy"] == 0.5
    assert report["per_class"]["included"] == {"support": 2, "precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3)}
    assert report["per_class"]["not included"]["precision"] == 0.5
    assert report["per_class"]["not included"]["recall"] == 1.0
    assert report["per_class"]["not enough information"]["f1"] == 0.0
    assert report["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert report["confusion_matrix"]["matrix"]["not enough information"][INVALID] == 1


def test_absent_label_is_not_scored():
    report = classification_report(["included"], ["included"], INCLUSION)
    assert report["per_class"]["not applicable"]["f1"] is None
    assert report["labels_absent_from_gold"] == ["not included", "not enough information", "not applicable"]
    assert report["macro_f1"] == 1.0
    assert report["macro_f1_absent_labels_as_zero"] == 0.25


@pytest.mark.parametrize("gold,pred,expected", [
    ((), (), (1.0, 1.0, 1.0)),
    ((), (1,), (0.0, 0.0, 0.0)),
    ((1,), (), (0.0, 0.0, 0.0)),
    ((0, 1), (1,), (1.0, 0.5, pytest.approx(2 / 3))),
])
def test_evidence_row_rules(gold, pred, expected):
    assert evidence_row_scores(gold, pred) == expected


def test_evidence_report_counts_and_gold_only_macro():
    report = evidence_report([(), (), (0,), (0,)], [(), (1,), (), (0,)])
    assert report["counts"] == {"both_empty": 1, "gold_empty_predicted_some": 1, "gold_some_predicted_empty": 1, "both_nonempty": 1}
    assert report["macro_all_rows"]["f1"] == 0.5
    assert report["macro_rows_with_gold_evidence"]["f1"] == 0.5
    assert report["macro_rows_with_gold_evidence"]["rows"] == 2


def test_full_report_splits_criterion_types():
    examples = [example(1, "a", "inclusion", "included"), example(2, "b", "exclusion", "excluded")]
    verified = [verify_output('{"label": "included", "evidence": []}', "inclusion", SENTENCES),
                verify_output('{"label": "not excluded", "evidence": []}', "exclusion", SENTENCES)]
    report = full_report(examples, verified)
    assert report["classification"]["inclusion"]["accuracy"] == 1.0
    assert report["classification"]["exclusion"]["accuracy"] == 0.0
    assert report["classification"]["overall"]["rows"] == 2


def test_bootstrap_is_repeatable_and_ordered():
    examples = [example(i, f"p{i % 3}", "inclusion", "included" if i % 2 else "not included") for i in range(12)]
    verified = [verify_output('{"label": "included", "evidence": []}', "inclusion", SENTENCES) for _ in examples]
    first = bootstrap_intervals(examples, verified, repeats=100)
    assert first == bootstrap_intervals(examples, verified, repeats=100)
    assert first["intervals"]["accuracy"]["low"] <= first["intervals"]["accuracy"]["high"]
    assert first["patients"] == 3


def test_majority_baseline_uses_training_counts_and_cites_nothing():
    train = [example(1, "a", "exclusion", "not excluded"), example(2, "a", "exclusion", "not excluded"), example(3, "a", "exclusion", "excluded"),
             example(4, "a", "inclusion", "included")]
    verified, majority = majority_baseline(train, [example(9, "b", "exclusion", "excluded")])
    assert majority == {"exclusion": "not excluded", "inclusion": "included"}
    assert verified[0].label == "not excluded" and verified[0].evidence == ()


def test_test_split_requires_final():
    with pytest.raises(PermissionError):
        check_split_access("test", False)
    with pytest.raises(ValueError):
        check_split_access("validation", True)
    check_split_access("test", True)
    check_split_access("validation", False)


def test_test_access_is_logged(tmp_path):
    path = tmp_path / "log.jsonl"
    log_test_access({"model": "m"}, path)
    entry = json.loads(path.read_text())
    assert entry["model"] == "m" and "time_utc" in entry and "commit" in entry


def test_prediction_record_links_evidence_to_the_note():
    item = example(1, "a", "inclusion", "included", (0,))
    record = prediction_record(item, verify_output('{"label": "included", "evidence": [0]}', "inclusion", SENTENCES))
    assert record["evidence_text"] == {"0": "First."}
    assert "[0] First." in record["explanation"]


def test_prediction_record_for_invalid_output():
    item = example(1, "a", "inclusion", "included")
    record = prediction_record(item, verify_output("garbage", "inclusion", SENTENCES))
    assert record["predicted_label"] == INVALID and "could not be verified" in record["explanation"]
