"""Common evaluation metrics for labels and evidence.

Rules used throughout:
- A label with no gold examples in a split shows "n/a" (None) and is left out of macro-F1.
  A second macro-F1 counting those labels as zero is also reported.
- Unusable model output is scored as a wrong answer in its own confusion-matrix column.
- Evidence, per row: both sets empty scores 1. If exactly one set is empty, precision, recall and
  F1 all score 0, because nothing correct was found or nothing correct existed to find.
"""
import random
from collections import defaultdict

from screener.data.prompts import ALL_LABELS, LABELS
from screener.eval.verifier import INVALID, Verified

_SCOPES = {"overall": ALL_LABELS, "inclusion": LABELS["inclusion"], "exclusion": LABELS["exclusion"]}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evidence_micro(gold_sets, predicted_sets):
    """Micro precision, recall and F1 over all sentence IDs."""
    true_positive = false_positive = false_negative = 0
    for gold, predicted in zip(gold_sets, predicted_sets):
        gold, predicted = set(gold), set(predicted)
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def evidence_row_scores(gold, predicted):
    """Precision, recall and F1 for one row, using the empty-set rules above."""
    gold, predicted = set(gold), set(predicted)
    if not gold and not predicted:
        return 1.0, 1.0, 1.0
    if not gold or not predicted:
        return 0.0, 0.0, 0.0
    overlap = len(gold & predicted)
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return precision, recall, _f1(precision, recall)


def _macro(rows):
    if not rows:
        return {"precision": None, "recall": None, "f1": None, "rows": 0}
    count = len(rows)
    return {
        "precision": sum(r[0] for r in rows) / count,
        "recall": sum(r[1] for r in rows) / count,
        "f1": sum(r[2] for r in rows) / count,
        "rows": count,
    }


def evidence_report(gold_sets, predicted_sets):
    gold_sets = [set(g) for g in gold_sets]
    predicted_sets = [set(p) for p in predicted_sets]
    rows = [evidence_row_scores(g, p) for g, p in zip(gold_sets, predicted_sets)]
    return {
        "micro": evidence_micro(gold_sets, predicted_sets),
        "macro_all_rows": _macro(rows),
        "macro_rows_with_gold_evidence": _macro([r for r, g in zip(rows, gold_sets) if g]),
        "counts": {
            "both_empty": sum(not g and not p for g, p in zip(gold_sets, predicted_sets)),
            "gold_empty_predicted_some": sum(not g and bool(p) for g, p in zip(gold_sets, predicted_sets)),
            "gold_some_predicted_empty": sum(bool(g) and not p for g, p in zip(gold_sets, predicted_sets)),
            "both_nonempty": sum(bool(g) and bool(p) for g, p in zip(gold_sets, predicted_sets)),
        },
    }


def classification_report(gold_labels, predicted_labels, label_set):
    """Accuracy, macro-F1, per-class scores and a confusion matrix for one group of rows."""
    label_set = list(label_set)
    columns = label_set + [INVALID]
    matrix = {gold: {column: 0 for column in columns} for gold in label_set}
    for gold, predicted in zip(gold_labels, predicted_labels):
        matrix[gold][predicted if predicted in matrix[gold] else INVALID] += 1

    per_class = {}
    for label in label_set:
        support = sum(matrix[label].values())
        true_positive = matrix[label][label]
        predicted_count = sum(matrix[gold][label] for gold in label_set)
        if support == 0:
            precision = 0.0 if predicted_count else None
            per_class[label] = {"support": 0, "precision": precision, "recall": None, "f1": None}
            continue
        precision, recall = _ratio(true_positive, predicted_count), true_positive / support
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": _f1(precision, recall)}

    scored = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    correct = sum(g == p for g, p in zip(gold_labels, predicted_labels))
    return {
        "rows": len(gold_labels),
        "accuracy": _ratio(correct, len(gold_labels)),
        "macro_f1": sum(scored) / len(scored) if scored else None,
        "macro_f1_absent_labels_as_zero": sum(scored) / len(label_set) if label_set else None,
        "labels_absent_from_gold": [k for k, v in per_class.items() if v["support"] == 0],
        "per_class": per_class,
        "confusion_matrix": {"rows_are_gold": True, "columns": columns, "matrix": matrix},
    }


def score_predictions(examples, verified):
    """The compact summary used to compare runs during model selection."""
    report = classification_report([e.label for e in examples], [v.label for v in verified], ALL_LABELS)
    evidence = evidence_micro([e.evidence for e in examples], [v.evidence for v in verified])
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"] or 0.0,
        "invalid_rate": sum(not v.valid for v in verified) / len(verified),
        "evidence_precision": evidence["precision"],
        "evidence_recall": evidence["recall"],
        "evidence_f1": evidence["f1"],
        "labels_scored": sorted(set(e.label for e in examples)),
    }


def selection_key(metrics):
    """Model selection: validation macro-F1, then evidence F1 as the tie-break."""
    return (round(metrics["macro_f1"], 6), round(metrics["evidence_f1"], 6))


def full_report(examples, verified):
    """Every required metric, overall and separately for inclusion and exclusion criteria."""
    report = {"classification": {}, "evidence": {}}
    for scope, labels in _SCOPES.items():
        pairs = [(e, v) for e, v in zip(examples, verified) if scope == "overall" or e.criterion_type == scope]
        report["classification"][scope] = classification_report([e.label for e, _ in pairs], [v.label for _, v in pairs], labels)
        report["evidence"][scope] = evidence_report([e.evidence for e, _ in pairs], [v.evidence for _, v in pairs])
    report["invalid_rate"] = sum(not v.valid for v in verified) / len(verified)
    report["rows_with_dropped_evidence_ids"] = sum(bool(v.dropped_ids) for v in verified)
    return report


def bootstrap_intervals(examples, verified, repeats=2000, seed=42):
    """95% intervals from resampling whole patients, since rows from one patient are not independent."""
    by_patient = defaultdict(list)
    for example, result in zip(examples, verified):
        by_patient[example.patient_id].append((example, result))
    patients = sorted(by_patient)
    rng = random.Random(seed)
    samples = {"accuracy": [], "macro_f1": [], "evidence_micro_f1": []}
    for _ in range(repeats):
        chosen = [pair for p in rng.choices(patients, k=len(patients)) for pair in by_patient[p]]
        report = classification_report([e.label for e, _ in chosen], [v.label for _, v in chosen], ALL_LABELS)
        samples["accuracy"].append(report["accuracy"])
        samples["macro_f1"].append(report["macro_f1"])
        samples["evidence_micro_f1"].append(evidence_micro([e.evidence for e, _ in chosen], [v.evidence for _, v in chosen])["f1"])
    intervals = {}
    for name, values in samples.items():
        values.sort()
        intervals[name] = {"low": values[int(0.025 * repeats)], "high": values[int(0.975 * repeats) - 1]}
    return {"method": "patient-level bootstrap", "repeats": repeats, "seed": seed, "patients": len(patients), "intervals": intervals}


def majority_baseline(train_examples, examples):
    """Predict the training-majority label for each criterion type and cite no evidence."""
    counts = defaultdict(lambda: defaultdict(int))
    for example in train_examples:
        counts[example.criterion_type][example.label] += 1
    majority = {kind: max(sorted(labels), key=labels.get) for kind, labels in counts.items()}
    return [Verified(majority[e.criterion_type], (), True, "baseline") for e in examples], majority
