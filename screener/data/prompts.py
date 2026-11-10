"""Model input and output format. The same format is used for training, evaluation and the app."""
import json

from screener.data.sentences import format_numbered

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = sorted({label for labels in LABELS.values() for label in labels})

SYSTEM_PROMPT = (
    "You screen patients against one clinical trial criterion at a time. "
    "Use only the numbered patient note. Do not assume facts the note does not support. "
    "Reply with one JSON object and nothing else."
)

_DEFINITIONS = {
    "inclusion": (
        '"included": the note shows the patient meets this criterion.\n'
        '"not included": the note shows the patient does not meet this criterion.\n'
    ),
    "exclusion": (
        '"excluded": the note shows the patient has this exclusion condition.\n'
        '"not excluded": the note shows the patient does not have this exclusion condition.\n'
    ),
}
_SHARED = (
    '"not enough information": the criterion is relevant but the note cannot settle it.\n'
    '"not applicable": the criterion does not apply to this patient.\n'
)
_FORMATS = {
    "label_first": '{"label": "<label>", "evidence": [<sentence ids>]}',
    "evidence_first": '{"evidence": [<sentence ids>], "label": "<label>"}',
}


def build_messages(sentences, criterion_type, criterion_text, order="label_first"):
    """Chat messages built only from patient sentences, criterion type and criterion text."""
    if criterion_type not in LABELS:
        raise ValueError(f"unknown criterion type: {criterion_type!r}")
    user = (
        f"Patient note:\n{format_numbered(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Choose one label:\n{_DEFINITIONS[criterion_type]}{_SHARED}\n"
        f"Evidence is the list of sentence ids that support the label. Use [] if no sentence does.\n"
        f"Answer format: {_FORMATS[order]}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_target(label, evidence, order="label_first"):
    """The exact JSON text the model is trained to produce."""
    ids = sorted(set(int(i) for i in evidence))
    fields = {"label": label, "evidence": ids}
    if order == "evidence_first":
        fields = {"evidence": ids, "label": label}
    elif order != "label_first":
        raise ValueError(f"unknown output order: {order!r}")
    return json.dumps(fields, ensure_ascii=False)
