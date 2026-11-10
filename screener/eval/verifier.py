"""Check a model's raw text against the output schema and the patient's real sentence IDs.

The verifier never repairs a verdict: an unusable label stays invalid and is scored as wrong,
and it never infers a label from missing evidence.
"""
import json
import re
from dataclasses import dataclass, field

from screener.data.prompts import LABELS

INVALID = "__invalid__"
_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass(frozen=True)
class Verified:
    label: str
    evidence: tuple
    valid: bool
    raw: str
    errors: tuple = field(default_factory=tuple)
    dropped_ids: tuple = field(default_factory=tuple)
    extra_text: bool = False


def _load_object(text):
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        return value, False
    except json.JSONDecodeError:
        pass
    match = _OBJECT.search(stripped)
    if match is None:
        return None, False
    try:
        return json.loads(match.group(0)), True
    except json.JSONDecodeError:
        return None, False


def _invalid(raw, reason, extra_text=False):
    return Verified(INVALID, (), False, raw, (reason,), (), extra_text)


def verify_output(raw, criterion_type, sentences):
    """Return a Verified result for one generated answer."""
    value, extra_text = _load_object(raw)
    if not isinstance(value, dict):
        return _invalid(raw, "output is not a JSON object", extra_text)
    label = value.get("label")
    if not isinstance(label, str):
        return _invalid(raw, "label is missing", extra_text)
    label = label.strip().lower()
    if label not in LABELS[criterion_type]:
        return _invalid(raw, f"label {label!r} is not allowed for {criterion_type} criteria", extra_text)
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return _invalid(raw, "evidence is not a list", extra_text)

    kept, dropped = [], []
    for item in evidence:
        if isinstance(item, int) and not isinstance(item, bool) and item in sentences:
            kept.append(item)
        else:
            dropped.append(item)
    errors = (f"dropped {len(dropped)} evidence id(s) not in the note",) if dropped else ()
    return Verified(label, tuple(sorted(set(kept))), True, raw, errors, tuple(dropped), extra_text)
