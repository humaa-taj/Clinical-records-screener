"""A deterministic comparison for simple numeric criteria, applied on top of the model.

The model is unreliable at comparing numbers, so plain code does the arithmetic when it safely can.
Staying silent is always acceptable: the criterion must be a bare comparison, the note must hold exactly
one clear value for that measure, and nothing nearby may suggest it is not the patient's current value.
"""
import re
from dataclasses import dataclass

_NUMBER = r"(\d+(?:\.\d+)?)"
_PREFIX = r"(?:(?:patients?|subjects?|participants?|adults?) (?:with |having )?|with |has |have )?(?:an? )?"
_LINK = r"\s*(?:level |value |score )?(?:of |is |was |:|=)?\s*"
_HEDGE = re.compile(
    r"\b(target|goal|desired|previous|previously|prior|formerly|last year|years? ago|months? ago|family|mother|father|sister|"
    r"brother|daughter|son|wife|husband|if|should|would|after|before|increased|decreased|rose|fell|dropped|improved|range|"
    r"normal|approximately|about|around|unknown|pending|not available)\b|~"
)
_RELATIVE = re.compile(r"\b(mother|father|sister|brother|daughter|son|wife|husband|grandmother|grandfather|parents?)\b")
_FOREIGN_UNIT = re.compile(r"\s*(?:mmol|µmol|umol|μmol|nmol|g/|mg|ml|l\b)")
_RANGE_AFTER = re.compile(r"\s*(?:-|to)\s*\d")

_AT_LEAST = r"(?:at least|>=|greater than or equal to|equal to or greater than|no less than)"
_ABOVE = r"(?:above|over|greater than|more than|higher than|>)"
_AT_MOST = r"(?:at most|<=|less than or equal to|equal to or less than|no more than|not more than)"
_BELOW = r"(?:below|under|less than|lower than|<)"
_COMPARISONS = ((_AT_LEAST, ">="), (_ABOVE, ">"), (_AT_MOST, "<="), (_BELOW, "<"))


@dataclass(frozen=True)
class Measure:
    alias: str
    unit: str
    note_needs_unit: bool = False


MEASURES = (
    Measure(r"bmi|body mass index", r"kg/m2|kg/m²|kg/m\^2"),
    Measure(r"hba1c|hb a1c|a1c", r"%|percent"),
    Measure(r"egfr|estimated glomerular filtration rate", r"ml/min(?:/1\.73 ?m(?:2|²|\^2))?"),
    Measure(r"(?:serum )?creatinine", r"mg/dl", note_needs_unit=True),
    Measure(r"(?:left ventricular )?ejection fraction|lvef", r"%|percent"),
)

_YEARS = r"(?:\s*years?(?: old| of age)?)"
_AGE_WORD = r"(?:(?:adults?|patients?|participants?|subjects?|people|men|women)\s+)?(?:aged?|age is)"
_PERSON = r"(?:(?:the )?(?:participants?|patients?|subjects?)(?: is| are)? |adults? |(?:must be )?)"
_AGE_RANGES = (
    rf"{_AGE_WORD}\s*(?:between\s*)?{_NUMBER}\s*(?:and|to|-)\s*{_NUMBER}{_YEARS}?",
    rf"{_PERSON}(?:between\s*)?{_NUMBER}\s*(?:and|to|-)\s*{_NUMBER}{_YEARS}",
)
_NOTE_AGE = re.compile(r"\b(\d{1,3})(?:[- ]years?[- ]old|\s*(?:y/o|yo\b|y\.o\.|yr old))|\baged (\d{1,3})\b(?!\s*(?:months?|weeks?|days?))")


@dataclass(frozen=True)
class NumberResult:
    label: str
    sentence_id: int
    value: float


def _normalize(text):
    text = text.lower().strip().rstrip(".;,").replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def _holds(value, rule):
    if rule[0] == "between":
        return rule[1] <= value <= rule[2]
    limit = rule[1]
    return {">=": value >= limit, ">": value > limit, "<=": value <= limit, "<": value < limit}[rule[0]]


def _measure_rule(text):
    """Return (measure, rule) when the whole criterion is one comparison on a known measure."""
    for measure in MEASURES:
        head = rf"{_PREFIX}(?:{measure.alias}){_LINK}"
        tail = rf"\s*(?:{measure.unit})?"
        between = re.fullmatch(rf"{head}(?:between\s*)?{_NUMBER}\s*(?:and|to|-)\s*{_NUMBER}{tail}", text)
        if between:
            return measure, ("between", float(between[1]), float(between[2]))
        for words, operator in _COMPARISONS:
            match = re.fullmatch(rf"{head}{words}\s*{_NUMBER}{tail}", text)
            if match:
                return measure, (operator, float(match[1]))
    return None


def _age_rule(text):
    for pattern in _AGE_RANGES:
        match = re.fullmatch(pattern, text)
        if match:
            return "between", float(match[1]), float(match[2])
    for words, operator in _COMPARISONS:
        match = re.fullmatch(rf"(?:{_AGE_WORD}\s*|{_PERSON}){words}\s*{_NUMBER}{_YEARS}", text) or re.fullmatch(rf"{_AGE_WORD}\s*{words}\s*{_NUMBER}", text)
        if match:
            return operator, float(match[1])
    match = re.fullmatch(rf"(?:{_AGE_WORD}\s*{_NUMBER}{_YEARS}?|{_NUMBER}{_YEARS})\s*(?:or|and)\s*(older|over|above|younger|under|below)", text)
    if match:
        return (">=" if match[3] in ("older", "over", "above") else "<="), float(match[1] or match[2])
    return None


def _note_age(sentences):
    found = []
    for sentence_id, sentence in sentences.items():
        text = sentence.lower()
        ages = [float(a or b) for a, b in _NOTE_AGE.findall(text)]
        if ages and _RELATIVE.search(text):
            return None
        found.extend((age, sentence_id) for age in ages)
    if len({age for age, _ in found}) != 1:
        return None
    return found[0]


def _note_value(measure, sentences):
    found = []
    for sentence_id, sentence in sentences.items():
        text = sentence.lower()
        if not re.search(rf"\b(?:{measure.alias})\b", text):
            continue
        matches = list(re.finditer(rf"\b(?:{measure.alias}){_LINK}{_NUMBER}", text))
        if len(matches) != 1 or _HEDGE.search(text):
            return None
        after = text[matches[0].end():]
        has_unit = re.match(rf"\s*(?:{measure.unit})", after) is not None
        if _RANGE_AFTER.match(after) or (not has_unit and (measure.note_needs_unit or _FOREIGN_UNIT.match(after))):
            return None
        found.append((float(matches[0][1]), sentence_id))
    if len({value for value, _ in found}) != 1:
        return None
    return found[0]


def check_number(criterion_type, criterion, sentences):
    """The verdict arithmetic gives for this criterion, or None if it cannot be decided safely."""
    text = _normalize(criterion)
    rule = _age_rule(text)
    found = _note_age(sentences) if rule else None
    if rule is None:
        parsed = _measure_rule(text)
        if parsed is None:
            return None
        measure, rule = parsed
        found = _note_value(measure, sentences)
    if found is None:
        return None
    value, sentence_id = found
    holds = _holds(value, rule)
    if criterion_type == "inclusion":
        label = "included" if holds else "not included"
    else:
        label = "excluded" if holds else "not excluded"
    return NumberResult(label, sentence_id, value)
