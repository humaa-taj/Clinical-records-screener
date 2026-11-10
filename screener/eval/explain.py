"""Explanations built directly from the sentences a prediction cites.

No model writes these. The dataset's only written explanations come from GPT-4 and are off limits,
and a tested base-model paraphrase invented patient facts, so each explanation states the verdict
in fixed wording and quotes the cited sentences exactly.
"""

NO_EVIDENCE = (
    "No sentence in the note was cited for this criterion, so the verdict is unconfirmed "
    "and needs human review. Missing information is not proof either way."
)
_VERDICTS = {
    "included": "The note indicates the patient meets this inclusion criterion.",
    "not included": "The note indicates the patient does not meet this inclusion criterion.",
    "excluded": "The note indicates the patient has this exclusion condition.",
    "not excluded": "The note indicates the patient does not have this exclusion condition.",
    "not enough information": "The note does not settle this criterion.",
    "not applicable": "This criterion does not apply to this patient.",
}


def explain(label, evidence, sentences):
    """Explain one verified prediction using only its cited sentences."""
    if not evidence:
        return NO_EVIDENCE
    quoted = " ".join(f"[{i}] {sentences[i]}" for i in evidence)
    return f"{_VERDICTS[label]} Cited sentences: {quoted}"
