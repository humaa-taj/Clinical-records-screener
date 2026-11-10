import pytest

from screener.data.prompts import ALL_LABELS
from screener.eval.explain import NO_EVIDENCE, explain

SENTENCES = {0: "HbA1c was 7.2 percent.", 1: "She has no fever."}


def test_no_evidence_is_flagged_as_unconfirmed():
    assert explain("not excluded", (), SENTENCES) == NO_EVIDENCE


def test_cited_sentences_are_quoted_exactly():
    assert "[0] HbA1c was 7.2 percent." in explain("included", (0,), SENTENCES)


def test_uncited_sentences_are_left_out():
    assert "no fever" not in explain("included", (0,), SENTENCES)


@pytest.mark.parametrize("label", ALL_LABELS)
def test_every_label_has_wording(label):
    assert explain(label, (1,), SENTENCES).endswith("[1] She has no fever.")
