from screener.eval.metrics import evidence_micro, score_predictions
from screener.eval.verifier import INVALID, verify_output
from screener.data.prepare import Example

SENTENCES = {0: "First.", 1: "Second.", 2: "Third."}


def test_valid_output():
    result = verify_output('{"label": "included", "evidence": [2, 0, 2]}', "inclusion", SENTENCES)
    assert result.valid and result.label == "included" and result.evidence == (0, 2)


def test_label_must_match_criterion_type():
    result = verify_output('{"label": "excluded", "evidence": []}', "inclusion", SENTENCES)
    assert not result.valid and result.label == INVALID


def test_invalid_output_is_never_repaired_into_a_verdict():
    for raw in ("", "not json", '{"evidence": [0]}', '{"label": "included"}', '{"label": "maybe", "evidence": []}', "[1, 2]"):
        result = verify_output(raw, "inclusion", SENTENCES)
        assert not result.valid and result.label == INVALID


def test_unknown_ids_are_dropped_and_reported():
    result = verify_output('{"label": "not excluded", "evidence": [1, 9, "x", true]}', "exclusion", SENTENCES)
    assert result.valid and result.evidence == (1,)
    assert len(result.dropped_ids) == 3 and result.errors


def test_extra_text_around_json_is_tolerated_and_flagged():
    result = verify_output('Sure!\n{"label": "not applicable", "evidence": []}\nDone', "inclusion", SENTENCES)
    assert result.valid and result.extra_text


def test_empty_evidence_is_not_turned_into_a_label():
    result = verify_output('{"label": "not enough information", "evidence": []}', "exclusion", SENTENCES)
    assert result.label == "not enough information" and result.evidence == ()


def test_evidence_micro_by_hand():
    scores = evidence_micro([(0, 1), (), (2,)], [(0,), (1,), ()])
    assert scores["precision"] == 0.5
    assert scores["recall"] == 1 / 3
    assert abs(scores["f1"] - 0.4) < 1e-12


def test_invalid_counts_as_wrong():
    example = Example(1, "p", "t", SENTENCES, "inclusion", "c", "included", ())
    bad = verify_output("nonsense", "inclusion", SENTENCES)
    good = verify_output('{"label": "included", "evidence": []}', "inclusion", SENTENCES)
    assert score_predictions([example], [bad])["accuracy"] == 0.0
    assert score_predictions([example], [good])["accuracy"] == 1.0
    assert score_predictions([example], [bad])["invalid_rate"] == 1.0
