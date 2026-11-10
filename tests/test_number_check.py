"""The number check must get simple cases right and stay silent on anything unclear."""
import pytest

from app.number_check import check_number
from app.pipeline import build_record, build_report, overall_result
from app.report import render_report_html
from screener.eval.verifier import verify_output


def note(*sentences):
    return dict(enumerate(sentences))


BMI = note("A 52-year-old man attends clinic.", "His BMI is 33 kg/m2.")


@pytest.mark.parametrize("kind, criterion, expected", [
    ("inclusion", "BMI of at least 30 kg/m2", "included"),
    ("inclusion", "BMI below 25 kg/m2", "not included"),
    ("inclusion", "BMI between 30 and 35", "included"),
    ("exclusion", "BMI > 40", "not excluded"),
    ("exclusion", "BMI ≥ 33", "excluded"),
    ("inclusion", "Age 18 years or older", "included"),
    ("inclusion", "Age 18 to 50 years", "not included"),
    ("exclusion", "Age under 18 years", "not excluded"),
])
def test_simple_comparisons(kind, criterion, expected):
    assert check_number(kind, criterion, BMI).label == expected


def test_it_cites_the_sentence_holding_the_value():
    assert check_number("inclusion", "BMI of at least 30", BMI).sentence_id == 1


@pytest.mark.parametrize("criterion", [
    "BMI of at least 30 or documented obesity",
    "Age 2 months to 20 years of age",
    "Stable BMI",
    "Creatinine clearance below 20 ml/min",
    "BMI of at least 30 kg/m3",
])
def test_anything_more_than_a_bare_comparison_is_left_alone(criterion):
    assert check_number("inclusion", criterion, BMI) is None


@pytest.mark.parametrize("sentence", [
    "Her target BMI is 25.",
    "Her BMI was 33 two years ago.",
    "His BMI rose to 33.",
    "BMI 30 to 33.",
    "BMI 33 today and BMI 35 at the first visit.",
    "BMI was not recorded.",
])
def test_unclear_note_values_are_left_alone(sentence):
    assert check_number("inclusion", "BMI of at least 30", note(sentence)) is None


def test_conflicting_values_across_sentences_are_left_alone():
    assert check_number("inclusion", "BMI of at least 30", note("BMI is 33.", "BMI is 28.")) is None


@pytest.mark.parametrize("sentence", ["Creatinine 120 umol/L.", "Creatinine was 1.4."])
def test_lab_values_need_the_expected_unit(sentence):
    assert check_number("exclusion", "Creatinine above 2.0 mg/dL", note(sentence)) is None


def test_lab_value_with_the_expected_unit_is_decided():
    assert check_number("exclusion", "Creatinine above 2.0 mg/dL", note("Creatinine was 1.4 mg/dL.")).label == "not excluded"


def test_a_relatives_age_is_not_the_patients():
    assert check_number("inclusion", "Age 18 years or older", note("Her 70-year-old mother has diabetes.")) is None


def test_ages_in_months_are_left_alone():
    assert check_number("inclusion", "Age under 12 years", note("An 18-month-old boy has a cough.")) is None


def verdict(label, evidence, kind, criterion, sentences=BMI):
    return build_record(0, kind, criterion, verify_output(f'{{"label": "{label}", "evidence": {evidence}}}', kind, sentences), sentences, {})


def test_a_wrong_verdict_is_corrected_and_cites_the_value():
    item = verdict("not included", "[1]", "inclusion", "BMI of at least 30 kg/m2")
    assert (item["label"], item["evidence_ids"], item["adjusted"], item["model_label"]) == ("included", [1], True, "not included")
    assert "meets this inclusion criterion" in item["explanation"]
    assert overall_result([item])["outcome"] == "Potentially eligible"


def test_an_agreeing_verdict_is_untouched():
    item = verdict("included", "[1]", "inclusion", "BMI of at least 30 kg/m2")
    assert item["adjusted"] is False and item["evidence_ids"] == [1]


def test_invalid_model_output_is_not_rescued():
    item = build_record(0, "inclusion", "BMI of at least 30", verify_output("junk", "inclusion", BMI), BMI, {})
    assert item["adjusted"] is False and item["valid"] is False


def test_the_report_carries_only_the_short_note():
    item = verdict("not included", "[1]", "inclusion", "BMI of at least 30 kg/m2")
    html = render_report_html(build_report("a" * 32, BMI, [item], "cached", {}))
    assert "Adjusted by number check." in html and "Model said" not in html
