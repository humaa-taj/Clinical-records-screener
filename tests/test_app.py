"""API tests with a stand-in engine, so no GPU or model is needed."""
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.pipeline import (
    NOTICE,
    InputError,
    build_record,
    missing_to_show,
    overall_result,
    parse_eligibility,
    build_report,
    split_note,
    validate_criteria,
    validate_note,
)
from app.engine import ROOT, pinned_versions
from app.report import render_report_html
from app.server import create_app
from screener.eval.verifier import verify_output

SENTENCES = ["She is 58 years old.", "HbA1c was 8.4 percent."]
NOTE = {0: SENTENCES[0], 1: SENTENCES[1]}


def record(label, evidence, kind="inclusion", index=0):
    verified = verify_output(f'{{"label": "{label}", "evidence": {evidence}}}', kind, NOTE)
    return build_record(index, kind, "criterion", verified, NOTE, {})


class FakeEngine:
    state = "ready"
    error = None

    def __init__(self, delay=0.0, failure=None):
        self.delay, self.failure = delay, failure

    def info(self):
        return {"state": self.state, "error": self.error}

    def provenance(self):
        return {"target_model": "fake", "adapter": None}

    def screen_one(self, index, sentences, kind, text, mode):
        time.sleep(self.delay)
        if self.failure:
            raise self.failure
        label = "included" if kind == "inclusion" else "not excluded"
        return build_record(index, kind, text, verify_output(f'{{"label": "{label}", "evidence": [0]}}', kind, sentences), sentences, {"mode": mode})

    def compare_decoders(self, *args):
        return {"rows": []}

    def compare_attention(self, length, tile):
        return {"length": length, "tile": tile}


def client_for(tmp_path, engine=None, trial_client=None):
    return TestClient(create_app(engine or FakeEngine(), tmp_path / "reports", tmp_path / "static", trial_client))


def wait(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


@pytest.mark.parametrize("text", ["", "   ", "1234 5678 90", "bad\x00file", "a" * 20_001])
def test_bad_notes_are_rejected(text):
    with pytest.raises(InputError):
        validate_note(text)


def test_note_is_split_into_numbered_order():
    assert split_note("She is 58 years old. HbA1c was 8.4 percent.") == SENTENCES


def test_criteria_validation():
    assert validate_criteria([" A ", ""], ["B"]) == [("inclusion", "A"), ("exclusion", "B")]
    for inclusion, exclusion in (([], []), ([""], [" "]), (["x"] * 41, []), (["x" * 1001], [])):
        with pytest.raises(InputError):
            validate_criteria(inclusion, exclusion)


def test_cited_failure_makes_likely_ineligible():
    records = [record("not included", "[0]"), record("included", "[1]")]
    assert overall_result(records)["outcome"] == "Likely ineligible"


def test_uncited_failure_only_needs_review():
    assert overall_result([record("not included", "[]")])["outcome"] == "Needs review"


def test_all_met_with_evidence_is_potentially_eligible():
    assert overall_result([record("included", "[0]"), record("not excluded", "[1]", "exclusion")])["outcome"] == "Potentially eligible"


@pytest.mark.parametrize("label,kind,evidence", [("not enough information", "inclusion", "[0]"), ("not excluded", "exclusion", "[]")])
def test_unknown_or_uncited_needs_review(label, kind, evidence):
    assert overall_result([record(label, evidence, kind)])["outcome"] == "Needs review"


def test_not_applicable_without_evidence_is_not_flagged():
    item = record("not applicable", "[]")
    assert item["unconfirmed"] is False
    assert overall_result([item])["outcome"] == "Potentially eligible"


def test_invalid_output_needs_review_and_is_labeled():
    item = build_record(0, "inclusion", "c", verify_output("junk", "inclusion", NOTE), NOTE, {})
    assert item["valid"] is False and "could not be verified" in item["explanation"]
    assert overall_result([item])["counts"]["unverified"] == 1
    assert overall_result([item])["outcome"] == "Needs review"


def test_overall_is_always_preliminary():
    result = overall_result([record("included", "[0]")])
    assert result["preliminary"] is True and "not a ground-truth" in result["rule"]


def test_report_html_has_notice_and_escapes_input():
    item = record("included", "[0]")
    item["criterion"] = "<script>alert(1)</script>"
    html = render_report_html(build_report("a" * 32, NOTE, [item], "cached", {"target_model": "m"}))
    assert NOTICE in html and "<script>alert" not in html and "&lt;script&gt;" in html


def test_parse_eligibility_sections():
    text = "Inclusion Criteria:\n\n* Age over 18\n* Diabetes\n\nExclusion Criteria:\n\n1. Pregnancy\n"
    assert parse_eligibility(text) == {"inclusion": ["Age over 18", "Diabetes"], "exclusion": ["Pregnancy"]}


def test_sentences_endpoint_error_is_readable(tmp_path):
    response = client_for(tmp_path).post("/api/sentences", json={"note": ""})
    assert response.status_code == 400 and response.json()["message"] == "Enter a patient note first."


def test_screen_flow_saves_a_downloadable_report(tmp_path):
    client = client_for(tmp_path)
    job_id = client.post("/api/screen", json={"sentences": SENTENCES, "inclusion": ["Adult"], "exclusion": ["Pregnancy"]}).json()["id"]
    job = wait(client, job_id)
    assert job["state"] == "done" and job["done"] == 2 and len(job["results"]) == 2
    assert job["results"][0]["evidence"] == [{"id": 0, "text": SENTENCES[0]}]
    report_id = job["report_id"]
    assert client.get("/api/reports").json()["reports"][0]["id"] == report_id
    assert client.get(f"/api/reports/{report_id}").json()["notice"] == NOTICE
    assert "attachment" in client.get(f"/api/reports/{report_id}/download.html").headers["content-disposition"]
    assert client.get(f"/api/reports/{report_id}/download.json").status_code == 200


def test_screen_rejects_bad_input_and_can_recover(tmp_path):
    client = client_for(tmp_path)
    assert client.post("/api/screen", json={"sentences": SENTENCES, "inclusion": [], "exclusion": []}).status_code == 400
    assert client.post("/api/screen", json={"sentences": SENTENCES, "inclusion": ["A"], "mode": "fast"}).status_code == 400
    assert client.post("/api/screen", json={"sentences": ["", "x"], "inclusion": ["A"]}).status_code == 400
    assert client.post("/api/screen", json={"sentences": SENTENCES, "inclusion": ["A"]}).status_code == 200


def test_only_one_job_runs_at_a_time_and_cancel_works(tmp_path):
    client = client_for(tmp_path, FakeEngine(delay=0.05))
    body = {"sentences": SENTENCES, "inclusion": ["A", "B", "C", "D", "E"]}
    job_id = client.post("/api/screen", json=body).json()["id"]
    busy = client.post("/api/screen", json=body)
    assert busy.status_code == 409 and "in progress" in busy.json()["message"]
    client.post(f"/api/jobs/{job_id}/cancel")
    job = wait(client, job_id)
    assert job["state"] == "cancelled" and job["done"] < 5
    assert client.post("/api/screen", json=body).status_code == 200


def test_engine_failure_is_reported_and_clears_the_queue(tmp_path):
    client = client_for(tmp_path, FakeEngine(failure=RuntimeError("boom")))
    body = {"sentences": SENTENCES, "inclusion": ["A"]}
    job = wait(client, client.post("/api/screen", json=body).json()["id"])
    assert job["state"] == "failed" and "boom" in job["error"]
    assert client.post("/api/screen", json=body).status_code == 200


def test_model_not_ready_returns_readable_error(tmp_path):
    engine = FakeEngine()
    engine.state = "loading"
    response = client_for(tmp_path, engine).post("/api/screen", json={"sentences": SENTENCES, "inclusion": ["A"]})
    assert response.status_code == 503 and "loading" in response.json()["message"]


@pytest.mark.parametrize("report_id", ["../secrets", "notahexid", "A" * 32])
def test_report_ids_cannot_escape_the_folder(tmp_path, report_id):
    assert client_for(tmp_path).get(f"/api/reports/{report_id}").status_code == 404


def test_comparison_endpoints_validate_and_run(tmp_path):
    client = client_for(tmp_path)
    assert client.post("/api/compare/attention", json={"length": 4, "tile": 8}).status_code == 400
    job = wait(client, client.post("/api/compare/attention", json={"length": 64, "tile": 16}).json()["id"])
    assert job["output"] == {"length": 64, "tile": 16}
    job = wait(client, client.post("/api/compare/decoding", json={"sentences": SENTENCES, "criterion": "Adult"}).json()["id"])
    assert job["state"] == "done"


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload or {}

    def json(self):
        return self._payload


class FakeTrialClient:
    def __init__(self, result):
        self.result = result

    def get(self, *args, **kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_trial_import(tmp_path):
    payload = {"protocolSection": {"eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n* Adult\nExclusion Criteria:\n* Pregnancy"},
                                   "identificationModule": {"briefTitle": "Study"}}}
    body = client_for(tmp_path, trial_client=FakeTrialClient(FakeResponse(200, payload))).get("/api/trial/nct01234567").json()
    assert body == {"id": "NCT01234567", "title": "Study", "inclusion": ["Adult"], "exclusion": ["Pregnancy"]}


@pytest.mark.parametrize("result,code", [(FakeResponse(404), 404), (httpx.ConnectError("offline"), 502), (FakeResponse(500), 502)])
def test_trial_failures_are_readable(tmp_path, result, code):
    response = client_for(tmp_path, trial_client=FakeTrialClient(result)).get("/api/trial/NCT01234567")
    assert response.status_code == code and response.json()["message"]


def test_bad_trial_id(tmp_path):
    assert client_for(tmp_path).get("/api/trial/abc").status_code == 400


def test_frontend_fallback_and_static_path_safety(tmp_path):
    client = client_for(tmp_path)
    (tmp_path / "static").mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    (tmp_path / "static" / "index.html").write_text("<p>app</p>")
    assert client.get("/results").text == "<p>app</p>"
    assert "nope" not in client.get("/%2e%2e/secret.txt").text
    assert client.get("/api/unknown").status_code == 404


def test_reason_names_each_problem_without_double_counting():
    records = [record("not enough information", "[]"), record("not excluded", "[]", "exclusion"), record("included", "[0]")]
    assert overall_result(records)["reason"] == "1 criterion has too little information in the note and 1 verdict cites no sentence."


def test_reason_for_a_cited_failure():
    assert overall_result([record("not included", "[0]")])["reason"] == "1 criterion failed with cited evidence."


def test_reason_for_a_clean_pass_and_for_nothing_applicable():
    assert overall_result([record("included", "[0]")])["reason"] == "Every applicable criterion was met with cited evidence."
    assert overall_result([record("not applicable", "[]")])["reason"] == "None of the criteria applied to this patient."


def test_uncited_not_applicable_does_not_ask_for_review():
    assert "review" not in record("not applicable", "[]")["explanation"]


def test_missing_information_is_not_repeated():
    assert missing_to_show(record("not enough information", "[0]")) == []
    assert missing_to_show(record("not enough information", "[]")) == ["The note does not contain enough to settle this criterion."]
    assert missing_to_show(record("not excluded", "[]", "exclusion")) == []


def full_report():
    records = [record("included", "[0]"), record("not excluded", "[]", "exclusion", 1)]
    return build_report("a" * 32, NOTE, records, "cached", {"target_model": "m"})


def test_report_is_sectioned_and_states_the_reason():
    html = render_report_html(full_report())
    for text in ("Inclusion criteria", "Exclusion criteria", "Patient note", "Target model", "1 verdict cites no sentence."):
        assert text in html


def test_report_is_self_contained():
    html = render_report_html(full_report())
    assert "font/woff2;base64," in html
    assert "http://" not in html and "https://" not in html and "<script" not in html


def test_report_marks_only_cited_sentences_and_does_not_repeat_quotes():
    html = render_report_html(full_report())
    assert html.count("<li class=cited>") == 1 and "Cited sentences:" not in html


def test_reports_saved_before_reasons_still_render():
    report = full_report()
    del report["overall"]["reason"]
    assert "1 verdict cites no sentence." in render_report_html(report)


def test_report_omits_an_empty_group():
    report = build_report("a" * 32, NOTE, [record("included", "[0]")], "cached", {})
    assert "Exclusion criteria" not in render_report_html(report)


def test_reports_record_exact_model_and_adapter_versions():
    lock = json.loads((ROOT / "adapters.lock.json").read_text(encoding="utf-8"))["crs-adapter"]
    versions = pinned_versions("models/qwen2.5-1.5b", "models/crs-adapter")
    assert versions["adapter_revision"] == f"{lock['repo']}@{lock['revision']}"
    assert versions["target_revision"].startswith("Qwen/Qwen2.5-1.5B-Instruct@")


def test_unknown_model_folders_are_not_given_a_version():
    assert pinned_versions("models/other", None) == {"target_revision": "not pinned", "adapter_revision": "not pinned"}
