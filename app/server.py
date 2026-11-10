"""FastAPI server for CRS. Serves the JSON API and the built browser application.

One job runs at a time, because a single GPU model is shared. Usage: python -m app.server
"""
import json
import re
import threading
import uuid
from pathlib import Path

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.pipeline import (
    EXAMPLE,
    TRIAL_ID,
    InputError,
    build_report,
    overall_result,
    parse_eligibility,
    split_note,
    validate_criteria,
    validate_mode,
    validate_sentences,
)
from app.report import render_report_html

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_STATIC = Path(__file__).resolve().parent / "web" / "dist"
_REPORT_ID = re.compile(r"^[0-9a-f]{32}$")
CLINICAL_TRIALS = "https://clinicaltrials.gov/api/v2/studies/{}"
MEMORY_MESSAGE = "The GPU ran out of memory. Close other GPU-heavy programs and try again."


class NoteIn(BaseModel):
    note: str


class ScreenIn(BaseModel):
    sentences: list[str]
    inclusion: list[str] = []
    exclusion: list[str] = []
    mode: str = "cached"


class CompareIn(BaseModel):
    sentences: list[str]
    criterion_type: str = "inclusion"
    criterion: str
    max_new_tokens: int = 64
    ignore_eos: bool = False
    draft_length: int = 4


class AttentionIn(BaseModel):
    length: int = 512
    tile: int = 64


def create_app(engine, reports_dir=DEFAULT_REPORTS, static_dir=DEFAULT_STATIC, trial_client=None):
    app = FastAPI(title="CRS")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    static_dir = Path(static_dir)
    lock = threading.Lock()
    jobs, active = {}, {"id": None}

    @app.exception_handler(InputError)
    async def on_input_error(_: Request, error: InputError):
        return JSONResponse(status_code=400, content={"message": str(error)})

    @app.exception_handler(Exception)
    async def on_unexpected(_: Request, error: Exception):
        return JSONResponse(status_code=500, content={"message": f"Something went wrong: {error}"})

    def require_ready():
        if engine.state != "ready":
            raise HTTPException(503, detail="The model is still loading." if engine.state == "loading" else engine.error)

    def start_job(kind, total, work):
        """Run work(job) on a background thread, refusing if another job is running."""
        with lock:
            if active["id"] is not None:
                raise HTTPException(409, detail="Another run is in progress. Wait for it or cancel it.")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "kind": kind, "state": "running", "done": 0, "total": total, "results": [], "cancel": False,
                   "error": None, "overall": None, "report_id": None, "output": None}
            jobs[job_id] = job
            active["id"] = job_id

        def runner():
            try:
                work(job)
                if job["state"] == "running":
                    job["state"] = "done"
            except InputError as error:
                job.update(state="failed", error=str(error))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                job.update(state="failed", error=MEMORY_MESSAGE)
            except Exception as error:
                job.update(state="failed", error=f"The run failed: {error}")
            finally:
                with lock:
                    active["id"] = None

        threading.Thread(target=runner, daemon=True).start()
        return {"id": job_id}

    @app.get("/api/status")
    def status():
        return {**engine.info(), "busy": active["id"] is not None}

    @app.get("/api/example")
    def example():
        return EXAMPLE

    @app.post("/api/sentences")
    def sentences(body: NoteIn):
        return {"sentences": split_note(body.note)}

    @app.get("/api/trial/{trial_id}")
    def trial(trial_id: str):
        trial_id = trial_id.strip().upper()
        if not TRIAL_ID.match(trial_id):
            raise InputError("Enter a trial ID such as NCT01234567.")
        fields = "protocolSection.eligibilityModule.eligibilityCriteria,protocolSection.identificationModule.briefTitle"
        client = trial_client or httpx
        try:
            response = client.get(CLINICAL_TRIALS.format(trial_id), params={"fields": fields}, timeout=15)
        except httpx.HTTPError as error:
            raise HTTPException(502, detail="ClinicalTrials.gov could not be reached. Enter the criteria by hand.") from error
        if response.status_code == 404:
            raise HTTPException(404, detail=f"No trial was found for {trial_id}.")
        if response.status_code != 200:
            raise HTTPException(502, detail="ClinicalTrials.gov returned an error. Enter the criteria by hand.")
        section = response.json().get("protocolSection", {})
        parsed = parse_eligibility(section.get("eligibilityModule", {}).get("eligibilityCriteria", ""))
        if not parsed["inclusion"] and not parsed["exclusion"]:
            raise HTTPException(422, detail="That trial's criteria could not be split automatically. Enter them by hand.")
        return {"id": trial_id, "title": section.get("identificationModule", {}).get("briefTitle", ""), **parsed}

    @app.post("/api/screen")
    def screen(body: ScreenIn):
        require_ready()
        mode = validate_mode(body.mode)
        note = validate_sentences(body.sentences)
        items = validate_criteria(body.inclusion, body.exclusion)

        def work(job):
            for index, (kind, text) in enumerate(items):
                if job["cancel"]:
                    job["state"] = "cancelled"
                    return
                job["results"].append(engine.screen_one(index, note, kind, text, mode))
                job["done"] = index + 1
            job["overall"] = overall_result(job["results"])
            report = build_report(uuid.uuid4().hex, note, job["results"], mode, engine.provenance())
            (reports_dir / f"{report['id']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            job["report_id"] = report["id"]

        return start_job("screen", len(items), work)

    @app.post("/api/compare/decoding")
    def compare_decoding(body: CompareIn):
        require_ready()
        note = validate_sentences(body.sentences)
        items = validate_criteria([body.criterion] if body.criterion_type == "inclusion" else [], [body.criterion] if body.criterion_type == "exclusion" else [])
        limit = max(1, min(body.max_new_tokens, 256))
        draft_length = max(1, min(body.draft_length, 16))

        def work(job):
            job["output"] = engine.compare_decoders(note, items[0][0], items[0][1], limit, body.ignore_eos, draft_length)
            job["done"] = 1

        return start_job("decoding", 1, work)

    @app.post("/api/compare/attention")
    def compare_attention(body: AttentionIn):
        require_ready()
        if not 8 <= body.length <= 4096:
            raise InputError("Sequence length must be between 8 and 4,096.")
        if not 8 <= body.tile <= body.length:
            raise InputError("Tile size must be between 8 and the sequence length.")

        def work(job):
            job["output"] = engine.compare_attention(body.length, body.tile)
            job["done"] = 1

        return start_job("attention", 1, work)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, detail="That run was not found.")
        return {key: value for key, value in jobs[job_id].items() if key != "cancel"}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, detail="That run was not found.")
        jobs[job_id]["cancel"] = True
        return {"cancelling": True}

    def report_path(report_id):
        if not _REPORT_ID.match(report_id):
            raise HTTPException(404, detail="That report was not found.")
        path = reports_dir / f"{report_id}.json"
        if not path.exists():
            raise HTTPException(404, detail="That report was not found.")
        return path

    @app.get("/api/reports")
    def reports():
        rows = []
        for path in sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"id": data["id"], "created_utc": data["created_utc"], "outcome": data["overall"]["outcome"],
                         "criteria": len(data["results"]), "mode": data["provenance"]["decoding_mode"]})
        return {"reports": rows}

    @app.get("/api/reports/{report_id}")
    def report_json(report_id: str):
        return json.loads(report_path(report_id).read_text(encoding="utf-8"))

    @app.get("/api/reports/{report_id}/download.json")
    def report_download_json(report_id: str):
        return FileResponse(report_path(report_id), media_type="application/json", filename=f"crs-report-{report_id[:8]}.json")

    @app.get("/api/reports/{report_id}/download.html")
    def report_download_html(report_id: str):
        report = json.loads(report_path(report_id).read_text(encoding="utf-8"))
        headers = {"Content-Disposition": f'attachment; filename="crs-report-{report_id[:8]}.html"'}
        return HTMLResponse(render_report_html(report), headers=headers)

    @app.exception_handler(HTTPException)
    async def on_http_error(_: Request, error: HTTPException):
        return JSONResponse(status_code=error.status_code, content={"message": error.detail})

    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, detail="Not found.")
        file = (static_dir / path).resolve()
        if path and file.is_file() and file.is_relative_to(static_dir.resolve()):
            return FileResponse(file)
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse("<p>The interface has not been built yet.</p>", status_code=503)

    return app


def main():
    import uvicorn

    from app.engine import Engine

    engine = Engine()
    threading.Thread(target=engine.load, daemon=True).start()
    print("CRS is running at http://127.0.0.1:8600  (close this window to stop it)", flush=True)
    uvicorn.run(create_app(engine), host="127.0.0.1", port=8600, log_level="warning")


if __name__ == "__main__":
    main()
