"""The downloadable HTML report: one self-contained, printable file.

Every value from the report is escaped. Fonts are embedded so the file looks the same on any machine
and needs no network connection.
"""
import base64
import html
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from app.pipeline import FAILED, MET, NOT_APPLICABLE, UNKNOWN, missing_to_show, overall_reason

FONTS = Path(__file__).resolve().parent / "assets" / "fonts"
ADJUSTED_NOTE = "Adjusted by number check."
_OUTCOME_CLASSES = {"Likely ineligible": "failed", "Needs review": "unknown", "Potentially eligible": "met"}
_COUNTS = (
    ("met", "Met"),
    ("failed", "Failed"),
    ("unknown", "Unknown"),
    ("not_applicable", "Not applicable"),
    ("unverified", "Unverified"),
    ("no_evidence", "No citation"),
)
_FIELD_NAMES = {"note_sha256": "Note SHA-256", "target_model": "Target model", "output_order": "Output order", "decoding_mode": "Decoding mode", "target_revision": "Target revision", "adapter_revision": "Adapter revision"}

_STYLE = """
:root{--ink:#16181d;--soft:#4a505b;--faint:#6b7280;--line:#e3e5e9;--wash:#f4f5f7}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;background:#eceef1;color:var(--ink);font:400 14px/1.55 Geist,'Segoe UI',system-ui,sans-serif}
.sheet{max-width:820px;margin:32px auto;padding:48px 56px;background:#fff;box-shadow:0 12px 40px -18px rgb(20 24 32/.35)}
h1,h2,h3,.brand,.outcome{font-family:'Space Grotesk',Geist,'Segoe UI',sans-serif;margin:0}
.mast{display:flex;justify-content:space-between;align-items:flex-end;gap:32px;padding-bottom:18px;border-bottom:2px solid var(--ink)}
.brand{font-size:13px;font-weight:600;letter-spacing:.16em}
h1{margin-top:4px;font-size:27px;font-weight:500;line-height:1.2}
.meta{margin:0;text-align:right;font-size:12.5px;color:var(--soft);white-space:nowrap}
.notice{margin:14px 0 0;font-size:12.5px;font-weight:500;color:var(--soft)}
.summary{display:grid;grid-template-columns:1fr auto;gap:8px 32px;align-items:end;margin-top:26px;padding:22px 24px;background:var(--wash);border-left:4px solid var(--accent)}
.summary.met{--accent:#1f8a5b}.summary.failed{--accent:#c23a3a}.summary.unknown{--accent:#b7861a}
.eyebrow{margin:0;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.outcome{margin-top:3px;font-size:30px;font-weight:500;line-height:1.15}
.reason{grid-column:1/-1;margin:2px 0 0;font-size:15px;color:var(--soft)}
.counts{display:flex;gap:18px;margin:0;padding:0;list-style:none}
.counts li{min-width:44px;text-align:center}
.counts b{display:block;font:500 21px/1.1 'Space Grotesk',Geist,sans-serif;font-variant-numeric:tabular-nums}
.counts span{display:block;margin-top:3px;font-size:10.5px;line-height:1.2;color:var(--faint)}
section.block{margin-top:30px}
h2{padding-bottom:7px;border-bottom:1px solid var(--line);font-size:13px;font-weight:600;letter-spacing:.1em;text-transform:uppercase}
.crit{padding:14px 0;border-bottom:1px solid var(--line);break-inside:avoid}
.crit:last-child{border-bottom:0}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}
h3{font:500 15px/1.4 Geist,'Segoe UI',sans-serif}
.tag{flex:none;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:500;white-space:nowrap}
.tag.met{background:#e2f4ea;color:#14663f}.tag.failed{background:#fbe6e6;color:#962a2a}.tag.unknown{background:#fbf1d9;color:#7d5a0c}.tag.na{background:#ebedf0;color:#4a505b}.tag.invalid{background:#efe8fa;color:#57359a}
.why{margin:5px 0 0;color:var(--soft)}
.missing{margin:5px 0 0;font-size:13px;color:var(--faint)}
.adjusted{margin:5px 0 0;font-size:12px;color:var(--faint)}
.quotes{margin:9px 0 0;padding:0;list-style:none}
.quotes li{display:flex;gap:10px;padding:2px 0 2px 12px;border-left:2px solid var(--line)}
.sid{flex:none;min-width:20px;font-size:12px;line-height:21px;color:var(--faint);font-variant-numeric:tabular-nums}
.note{margin:12px 0 0;padding:0;list-style:none}
.note li{display:flex;gap:10px;padding:3px 8px;border-radius:4px}
.note li.cited{background:var(--wash)}
.note li.cited .sid{color:var(--ink);font-weight:600}
.key{margin:8px 0 0;font-size:12px;color:var(--faint)}
.method{margin-top:34px;padding-top:12px;border-top:2px solid var(--ink);font-size:12px;color:var(--soft)}
.method dl{display:grid;grid-template-columns:max-content 1fr;gap:3px 18px;margin:0}
.method dt{color:var(--faint)}.method dd{margin:0;overflow-wrap:anywhere}
.method p{margin:10px 0 0}
@media(max-width:640px){.sheet{margin:0;padding:28px 20px}.mast{display:block}.meta{margin-top:10px;text-align:left}.summary{grid-template-columns:1fr}.counts{flex-wrap:wrap}.head{flex-direction:column;gap:6px}}
@page{size:A4;margin:14mm}
@media print{body{background:#fff;font-size:12.5px}.sheet{max-width:none;margin:0;padding:0;box-shadow:none}h2{break-after:avoid}}
"""


@lru_cache(maxsize=1)
def _font_faces():
    faces = []
    for family, file in (("Geist", "geist-latin-wght-normal.woff2"), ("Space Grotesk", "space-grotesk-latin-wght-normal.woff2")):
        path = FONTS / file
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            faces.append(f"@font-face{{font-family:'{family}';font-weight:300 700;src:url(data:font/woff2;base64,{data}) format('woff2')}}")
    return "".join(faces)


def _tag_class(label):
    if label in MET:
        return "met"
    if label in FAILED:
        return "failed"
    if label == UNKNOWN:
        return "unknown"
    return "na" if label == NOT_APPLICABLE else "invalid"


def _label_text(label):
    text = "Unverified output" if label == "__invalid__" else label
    return text[0].upper() + text[1:]


def _date(stamp):
    try:
        moment = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return str(stamp)
    return f"{moment.day} {moment:%B %Y, %H:%M} UTC"


def _criterion(record, e):
    quotes = "".join(f"<li><span class='sid'>{e(item['id'])}</span><span>{e(item['text'])}</span></li>" for item in record["evidence"])
    quotes = f"<ul class='quotes'>{quotes}</ul>" if quotes else ""
    # The cited sentences are quoted just above, so the explanation does not repeat them.
    why = record["explanation"].split(" Cited sentences:")[0]
    missing = "".join(f"<p class='missing'>{e(text)}</p>" for text in missing_to_show(record))
    adjusted = f"<p class='adjusted'>{ADJUSTED_NOTE}</p>" if record.get("adjusted") else ""
    return (
        f"<article class='crit'><div class='head'><h3>{e(record['criterion'])}</h3>"
        f"<span class='tag {_tag_class(record['label'])}'>{e(_label_text(record['label']))}</span></div>"
        f"{quotes}<p class='why'>{e(why)}</p>{missing}{adjusted}</article>"
    )


def _group(title, kind, records, e):
    rows = "".join(_criterion(r, e) for r in records if r["criterion_type"] == kind)
    return f"<section class='block'><h2>{title}</h2>{rows}</section>" if rows else ""


def render_report_html(report):
    e = lambda value: html.escape(str(value), quote=True)
    overall, records = report["overall"], report["results"]
    reason = overall.get("reason") or overall_reason(records)
    cited = {i for r in records for i in r["evidence_ids"]}
    counts = "".join(f"<li><b>{e(overall['counts'].get(key, 0))}</b><span>{name}</span></li>" for key, name in _COUNTS)
    note = "".join(
        f"<li{' class=cited' if s['id'] in cited else ''}><span class='sid'>{e(s['id'])}</span><span>{e(s['text'])}</span></li>"
        for s in report["patient_sentences"]
    )
    fields = [("Report ID", report["id"]), ("Created", _date(report["created_utc"]))]
    fields += [(_FIELD_NAMES.get(k, k.replace("_", " ").capitalize()), v) for k, v in report["provenance"].items()]
    method = "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in fields)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CRS report {e(report['id'][:8])}</title><style>{_font_faces()}{_STYLE}</style></head>
<body><main class="sheet">
<header class="mast"><div><p class="brand">CRS</p><h1>Eligibility screening report</h1></div>
<p class="meta">{e(_date(report['created_utc']))}<br>Report {e(report['id'][:8])}</p></header>
<p class="notice">{e(report['notice'])}</p>
<section class="summary {_OUTCOME_CLASSES.get(overall['outcome'], 'unknown')}">
<div><p class="eyebrow">Overall result, preliminary</p><p class="outcome">{e(overall['outcome'])}</p></div>
<ul class="counts">{counts}</ul><p class="reason">{e(reason)}</p></section>
{_group('Inclusion criteria', 'inclusion', records, e)}
{_group('Exclusion criteria', 'exclusion', records, e)}
<section class="block"><h2>Patient note</h2><ol class="note">{note}</ol><p class="key">Shaded sentences were cited by at least one verdict.</p></section>
<footer class="method"><dl>{method}</dl><p>{e(overall['rule'])}</p></footer>
</main></body></html>"""
