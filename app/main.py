import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import ai, db, rules
from .retrieval import RetrievalError, kb

STATIC = Path(__file__).parent / "static"
LEVELS = rules.LEVELS


@asynccontextmanager
async def lifespan(app):
    db.init()
    seed_demo()
    try:
        kb.load()
    except RetrievalError:
        pass  # surfaced per-report as a retrieval failure
    yield


app = FastAPI(title="Equipment Maintenance Triage Assistant", lifespan=lifespan)


class SensorIn(BaseModel):
    sensor: str
    value: Optional[float] = None


class ReportIn(BaseModel):
    equipment_type: str
    equipment_id: str
    issue: str
    events: str = ""
    sensors: list[SensorIn] = []


class WorkOrderEdit(BaseModel):
    draft: dict


class Decision(BaseModel):
    decision: str  # approve | reject
    note: str = ""


class FindingIn(BaseModel):
    text: str
    recorded_by: str = "technician"


def split_events(text):
    return [e.strip(" -•\t") for e in re.split(r"[\n;]+", text) if e.strip(" -•\t")]


def build_evidence(issue, events, rule_res, chunks):
    ev = {"I1": {"type": "issue", "label": "Reported issue description", "text": issue}}
    for i, e in enumerate(events, 1):
        ev[f"E{i}"] = {"type": "event", "label": f"Operating event {i}", "text": e}
    for r in rule_res["items"]:
        txt = f"{r['label']}: status {r['status'].upper()}. {r['note']}"
        if r["values"]:
            txt += f" Readings: {', '.join(f'{v:g}' for v in r['values'])} {r['unit']}."
        ev[r["id"]] = {"type": "threshold", "label": f"Threshold check: {r['label']}", "text": txt}
    for i, c in enumerate(chunks, 1):
        ev[f"M{i}"] = {"type": "manual", "label": f"{c['doc']} / {c['section']}", "text": c["text"]}
    return ev


def build_prompt(etype, eid, ev, rule_res, notes, flags):
    lines = [f"Equipment: {etype} {eid}", "", "EVIDENCE:"]
    lines += [f"[{k}] ({v['type']}) {v['text']}" for k, v in ev.items()]
    if notes:
        lines += ["", "DATA QUALITY / CONFLICT NOTES (computed by code):"] + [f"- {n}" for n in notes]
    if flags:
        lines += ["", "SAFETY KEYWORDS DETECTED: " + ", ".join(flags)]
    if not any(k.startswith("M") for k in ev):
        lines += ["", "NOTE: no manual sections matched. Do not invent manual content; rely on events and checks only."]
    return "\n".join(lines)


def template_work_order(etype, eid, issue, rule_res, flags, floor):
    acts = [f"Verify {r['label']} ({r['status'].upper()}): {r['note']}" for r in rule_res["items"]
            if r["status"] in ("warning", "alarm")]
    acts.append("Technician to inspect the equipment and record findings.")
    return {"title": f"{etype.title()} {eid}: reported issue", "description": issue,
            "priority": floor or "", "recommended_actions": acts, "parts_tools": [],
            "safety_notes": "Lock out / tag out before inspection." + (f" Safety keywords reported: {', '.join(flags)}." if flags else ""),
            "citations": []}


def analyse(etype, eid, issue, events_text, sensors):
    events = split_events(events_text)
    rule_res = rules.evaluate(etype, [s if isinstance(s, dict) else s.model_dump() for s in sensors])
    conflicts = rules.text_conflicts(issue + ". " + " ".join(events), rule_res["items"])
    flags = rules.safety_flags(issue, events_text)
    floor, floor_reasons = rules.priority_floor(rule_res, flags)
    notes = rule_res["data_quality_notes"] + conflicts

    query = " ".join([issue, events_text] + [r["label"] for r in rule_res["items"] if r["status"] in ("warning", "alarm")])
    retrieval = {"status": "ok", "message": "", "chunks": []}
    try:
        retrieval["chunks"] = kb.search(query, etype)
        if not retrieval["chunks"]:
            retrieval.update(status="no_match", message="No relevant manual section found for this report. "
                             "The AI was told not to invent manual content.")
    except RetrievalError as e:
        retrieval.update(status="failed", message=f"Manual retrieval failed: {e}")

    ev = build_evidence(issue, events, rule_res, retrieval["chunks"])
    ai_out = ai.run_ai(build_prompt(etype, eid, ev, rule_res, notes, flags), ev.keys())
    return assemble(ai_out, rule_res, conflicts, flags, floor, floor_reasons, retrieval, ev, events)


def assemble(ai_out, rule_res, conflicts, flags, floor, floor_reasons, retrieval, ev, events):
    prio = {"floor": floor, "floor_reasons": floor_reasons, "ai": None, "final": floor, "escalated": False}
    if ai_out["status"] == "ok":
        lvl = ai_out["result"]["priority"]["level"]
        prio["ai"] = lvl
        prio["final"] = lvl
        if floor and LEVELS.index(floor) > LEVELS.index(lvl):
            prio.update(final=floor, escalated=True)
    return {"events": events, "rules": rule_res, "text_conflicts": conflicts, "safety_flags": flags,
            "retrieval": retrieval, "evidence": ev, "ai": ai_out, "priority": prio}


def make_work_order(report_id, etype, eid, issue, a):
    r = a["ai"]
    if r["status"] == "ok":
        d = dict(r["result"]["work_order"])
        d["priority"] = a["priority"]["final"]
        d["citations"] = r["result"]["priority"]["citations"]
        return db.insert_work_order(report_id, eid, "ai", d)
    d = template_work_order(etype, eid, issue, a["rules"], a["safety_flags"], a["priority"]["final"])
    return db.insert_work_order(report_id, eid, "template", d)


@app.post("/api/reports")
def create_report(p: ReportIn):
    etype, eid = p.equipment_type.strip().lower(), p.equipment_id.strip()
    if etype not in rules.THRESHOLDS:
        raise HTTPException(422, f"Unknown equipment type '{etype}'.")
    if not eid or len(p.issue.strip()) < 5:
        raise HTTPException(422, "Equipment identifier and an issue description are required.")
    sensors = [s.model_dump() for s in p.sensors]
    a = analyse(etype, eid, p.issue.strip(), p.events, sensors)
    rid = db.insert_report(etype, eid, p.issue.strip(), a["events"], sensors, a)
    make_work_order(rid, etype, eid, p.issue, a)
    return db.get_report(rid)


@app.get("/api/reports/{rid}")
def get_report(rid: int):
    r = db.get_report(rid)
    if not r:
        raise HTTPException(404, "Report not found")
    return r


@app.post("/api/reports/{rid}/retry-ai")
def retry_ai(rid: int):
    r = db.get_report(rid)
    if not r:
        raise HTTPException(404, "Report not found")
    if r["work_order"] and r["work_order"]["status"] != "draft":
        raise HTTPException(409, "Work order already decided; create a new report instead.")
    a = analyse(r["equipment_type"], r["equipment_id"], r["issue"], "\n".join(r["events"]), r["sensors"])
    db.update_analysis(rid, a)
    if r["work_order"]:
        if a["ai"]["status"] == "ok":
            d = dict(a["ai"]["result"]["work_order"])
            d["priority"], d["citations"] = a["priority"]["final"], a["ai"]["result"]["priority"]["citations"]
            db.update_work_order(r["work_order"]["id"], draft=d, source="ai")
    return db.get_report(rid)


@app.patch("/api/work-orders/{wid}")
def edit_work_order(wid: int, body: WorkOrderEdit):
    wo = db.get_work_order(wid)
    if not wo:
        raise HTTPException(404, "Work order not found")
    if wo["status"] != "draft":
        raise HTTPException(409, f"Work order is {wo['status']} and can no longer be edited.")
    d = {**wo["draft"], **{k: v for k, v in body.draft.items() if k in wo["draft"] or k in ("title", "description")}}
    if d.get("priority") and d["priority"] not in LEVELS:
        raise HTTPException(422, "priority must be Low, Medium, High or Critical")
    d["edited"] = True
    db.update_work_order(wid, draft=d)
    return db.get_work_order(wid)


@app.post("/api/work-orders/{wid}/decision")
def decide(wid: int, body: Decision):
    wo = db.get_work_order(wid)
    if not wo:
        raise HTTPException(404, "Work order not found")
    if body.decision not in ("approve", "reject"):
        raise HTTPException(422, "decision must be 'approve' or 'reject'")
    if wo["status"] != "draft":
        raise HTTPException(409, f"Work order already {wo['status']}.")
    if body.decision == "reject" and not body.note.strip():
        raise HTTPException(422, "Please give a reason when rejecting.")
    db.update_work_order(wid, status="approved" if body.decision == "approve" else "rejected", note=body.note.strip())
    return db.get_work_order(wid)


@app.post("/api/reports/{rid}/findings")
def add_finding(rid: int, body: FindingIn):
    r = db.get_report(rid)
    if not r:
        raise HTTPException(404, "Report not found")
    if not body.text.strip():
        raise HTTPException(422, "Finding text is required.")
    db.insert_finding(rid, r["equipment_id"], body.text.strip(), body.recorded_by.strip() or "technician")
    return db.get_report(rid)["findings"]


@app.get("/api/equipment")
def equipment():
    return db.equipment_list()


@app.get("/api/equipment/{eid}/history")
def equipment_history(eid: str):
    return db.history(eid)


@app.get("/api/meta")
def meta():
    return {"types": {t: {k: {"label": v["label"], "unit": v["unit"]} for k, v in s.items()}
                      for t, s in rules.THRESHOLDS.items()},
            "ai_configured": bool(os.environ.get("GROQ_API_KEY", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()),
            "kb_sections": len(kb.chunks)}


@app.get("/api/health")
def health():
    return {"ok": True}


def seed_demo():
    with db.conn() as c:
        if c.execute("SELECT COUNT(*) FROM reports").fetchone()[0]:
            return
    ts1, ts2 = "2026-08-14T09:30:00+00:00", "2026-09-22T14:05:00+00:00"
    rr = rules.evaluate("compressor", [{"sensor": "discharge_temp", "value": 124}])
    a1 = {"events": ["Tripped on high temperature twice"], "rules": rr, "text_conflicts": [], "safety_flags": [],
          "retrieval": {"status": "ok", "message": "", "chunks": []}, "evidence": {},
          "ai": {"status": "unavailable", "message": "Seeded historical record (no AI run).", "result": None},
          "priority": {"floor": "Medium", "floor_reasons": [], "ai": None, "final": "High", "escalated": False}}
    r1 = db.insert_report("compressor", "C-102", "Compressor trips on high discharge temperature", ["Tripped on high temperature twice"],
                          [{"sensor": "discharge_temp", "value": 124}], a1, ts1)
    db.insert_work_order(r1, "C-102", "template", {"title": "Clean oil cooler on C-102", "description": "Cooler fins clogged with dust.",
                         "priority": "High", "recommended_actions": ["Clean oil cooler", "Check oil level"], "parts_tools": ["Air lance"],
                         "safety_notes": "Lock out / tag out.", "citations": []}, "approved", "Done during planned stop.", ts1)
    db.insert_finding(r1, "C-102", "Oil cooler fins blocked with dust; cleaned. Temperature back to 96 °C.", "demo technician", ts1)
    a2 = dict(a1, events=["Seal weeping since Monday"], rules=rules.evaluate("pump", []))
    r2 = db.insert_report("pump", "P-201", "Small leak at pump shaft seal", ["Seal weeping since Monday"], [], a2, ts2)
    db.insert_work_order(r2, "P-201", "template", {"title": "Inspect seal on P-201", "description": "Leak at shaft.", "priority": "Medium",
                         "recommended_actions": ["Check seal flush line"], "parts_tools": [], "safety_notes": "", "citations": []},
                         "rejected", "Duplicate of work order already planned.", ts2)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
