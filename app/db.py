import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY AUTOINCREMENT, equipment_type TEXT, equipment_id TEXT, issue TEXT,
  events TEXT, sensors TEXT, analysis TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS work_orders(
  id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER, equipment_id TEXT, source TEXT, draft TEXT,
  status TEXT, decision_note TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS findings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER, equipment_id TEXT, text TEXT,
  recorded_by TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_reports_eq ON reports(equipment_id);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def path():
    return os.environ.get("DB_PATH", "data/triage.db")


@contextmanager
def conn():
    c = sqlite3.connect(path())
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    os.makedirs(os.path.dirname(os.path.abspath(path())), exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)


def _j(row, *cols):
    d = dict(row)
    for k in cols:
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d


def insert_report(etype, eid, issue, events, sensors, analysis, created_at=None):
    with conn() as c:
        cur = c.execute("INSERT INTO reports(equipment_type,equipment_id,issue,events,sensors,analysis,created_at) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (etype, eid, issue, json.dumps(events), json.dumps(sensors), json.dumps(analysis),
                         created_at or now()))
        return cur.lastrowid


def update_analysis(report_id, analysis):
    with conn() as c:
        c.execute("UPDATE reports SET analysis=? WHERE id=?", (json.dumps(analysis), report_id))


def insert_work_order(report_id, eid, source, draft, status="draft", note=None, ts=None):
    with conn() as c:
        cur = c.execute("INSERT INTO work_orders(report_id,equipment_id,source,draft,status,decision_note,created_at,updated_at)"
                        " VALUES(?,?,?,?,?,?,?,?)", (report_id, eid, source, json.dumps(draft), status, note, ts or now(), ts or now()))
        return cur.lastrowid


def get_work_order(wid):
    with conn() as c:
        r = c.execute("SELECT * FROM work_orders WHERE id=?", (wid,)).fetchone()
    return _j(r, "draft") if r else None


def update_work_order(wid, draft=None, status=None, note=None, source=None):
    wo = get_work_order(wid)
    with conn() as c:
        c.execute("UPDATE work_orders SET draft=?, status=?, decision_note=?, source=?, updated_at=? WHERE id=?",
                  (json.dumps(draft if draft is not None else wo["draft"]), status or wo["status"],
                   note if note is not None else wo["decision_note"], source or wo["source"], now(), wid))


def insert_finding(report_id, eid, text, by, ts=None):
    with conn() as c:
        c.execute("INSERT INTO findings(report_id,equipment_id,text,recorded_by,created_at) VALUES(?,?,?,?,?)",
                  (report_id, eid, text, by, ts or now()))


def get_report(rid):
    with conn() as c:
        r = c.execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
        if not r:
            return None
        rep = _j(r, "events", "sensors", "analysis")
        wo = c.execute("SELECT * FROM work_orders WHERE report_id=? ORDER BY id DESC LIMIT 1", (rid,)).fetchone()
        rep["work_order"] = _j(wo, "draft") if wo else None
        rep["findings"] = [dict(x) for x in c.execute("SELECT * FROM findings WHERE report_id=? ORDER BY id", (rid,))]
    return rep


def equipment_list():
    with conn() as c:
        rows = c.execute("SELECT equipment_id, equipment_type, COUNT(*) n, MAX(created_at) last "
                         "FROM reports GROUP BY equipment_id ORDER BY last DESC").fetchall()
    return [dict(r) for r in rows]


def history(eid):
    with conn() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM reports WHERE equipment_id=? ORDER BY id DESC", (eid,))]
    out = []
    for i in ids:
        r = get_report(i)
        a = r["analysis"] or {}
        out.append({"id": r["id"], "created_at": r["created_at"], "equipment_type": r["equipment_type"],
                    "issue": r["issue"], "rule_overall": (a.get("rules") or {}).get("overall"),
                    "priority": (a.get("priority") or {}).get("final"),
                    "work_order": r["work_order"], "findings": r["findings"]})
    return out
