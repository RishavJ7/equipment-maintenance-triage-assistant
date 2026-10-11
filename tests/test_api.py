import json
import pytest
from fastapi.testclient import TestClient

from app import ai


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    from app.main import app
    with TestClient(app) as c:
        yield c


REPORT = {"equipment_type": "compressor", "equipment_id": "C-900",
          "issue": "Compressor trips and makes a grinding noise after restart",
          "events": "Tripped twice yesterday\nLoud noise after restart",
          "sensors": [{"sensor": "discharge_temp", "value": 125}, {"sensor": "vibration", "value": 8.0}]}


def good_llm(system, user):
    return json.dumps({
        "possible_causes": [{"cause": "Possible worn bearing", "rationale": "grinding + vibration", "confidence": "medium",
                             "citations": ["E2", "M1", "BOGUS"]},
                            {"cause": "Possible overload", "rationale": "x", "confidence": "high", "citations": []}],
        "follow_up_questions": [{"question": "Any recent lubrication?", "why": "w", "citations": ["M1"]}],
        "inspection_steps": [{"step": "Check mounting bolts", "citations": ["M1"]}],
        "priority": {"level": "Low", "reason": "ai thinks low", "citations": ["R2"]},
        "work_order": {"title": "Inspect C-900", "description": "d", "recommended_actions": ["a"], "parts_tools": [], "safety_notes": "LOTO"}})


def test_no_key_still_gives_rules_and_template(client):
    r = client.post("/api/reports", json=REPORT).json()
    assert r["analysis"]["ai"]["status"] == "unavailable"
    assert r["analysis"]["rules"]["overall"] == "alarm"
    assert r["work_order"]["source"] == "template"
    assert r["analysis"]["priority"]["final"] == "High"  # one ALARM (vibration) -> floor High


def test_ai_success_validates_and_escalates(client, monkeypatch):
    monkeypatch.setattr(ai, "call_llm", good_llm)
    r = client.post("/api/reports", json=REPORT).json()
    a = r["analysis"]
    assert a["ai"]["status"] == "ok"
    causes = a["ai"]["result"]["possible_causes"]
    assert causes[0]["citations"] == ["E2", "M1"]          # BOGUS dropped
    assert causes[1]["uncited"] and causes[1]["confidence"] == "low"  # uncited -> flagged, downgraded
    assert a["priority"]["ai"] == "Low" and a["priority"]["final"] == "High" and a["priority"]["escalated"]
    assert r["work_order"]["source"] == "ai" and r["work_order"]["draft"]["priority"] == "High"


def test_ai_garbage_shows_invalid_output(client, monkeypatch):
    monkeypatch.setattr(ai, "call_llm", lambda s, u: "sorry, I cannot help")
    a = client.post("/api/reports", json=REPORT).json()["analysis"]
    assert a["ai"]["status"] == "invalid_output" and a["ai"]["result"] is None
    assert a["rules"]["overall"] == "alarm"


def test_ai_timeout(client, monkeypatch):
    def boom(s, u):
        raise ai.AIError("timeout", "The AI request timed out after 45s.")
    monkeypatch.setattr(ai, "call_llm", boom)
    a = client.post("/api/reports", json=REPORT).json()["analysis"]
    assert a["ai"]["status"] == "timeout"


def test_retrieval_no_match(client):
    body = dict(REPORT, issue="The cafeteria coffee machine is broken again", events="", sensors=[])
    a = client.post("/api/reports", json=body).json()["analysis"]
    assert a["retrieval"]["status"] == "no_match"


def test_retry_ai_recovers(client, monkeypatch):
    rid = client.post("/api/reports", json=REPORT).json()["id"]
    monkeypatch.setattr(ai, "call_llm", good_llm)
    r = client.post(f"/api/reports/{rid}/retry-ai").json()
    assert r["analysis"]["ai"]["status"] == "ok" and r["work_order"]["source"] == "ai"


def test_work_order_edit_approve_lock(client):
    r = client.post("/api/reports", json=REPORT).json()
    wid = r["work_order"]["id"]
    e = client.patch(f"/api/work-orders/{wid}", json={"draft": {"title": "Edited title"}}).json()
    assert e["draft"]["title"] == "Edited title" and e["draft"]["edited"]
    assert client.post(f"/api/work-orders/{wid}/decision", json={"decision": "reject"}).status_code == 422
    ok = client.post(f"/api/work-orders/{wid}/decision", json={"decision": "approve"})
    assert ok.json()["status"] == "approved"
    assert client.patch(f"/api/work-orders/{wid}", json={"draft": {"title": "x"}}).status_code == 409


def test_findings_and_history(client):
    r = client.post("/api/reports", json=REPORT).json()
    client.post(f"/api/reports/{r['id']}/findings", json={"text": "Bearing worn", "recorded_by": "Sam"})
    h = client.get("/api/equipment/C-900/history").json()
    assert h[0]["findings"][0]["text"] == "Bearing worn"
    seeded = client.get("/api/equipment/C-102/history").json()
    assert seeded and seeded[0]["work_order"]["status"] == "approved"


def test_validation_errors(client):
    assert client.post("/api/reports", json=dict(REPORT, equipment_type="rocket")).status_code == 422
    assert client.post("/api/reports", json=dict(REPORT, issue="x")).status_code == 422


def test_meta_endpoint_reflects_ai_configured(client, monkeypatch):
    assert client.get("/api/meta").json()["ai_configured"] is False
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert client.get("/api/meta").json()["ai_configured"] is True


def test_call_llm_routes_to_groq(monkeypatch):
    called = {}

    def mock_post(url, headers=None, json=None, timeout=None):
        called["url"] = url
        called["headers"] = headers
        called["json"] = json
        class MockResp:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}
        return MockResp()

    monkeypatch.setattr("httpx.post", mock_post)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    res = ai.call_llm("sys", "usr")
    assert res == "{\"ok\": true}"
    assert "api.groq.com" in called["url"]
    assert called["headers"]["Authorization"] == "Bearer gsk_test_key_123"

