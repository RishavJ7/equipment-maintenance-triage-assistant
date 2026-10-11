"""LLM step. Everything here is allowed to fail; the caller always gets a status, never an exception."""
import json
import os
import re

import httpx

LEVELS = ["Low", "Medium", "High", "Critical"]

SYSTEM = """You are a maintenance triage assistant helping a technician. You only SUGGEST; a human decides.
Rules:
1. Use ONLY the evidence provided. Each evidence item has an ID in brackets (M=manual section, E=operating event, R=threshold check result, I=issue description).
2. Causes are hypotheses. Never state a cause as confirmed or certain. Word them as "possible ...".
3. Every cause, question, inspection step and the priority MUST list citations: IDs from the evidence that support it. Never invent IDs.
4. Threshold check results (R*) are facts computed by code. Do not recompute or override them.
5. If evidence is thin or conflicting, say so and return fewer items rather than guessing.
6. Priority must be one of: Low, Medium, High, Critical.
7. Respond with ONLY a JSON object, no prose, no markdown fences, matching exactly:
{"possible_causes":[{"cause":str,"rationale":str,"confidence":"low"|"medium"|"high","citations":[str]}],
 "follow_up_questions":[{"question":str,"why":str,"citations":[str]}],
 "inspection_steps":[{"step":str,"citations":[str]}],
 "priority":{"level":str,"reason":str,"citations":[str]},
 "work_order":{"title":str,"description":str,"recommended_actions":[str],"parts_tools":[str],"safety_notes":str}}"""


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class AIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def call_llm(system, user):
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    # If ANTHROPIC_API_KEY was configured with a Groq key (starts with 'gsk_'), route to Groq
    if not groq_key and anthropic_key.startswith("gsk_"):
        groq_key = anthropic_key

    timeout = float(os.environ.get("AI_TIMEOUT_SECONDS", "45"))

    if groq_key:
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        try:
            r = httpx.post("https://api.groq.com/openai/v1/chat/completions", timeout=timeout,
                           headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                           json={"model": model,
                                 "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": user}]})
        except httpx.TimeoutException:
            raise AIError("timeout", f"The AI request timed out after {timeout:.0f}s.")
        except httpx.HTTPError as e:
            raise AIError("error", f"Could not reach the AI service: {type(e).__name__}.")
        if r.status_code != 200:
            raise AIError("error", f"AI service returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            raise AIError("invalid_output", "AI response had an unexpected shape.")

    if anthropic_key:
        try:
            r = httpx.post("https://api.anthropic.com/v1/messages", timeout=timeout,
                           headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                           json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": 2500,
                                 "system": system, "messages": [{"role": "user", "content": user}]})
        except httpx.TimeoutException:
            raise AIError("timeout", f"The AI request timed out after {timeout:.0f}s.")
        except httpx.HTTPError as e:
            raise AIError("error", f"Could not reach the AI service: {type(e).__name__}.")
        if r.status_code != 200:
            raise AIError("error", f"AI service returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        except Exception:
            raise AIError("invalid_output", "AI response had an unexpected shape.")

    raise AIError("unavailable", "Neither GROQ_API_KEY nor ANTHROPIC_API_KEY is set, so the AI step was skipped.")


def _parse(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object found")
    return json.loads(t[start:end + 1])


def _cites(item, allowed):
    raw = item.get("citations") if isinstance(item, dict) else None
    good = [c for c in (raw if isinstance(raw, list) else []) if c in allowed]
    return good, len(good) < len(raw if isinstance(raw, list) else [])


def validate(data, allowed):
    """Normalise and check model output. Raises ValueError if unusable. Flags (never hides) uncited items."""
    if not isinstance(data, dict):
        raise ValueError("top-level value is not an object")
    out, warnings = {}, []

    def items(key, fields):
        res = []
        for it in data.get(key) or []:
            if not isinstance(it, dict) or not isinstance(it.get(fields[0]), str) or not it[fields[0]].strip():
                continue
            cites, dropped = _cites(it, allowed)
            row = {f: it.get(f) for f in fields}
            row["citations"], row["uncited"] = cites, not cites
            if dropped:
                warnings.append(f"Removed unknown citation IDs from one {key} item.")
            res.append(row)
        return res

    out["possible_causes"] = items("possible_causes", ["cause", "rationale", "confidence"])
    for c in out["possible_causes"]:
        if c["confidence"] not in ("low", "medium", "high") or c["uncited"]:
            c["confidence"] = "low"
    out["follow_up_questions"] = items("follow_up_questions", ["question", "why"])
    out["inspection_steps"] = items("inspection_steps", ["step"])
    p = data.get("priority") or {}
    if p.get("level") not in LEVELS:
        raise ValueError("priority.level missing or invalid")
    pc, _ = _cites(p, allowed)
    out["priority"] = {"level": p["level"], "reason": str(p.get("reason", "")), "citations": pc, "uncited": not pc}
    wo = data.get("work_order")
    if not isinstance(wo, dict) or not str(wo.get("title", "")).strip():
        raise ValueError("work_order missing")
    lst = lambda v: [str(x) for x in v] if isinstance(v, list) else []
    out["work_order"] = {"title": str(wo["title"]), "description": str(wo.get("description", "")),
                         "recommended_actions": lst(wo.get("recommended_actions")),
                         "parts_tools": lst(wo.get("parts_tools")), "safety_notes": str(wo.get("safety_notes", ""))}
    if not out["possible_causes"] and not out["inspection_steps"]:
        raise ValueError("no causes or inspection steps returned")
    out["warnings"] = sorted(set(warnings))
    return out


def run_ai(context_text, allowed_ids):
    """Returns {"status": ok|unavailable|timeout|error|invalid_output, "message": str, "result": dict|None}."""
    last = "unknown"
    for attempt in range(2):
        try:
            prompt = context_text if attempt == 0 else (
                context_text + f"\n\nYour previous reply was rejected ({last}). Reply again with ONLY the JSON object.")
            raw = call_llm(SYSTEM, prompt)
            return {"status": "ok", "message": "", "result": validate(_parse(raw), set(allowed_ids))}
        except AIError as e:
            if e.status != "invalid_output":
                return {"status": e.status, "message": e.message, "result": None}
            last = e.message
        except (ValueError, json.JSONDecodeError) as e:
            last = str(e)
    return {"status": "invalid_output", "result": None,
            "message": f"The AI returned output that failed validation twice ({last}). No AI suggestions are shown."}
