# Equipment Maintenance Triage Assistant

A technician reports an equipment problem. The app runs **fixed threshold rules**, retrieves **manual sections**, and asks an **LLM** (Groq or Claude) for *suggestions*: possible causes, follow-up questions, inspection steps, a priority and a draft work order. A human technician edits, approves or rejects the draft. The AI never confirms anything.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # put your key in GROQ_API_KEY (or ANTHROPIC_API_KEY)
export GROQ_API_KEY=gsk_...                            # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."
uvicorn app.main:app --reload
```
Open http://localhost:8000. Without a key the app still works and shows an "AI not configured" state (see *Failure handling*).

Tests: `pip install -r requirements-dev.txt && pytest -q` (20 tests; the LLM is mocked, no key needed).

Docker: `docker build -t triage . && docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... triage`

## Deploy (Render, free tier)
1. Push this repo to GitHub.
2. Render → New → Web Service → connect the repo → Runtime **Docker** (the Dockerfile is detected).
3. Add environment variable `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`). Deploy. The public URL is your deployed link.

Free tiers sleep when idle (first request takes ~30s) and have an ephemeral disk, so SQLite history resets on redeploy. Demo history is re-seeded on startup. For persistence, attach a disk and set `DB_PATH` to it, or swap in Postgres.

## How it maps to the brief

| Requirement | Where |
|---|---|
| Report fields (type, ID, issue, events, optional sensors) | Form in `app/static/index.html`, `POST /api/reports` |
| Knowledge base (manuals, faults, maintenance) | `knowledge/*.md` (4 equipment types + general guide) |
| Retrieve relevant manual sections | `app/retrieval.py`: BM25 over `##` sections, filtered by equipment type, minimum score cut-off |
| Possible causes, not confirmed | LLM prompt rule + UI label "Possible causes" with dashed border; confidence forced to *low* if uncited |
| Targeted follow-up questions / inspection steps / priority / draft work order | `app/ai.py` single structured-JSON call, validated in code |
| Cite manual or event evidence | Every item carries evidence IDs (M=manual, E=event, R=threshold result, I=issue). Unknown IDs are stripped, uncited items are flagged "no evidence cited". Click a chip to see the source text |
| Deterministic threshold checks | `app/rules.py` (no AI), unit-tested |
| Edit / approve / reject work order | `PATCH /api/work-orders/{id}`, `POST .../decision`. Decided orders are locked; rejection needs a reason |
| Preserve issue and maintenance history | SQLite: `reports`, `work_orders`, `findings`. History tab per equipment ID |
| Observations vs possible causes vs confirmed findings | Three visually distinct sections. *Confirmed findings* are only written by a technician (`POST /api/reports/{id}/findings`); no code path lets the AI write them |
| Missing or conflicting sensor data | See below |
| Show AI and retrieval failures | See below |

### Design decisions
- **Code decides facts, AI suggests.** Threshold results are computed in code and passed to the LLM as evidence it must not override. After the AI answers, a **priority floor** is applied in code (any ALARM → at least High, 2+ alarms or a safety keyword such as "smoke" → Critical). If the AI proposed a lower priority, the UI shows "Raised by fixed rules" and why.
- **Conflicting data is surfaced, not resolved.** Two readings of the same sensor that differ by more than 10% are flagged and evaluated at the worst case. Implausible values (outside physical range) are ignored and reported. Free text such as "temperature is normal" contradicting an alarming sensor is flagged.
- **Missing data is explicit.** Sensors not provided are listed as "not provided (not checked)"; they are never treated as normal. With no usable readings the overall status is "insufficient data".
- **Validation over trust.** Model output must be valid JSON with a valid priority and work order; one automatic retry; otherwise the user sees "AI output rejected".
- **Why BM25, not embeddings:** the corpus is small and keyword-rich; BM25 is dependency-free, deterministic and easy to explain. The interface (`kb.search`) can be swapped for a vector store.

### Failure handling
| Situation | What the user sees |
|---|---|
| No API key | Banner "AI is not configured". Rules, safety flags, retrieval and a template work order still work |
| Timeout / HTTP error | Banner naming the failure, with a **Retry AI analysis** button |
| Malformed or invalid JSON | Banner "AI output rejected" after one retry; no partial suggestions are shown |
| No relevant manual section | "No manual match found"; the AI is told not to invent manual content |
| Knowledge base missing | "Manual retrieval failed" banner |

A template work order (clearly labelled *Template (no AI)*) is created when the AI is unavailable so the technician is never blocked.

## Demo scenarios (buttons in the UI)
1. **Compressor tripping**: grinding noise + vibration alarm → bearing/coupling hypotheses with citations, High priority.
2. **Conflicting sensors**: two temperature readings (134 / 98) and text saying temperature is normal → conflict warnings, worst case used.
3. **Pump, no sensors**: missing data handled, manual-driven suggestions (cavitation).
4. **Off-topic**: "No manual match found".
5. **Safety keyword**: smoke / sparks → forced Critical.

## Project layout
```
app/main.py       API + orchestration     app/rules.py      deterministic checks, priority floor
app/ai.py         LLM call + validation   app/retrieval.py  BM25 over knowledge/
app/db.py         SQLite persistence      app/static/       single-page UI
knowledge/        manuals (markdown)      tests/            rules + API tests (LLM mocked)
```

## Limitations
Single-user, no authentication. Thresholds and manuals are illustrative sample content, not real vendor data. The LLM prompt is tested with mocked responses in CI; verify live output with your own key.
