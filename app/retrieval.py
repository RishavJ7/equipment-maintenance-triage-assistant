"""BM25 retrieval over markdown manuals in /knowledge. One chunk per '## ' section."""
import math
import os
import re
from collections import Counter
from pathlib import Path

KB_DIR = Path(os.environ.get("KB_DIR", Path(__file__).resolve().parent.parent / "knowledge"))
MIN_SCORE = float(os.environ.get("RETRIEVAL_MIN_SCORE", "3.0"))
STOP = set("a an the and or of to in on at is are was were be been it its this that with for from by as but not no "
           "has have had after before then than so if when while about into out up down over under again also".split())


class RetrievalError(Exception):
    pass


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokenize(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


class KnowledgeBase:
    def __init__(self):
        self.chunks, self.df, self.avg_len = [], Counter(), 0.0

    def load(self, directory=KB_DIR):
        files = sorted(Path(directory).glob("*.md")) if Path(directory).is_dir() else []
        if not files:
            raise RetrievalError(f"Knowledge base is empty or missing at {directory}")
        chunks = []
        for f in files:
            raw = f.read_text(encoding="utf-8")
            title = (re.search(r"^# (.+)$", raw, re.M) or [None, f.stem])[1]
            for part in re.split(r"^## ", raw, flags=re.M)[1:]:
                head, _, body = part.partition("\n")
                chunks.append({"doc": title, "equipment": f.stem, "section": head.strip(), "text": body.strip(),
                               "tokens": tokenize((head + " ") * 3 + body)})
        self.chunks = chunks
        self.df = Counter(t for c in chunks for t in set(c["tokens"]))
        self.avg_len = sum(len(c["tokens"]) for c in chunks) / max(len(chunks), 1)
        return len(chunks)

    def search(self, query, equipment_type, k=4):
        if not self.chunks:
            raise RetrievalError("Knowledge base not loaded")
        q, n, scored = tokenize(query), len(self.chunks), []
        for c in self.chunks:
            if c["equipment"] not in (equipment_type, "general"):
                continue
            tf, dl, s = Counter(c["tokens"]), len(c["tokens"]), 0.0
            for t in set(q):
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * dl / self.avg_len))
            if s >= MIN_SCORE:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [{"doc": c["doc"], "section": c["section"], "text": c["text"], "score": round(s, 2)}
                for s, c in scored[:k]]


kb = KnowledgeBase()
