"""Keyword/phrase search over the curated reportable-conditions database.

Tier 4 of the deterministic stack: when the lookup table doesn't have an
alias, the agent calls this to retrieve top-k candidate codes from a wider
authoritative knowledge base (CDC NNDSS, WHO IDSR). Each result carries a
source_url so the demo UI can show "I extracted X because the CDC NNDSS
reports it under display name Y — source: <url>".

Scoring is intentionally simple (no embeddings, no ML model):
  - Exact-phrase match on display or any alt_name → +1.0
  - Word-overlap fraction over both query and entry tokens     → 0.0–0.5
  - Bonus for matching long discriminative tokens (≥6 chars)   → +0.1 each
The result is bounded in [0, ~2.0] and sorted descending.

Why not TF-IDF / sentence-transformer?
  - 60-entry database, deterministic ranking is fine
  - No embedding inference cost on-device
  - Easy to debug (every score is human-readable)
  - Honest scope for a hackathon: ML-free RAG
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9-]+\b", re.IGNORECASE)
_APOSTROPHE_RE = re.compile(r"['‘’]s?\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Strip apostrophes and possessives so 'Legionnaires'' matches 'Legionnaires'."""
    return _APOSTROPHE_RE.sub("", text)
_DEFAULT_DB_PATH = Path(__file__).parent / "reportable_conditions.json"
_DB_CACHE: list[dict] | None = None


def _load_db(path: Path = _DEFAULT_DB_PATH) -> list[dict]:
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE
    raw = json.loads(path.read_text())
    _DB_CACHE = raw.get("conditions", [])
    return _DB_CACHE


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


@dataclass(frozen=True)
class RagHit:
    code: str
    system: str
    display: str
    score: float
    matched_phrase: str | None
    source: str
    source_url: str
    category: str | None
    alt_names: list[str]

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "system": self.system,
            "display": self.display,
            "score": round(self.score, 3),
            "matched_phrase": self.matched_phrase,
            "source": self.source,
            "source_url": self.source_url,
            "category": self.category,
            "alt_names": list(self.alt_names),
        }


def search(query: str, *, top_k: int = 3, min_score: float = 0.2) -> list[RagHit]:
    """Return up to top_k entries scoring above min_score for the query."""
    if not query.strip():
        return []
    db = _load_db()
    q_norm = _normalize(query)
    q_lower = q_norm.lower()
    q_tokens = _tokens(q_norm)

    results: list[tuple[float, str | None, dict]] = []
    for entry in db:
        display = entry["display"]
        alt_names = entry.get("alt_names", []) or []

        # 1. Exact-phrase bonus — strongest signal
        matched_phrase: str | None = None
        phrase_bonus = 0.0
        for candidate in [display] + alt_names:
            if _normalize(candidate).lower() in q_lower:
                phrase_bonus = max(phrase_bonus, 1.0)
                matched_phrase = matched_phrase or candidate

        # 2. Token overlap
        e_tokens = _tokens(_normalize(display))
        for alt in alt_names:
            e_tokens |= _tokens(_normalize(alt))
        overlap = q_tokens & e_tokens
        token_score = 0.0
        if e_tokens:
            recall = len(overlap) / max(len(q_tokens), 1)
            precision = len(overlap) / max(len(e_tokens), 1)
            # Harmonic mean keeps short queries from over-matching catch-all entries
            token_score = (
                2 * recall * precision / (recall + precision)
                if (recall + precision) > 0
                else 0.0
            ) * 0.5

        # 3. Long-token bonus for discriminative names
        long_token_bonus = 0.1 * sum(
            1 for t in overlap if len(t) >= 6
        )

        score = phrase_bonus + token_score + long_token_bonus
        if score >= min_score:
            results.append((score, matched_phrase, entry))

    results.sort(key=lambda x: (-x[0], x[2]["display"]))
    out: list[RagHit] = []
    for score, matched_phrase, entry in results[:top_k]:
        out.append(RagHit(
            code=entry["code"],
            system=entry["system"],
            display=entry["display"],
            score=score,
            matched_phrase=matched_phrase,
            source=entry.get("source", "unknown"),
            source_url=entry.get("source_url", ""),
            category=entry.get("category"),
            alt_names=list(entry.get("alt_names", [])),
        ))
    return out


def main() -> None:
    """Quick CLI for poking at the index. Run any query."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.2)
    args = ap.parse_args()

    q = " ".join(args.query)
    print(f"query: {q!r}\n")
    hits = search(q, top_k=args.top_k, min_score=args.min_score)
    if not hits:
        print("(no matches above min-score)")
        return
    for i, h in enumerate(hits, 1):
        print(
            f"  {i}. score={h.score:.3f}  {h.code} {h.system}  {h.display}"
        )
        if h.matched_phrase:
            print(f"      matched_phrase: '{h.matched_phrase}'")
        print(f"      alt: {', '.join(h.alt_names)}")
        print(f"      src: {h.source} — {h.source_url}")


if __name__ == "__main__":
    main()
