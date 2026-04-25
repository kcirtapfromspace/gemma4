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
from typing import Callable

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


# -----------------------------------------------------------------------------
# Fast-path gate (c19 Rank 2)
#
# Single-turn shortcut: when tier-1 (deterministic_extract) is empty AND RAG
# returns a high-confidence hit AND the matched phrase isn't in NegEx scope,
# skip the agent loop and synthesise the extraction directly from the RAG
# hit. Drops agent-tier-only median latency from ~13 s to <1 s on the long-
# tail cases (Legionnaires, C diff, valley fever, Marburg, RMSF, etc.).
#
# Mirror of `RagSearch.fastPathHit` in
# apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/RagSearch.swift. Keep them in
# sync; precision must stay 1.000 — any new false positive kills the
# proposal per Rank 2's kill criterion.

# Threshold mirrors RagSearch.fastPathThreshold in Swift.
FAST_PATH_THRESHOLD: float = 0.70

# Negation predicate: (text, match_start, match_end) -> bool.
# Caller injects so this module stays free of regex_preparser imports
# (avoids circular imports — regex_preparser is the lookup tier, not the
# RAG tier).
IsNegated = Callable[[str, int, int], bool]


def _never_negated(text: str, start: int, end: int) -> bool:
    """No-op negation predicate. Tests / CLIs that don't want NegEx pass this."""
    return False


@dataclass(frozen=True)
class FastPathSpan:
    """Where in the narrative the matched phrase landed.

    Mirror of `RagSearch.FastPathSpan` in Swift. UTF-16 offsets in Swift; in
    Python we use plain `str` indices (UTF-32 codepoints) because that's
    what consumers in `agent_pipeline.py` and downstream callers use. The
    location/length are equivalent for ASCII clinical text — every case in
    the bench is ASCII — but if a non-ASCII narrative ever flows through,
    the Python span won't byte-for-byte match the Swift span. Acceptable
    for current scope.
    """
    text: str
    location: int
    length: int


@dataclass(frozen=True)
class FastPathHit:
    """One fast-path hit + the literal matched span.

    Returned only when:
      1. top RAG score >= threshold (default 0.70)
      2. there exists at least one occurrence of `hit.matched_phrase` in
         the narrative that is NOT in NegEx scope.
    Caller must ensure deterministic extraction is empty before calling.
    """
    hit: RagHit
    span: FastPathSpan


def _first_non_negated_span(
    phrase: str,
    narrative: str,
    is_negated: IsNegated,
) -> FastPathSpan | None:
    """Scan `narrative` for every case-insensitive occurrence of `phrase`.

    Return the first occurrence that is NOT in NegEx scope per the
    caller-provided predicate. Returns None when every occurrence is
    suppressed ("ruled out Legionnaires" with no other mention).

    Uses literal substring search (not regex) so phrases containing
    metacharacters ("C. trachomatis", "Plasmodium malariae malaria")
    match without escaping.
    """
    trimmed = phrase.strip()
    if not trimmed:
        return None
    needle = trimmed.lower()
    haystack = narrative.lower()
    n = len(needle)
    start = 0
    while start <= len(haystack) - n:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        end = idx + n
        if not is_negated(narrative, idx, end):
            return FastPathSpan(text=narrative[idx:end], location=idx, length=n)
        start = end
    return None


def first_asserted_span(
    narrative: str,
    hit: RagHit,
    is_negated: IsNegated = _never_negated,
) -> FastPathSpan | None:
    """First non-negated occurrence of `hit.matched_phrase` in `narrative`.

    Why only `matched_phrase` (not alt_names + display)? altName fallback
    is too loose: a short alias like "Cocci" (for Coccidioidomycosis)
    matches the genus prefix in "Coccidioides serology negative for valley
    fever" at an unrelated offset, defeating NegEx on the actual diagnosis.
    `matched_phrase` is the exact phrase that earned RAG's score in the
    first place — if THAT span is suppressed, the hit is suppressed. If
    `matched_phrase` is None (token-overlap-only hit, never an exact phrase
    match), the fast-path declines and falls through to the agent.

    Mirrors RagSearch.firstAssertedSpan in Swift. Conservative rule per
    ios-eng's c19 fix (commit 926c8ef).
    """
    matched = hit.matched_phrase
    if not matched:
        return None
    return _first_non_negated_span(matched, narrative, is_negated)


def fast_path_hit(
    narrative: str,
    threshold: float = FAST_PATH_THRESHOLD,
    is_negated: IsNegated = _never_negated,
) -> FastPathHit | None:
    """Top-level fast-path gate.

    Returns a FastPathHit if every condition holds:
      - top RAG score >= threshold
      - at least one non-negated occurrence of the matched phrase per
        the caller-provided `is_negated` predicate

    Otherwise returns None — caller falls through to the agent loop.

    Mirrors RagSearch.fastPathHit in Swift.
    """
    hits = search(narrative, top_k=1, min_score=threshold)
    if not hits:
        return None
    top = hits[0]
    if top.score < threshold:
        return None
    span = first_asserted_span(narrative, top, is_negated=is_negated)
    if span is None:
        return None
    return FastPathHit(hit=top, span=span)


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
