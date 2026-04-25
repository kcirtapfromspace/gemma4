"""Parity probes for the c19 single-turn fast-path gate.

Python mirror of the 8 fast-path probes in
apps/mobile/ios-app/validate_rag.swift. Same narratives, same expected
behavior, same hand-rolled NegEx predicate. If Python and Swift gates
agree on all 8, the precision-1.000 contract holds across both runtimes.

Usage:
  python apps/mobile/convert/validate_fast_path.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_search import (  # noqa: E402
    FAST_PATH_THRESHOLD,
    fast_path_hit,
)

# Negation triggers: same set as `cliIsNegated` in validate_rag.swift.
_NEG_TRIGGERS = (
    "ruled out", "negative for", "no evidence of",
    "no signs of", "no sign of", "no history of",
    "denies", "without", "absent", "not detected",
    "not positive for", "not suspected",
    "exclude", "excluded", "excludes", "excluding",
    "differential diagnosis", "differential dx",
)
_NEG_WINDOW = 60


def _cli_is_negated(text: str, match_start: int, _match_end: int) -> bool:
    """60-char before-window NegEx, matching validate_rag.swift's predicate.

    Crude vs. the real `regex_preparser._is_negated` (no terminator
    clipping), but matches the Swift CLI exactly so the parity bench
    apples-to-apples.
    """
    start = max(0, match_start - _NEG_WINDOW)
    if start >= match_start:
        return False
    window = text[start:match_start].lower()
    return any(trigger in window for trigger in _NEG_TRIGGERS)


# 8 cases — labels and contracts match validate_rag.swift exactly.
_CASES = [
    {
        "label": "valley fever (asserted)",
        "narrative": "Patient with classic valley fever from a California desert vacation. Cough, fatigue.",
        "should_fire": True,
        "expected_code": "37436014",
    },
    {
        "label": "Marburg outbreak (asserted)",
        "narrative": "Returning traveler from Uganda, suspected Marburg hemorrhagic fever, isolation initiated.",
        "should_fire": True,
        "expected_code": "418182002",
    },
    {
        "label": "C diff colitis (asserted)",
        "narrative": "Severe diarrhea after broad-spectrum antibiotics. C diff colitis confirmed by toxin assay.",
        "should_fire": True,
        "expected_code": "186431008",
    },
    {
        "label": "Legionnaires (token-overlap)",
        "narrative": "Outbreak investigation suggests Legionnaires' disease via cooling tower aerosol.",
        "should_fire": False,
        "expected_code": None,
    },
    {
        "label": "ruled out Legionnaires (negated)",
        "narrative": "Legionnaires' disease ruled out per negative urinary antigen.",
        "should_fire": False,
        "expected_code": None,
    },
    {
        "label": "negative for valley fever (negated)",
        "narrative": "Coccidioides serology negative for valley fever; alternate workup pending.",
        "should_fire": False,
        "expected_code": None,
    },
    {
        "label": "no fast-path on bare narrative without hit",
        "narrative": "Patient reports headache and fatigue. No specific exposures identified.",
        "should_fire": False,
        "expected_code": None,
    },
    {
        "label": "Plasmodium malariae (asserted)",
        "narrative": "Returning traveler with intermittent fevers and Plasmodium malariae malaria diagnosed.",
        "should_fire": True,
        "expected_code": "186946009",
    },
]


def main() -> int:
    print(f"=== fast-path probes (threshold {FAST_PATH_THRESHOLD}) ===")
    pass_count = 0
    fail_count = 0
    for i, case in enumerate(_CASES):
        label = f"{case['label']:<40s}"
        fired = fast_path_hit(
            case["narrative"], is_negated=_cli_is_negated
        )
        should_fire = case["should_fire"]
        expected_code = case["expected_code"]
        if should_fire and fired is not None:
            code_ok = expected_code is None or fired.hit.code == expected_code
            if code_ok:
                print(
                    f"  [{i}] PASS {label} → fired, code={fired.hit.code}, "
                    f"score={fired.hit.score:.3f}"
                )
                pass_count += 1
            else:
                print(
                    f"  [{i}] FAIL {label} → fired but code={fired.hit.code} "
                    f"(expected {expected_code})"
                )
                fail_count += 1
        elif not should_fire and fired is None:
            print(f"  [{i}] PASS {label} → did not fire (correct)")
            pass_count += 1
        elif should_fire and fired is None:
            print(f"  [{i}] FAIL {label} → expected fire but did not")
            fail_count += 1
        else:
            assert fired is not None  # for mypy; should_fire=False, fired non-None
            print(
                f"  [{i}] FAIL {label} → unexpectedly fired, "
                f"code={fired.hit.code}, score={fired.hit.score:.3f}"
            )
            fail_count += 1

    total = pass_count + fail_count
    print(f"\nfast-path: {pass_count}/{total} pass")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
