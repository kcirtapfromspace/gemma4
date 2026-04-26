"""Deterministic eICR code extractor.

Extracts SNOMED, LOINC, and RxNorm codes from eICR text using regex.
Returns the same JSON shape that the LLM is asked to produce so that
score_extraction() in validate_all_cases.py can score it directly.

This is the offline prototype for the autoresearch Rank 2 (regex pre-pass)
proposal. Once validated, the same patterns port to Swift in the iOS app.

Patterns covered:
- "(SNOMED 12345)" / "SNOMED 12345" → 6-9 digit numeric SNOMED CT
- "(LOINC 12345-6)" / "LOINC 12345-6" → LOINC code with check digit
- "(RxNorm 12345)" / "RxNorm 12345" → RxNorm RXCUI

Patterns NOT covered (out of scope for Rank 2 first pass — handled by LLM
or a follow-up adversarial pass):
- Codes inside CDA XML attributes (<code code="...">)
- Codes referenced by displayName only, no numeric
- ICD-10 codes (need different shape: A00.0)
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Capture the *whole* "(SNOMED 76272004)" match span so provenance can quote it
# back to the user. group(1) is still the bare numeric code.
SNOMED_RE = re.compile(r"\bSNOMED[\s:]+(\d{6,9})\b", re.IGNORECASE)
LOINC_RE = re.compile(r"\bLOINC[\s:]+(\d{1,7}-\d)\b", re.IGNORECASE)
RXNORM_RE = re.compile(r"\bRxNorm[\s:]+(\d{2,9})\b", re.IGNORECASE)

# Confidence floors per provenance tier. CDA XML and inline parenthesized
# codes are explicit assertions in the source — nearly perfect. Lookup-table
# alias matches carry inherent ambiguity (alias→code mapping is curator
# judgment, drug doses may not match exactly) so they default lower.
_TIER_CONFIDENCE = {
    "inline": 0.99,
    "cda": 0.99,
    "lookup": 0.85,
}
SYSTEM_NAMES = {
    "snomed": "SNOMED",
    "loinc": "LOINC",
    "rxnorm": "RxNorm",
}

# CDA <code code="..." codeSystem="..."/> attribute order is unspecified, so
# match the codeSystem OID separately and then pull the partner code attr from
# the same tag. CDA OIDs:
#   SNOMED CT: 2.16.840.1.113883.6.96
#   LOINC:     2.16.840.1.113883.6.1
#   RxNorm:    2.16.840.1.113883.6.88
_CDA_TAG_RE = re.compile(r"<[^>]*\bcodeSystem=\"([^\"]+)\"[^>]*>")
_CODE_ATTR_RE = re.compile(r"\bcode=\"([^\"]+)\"")
_CDA_OID_TO_BUCKET = {
    "2.16.840.1.113883.6.96": "snomed",
    "2.16.840.1.113883.6.1": "loinc",
    "2.16.840.1.113883.6.88": "rxnorm",
}


@dataclass(frozen=True)
class CodeMatch:
    """Per-code provenance record. One per emitted code."""
    code: str
    display: str
    system: str            # 'SNOMED' / 'LOINC' / 'RxNorm'
    bucket: str            # 'snomed' / 'loinc' / 'rxnorm' (lowercase, for routing)
    tier: str              # 'inline' / 'cda' / 'lookup' / 'rag'
    confidence: float      # 0.0–1.0
    source_text: str       # the substring in the input that proved this code
    source_offset: int     # char offset in the input
    source_length: int     # length of the source span
    alias: str | None = None       # for lookup tier: which alias matched
    source_url: str | None = None  # for rag tier: source document

    def to_dict(self) -> dict:
        d = {
            "code": self.code,
            "display": self.display,
            "system": self.system,
            "tier": self.tier,
            "confidence": round(self.confidence, 3),
            "source_text": self.source_text,
            "source_offset": self.source_offset,
            "source_length": self.source_length,
        }
        if self.alias is not None:
            d["alias"] = self.alias
        if self.source_url is not None:
            d["source_url"] = self.source_url
        return d


_DISPLAY_NAME_ATTR_RE = re.compile(r'\bdisplayName="([^"]+)"')

# c20 Bug 7: clinical-statement label prefixes that legitimize an inline
# `(SNOMED 12345)` / `(LOINC ...)` / `(RxNorm ...)` annotation. When NONE of
# these labels appears within `_INLINE_LABEL_WINDOW` chars before the inline
# code AND the code is not in the curated lookup table, the inline tier
# refuses to emit. This blocks adversarial bait substrings like
# "patient read on the internet that 'condition (SNOMED 99999999) has 90% mortality'"
# from leaking into the extraction.
#
# Every legitimate combined-27/adv1-5 case uses one of these prefixes —
# `Dx: Syphilis (SNOMED 76272004)`, `Lab: ... (LOINC ...)`, `Meds: ... (RxNorm ...)` —
# so the gate keeps recall at 1.000 on those benches.
_INLINE_LABEL_RE = re.compile(
    r"\b(?:"
    r"Dx|Diagnosis|Diagnoses|Final|Final\s+admission\s+diagnosis"
    r"|Reason(?:\s+for\s+(?:visit|ED\s+visit|admission))?"
    r"|Assessment|Assessment\s+and\s+Plan|A/P|Plan|Impression"
    r"|Clinical\s+impression|Clinical\s+assessment"
    r"|Lab|Labs|Laboratory|Workup"
    r"|Med|Meds|Medication|Medications|Rx|Treatment"
    r"|Imaging|Microbiology|Pathology|Procedure"
    r"|Problem(?:\s+list)?|History|PMH"
    r"|Encounter|Vitals|Vital\s+signs"
    r")\s*:",
    re.IGNORECASE,
)
# Look at the 60 chars immediately preceding the inline code. Most labeled
# clinical lines have the colon within ~30 chars of the value, but we widen
# slightly to accommodate `Final admission diagnosis: Measles (SNOMED ...)`
# style headers without losing precision. The window is anchored to the
# match start (NOT to start-of-line) so multi-clause lines like
# "Workup: TB ruled out. Dx: Sarcoidosis (SNOMED ...)" still pass — the
# closest preceding label "Dx:" is what counts.
_INLINE_LABEL_WINDOW = 60


def _is_curated_code(code: str, bucket: str) -> bool:
    """True if `code` exists in lookup_table.json under the given bucket.

    Used by `_extract_inline` to allow well-known curated codes through
    even when they're not preceded by a clinical-statement label. This
    keeps the inline tier honest on legitimate prose that names a known
    code without the `Dx:` / `Lab:` framing.
    """
    table = _load_lookup()
    cat = {"snomed": "snomed", "loinc": "loincs", "rxnorm": "rxnorms"}.get(bucket)
    if cat is None:
        return False
    for entry_code, _patterns in table.get(cat, []):
        if entry_code == code:
            return True
    return False


def _extract_inline(text: str) -> list[CodeMatch]:
    out: list[CodeMatch] = []
    for pattern, bucket in (
        (SNOMED_RE, "snomed"),
        (LOINC_RE, "loinc"),
        (RXNORM_RE, "rxnorm"),
    ):
        for m in pattern.finditer(text):
            code = m.group(1)
            # Display: text on the same line, after the LAST "Label:" header
            # before the code. Lines like "Workup: TB ruled out. Dx: Sarcoidosis
            # (SNOMED ...)" have multiple headers — keep only the closest one.
            line_start = text.rfind("\n", 0, m.start()) + 1
            preceding = text[line_start:m.start()].strip(" \t")
            # Find the last "Word:" header in the preceding text and slice after it.
            label_iter = list(re.finditer(r"\b[A-Za-z][A-Za-z ]{0,30}?:\s*", preceding))
            if label_iter:
                preceding = preceding[label_iter[-1].end():]
            preceding = preceding.strip(" \t,;-(")
            display = preceding or code

            # c20 Bug 7 gate: emit ONLY when (a) the code is curated in
            # `lookup_table.json` (we already know about it) OR (b) a
            # clinical-statement label appears within
            # `_INLINE_LABEL_WINDOW` chars before the match. Adversarial
            # bait substrings like "(SNOMED 99999999)" inside quoted
            # internet-rumor prose carry no preceding label, so they fall
            # through. CDA `<code code="..." codeSystem="..."/>` matches are
            # handled by `_extract_cda_matches` and unaffected.
            window_start = max(0, m.start() - _INLINE_LABEL_WINDOW)
            window = text[window_start:m.start()]
            has_label = _INLINE_LABEL_RE.search(window) is not None
            if not has_label and not _is_curated_code(code, bucket):
                continue

            out.append(CodeMatch(
                code=code,
                display=display,
                system=SYSTEM_NAMES[bucket],
                bucket=bucket,
                tier="inline",
                confidence=_TIER_CONFIDENCE["inline"],
                source_text=text[m.start():m.end()],
                source_offset=m.start(),
                source_length=m.end() - m.start(),
            ))
    return out


def _extract_cda_matches(text: str) -> list[CodeMatch]:
    out: list[CodeMatch] = []
    for tag in _CDA_TAG_RE.finditer(text):
        oid = tag.group(1)
        bucket = _CDA_OID_TO_BUCKET.get(oid)
        if bucket is None:
            continue
        tag_text = tag.group(0)
        code_match = _CODE_ATTR_RE.search(tag_text)
        if not code_match:
            continue
        code = code_match.group(1)
        display_match = _DISPLAY_NAME_ATTR_RE.search(tag_text)
        display = display_match.group(1) if display_match else code
        out.append(CodeMatch(
            code=code,
            display=display,
            system=SYSTEM_NAMES[bucket],
            bucket=bucket,
            tier="cda",
            confidence=_TIER_CONFIDENCE["cda"],
            source_text=tag_text,
            source_offset=tag.start(),
            source_length=tag.end() - tag.start(),
        ))
    return out


# Legacy code-list shims kept so the in-process tests that still import these
# helpers continue to work. New callers should use _extract_inline /
# _extract_cda_matches and read CodeMatch fields directly.
def _extract_cda(text: str) -> dict[str, list[str]]:
    """Backward-compat: dict-of-lists view of CDA XML matches."""
    buckets: dict[str, list[str]] = {"snomed": [], "loinc": [], "rxnorm": []}
    for cm in _extract_cda_matches(text):
        buckets[cm.bucket].append(cm.code)
    return buckets


# NegEx-style negation triggers (Chapman et al., 2001 — adapted).
# Pre-triggers appear BEFORE the asserted concept and negate forward;
# post-triggers appear AFTER and negate backward. We don't distinguish here
# — we scan a window on both sides of each lookup match.
#
# Scope of a trigger ends at a TERMINATOR (clause boundary). This matters
# for cases like "negative for COVID, positive for influenza" where the
# scope of "negative for" must end at the comma so influenza isn't suppressed.
_NEG_TRIGGERS = re.compile(
    r"\b(?:"
    r"ruled\s+out"
    r"|negative\s+for"
    r"|no\s+evidence\s+of"
    r"|no\s+current\s+evidence\s+of"
    r"|no\s+signs?\s+of"
    r"|no\s+history\s+of"
    r"|denies"
    r"|without"
    r"|absent"
    r"|not\s+detected"
    r"|not\s+positive\s+for"
    r"|not\s+suspected"
    r"|not\s+invoking"
    r"|do(?:es)?\s+not\s+have"
    r"|did\s+not\s+have"
    r"|not\s+eligible\s+for"
    r"|exclud(?:e|ed|es|ing)"
    r"|excluded"
    r"|differential\s+(?:diagnosis|dx|includ(?:ed|es|ing))"
    r")\b",
    re.IGNORECASE,
)
# Tokens that close the scope of a negation trigger.
# c20 adv6 fix: comma added so "no history of stroke, history of HIV"
# does NOT leak the "no history of" trigger past the comma into the
# next clause and incorrectly suppress HIV (`adv6_neg_enumeration_*`).
_NEG_TERMINATORS = re.compile(
    r"(?:\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|\.\s|;|\n|<|,)",
    re.IGNORECASE,
)
# Window size in characters (≈ 6-8 words on typical clinical prose).
_NEG_WINDOW_BEFORE = 60
_NEG_WINDOW_AFTER = 30

# Post-hoc NegEx triggers — phrases where the disease name precedes the
# negation marker ("Zika serology came back negative", "Influenza A returned
# negative"). The standard `_NEG_TRIGGERS` pattern is checked in a 30-char
# after-window, but post-hoc constructions often have intervening lab/method
# tokens between the disease name and the trigger ("Zika virus IgM antibody
# serology came back negative" — "came back negative" is ~33 chars after
# "Zika"). To handle these without widening the broad `_NEG_TRIGGERS` window,
# we add a focused post-hoc scan with a slightly larger window (80 chars),
# clause-bounded, that ONLY matches these very specific multi-word
# constructions — over-firing risk is minimal because each pattern is highly
# discriminative.
_POSTHOC_NEG_TRIGGERS = re.compile(
    r"\b(?:"
    r"came\s+back\s+negative"
    r"|returned\s+negative"
    r"|(?:was|is|were|are)\s+negative"
    r"|reported\s+negative"
    r"|results?\s+(?:was|were|is|are)\s+negative"
    r"|IgM\s+negative"
    r"|IgG\s+negative"
    r")\b",
    re.IGNORECASE,
)
# Conservative post-hoc window: scan up to next clause terminator (.; \n,)
# capped at 80 chars. The cap is wider than the broad 30-char window because
# the post-hoc triggers above are highly specific multi-word phrases — false
# positives on these in non-clinical prose are vanishingly rare.
_POSTHOC_WINDOW_AFTER = 80
_POSTHOC_TERMINATORS = re.compile(r"(?:\.\s|;|\n|,)", re.IGNORECASE)

# c20 final pass: vaccine-context negation. "No varicella series" / "no MMR
# booster" / "never received Tdap vaccine" — the disease/agent name is named
# only as a vaccine target, NOT as an active diagnosis. Combines a tight
# pre-window check (`No`/`never received`/`no record of` before the alias)
# with a post-window check that the immediate next 1-2 words include
# `series`/`booster`/`vaccine`/`vaccination`/`immunization`/`shot`. The
# combined-context requirement keeps the rule safe — bare "No varicella"
# without a vaccine-context noun still fires (an active rule-out would say
# "No evidence of varicella" / "varicella ruled out").
_VAX_PRE_TRIGGERS = re.compile(
    r"\b(?:"
    r"no(?:t)?(?:\s+up\s+to\s+date(?:\s+on)?)?"
    r"|never(?:\s+received|\s+had|\s+got)?"
    r"|did\s+not\s+(?:receive|have|get)"
    r"|missed"
    r"|declined"
    r"|refused"
    r"|skipped"
    r")\b",
    re.IGNORECASE,
)
_VAX_POST_NOUNS = re.compile(
    r"^\W{0,3}\S{0,30}?\s*\b(?:series|booster|vaccine|vaccination|immuniz\w*|shot|dose|MMR\b)",
    re.IGNORECASE,
)


def _is_vaccine_context_negation(text: str, match_start: int, match_end: int) -> bool:
    """True when the alias is named in a 'No <X> series/booster/vaccine' context.

    Pre-window (10 chars): one of `no`/`never`/`missed`/`declined`/etc.
    Post-window (40 chars from match_end): a vaccine-context noun
    (series/booster/vaccine/vaccination/immunization/shot/dose).
    Both must be present — either alone is too loose to suppress an alias.
    """
    pre = text[max(0, match_start - 12):match_start]
    if not _VAX_PRE_TRIGGERS.search(pre):
        return False
    post = text[match_end:match_end + 40]
    return _VAX_POST_NOUNS.match(post) is not None


def _is_negated(text: str, match_start: int, match_end: int) -> bool:
    """True if the alias span at [match_start, match_end) sits in negation scope.

    Scans a window on each side, finding the nearest terminator first to clip
    the search range, then checks for any negation trigger in the clipped span.
    Also runs a focused post-hoc scan for phrasings like "X came back
    negative" where the trigger sits after the disease name and may exceed the
    standard 30-char after-window.
    """
    # Backward window — clip to nearest terminator on the left of the match.
    left_window = text[max(0, match_start - _NEG_WINDOW_BEFORE):match_start]
    last_term = -1
    for term in _NEG_TERMINATORS.finditer(left_window):
        last_term = term.end()
    pre_span = left_window[last_term:] if last_term >= 0 else left_window
    if _NEG_TRIGGERS.search(pre_span):
        return True

    # Forward window — clip to nearest terminator on the right.
    right_window = text[match_end:min(len(text), match_end + _NEG_WINDOW_AFTER)]
    first_term = _NEG_TERMINATORS.search(right_window)
    post_span = right_window[:first_term.start()] if first_term else right_window
    if _NEG_TRIGGERS.search(post_span):
        return True

    # Post-hoc forward scan — wider window for very specific multi-word
    # negation constructions. Clause-bounded by the post-hoc terminator set
    # (which includes ',' since "X came back negative, so we treated Y..."
    # closes the X clause).
    posthoc_window = text[match_end:min(len(text), match_end + _POSTHOC_WINDOW_AFTER)]
    posthoc_term = _POSTHOC_TERMINATORS.search(posthoc_window)
    posthoc_span = posthoc_window[:posthoc_term.start()] if posthoc_term else posthoc_window
    if _POSTHOC_NEG_TRIGGERS.search(posthoc_span):
        return True

    # c20 final pass: vaccine-context negation
    # ("No varicella series", "never received MMR vaccine"). Suppresses the
    # alias when both a denial trigger precedes it AND a vaccine-context
    # noun follows. Long admission notes enumerate immunization history in
    # this form; without this rule they leak vaccine-target diseases as if
    # actively diagnosed (`adv6_long_form_admission_note`'s varicella).
    if _is_vaccine_context_negation(text, match_start, match_end):
        return True

    return False


_DEFAULT_LOOKUP_PATH = Path(__file__).parent / "lookup_table.json"
_LOOKUP_CACHE: dict[str, list[tuple[str, list[tuple[str, re.Pattern[str]]]]]] | None = None


def _load_lookup(path: Path = _DEFAULT_LOOKUP_PATH) -> dict[str, list[tuple[str, list[tuple[str, re.Pattern[str]]]]]]:
    """Compile the displayName lookup table once and cache it.

    Returns dict of category -> list of (code, [(alias_str, compiled_pattern)]).
    Aliases retained alongside their patterns so provenance can record which
    alias triggered the match.
    """
    global _LOOKUP_CACHE
    if _LOOKUP_CACHE is not None:
        return _LOOKUP_CACHE
    raw = json.loads(path.read_text())
    out: dict[str, list[tuple[str, list[tuple[str, re.Pattern[str]]]]]] = {}
    for cat in ("snomed", "loincs", "rxnorms"):
        compiled: list[tuple[str, list[tuple[str, re.Pattern[str]]]]] = []
        for entry in raw.get(cat, []):
            patterns = [
                (a, re.compile(r"\b" + re.escape(a) + r"\b", re.IGNORECASE))
                for a in entry.get("aliases", [])
            ]
            compiled.append((entry["code"], patterns))
        out[cat] = compiled
    _LOOKUP_CACHE = out
    return out


def _mask_cda_attributes(text: str) -> str:
    """Replace CDA `<...code="..." codeSystem="..." displayName="..."/>` tags
    AND any same-string narrative duplicates with whitespace so the lookup
    tier doesn't cross-axis match on text already represented by an
    explicit-tier CDA code.

    Cross-axis bleed example (HL7 eICR pertussis sample):
      `<observation><code code="80825-3" codeSystem="<LOINC>" displayName="Zika virus envelope (E) gene [Presence] in Serum"/>`
      ...later in the same CDA's <text><table> cells...
      `<td>Zika virus envelope (E) gene [Presence] in Serum...</td>`
    Without masking, the lookup tier scans the whole CDA XML and matches
    the curated `Zika` alt_name inside both the structured `displayName`
    attribute AND the rendered `<td>` cell that duplicates it, emitting
    SNOMED 3928002 (Zika virus disease) — a false positive that has
    nothing to do with the actual lab being reported.

    Only masks `<code ...>` tags whose `code="..."` attribute is NON-empty
    (i.e. the CDA tier extracts something concrete from the tag). Tags
    without a non-empty `code` attribute, or non-`<code>` elements, fall
    through so freeform narrative inside the CDA continues to be scanned
    (cf. `adv5_cda_displayname_no_code_attr` where lookup needs to fire on
    `displayName="Whooping cough"` because `code=""`).

    Replacement uses spaces of the same length to preserve byte/character
    offsets for downstream provenance tracking.
    """
    out_chars = list(text)
    masked_displays: list[str] = []
    for tag in _CDA_TAG_RE.finditer(text):
        tag_text = tag.group(0)
        oid = tag.group(1)
        if oid not in _CDA_OID_TO_BUCKET:
            continue
        code_match = _CODE_ATTR_RE.search(tag_text)
        if not code_match or not code_match.group(1):
            # Empty / missing code attribute — keep visible so lookup can fire
            # on the displayName (e.g. adv5_cda_displayname_no_code_attr).
            continue
        # Capture the displayName (if present) before masking the tag — we'll
        # also mask any duplicate occurrences of this string in CDA narrative
        # cells (`<td>Zika virus envelope...</td>` etc.) below.
        disp = _DISPLAY_NAME_ATTR_RE.search(tag_text)
        if disp and len(disp.group(1)) >= 8:
            # 8-char floor avoids masking generic tokens like "Final" /
            # "Active" that appear in many displayName attributes.
            masked_displays.append(disp.group(1))
        # Mask everything *inside the angle brackets* but keep the brackets
        # themselves so any further regex anchored on `<` boundaries still
        # works. We replace the inner span with spaces to preserve length.
        s, e = tag.start() + 1, tag.end() - 1
        for i in range(s, e):
            out_chars[i] = " "

    # Second pass: blank out same-string duplicates in narrative cells. CDA
    # `<text><table>` blocks are rendered duplicates of the structured
    # entries; if a displayName already belongs to an extracted CDA code,
    # any verbatim copy of that displayName is also accounted for and must
    # not seed an alt_name match elsewhere. We tolerate whitespace / inline
    # tag variation (e.g. `Zika virus envelope ... in Serum<br />by Probe...`
    # vs. the attribute form `Zika virus envelope ... in Serum by Probe...`)
    # by matching the displayName as a sequence of tokens separated by
    # arbitrary non-alpha-numeric whitespace/tag content.
    if masked_displays:
        masked_str = "".join(out_chars)
        # Separator pattern: any non-letter/non-digit run, optionally including
        # short inline tags like `<br />`, `<br/>`, `</span>`. Cap span at 32
        # chars per gap so we don't sprawl across unrelated table cells.
        sep = r"(?:[^A-Za-z0-9]|<[^>]{0,16}>){1,32}?"
        for disp in masked_displays:
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", disp)
            if len(tokens) < 3:
                # Single/two-word displays risk over-masking shared prose.
                continue
            pattern = r"\b" + sep.join(re.escape(t) for t in tokens) + r"\b"
            try:
                for m in re.finditer(pattern, masked_str):
                    for i in range(m.start(), m.end()):
                        out_chars[i] = " "
            except re.error:
                continue
            masked_str = "".join(out_chars)
    return "".join(out_chars)


def _is_label_header_use(alias: str, text: str, match_end: int) -> bool:
    """True when a short uppercase acronym alias (≤4 chars) is being used
    as a section/data label rather than a clinical assertion.

    Example: `CBC: WBC 4.2 (low)` — `CBC` is a label introducing lab values,
    NOT a clinical assertion that the patient *has* a complete blood count
    diagnosis. Test cases authored without explicit `(LOINC 57021-8)`
    annotation expect the routine-lab CBC to NOT extract as a reportable
    LOINC. Same pattern for `CMP:`, `CRP:`, etc.

    Conservative scope: only triggers for ≤4-char uppercase alias followed
    immediately by `:` (allowing optional whitespace between). Long display
    names like "Complete blood count" don't go through this path.
    """
    if len(alias) > 4 or not alias.isupper():
        return False
    # Allow up to 2 chars of whitespace between alias and colon
    return bool(re.match(r"\s{0,2}:", text[match_end:match_end + 4]))


def _extract_lookup_matches(text: str, *, detect_negation: bool = True) -> list[CodeMatch]:
    """Match known displayNames in text, return per-match provenance records.

    Suppresses matches in NegEx-style negation scope. First non-negated alias
    per code wins; the alias string and source span are preserved for the UI.
    """
    table = _load_lookup()
    cat_to_bucket = {"snomed": "snomed", "loincs": "loinc", "rxnorms": "rxnorm"}
    out: list[CodeMatch] = []
    # c20 cross-axis bleed fix: mask CDA `<code .../>` attribute values so
    # the lookup tier never matches an alt_name (e.g. `Zika`) inside a
    # `displayName="..."` attribute that already belongs to a different-axis
    # explicit code (e.g. a LOINC for `Zika virus envelope...`). Offsets
    # preserved via length-equal whitespace replacement.
    scan_text = _mask_cda_attributes(text)
    for cat, bucket in cat_to_bucket.items():
        for code, patterns in table.get(cat, []):
            picked: tuple[str, re.Match[str]] | None = None
            for alias, p in patterns:
                for m in p.finditer(scan_text):
                    if detect_negation and _is_negated(scan_text, m.start(), m.end()):
                        continue
                    # c20 final pass: skip short acronym aliases used as
                    # data-label headers (`CBC:`, `CMP:`). Lookup tier
                    # treats them as label markers; if the case actually
                    # wants the lab coded, the inline tier handles
                    # `Lab: Complete blood count (LOINC 57021-8)`.
                    if _is_label_header_use(alias, scan_text, m.end()):
                        continue
                    picked = (alias, m)
                    break
                if picked is not None:
                    break
            if picked is None:
                continue
            alias, m = picked
            out.append(CodeMatch(
                code=code,
                display=alias,
                system=SYSTEM_NAMES[bucket],
                bucket=bucket,
                tier="lookup",
                confidence=_TIER_CONFIDENCE["lookup"],
                # Source span quoted from the ORIGINAL text so any CDA-attr
                # masking is invisible to provenance consumers.
                source_text=text[m.start():m.end()],
                source_offset=m.start(),
                source_length=m.end() - m.start(),
                alias=alias,
            ))
    return out


def _extract_lookup(text: str, *, detect_negation: bool = True) -> dict[str, list[str]]:
    """Backward-compat: dict-of-lists view of lookup matches."""
    buckets: dict[str, list[str]] = {"snomed": [], "loinc": [], "rxnorm": []}
    for cm in _extract_lookup_matches(text, detect_negation=detect_negation):
        buckets[cm.bucket].append(cm.code)
    return buckets


@dataclass(frozen=True)
class Extraction:
    conditions: list[str]
    loincs: list[str]
    rxnorms: list[str]
    # Per-code provenance — one CodeMatch per unique (code, bucket) emitted.
    # First-seen wins for dedup; later-tier matches for the same code are
    # dropped, so order matters: inline > cda > lookup > rag.
    matches: list[CodeMatch] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "conditions": list(self.conditions),
                "loincs": list(self.loincs),
                "rxnorms": list(self.rxnorms),
            }
        )

    def to_provenance_dict(self) -> dict:
        """Rich output: code arrays + per-code provenance for the agent/UI."""
        return {
            "conditions": list(self.conditions),
            "loincs": list(self.loincs),
            "rxnorms": list(self.rxnorms),
            "matches": [m.to_dict() for m in self.matches],
        }


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def extract(
    text: str,
    *,
    use_lookup: bool = True,
    detect_negation: bool = True,
) -> Extraction:
    """Run the deterministic 3-tier extractor with full provenance tracking.

    Tier order (also dedup priority — first occurrence of a (bucket, code)
    tuple wins, later ones are dropped):

      inline   parenthesized "(SNOMED 76272004)" labels
      cda      <code code="…" codeSystem="…"/>
      lookup   curated displayName aliases (NegEx-suppressed)
    """
    # c20 adv6 fix: NFKC normalize so non-breaking spaces (U+00A0),
    # smart quotes, em/en dashes, and other typographic Unicode are
    # collapsed to ASCII forms before regex matching. Lookup-table
    # aliases use ASCII `\s` / `\b`; without normalization a phrase
    # like "Bordetella\xa0pertussis" silently misses
    # (`adv6_unicode_typography_*`).
    text = unicodedata.normalize("NFKC", text)
    matches: list[CodeMatch] = []
    matches.extend(_extract_inline(text))
    matches.extend(_extract_cda_matches(text))
    if use_lookup:
        matches.extend(_extract_lookup_matches(text, detect_negation=detect_negation))

    # Dedup keeping the highest-priority (earliest-tier) record per code.
    seen: set[tuple[str, str]] = set()
    deduped: list[CodeMatch] = []
    by_bucket: dict[str, list[str]] = {"snomed": [], "loinc": [], "rxnorm": []}
    for cm in matches:
        key = (cm.bucket, cm.code)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cm)
        by_bucket[cm.bucket].append(cm.code)

    return Extraction(
        conditions=by_bucket["snomed"],
        loincs=by_bucket["loinc"],
        rxnorms=by_bucket["rxnorm"],
        matches=deduped,
    )


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "case_file",
        help="JSONL with test cases (e.g. scripts/test_cases.jsonl)",
    )
    ap.add_argument(
        "--show-output",
        action="store_true",
        help="Print extractor output JSON for each case",
    )
    ap.add_argument(
        "--no-lookup",
        action="store_true",
        help="Disable the displayName lookup tier (regex + CDA XML only)",
    )
    ap.add_argument(
        "--no-negation",
        action="store_true",
        help="Disable NegEx-style negation detection on lookup matches",
    )
    args = ap.parse_args()

    rows = []
    for ln in open(args.case_file):
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))

    total_expected = 0
    total_matched = 0
    total_extracted = 0
    total_false_pos = 0
    perfect_cases = 0
    print(f"Running {len(rows)} case(s) from {args.case_file}\n")
    for idx, case in enumerate(rows, 1):
        case_id = case.get("case_id", f"case_{idx}")
        result = extract(
            case["user"],
            use_lookup=not args.no_lookup,
            detect_negation=not args.no_negation,
        )

        exp_c = case.get("expected_conditions", []) or []
        exp_l = case.get("expected_loincs", []) or []
        exp_r = case.get("expected_rxnorms", []) or []

        m_c = sum(1 for c in exp_c if c in result.conditions)
        m_l = sum(1 for c in exp_l if c in result.loincs)
        m_r = sum(1 for c in exp_r if c in result.rxnorms)

        case_expected = len(exp_c) + len(exp_l) + len(exp_r)
        case_matched = m_c + m_l + m_r
        case_extracted = (
            len(result.conditions) + len(result.loincs) + len(result.rxnorms)
        )
        # False positives: codes the parser emitted that weren't expected.
        # Computed per category against that category's expected set, so a
        # SNOMED match never excuses a spurious LOINC.
        fp_c = [c for c in result.conditions if c not in exp_c]
        fp_l = [c for c in result.loincs if c not in exp_l]
        fp_r = [c for c in result.rxnorms if c not in exp_r]
        case_false_pos = len(fp_c) + len(fp_l) + len(fp_r)

        total_expected += case_expected
        total_matched += case_matched
        total_extracted += case_extracted
        total_false_pos += case_false_pos
        if case_expected and case_matched == case_expected and case_false_pos == 0:
            perfect_cases += 1

        status = "OK " if (case_matched == case_expected and case_false_pos == 0) else "MISS"
        fp_marker = f" FP={case_false_pos}" if case_false_pos else ""
        print(
            f"  {status} {idx}/{len(rows)} {case_id}: "
            f"{case_matched}/{case_expected} "
            f"(C {m_c}/{len(exp_c)}, L {m_l}/{len(exp_l)}, R {m_r}/{len(exp_r)})"
            f"{fp_marker}"
        )
        if args.show_output:
            print(f"        out: {result.to_json()}")
        if case_matched != case_expected:
            missing_c = [c for c in exp_c if c not in result.conditions]
            missing_l = [c for c in exp_l if c not in result.loincs]
            missing_r = [c for c in exp_r if c not in result.rxnorms]
            if missing_c:
                print(f"        missing SNOMED: {missing_c}")
            if missing_l:
                print(f"        missing LOINC : {missing_l}")
            if missing_r:
                print(f"        missing RxNorm: {missing_r}")
        if case_false_pos:
            if fp_c:
                print(f"        spurious SNOMED (FP): {fp_c}")
            if fp_l:
                print(f"        spurious LOINC  (FP): {fp_l}")
            if fp_r:
                print(f"        spurious RxNorm (FP): {fp_r}")

    recall = total_matched / total_expected if total_expected else 0.0
    precision = total_matched / total_extracted if total_extracted else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    print(
        f"\nAggregate: {total_matched}/{total_expected} matched, "
        f"{total_extracted} extracted, {total_false_pos} false-positive"
    )
    print(
        f"  recall = {recall:.3f}  precision = {precision:.3f}  F1 = {f1:.3f}  "
        f"({perfect_cases}/{len(rows)} cases perfect)"
    )
    # Exit code: 0 only if recall == 1.0 AND no false positives.
    sys.exit(0 if total_matched == total_expected and total_false_pos == 0 else 1)


if __name__ == "__main__":
    main()
