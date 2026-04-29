"""Longitudinal case diff — EZeCR-style flat CSV at the edge.

Mirrors the CDC EZeCR (Easy electronic Case Reporting) MVP "flat CSV diff
between case versions for the same patient" feature documented in the
ThoughtWorks/Ad Hoc CDC D2E Workshop Readout (page 10–11). EZeCR's MVP
deliberately avoids structured longitudinal patient records: instead it
emits one row per axis change (added / removed / unchanged) keyed on a
patient hash, so a clinician can see "what's new this visit" without
maintaining a server-side longitudinal record. We do the same on-device.

Patient identity (edge, no Verato-style probabilistic matching):

    patient_hash = sha256(lower(given) || "|" || lower(family) ||
                          "|" || iso8601(dob))[:16]

Exact match only — clinician confirms at intake. Probabilistic matching
belongs server-side; on-device we trade recall for zero-PHI on the wire.

Axis routing (from FHIR resource type + Observation.category):
    Condition resource                              → 'condition' axis
    Observation with category=laboratory            → 'lab' axis
    Observation with category=vital-signs           → 'vital' axis
    MedicationStatement / MedicationRequest         → 'medication' axis

Diff per axis is set-membership over (code_system, code) keys:
    added       = current - prior
    removed     = prior - current
    unchanged   = current ∩ prior

The Swift mirror lives in `apps/mobile/ios-app/ClinIQ/ClinIQ/Diff/CaseDiff.swift`
(owned by the iOS UX agent). The algorithm and `CodedEntity` shape MUST
stay byte-for-byte identical — both sides produce the same CSV row order,
same patient_hash for the same Patient.name + Patient.birthDate, same
axis labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Iterable

# FHIR canonical system URIs (mirror of `fhir_bundle.SYSTEM_URI` values).
SNOMED_URI = "http://snomed.info/sct"
LOINC_URI = "http://loinc.org"
RXNORM_URI = "http://www.nlm.nih.gov/research/umls/rxnorm"

# `Observation.category` codes per FHIR R4 ObservationCategory value set.
_CAT_LAB = "laboratory"
_CAT_VITAL = "vital-signs"

# Axis labels — keep these stable; the Swift CaseDiff enum uses the same
# strings, and the EZeCR CSV `axis` column is keyed off them downstream.
AXIS_CONDITION = "condition"
AXIS_LAB = "lab"
AXIS_VITAL = "vital"
AXIS_MEDICATION = "medication"

# CSV header order — fixed by the EZeCR MVP spec; do not reorder without
# coordinating with the iOS CaseDiff exporter.
CSV_HEADER = [
    "patient_hash",
    "case_id",
    "case_dt",
    "axis",
    "code_system",
    "code",
    "display",
    "change_type",
]


# ---------------------------------------------------------------------------
# Data shapes


@dataclass(frozen=True)
class CodedEntity:
    """One axis-tagged code with its FHIR system URI + optional display.

    `display` is best-effort (pulled from `Coding.display` if present, else
    from the curated lookup in `fhir_bundle._display_for`). `axis` is one of
    AXIS_CONDITION / AXIS_LAB / AXIS_VITAL / AXIS_MEDICATION.
    """

    axis: str
    code_system: str
    code: str
    display: str | None = None

    def key(self) -> tuple[str, str, str]:
        return (self.axis, self.code_system, self.code)


@dataclass
class AxisDiff:
    """Per-axis added/removed/unchanged buckets."""

    axis: str
    added: list[CodedEntity] = field(default_factory=list)
    removed: list[CodedEntity] = field(default_factory=list)
    unchanged: list[CodedEntity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "added": [_entity_dict(e) for e in self.added],
            "removed": [_entity_dict(e) for e in self.removed],
            "unchanged": [_entity_dict(e) for e in self.unchanged],
        }


@dataclass
class CaseDiff:
    """Per-pair (prior → current) diff across all four axes.

    `prior_case_id` / `current_case_id` are the raw case ids the caller
    passed in (e.g. `p1_v1`, `p1_v2`). `patient_hash` is included on every
    diff so downstream consumers (CSV writer, iOS detail view) can group
    diffs by patient without re-hashing.
    """

    patient_hash: str
    prior_case_id: str
    current_case_id: str
    axes: dict[str, AxisDiff]

    def to_dict(self) -> dict:
        return {
            "patient_hash": self.patient_hash,
            "prior_case_id": self.prior_case_id,
            "current_case_id": self.current_case_id,
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
        }


def _entity_dict(e: CodedEntity) -> dict:
    return {
        "axis": e.axis,
        "code_system": e.code_system,
        "code": e.code,
        "display": e.display,
    }


# ---------------------------------------------------------------------------
# Patient identity


def patient_hash(given: str, family: str, dob: str) -> str:
    """sha256(lower(given) || '|' || lower(family) || '|' || iso8601(dob))[:16].

    `dob` is expected as an ISO-8601 date or datetime string; we don't
    re-parse it — caller is responsible for canonicalizing. Lowercasing the
    name parts is intentional; DOB is kept as-passed so callers control
    timezone semantics.
    """
    raw = f"{given.lower()}|{family.lower()}|{dob}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def patient_hash_from_bundle(bundle: dict) -> str | None:
    """Pull (given, family, birthDate) from a Bundle's Patient entry and hash.

    Returns None if the Bundle has no Patient with both name and birthDate
    populated. The minimal Patient entry produced by `fhir_bundle.to_bundle`
    is id-only (no name/birthDate) — callers that want a real patient hash
    must build the Bundle through a path that carries demographics, OR pass
    (given, family, dob) directly to `patient_hash()`.
    """
    for entry in bundle.get("entry") or []:
        res = entry.get("resource") or {}
        if res.get("resourceType") != "Patient":
            continue
        names = res.get("name") or []
        dob = res.get("birthDate")
        if not names or not dob:
            return None
        n = names[0]
        given_list = n.get("given") or []
        given = (given_list[0] if given_list else "").strip()
        family = (n.get("family") or "").strip()
        if not given or not family:
            return None
        return patient_hash(given, family, dob)
    return None


# ---------------------------------------------------------------------------
# Bundle → CodedEntity[] extraction


def _coding_first(codeable: dict | None) -> tuple[str, str, str | None] | None:
    """Pull the first (system, code, display) from a CodeableConcept-shaped dict."""
    if not codeable:
        return None
    codings = codeable.get("coding") or []
    if not codings:
        return None
    c = codings[0]
    system = c.get("system")
    code = c.get("code")
    if not system or not code:
        return None
    return system, code, c.get("display")


def _category_codes(resource: dict) -> set[str]:
    """Flatten Observation.category[].coding[].code into a set."""
    out: set[str] = set()
    for cat in resource.get("category") or []:
        for c in cat.get("coding") or []:
            code = c.get("code")
            if code:
                out.add(code)
    return out


def entities_from_bundle(bundle: dict) -> list[CodedEntity]:
    """Walk a FHIR Bundle and emit one CodedEntity per axis-relevant resource.

    Unsupported resource types are silently skipped. Observations without a
    laboratory or vital-signs category are also skipped — we only diff the
    four EZeCR axes.
    """
    out: list[CodedEntity] = []
    for entry in bundle.get("entry") or []:
        res = entry.get("resource") or {}
        rtype = res.get("resourceType")
        if rtype == "Condition":
            picked = _coding_first(res.get("code"))
            if picked is None:
                continue
            system, code, display = picked
            out.append(CodedEntity(AXIS_CONDITION, system, code, display))
        elif rtype == "Observation":
            picked = _coding_first(res.get("code"))
            if picked is None:
                continue
            system, code, display = picked
            cats = _category_codes(res)
            if _CAT_VITAL in cats:
                axis = AXIS_VITAL
            elif _CAT_LAB in cats:
                axis = AXIS_LAB
            else:
                # No explicit category → default to `lab`. Our `to_bundle()`
                # only stamps `vital-signs` on the curated `_VITAL_SIGN_LOINCS`
                # set; anything else is a lab Observation by construction
                # (the extraction surface only carries SNOMED conditions,
                # LOINC labs/vitals, and RxNorm meds — there is no third
                # Observation flavor in scope). Mirror this in the Swift
                # CaseDiff: same default-to-lab rule, same axis label.
                axis = AXIS_LAB
            out.append(CodedEntity(axis, system, code, display))
        elif rtype in ("MedicationStatement", "MedicationRequest"):
            picked = _coding_first(res.get("medicationCodeableConcept"))
            if picked is None:
                continue
            system, code, display = picked
            out.append(CodedEntity(AXIS_MEDICATION, system, code, display))
    return out


# ---------------------------------------------------------------------------
# Diff


def diff_codes(
    prior: list[CodedEntity], current: list[CodedEntity]
) -> dict[str, list[CodedEntity]]:
    """Set-membership diff over (code_system, code) keys.

    `prior` and `current` are flat lists of any axis. The returned dict has
    keys `added` / `removed` / `unchanged`, with `added` and `unchanged`
    drawn from `current` (so display strings reflect the current case) and
    `removed` drawn from `prior`.
    """
    prior_keys = {(e.code_system, e.code) for e in prior}
    current_keys = {(e.code_system, e.code) for e in current}
    added = [e for e in current if (e.code_system, e.code) not in prior_keys]
    removed = [e for e in prior if (e.code_system, e.code) not in current_keys]
    unchanged = [e for e in current if (e.code_system, e.code) in prior_keys]
    return {"added": added, "removed": removed, "unchanged": unchanged}


def compute_diff(
    prior_bundle: dict,
    current_bundle: dict,
    *,
    patient_hash_value: str,
    prior_case_id: str,
    current_case_id: str,
) -> CaseDiff:
    """Compute a per-axis diff between two FHIR Bundles for the same patient.

    Caller passes the patient_hash explicitly because the minimal Bundle
    shape `fhir_bundle.to_bundle` emits doesn't carry demographics — the
    hash has to come from the agent_pipeline JSONL row's `patient` block.
    """
    prior_entities = entities_from_bundle(prior_bundle)
    current_entities = entities_from_bundle(current_bundle)

    axes: dict[str, AxisDiff] = {}
    for axis in (AXIS_CONDITION, AXIS_LAB, AXIS_VITAL, AXIS_MEDICATION):
        prior_axis = [e for e in prior_entities if e.axis == axis]
        current_axis = [e for e in current_entities if e.axis == axis]
        d = diff_codes(prior_axis, current_axis)
        axes[axis] = AxisDiff(
            axis=axis,
            added=d["added"],
            removed=d["removed"],
            unchanged=d["unchanged"],
        )
    return CaseDiff(
        patient_hash=patient_hash_value,
        prior_case_id=prior_case_id,
        current_case_id=current_case_id,
        axes=axes,
    )


# ---------------------------------------------------------------------------
# EZeCR flat CSV emission


def _csv_rows_for_case(
    *,
    patient_hash_value: str,
    case_id: str,
    case_dt: str,
    entities: list[CodedEntity],
    change_type: str,
) -> list[list[str]]:
    """Project one batch of entities into CSV rows for a single (case, change_type)."""
    out: list[list[str]] = []
    for e in entities:
        out.append([
            patient_hash_value,
            case_id,
            case_dt,
            e.axis,
            e.code_system,
            e.code,
            e.display or "",
            change_type,
        ])
    return out


def _csv_sort_key(row: list[str]) -> tuple:
    # Order: case_dt, axis, change_type, code_system, code. Stable across
    # platforms because all keys are strings.
    return (row[2], row[3], row[7], row[4], row[5])


def emit_csv_rows(
    series: Iterable[dict],
) -> list[list[str]]:
    """Flatten a longitudinal series into EZeCR flat CSV rows.

    `series` is an ordered iterable of dicts with shape:

        {
            "patient_hash": "abc1234567890def",
            "case_id":      "p1_v1",
            "case_dt":      "2026-04-15T09:00:00Z",
            "bundle":       {<FHIR Bundle dict>},
            "prior_bundle": {<FHIR Bundle dict>} | None,  # None for the first case
        }

    For the FIRST case (prior_bundle is None) every entity is emitted as
    `unchanged` — there is no prior to diff against, but downstream consumers
    expect every entity to appear in the CSV at least once so the patient's
    starting state is captured. For SUBSEQUENT cases we emit added /
    removed / unchanged per the diff.

    Rows are sorted by (case_dt, axis, change_type, code_system, code).
    Header row is NOT included; caller adds it.
    """
    rows: list[list[str]] = []
    for visit in series:
        patient_hash_value = visit["patient_hash"]
        case_id = visit["case_id"]
        case_dt = visit["case_dt"]
        bundle = visit["bundle"]
        prior_bundle = visit.get("prior_bundle")

        current_entities = entities_from_bundle(bundle)
        if prior_bundle is None:
            rows.extend(_csv_rows_for_case(
                patient_hash_value=patient_hash_value,
                case_id=case_id,
                case_dt=case_dt,
                entities=current_entities,
                change_type="unchanged",
            ))
            continue

        prior_entities = entities_from_bundle(prior_bundle)
        d = diff_codes(prior_entities, current_entities)
        rows.extend(_csv_rows_for_case(
            patient_hash_value=patient_hash_value,
            case_id=case_id,
            case_dt=case_dt,
            entities=d["added"],
            change_type="added",
        ))
        rows.extend(_csv_rows_for_case(
            patient_hash_value=patient_hash_value,
            case_id=case_id,
            case_dt=case_dt,
            entities=d["removed"],
            change_type="removed",
        ))
        rows.extend(_csv_rows_for_case(
            patient_hash_value=patient_hash_value,
            case_id=case_id,
            case_dt=case_dt,
            entities=d["unchanged"],
            change_type="unchanged",
        ))
    rows.sort(key=_csv_sort_key)
    return rows


def write_csv(rows: list[list[str]], path: str | Path) -> None:
    """Write `rows` (without header) plus the canonical CSV_HEADER to `path`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for row in rows:
            w.writerow(row)


def to_csv_string(rows: list[list[str]]) -> str:
    """Same as write_csv but to a string buffer (used by smoke tests)."""
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_HEADER)
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Manifest helpers (used by score_fhir.py --diff-csv)


def load_manifest(path: str | Path) -> list[dict]:
    """Load a longitudinal manifest JSON.

    Manifest shape (one entry per case, ordered by case_dt):

        [
            {"case_id": "p1_v1", "patient_hash": "abc...", "case_dt": "...",
             "bundle_path": "build/bundles/p1_v1.json"},
            ...
        ]

    `bundle_path` is resolved relative to the manifest file unless absolute.
    """
    p = Path(path)
    raw = json.loads(p.read_text())
    base = p.parent
    out: list[dict] = []
    for row in raw:
        bp = Path(row["bundle_path"])
        if not bp.is_absolute():
            bp = base / bp
        out.append({
            "case_id": row["case_id"],
            "patient_hash": row["patient_hash"],
            "case_dt": row["case_dt"],
            "bundle_path": str(bp),
        })
    return out


def emit_csv_from_manifest(manifest_path: str | Path, csv_out: str | Path) -> int:
    """Load each manifest row's Bundle, build the prior-current series, write CSV.

    Cases are paired in manifest order — entry N's prior is entry N-1.
    Returns the number of rows written (excluding the header).
    """
    manifest = load_manifest(manifest_path)
    series: list[dict] = []
    prior_bundle: dict | None = None
    for row in manifest:
        bundle = json.loads(Path(row["bundle_path"]).read_text())
        series.append({
            "patient_hash": row["patient_hash"],
            "case_id": row["case_id"],
            "case_dt": row["case_dt"],
            "bundle": bundle,
            "prior_bundle": prior_bundle,
        })
        prior_bundle = bundle
    rows = emit_csv_rows(series)
    write_csv(rows, csv_out)
    return len(rows)
