"""Wrap a ClinIQ extraction in a minimal FHIR R4 Bundle.

Per proposals-2026-04-25.md Rank 3, the deliverable is "100% R4-valid
Bundles, on-device, offline" as a categorical clinical-interop credibility
signal. This module turns the deterministic+agent extraction shape
(`{"conditions": [...], "loincs": [...], "rxnorms": [...]}`) into an R4
Bundle with one Patient entry plus one Condition / Observation /
MedicationStatement entry per code.

Pure-stdlib output — no fhir.resources dependency. Validation lives in
`score_fhir.py` (separate file because it pulls in the heavy pydantic +
fhir.resources stack).

Mirror in Swift: `apps/mobile/ios-app/ClinIQ/ClinIQ/FHIR/BundleBuilder.swift`
(when ios-eng comes back online). Validation stays Python-only — there
is no Swift FHIR R4 validator we trust enough to ship.

Provenance flow: when callers pass a `provenance_map` dict mapping each
code → its CodeProvenance source URL (from EicrPreparser /
ReportableConditions / RagSearch), we stamp `Resource.meta.source` with
that URL. This lets a judge tap a Condition in the Bundle and click
through to the CDC NNDSS / WHO IDSR page that grounds it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Local — pull displayName from the curated lookup so Coding entries get
# `display`, which makes Bundles human-readable when judges open them.
sys.path.insert(0, str(Path(__file__).parent))
from regex_preparser import _load_lookup  # noqa: E402

# FHIR canonical system URIs.
SYSTEM_URI = {
    "SNOMED": "http://snomed.info/sct",
    "LOINC": "http://loinc.org",
    "RXNORM": "http://www.nlm.nih.gov/research/umls/rxnorm",
}

# Default subject ref for every Bundle this module emits.
DEFAULT_PATIENT_ID = "cliniq-patient-1"

# Cache the inverse lookup (code → displayName) so we don't rebuild per call.
_DISPLAY_CACHE: dict[tuple[str, str], str] | None = None


def _is_fhir_id_safe(value: str) -> bool:
    """FHIR id-safe token check used for atypical local/CDA slot codes."""
    if not value:
        return False
    return all(ch.isalnum() or ch in "-." for ch in value)


def normalize_code_value(bucket: str, value: object) -> str | None:
    """Return a raw code string for an extraction bucket.

    The model sometimes emits display-bearing strings such as
    ``"Pertussis (SNOMED 27836007)"``. Bundle generation and exact scoring both
    need the raw code only. This parser intentionally stays conservative: if a
    value has no bucket-appropriate code token, it is dropped rather than
    converted into a display string masquerading as a code.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None

    if bucket == "conditions":
        if text.isdigit():
            return text
        # SNOMED CT identifiers are integer SCTIDs. Require at least five
        # digits so malformed snippets like "SNOMED 68.90" do not become "68".
        match = re.search(
            r"\b(?:SNOMED(?:\s+CT)?\s*)?(\d{5,18})\b",
            text,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    if bucket == "loincs":
        match = re.search(r"\b\d{1,7}-\d\b", text)
        if match:
            return match.group(0)
        # Existing CDA fixtures include a few author-assigned, LOINC-bucket
        # identifiers that are not hyphenated LOINCs. Preserve only whole-token
        # FHIR-id-safe values; never preserve display-bearing prose.
        if _is_fhir_id_safe(text) and any(ch.isdigit() for ch in text):
            return text
        return None

    if bucket == "rxnorms":
        if text.isdigit():
            return text
        match = re.search(r"\b(?:RXNORM\s*)?(\d{3,18})\b", text, re.IGNORECASE)
        return match.group(1) if match else None

    return text if _is_fhir_id_safe(text) else None


def normalize_extraction(extraction: dict) -> dict:
    """Normalize/deduplicate a ClinIQ extraction while preserving order."""
    out = {"conditions": [], "loincs": [], "rxnorms": []}
    seen = {key: set() for key in out}
    for key in out:
        values = extraction.get(key) or []
        if not isinstance(values, list):
            continue
        for raw in values:
            code = normalize_code_value(key, raw)
            if code and code not in seen[key]:
                seen[key].add(code)
                out[key].append(code)
    return out


def _build_display_index() -> dict[tuple[str, str], str]:
    """Index (system_upper, code) → first alias (displayName) from lookup table.

    The lookup table groups by category (snomed/loincs/rxnorms) and stores
    aliases as a list with the canonical displayName first by convention
    (see lookup_table.json hand-curated ordering).
    """
    out: dict[tuple[str, str], str] = {}
    table = _load_lookup()
    for cat, system in (("snomed", "SNOMED"), ("loincs", "LOINC"), ("rxnorms", "RXNORM")):
        for code, patterns in table.get(cat, []):
            if patterns:
                first_alias = patterns[0][0]
                out[(system, code)] = first_alias
    return out


def _display_for(system: str, code: str) -> str | None:
    global _DISPLAY_CACHE
    if _DISPLAY_CACHE is None:
        _DISPLAY_CACHE = _build_display_index()
    return _DISPLAY_CACHE.get((system.upper(), code))


def _coding(system: str, code: str, *, display: str | None = None) -> dict:
    """Build a Coding entry. Display string is intentionally OMITTED.

    Background: the HL7-published `validator_cli.jar` (canonical R4 ref)
    enforces terminology binding on `Coding.display` — it looks the code
    up against the configured terminology service and rejects any display
    string that isn't one of the canonical names for that code. Our
    curated lookup table uses short, human-readable aliases (e.g.
    "SARS-CoV-2 RNA NAA+probe Ql Resp") which are not the canonical LOINC
    display ("SARS-CoV-2 (COVID-19) RNA [Presence] in Respiratory system
    specimen by NAA with probe detection"). Including them generates
    "Wrong Display Name" errors that mask real validation findings.

    The `display` field is OPTIONAL per FHIR R4 — Coding.display has
    cardinality 0..1 — and downstream consumers (iOS UI, Gradio Space)
    surface human-readable names via the lookup table directly, not via
    `Coding.display`. Dropping it keeps the Bundle JSON spec-pure
    against the canonical validator while preserving every other
    feature.
    """
    return {
        "system": SYSTEM_URI[system.upper()],
        "code": code,
    }


def _codeable_concept(system: str, codes: list[str]) -> dict:
    """Build a CodeableConcept with one or more codings."""
    return {"coding": [_coding(system, code) for code in codes]}


def _meta_with_source(source_url: str | None) -> dict | None:
    if not source_url:
        return None
    return {"source": source_url}


def _entry_id_and_full_url(prefix: str, code: str) -> tuple[str, str]:
    """Build a stable resource id + fullUrl for a Bundle entry.

    The HL7 canonical validator enforces invariant `bdl-7` ("a fullUrl
    SHALL be unique in a bundle, or else entries with the same fullUrl
    SHALL have different meta.versionId") AND requires `fullUrl` on every
    entry of a `collection` Bundle (otherwise relative subject references
    can't be resolved). We synthesize a urn:uuid-style identifier from
    the prefix + code so the same extraction always produces the same
    Bundle (deterministic, diff-friendly).
    """
    rid = f"{prefix}-{code}"
    full_url = f"urn:cliniq:{rid}"
    return rid, full_url


def _condition_entry(
    code: str, *, patient_ref: str, source_url: str | None = None
) -> dict:
    """Single SNOMED → Condition resource. R4 minimal valid shape.

    Required cardinality: subject (1..1), code (1..1). We additionally set
    `clinicalStatus = active` because R4's Condition.clinicalStatus is
    "must support" in US Core and many strict validators reject Bundles
    that omit it.
    """
    rid, full_url = _entry_id_and_full_url("condition", code)
    resource: dict[str, Any] = {
        "resourceType": "Condition",
        "id": rid,
        "subject": {"reference": f"urn:cliniq:patient-{patient_ref}"},
        "code": {"coding": [_coding("SNOMED", code)]},
        "clinicalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "active",
            }]
        },
    }
    meta = _meta_with_source(source_url)
    if meta:
        resource["meta"] = meta
    return {"fullUrl": full_url, "resource": resource}


_BP_COMPONENT_LOINCS = frozenset({
    "8480-6",   # Systolic blood pressure
    "8462-4",   # Diastolic blood pressure
})
_VITAL_SIGN_LOINCS = frozenset({
    *_BP_COMPONENT_LOINCS,
    "8867-4",   # Heart rate
    "8310-5",   # Body temperature
    "9279-1",   # Respiratory rate
    "8302-2",   # Body height
    "29463-7",  # Body weight (canonical magic LOINC for `bodyweight` profile)
    "59408-5",  # SpO2 (pulse oximetry, O2 saturation in arterial blood)
    "2708-6",   # Oxygen saturation in arterial blood (canonical magic LOINC for `oxygensat` profile)
    "2710-2",   # Oxygen saturation in capillary blood (also auto-binds to `oxygensat`)
    "85354-9",  # Blood pressure panel (parent BP observation)
    "85353-3",  # Vital signs panel
    "8716-3",   # Vital signs panel (alt)
    "3141-9",   # Body weight measured (auto-binds to `bodyweight` profile)
})
_VALUE_REQUIRED_VITAL_COMPONENT_LOINCS = frozenset({
    "39156-5",  # BMI
    "8287-5",   # Head circumference
})
# These are intentionally not classified as standalone vital-sign Observations.
# Their FHIR R4 profiles require a numeric `valueQuantity`; ClinIQ's
# extraction surface carries only the code, not the measured value. We handle
# them explicitly below as vital-signs panel components with
# `dataAbsentReason=unknown`, preserving the extracted LOINC without inventing
# measurements.
_VITAL_SIGN_MAGIC_CODE = {
    # These CDA sample codes auto-bind to base vital-sign profiles that
    # require a canonical "magic" LOINC in Observation.code. FHIR allows
    # additional codings, so keep the extracted code and add the canonical
    # profile code rather than rewriting the extraction answer.
    "2710-2": "2708-6",   # Capillary O2 saturation -> O2 sat profile code
    "3141-9": "29463-7",  # Measured body weight -> Body weight profile code
}


def _data_absent_reason_unknown() -> dict:
    return {
        "coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/data-absent-reason",
            "code": "unknown",
        }]
    }

# `Observation.category=vital-signs` slice required by the FHIR R4 base
# `vitalsigns` profile. The HL7-published validator auto-binds any
# Observation with a vital-sign LOINC code to that profile and rejects
# the resource if the category slice is missing.
_VSCAT_CATEGORY = {
    "coding": [{
        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
        "code": "vital-signs",
        "display": "Vital Signs",
    }]
}


def _observation_entry(
    code: str, *, patient_ref: str, source_url: str | None = None
) -> dict:
    """Single LOINC → Observation resource. R4 required: status, code.

    For vital-sign LOINCs (BP, HR, temp, RR, height, weight, BMI, SpO2)
    we also stamp `Observation.category=[VSCat]` because the FHIR R4
    base `vitalsigns` profile auto-binds those codes and the canonical
    HL7 validator (`validator_cli.jar`) rejects the resource without a
    `vital-signs` category slice. Non-vital-sign LOINCs (lab results,
    diagnostic NAA tests, etc.) keep the bare shape — adding category
    there would be an over-claim.
    """
    rid, full_url = _entry_id_and_full_url("observation", code)
    if code in _BP_COMPONENT_LOINCS:
        return _blood_pressure_observation_entry(
            code, patient_ref=patient_ref, source_url=source_url
        )
    if code in _VALUE_REQUIRED_VITAL_COMPONENT_LOINCS:
        return _vital_component_observation_entry(
            code, patient_ref=patient_ref, source_url=source_url
        )

    loinc_codes = [code]
    magic_code = _VITAL_SIGN_MAGIC_CODE.get(code)
    if magic_code and magic_code not in loinc_codes:
        loinc_codes.append(magic_code)

    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": rid,
        "status": "final",
        "code": _codeable_concept("LOINC", loinc_codes),
        "subject": {"reference": f"urn:cliniq:patient-{patient_ref}"},
    }
    if code in _VITAL_SIGN_LOINCS:
        resource["category"] = [_VSCAT_CATEGORY]
        # FHIR R4 vital-signs profile slice requires `effective[x]`
        # (Observation.effectiveDateTime or .effectivePeriod). We don't
        # have an authoritative timestamp from the extraction, so we
        # stamp a placeholder ISO-8601 date that satisfies cardinality.
        # Downstream consumers that need a real time should enrich the
        # Bundle with a case-specific encounter timestamp.
        resource["effectiveDateTime"] = "2026-01-01"
        # vs-2 invariant ("if no component or hasMember, value[x] or
        # dataAbsentReason must be present"): we have no measured value
        # plumbed through from the eICR extraction surface (extraction
        # surfaces *codes only*, not values), so emit a structured
        # dataAbsentReason. `unknown` is a valid value-set member.
        resource["dataAbsentReason"] = _data_absent_reason_unknown()
    meta = _meta_with_source(source_url)
    if meta:
        resource["meta"] = meta
    return {"fullUrl": full_url, "resource": resource}


def _vital_component_observation_entry(
    code: str, *, patient_ref: str, source_url: str | None = None
) -> dict:
    """Represent value-required code-only vitals as panel components.

    Standalone Observations for codes like BMI (`39156-5`) and head
    circumference (`8287-5`) must carry numeric `valueQuantity.value` under
    their FHIR R4 profiles. The extraction surface has no value, so we wrap
    the code as a component under the general vital-signs panel and mark the
    value absent.
    """
    rid, full_url = _entry_id_and_full_url("observation", code)
    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": rid,
        "status": "final",
        "category": [_VSCAT_CATEGORY],
        "code": _codeable_concept("LOINC", ["85353-3"]),
        "subject": {"reference": f"urn:cliniq:patient-{patient_ref}"},
        "effectiveDateTime": "2026-01-01",
        "component": [
            {
                "code": _codeable_concept("LOINC", [code]),
                "dataAbsentReason": _data_absent_reason_unknown(),
            }
        ],
    }
    meta = _meta_with_source(source_url)
    if meta:
        resource["meta"] = meta
    return {"fullUrl": full_url, "resource": resource}


def _blood_pressure_observation_entry(
    code: str, *, patient_ref: str, source_url: str | None = None
) -> dict:
    """Represent systolic/diastolic LOINCs as a BP panel Observation.

    The FHIR R4 base BP profile requires `Observation.code=85354-9` and
    both systolic + diastolic component slices. ClinIQ's extraction surface
    only carries codes, not measured values, so each component uses
    `dataAbsentReason=unknown`. The triggering code remains represented as a
    component, and the missing paired component is explicit rather than
    invented.
    """
    rid, full_url = _entry_id_and_full_url("observation", code)
    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": rid,
        "status": "final",
        "category": [_VSCAT_CATEGORY],
        "code": _codeable_concept("LOINC", ["85354-9"]),
        "subject": {"reference": f"urn:cliniq:patient-{patient_ref}"},
        "effectiveDateTime": "2026-01-01",
        "component": [
            {
                "code": _codeable_concept("LOINC", ["8480-6"]),
                "dataAbsentReason": _data_absent_reason_unknown(),
            },
            {
                "code": _codeable_concept("LOINC", ["8462-4"]),
                "dataAbsentReason": _data_absent_reason_unknown(),
            },
        ],
    }
    meta = _meta_with_source(source_url)
    if meta:
        resource["meta"] = meta
    return {"fullUrl": full_url, "resource": resource}


def _medication_statement_entry(
    code: str, *, patient_ref: str, source_url: str | None = None
) -> dict:
    """Single RxNorm → MedicationStatement resource.

    R4 required: status, medication[x], subject. We use
    medicationCodeableConcept; status='recorded' is appropriate for an
    extracted entity (we know it was recorded in the eICR; we don't have
    administration / completion semantics).
    """
    rid, full_url = _entry_id_and_full_url("medication-statement", code)
    # R4 MedicationStatement.status value set:
    #   active | completed | entered-in-error | intended | stopped |
    #   on-hold | unknown | not-taken
    # ('recorded' is an R5 value — R4's HL7-validator rejects it.) Use
    # `unknown` because we extracted the medication code but have no
    # signal on whether the patient is currently taking it.
    resource: dict[str, Any] = {
        "resourceType": "MedicationStatement",
        "id": rid,
        "status": "unknown",
        "medicationCodeableConcept": {"coding": [_coding("RXNORM", code)]},
        "subject": {"reference": f"urn:cliniq:patient-{patient_ref}"},
    }
    meta = _meta_with_source(source_url)
    if meta:
        resource["meta"] = meta
    return {"fullUrl": full_url, "resource": resource}


def _patient_entry(patient_id: str = DEFAULT_PATIENT_ID) -> dict:
    """Minimal Patient stub — id only.

    No PHI in our Bundle; the eICR Patient block contains demographics but
    we deliberately don't carry them through. Judges care that resources
    resolve their `subject` references, not about specific demographics.
    """
    full_url = f"urn:cliniq:patient-{patient_id}"
    return {
        "fullUrl": full_url,
        "resource": {
            "resourceType": "Patient",
            "id": patient_id,
        },
    }


def to_bundle(
    extraction: dict,
    *,
    patient_id: str = DEFAULT_PATIENT_ID,
    provenance_map: dict[str, str] | None = None,
) -> dict:
    """Wrap a `{"conditions": [...], "loincs": [...], "rxnorms": [...]}`
    extraction in a minimal R4 Bundle dict.

    Bundle.type='collection' (per FHIR R4 Bundle.type binding for "set of
    resources collected together for a specific purpose"). Each entry has
    both `fullUrl` and `resource` so relative subject references resolve
    cleanly in the canonical HL7 validator.

    `provenance_map` is an optional `code -> source_url` dict; entries get
    `meta.source` stamped with the URL. Use the values produced by
    `EicrPreparser.extractWithProvenance` (Swift) or
    `regex_preparser.extract(...).to_provenance_dict()` (Python).
    """
    provenance_map = provenance_map or {}
    extraction = normalize_extraction(extraction)
    entries: list[dict] = [_patient_entry(patient_id)]

    for code in extraction.get("conditions") or []:
        entries.append(
            _condition_entry(
                code, patient_ref=patient_id,
                source_url=provenance_map.get(code),
            )
        )
    for code in extraction.get("loincs") or []:
        entries.append(
            _observation_entry(
                code, patient_ref=patient_id,
                source_url=provenance_map.get(code),
            )
        )
    for code in extraction.get("rxnorms") or []:
        entries.append(
            _medication_statement_entry(
                code, patient_ref=patient_id,
                source_url=provenance_map.get(code),
            )
        )

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }


def main() -> None:
    """CLI: read an extraction JSON from stdin, write a Bundle to stdout.

    Usage:
        echo '{"conditions": ["840539006"], "loincs": [], "rxnorms": []}' \\
            | python apps/mobile/convert/fhir_bundle.py
    """
    extraction = json.loads(sys.stdin.read())
    bundle = to_bundle(extraction)
    json.dump(bundle, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
