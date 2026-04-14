use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::LazyLock;

use crate::extraction::Extraction;

static RE_FENCE_OPEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^```(?:json)?\s*").unwrap());
static RE_FENCE_CLOSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)```\s*$").unwrap());

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Bundle {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(rename = "type", default)]
    pub bundle_type: String,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub entry: Vec<BundleEntry>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BundleEntry {
    #[serde(rename = "fullUrl", default)]
    pub full_url: String,
    #[serde(default)]
    pub resource: serde_json::Value,
}

/// Strip markdown fences and trailing prose from LLM output, then parse as FHIR Bundle.
#[allow(dead_code)]
pub fn parse_bundle(raw: &str) -> Result<Bundle> {
    let stripped = strip_fences(raw);
    serde_json::from_str(&stripped).context("failed to parse FHIR Bundle JSON")
}

/// Strip markdown code fences and truncate after last closing brace/bracket.
/// Used by both FHIR and extraction parsing.
pub fn strip_fences(content: &str) -> String {
    let s = RE_FENCE_OPEN.replace_all(content, "");
    let s = RE_FENCE_CLOSE.replace_all(&s, "");
    let s = s.trim();

    let last_brace = s.rfind('}').unwrap_or(0);
    let last_bracket = s.rfind(']').unwrap_or(0);
    let last = last_brace.max(last_bracket);
    if last > 0 {
        s[..=last].to_string()
    } else {
        s.to_string()
    }
}

fn uid() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn gender_display(code: &str) -> &str {
    match code {
        "F" => "female",
        "M" => "male",
        _ => "unknown",
    }
}

/// Build a FHIR R4 Bundle from the model's Extraction output.
/// Every code, name, and field comes from what the fine-tuned Gemma 4 extracted.
pub fn build_bundle_from_extraction(ext: &Extraction) -> serde_json::Value {
    let patient_id = uid();
    let encounter_id = uid();
    let org_id = uid();

    let p = &ext.patient;
    let enc = &ext.encounter;
    let (given, family) = p.name_parts();
    let (city, state, zip) = p.addr_parts();

    // Patient resource
    let mut patient = json!({
        "resourceType": "Patient",
        "name": [{"use": "official", "family": family, "given": [given]}],
        "gender": gender_display(&p.sex),
        "birthDate": p.dob,
        "address": [{"city": city, "state": state, "postalCode": zip, "country": "US"}],
    });
    if let Some(phone) = &p.phone {
        if !phone.is_empty() {
            patient["telecom"] = json!([{"system": "phone", "value": phone}]);
        }
    }
    if let Some(race) = &p.race {
        if !race.is_empty() {
            patient["extension"] = json!([{
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                "extension": [{"url": "text", "valueString": race}]
            }]);
        }
    }

    // Encounter
    let encounter = json!({
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
        "type": [{"coding": [{"system": "http://www.ama-assn.org/go/cpt", "code": "99213", "display": if enc.enc_type.is_empty() { "Office visit" } else { &enc.enc_type }}]}],
        "subject": {"reference": format!("urn:uuid:{patient_id}")},
        "period": {"start": enc.date},
    });

    // Organization
    let org = json!({
        "resourceType": "Organization",
        "identifier": if enc.npi.is_empty() { json!([]) } else {
            json!([{"system": "http://hl7.org/fhir/sid/us-npi", "value": enc.npi}])
        },
        "name": enc.facility,
    });

    let mut entries = vec![
        json!({"fullUrl": format!("urn:uuid:{patient_id}"), "resource": patient}),
        json!({"fullUrl": format!("urn:uuid:{encounter_id}"), "resource": encounter}),
        json!({"fullUrl": format!("urn:uuid:{org_id}"), "resource": org}),
    ];

    let mut problem_refs = Vec::new();
    let mut result_refs = Vec::new();
    let mut med_refs = Vec::new();

    // Conditions — from model's extraction with SNOMED + ICD-10
    for c in &ext.conditions {
        let cid = uid();
        let mut codings = vec![json!({"system": "http://snomed.info/sct", "code": c.snomed, "display": c.name})];
        if !c.icd10.is_empty() {
            codings.push(json!({"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": c.icd10, "display": c.name}));
        }
        let res = json!({
            "resourceType": "Condition",
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": c.status}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis"}]}],
            "code": {"coding": codings},
            "subject": {"reference": format!("urn:uuid:{patient_id}")},
            "encounter": {"reference": format!("urn:uuid:{encounter_id}")},
            "onsetDateTime": c.onset,
        });
        problem_refs.push(json!({"reference": format!("urn:uuid:{cid}")}));
        entries.push(json!({"fullUrl": format!("urn:uuid:{cid}"), "resource": res}));
    }

    // Lab observations — from model's extraction with LOINC + specimen
    for lab in &ext.labs {
        let lid = uid();
        let mut res = json!({
            "resourceType": "Observation",
            "status": if lab.lab_status.is_empty() { "final" } else { &lab.lab_status },
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": lab.loinc, "display": lab.name}]},
            "subject": {"reference": format!("urn:uuid:{patient_id}")},
            "effectiveDateTime": if lab.date.is_empty() { &enc.date } else { &lab.date },
        });
        if !lab.result.is_empty() {
            res["valueCodeableConcept"] = json!({"coding": [{
                "system": "http://snomed.info/sct",
                "code": lab.result_snomed,
                "display": lab.result
            }]});
        }
        if !lab.specimen.is_empty() {
            res["specimen"] = json!({"display": lab.specimen});
        }
        result_refs.push(json!({"reference": format!("urn:uuid:{lid}")}));
        entries.push(json!({"fullUrl": format!("urn:uuid:{lid}"), "resource": res}));
    }

    // Vital signs — from model's extraction
    if let Some(v) = &ext.vitals {
        let vital_defs: Vec<(&str, &str, Option<f64>, &str)> = vec![
            ("8310-5", "Body temperature", v.temp, "Cel"),
            ("8867-4", "Heart rate", v.hr, "/min"),
            ("9279-1", "Respiratory rate", v.rr, "/min"),
            ("2708-6", "Oxygen saturation", v.spo2, "%"),
            ("8480-6", "Systolic blood pressure", v.sbp, "mm[Hg]"),
        ];
        for (loinc, name, val, unit) in vital_defs {
            if let Some(value) = val {
                let vid = uid();
                let res = json!({
                    "resourceType": "Observation",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": name}]},
                    "subject": {"reference": format!("urn:uuid:{patient_id}")},
                    "effectiveDateTime": enc.date,
                    "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
                });
                result_refs.push(json!({"reference": format!("urn:uuid:{vid}")}));
                entries.push(json!({"fullUrl": format!("urn:uuid:{vid}"), "resource": res}));
            }
        }
    }

    // Medications — from model's extraction with RxNorm
    for med in &ext.meds {
        let mid = uid();
        let res = json!({
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": med.rxnorm, "display": med.name}]},
            "subject": {"reference": format!("urn:uuid:{patient_id}")},
            "effectiveDateTime": enc.date,
        });
        med_refs.push(json!({"reference": format!("urn:uuid:{mid}")}));
        entries.push(json!({"fullUrl": format!("urn:uuid:{mid}"), "resource": res}));
    }

    // Composition (eICR document header)
    let comp_id = uid();
    let mut sections = vec![
        json!({"title": "Problems", "code": {"coding": [{"system": "http://loinc.org", "code": "11450-4"}]}, "entry": problem_refs}),
        json!({"title": "Results", "code": {"coding": [{"system": "http://loinc.org", "code": "30954-2"}]}, "entry": result_refs}),
    ];
    if !med_refs.is_empty() {
        sections.push(json!({"title": "Medications", "code": {"coding": [{"system": "http://loinc.org", "code": "10160-0"}]}, "entry": med_refs}));
    }

    let comp = json!({
        "resourceType": "Composition",
        "status": "final",
        "type": {"coding": [{"system": "http://loinc.org", "code": "55751-2", "display": "Public Health Case Report"}]},
        "subject": {"reference": format!("urn:uuid:{patient_id}")},
        "encounter": {"reference": format!("urn:uuid:{encounter_id}")},
        "date": enc.date,
        "author": [{"reference": format!("urn:uuid:{org_id}")}],
        "title": "Initial Public Health Case Report - eICR",
        "section": sections,
    });
    entries.insert(0, json!({"fullUrl": format!("urn:uuid:{comp_id}"), "resource": comp}));

    json!({
        "resourceType": "Bundle",
        "type": "document",
        "timestamp": enc.date,
        "entry": entries,
    })
}
