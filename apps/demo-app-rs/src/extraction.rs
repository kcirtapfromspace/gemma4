//! Typed structs for the fine-tuned Gemma 4 model's extraction output.
//!
//! The model is trained to output JSON matching this schema exactly.
//! This is the single source of truth for all downstream processing:
//! FHIR bundle building, entity extraction, dedup hashing, and jurisdiction routing.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::fhir::strip_fences;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Extraction {
    #[serde(default)]
    pub patient: ExtPatient,
    #[serde(default)]
    pub encounter: ExtEncounter,
    #[serde(default)]
    pub conditions: Vec<ExtCondition>,
    #[serde(default)]
    pub labs: Vec<ExtLab>,
    #[serde(default)]
    pub vitals: Option<ExtVitals>,
    #[serde(default)]
    pub meds: Vec<ExtMed>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub reportable: bool,
    #[serde(default)]
    pub jurisdiction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtPatient {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub dob: String,
    #[serde(default)]
    pub sex: String,
    #[serde(default)]
    pub race: Option<String>,
    #[serde(default)]
    pub ethnicity: Option<String>,
    #[serde(default)]
    pub addr: String,
    #[serde(default)]
    pub phone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtEncounter {
    #[serde(default)]
    pub date: String,
    #[serde(rename = "type", default)]
    pub enc_type: String,
    #[serde(default)]
    pub facility: String,
    #[serde(default)]
    pub npi: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtCondition {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub snomed: String,
    #[serde(default)]
    pub icd10: String,
    #[serde(default)]
    pub onset: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub conf: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtLab {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub loinc: String,
    #[serde(default)]
    pub result: String,
    #[serde(default)]
    pub result_snomed: String,
    #[serde(default)]
    pub specimen: String,
    #[serde(default)]
    pub lab_status: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub conf: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtVitals {
    #[serde(default)]
    pub temp: Option<f64>,
    #[serde(default)]
    pub hr: Option<f64>,
    #[serde(default)]
    pub rr: Option<f64>,
    #[serde(default)]
    pub spo2: Option<f64>,
    #[serde(default)]
    pub sbp: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtMed {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub rxnorm: String,
    #[serde(default)]
    pub conf: f64,
}

/// Parse the model's raw output into a typed Extraction.
/// Handles markdown fences, trailing prose, and alternative JSON schemas
/// (the base model uses different field names than the fine-tuned training format).
pub fn parse(raw: &str) -> Result<Extraction> {
    let cleaned = strip_fences(raw);

    // Try strict parse first (fine-tuned model format)
    if let Ok(ext) = serde_json::from_str::<Extraction>(&cleaned) {
        // Only accept if the model actually populated key fields
        let has_patient = !ext.patient.name.is_empty();
        let has_conditions = ext.conditions.iter().any(|c| !c.name.is_empty() || !c.snomed.is_empty());
        if has_patient || has_conditions {
            return Ok(ext);
        }
    }

    // Flexible parse: handle base model's alternative field names
    let v: serde_json::Value =
        serde_json::from_str(&cleaned).context("failed to parse model output as JSON")?;

    let mut ext = Extraction::default();

    // Patient: try "patient" then "patient_demographics"
    let pd = v.get("patient").or(v.get("patient_demographics"));
    if let Some(p) = pd {
        ext.patient.name = p
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        ext.patient.dob = p
            .get("dob")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        ext.patient.sex = p
            .get("sex")
            .or(p.get("gender"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        ext.patient.race = p.get("race").and_then(|v| v.as_str()).map(|s| s.to_string());
        ext.patient.ethnicity = p
            .get("ethnicity")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        ext.patient.addr = p
            .get("addr")
            .or(p.get("location"))
            .or(p.get("address"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        ext.patient.phone = p
            .get("phone")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
    }

    // Encounter
    let enc = v.get("encounter");
    if let Some(e) = enc {
        ext.encounter.date = e
            .get("date")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        ext.encounter.facility = e
            .get("facility")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        ext.encounter.npi = e
            .get("npi")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
    }
    // If no encounter block, try to pull facility from patient_demographics
    if ext.encounter.facility.is_empty() {
        if let Some(fac) = pd
            .and_then(|p| p.get("facility"))
            .and_then(|v| v.as_str())
        {
            ext.encounter.facility = fac.to_string();
        }
    }

    // Conditions: handle [{name, snomed, ...}] and [{condition, code, ...}]
    if let Some(conds) = v.get("conditions").and_then(|v| v.as_array()) {
        for c in conds {
            let name = c
                .get("name")
                .or(c.get("condition"))
                .or(c.get("description"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let raw_code = c
                .get("snomed")
                .or(c.get("code"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            // Strip "SNOMED " prefix if present
            let snomed = raw_code
                .strip_prefix("SNOMED ")
                .unwrap_or(raw_code)
                .trim()
                .to_string();
            let icd10 = c
                .get("icd10")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let conf = c
                .get("conf")
                .or(c.get("confidence"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.9);
            ext.conditions.push(ExtCondition {
                name,
                snomed,
                icd10,
                onset: c
                    .get("onset")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                status: c
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("active")
                    .to_string(),
                conf,
            });
        }
    }

    // Labs: handle [{name, loinc, ...}] and [{test, code, ...}]
    if let Some(labs) = v.get("labs").and_then(|v| v.as_array()) {
        for l in labs {
            let name = l
                .get("name")
                .or(l.get("test"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let loinc = l
                .get("loinc")
                .or(l.get("code"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let conf = l
                .get("conf")
                .or(l.get("confidence"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.9);
            ext.labs.push(ExtLab {
                name,
                loinc,
                result: l
                    .get("result")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                result_snomed: l
                    .get("result_snomed")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                specimen: l
                    .get("specimen")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                lab_status: l
                    .get("lab_status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("final")
                    .to_string(),
                date: l
                    .get("date")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                conf,
            });
        }
    }

    // Meds: handle [{name, rxnorm, ...}] and [{medication, code, ...}]
    if let Some(meds) = v
        .get("meds")
        .or(v.get("medications"))
        .and_then(|v| v.as_array())
    {
        for m in meds {
            let name = m
                .get("name")
                .or(m.get("medication"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let rxnorm = m
                .get("rxnorm")
                .or(m.get("code"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let conf = m
                .get("conf")
                .or(m.get("confidence"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.9);
            ext.meds.push(ExtMed { name, rxnorm, conf });
        }
    }

    // Vitals: handle both structured and string values
    if let Some(vit) = v.get("vitals") {
        let parse_vital = |key: &str| -> Option<f64> {
            vit.get(key).and_then(|v| {
                v.as_f64().or_else(|| {
                    v.as_str().and_then(|s| {
                        s.trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
                            .parse::<f64>()
                            .ok()
                    })
                })
            })
        };
        ext.vitals = Some(ExtVitals {
            temp: parse_vital("temp").or(parse_vital("temperature")),
            hr: parse_vital("hr").or(parse_vital("heart_rate")),
            rr: parse_vital("rr").or(parse_vital("respiratory_rate")),
            spo2: parse_vital("spo2").or(parse_vital("oxygen_saturation")),
            sbp: parse_vital("sbp").or(parse_vital("blood_pressure")),
        });
    }

    // Summary
    ext.summary = v
        .get("summary")
        .or(v.get("case_summary"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    ext.reportable = v
        .get("reportable")
        .and_then(|v| v.as_bool())
        .unwrap_or(!ext.conditions.is_empty());
    ext.jurisdiction = v
        .get("jurisdiction")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(ext)
}

/// Parse a stored extraction JSON string (already clean, no fences).
pub fn parse_stored(json_str: &str) -> Result<Extraction> {
    serde_json::from_str(json_str).context("failed to parse stored extraction JSON")
}

impl ExtPatient {
    /// Split name into (given, family) for dedup hashing and FHIR Patient.
    pub fn name_parts(&self) -> (&str, &str) {
        match self.name.split_once(' ') {
            Some((given, family)) => (given, family),
            None => (self.name.as_str(), ""),
        }
    }

    /// Parse "City, ST ZIP" address into (city, state, zip).
    pub fn addr_parts(&self) -> (&str, &str, &str) {
        let parts: Vec<&str> = self.addr.splitn(2, ", ").collect();
        let city = parts.first().copied().unwrap_or("");
        let state_zip: Vec<&str> = parts
            .get(1)
            .unwrap_or(&"")
            .splitn(2, ' ')
            .collect();
        let state = state_zip.first().copied().unwrap_or("");
        let zip = if state_zip.len() > 1 { state_zip[1] } else { "" };
        (city, state, zip)
    }
}
