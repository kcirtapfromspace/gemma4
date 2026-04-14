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
/// Handles markdown fences and trailing prose from LLM responses.
pub fn parse(raw: &str) -> Result<Extraction> {
    let cleaned = strip_fences(raw);
    serde_json::from_str(&cleaned).context("failed to parse model extraction JSON")
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
