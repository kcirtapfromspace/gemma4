use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueStatus {
    Intake,
    Processed,
    Flagged,
    Graylist,
    OutOfState,
    QA,
    Closed,
}

impl QueueStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Intake => "Intake",
            Self::Processed => "Processed",
            Self::Flagged => "Flagged",
            Self::Graylist => "Graylist",
            Self::OutOfState => "OutOfState",
            Self::QA => "QA",
            Self::Closed => "Closed",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Intake" => Some(Self::Intake),
            "Processed" => Some(Self::Processed),
            "Flagged" => Some(Self::Flagged),
            "Graylist" => Some(Self::Graylist),
            "OutOfState" => Some(Self::OutOfState),
            "QA" => Some(Self::QA),
            "Closed" => Some(Self::Closed),
            _ => None,
        }
    }
}

impl std::fmt::Display for QueueStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Case {
    pub case_id: String,
    pub patient_name: String,
    pub patient_dob: String,
    pub patient_gender: String,
    pub patient_state: String,
    pub patient_city: String,
    pub condition_snomed: String,
    pub condition_display: String,
    pub lab_loinc: String,
    pub lab_result: String,
    pub jurisdiction: String,
    pub queue_status: String,
    pub dedup_hash: String,
    pub ingested_at: String,
    pub processed_at: Option<String>,
    pub inference_ms: i64,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub tokens_per_second: Option<f64>,
    pub raw_eicr_xml: String,
    pub extraction_json: String,
    pub fhir_bundle: String,
    pub entities_json: String,
}

/// Compute dedup hash: sha256(lower(family+given) || birthDate || snomed)
pub fn dedup_hash(family: &str, given: &str, dob: &str, snomed: &str) -> String {
    let input = format!(
        "{}{}{}{}",
        family.to_lowercase(),
        given.to_lowercase(),
        dob,
        snomed
    );
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}
