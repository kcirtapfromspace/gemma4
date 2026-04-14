//! Format detection and dispatch for multi-format clinical data ingest.
//! Supports eICR CDA/XML, HL7 v2.5.1 pipe-delimited, and FHIR R4 JSON.
//!
//! All formats are converted to the same prompt text the model was trained on,
//! matching the output of `EicrSummary::to_prompt_text()`.

use serde::Serialize;

/// Supported input formats for clinical data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InputFormat {
    EicrXml,
    Hl7v2,
    FhirJson,
}

impl InputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EicrXml => "eicr",
            Self::Hl7v2 => "hl7",
            Self::FhirJson => "fhir",
        }
    }
}

impl std::fmt::Display for InputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Auto-detect input format from content.
pub fn detect_format(input: &str) -> InputFormat {
    let trimmed = input.trim();
    if trimmed.starts_with("MSH|") {
        InputFormat::Hl7v2
    } else if trimmed.starts_with('{') && trimmed.contains("\"resourceType\"") {
        InputFormat::FhirJson
    } else {
        InputFormat::EicrXml
    }
}

/// Parse a format hint string into an InputFormat.
pub fn parse_format_hint(hint: &str) -> InputFormat {
    match hint.to_lowercase().as_str() {
        "hl7" | "hl7v2" | "hl7_v2" => InputFormat::Hl7v2,
        "fhir" | "fhir_json" | "fhir-json" => InputFormat::FhirJson,
        _ => InputFormat::EicrXml,
    }
}

/// Convert any supported format into the prompt text the model expects.
/// Returns the prompt text or an error message.
pub fn format_for_prompt(input: &str, format: InputFormat) -> Result<String, String> {
    match format {
        InputFormat::EicrXml => {
            let summary = crate::eicr::extract(input);
            let text = summary.to_prompt_text();
            if text.is_empty() {
                Err("Could not extract any data from eICR XML".into())
            } else {
                Ok(text)
            }
        }
        InputFormat::Hl7v2 => crate::hl7::format_prompt(input),
        InputFormat::FhirJson => crate::fhir::format_prompt_from_fhir_input(input),
    }
}
