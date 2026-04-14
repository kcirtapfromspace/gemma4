use serde::Serialize;

use crate::extraction::Extraction;
use crate::fhir::Bundle;

#[derive(Debug, Clone, Serialize)]
pub struct Entity {
    pub kind: String,
    pub system: String,
    pub code: String,
    pub display: String,
    pub conf: Option<f64>,
}

/// Build entities from the model's Extraction output, preserving confidence scores.
pub fn from_extraction(ext: &Extraction) -> Vec<Entity> {
    let mut entities = Vec::new();

    for c in &ext.conditions {
        entities.push(Entity {
            kind: "Condition".into(),
            system: "http://snomed.info/sct".into(),
            code: c.snomed.clone(),
            display: c.name.clone(),
            conf: Some(c.conf),
        });
        if !c.icd10.is_empty() {
            entities.push(Entity {
                kind: "Condition".into(),
                system: "http://hl7.org/fhir/sid/icd-10-cm".into(),
                code: c.icd10.clone(),
                display: c.name.clone(),
                conf: Some(c.conf),
            });
        }
    }

    for lab in &ext.labs {
        entities.push(Entity {
            kind: "Observation (Lab)".into(),
            system: "http://loinc.org".into(),
            code: lab.loinc.clone(),
            display: format!("{} - {}", lab.name, lab.result),
            conf: Some(lab.conf),
        });
    }

    if let Some(v) = &ext.vitals {
        let vital_defs: Vec<(&str, &str, Option<f64>, &str)> = vec![
            ("8310-5", "Body temperature", v.temp, "C"),
            ("8867-4", "Heart rate", v.hr, "/min"),
            ("9279-1", "Respiratory rate", v.rr, "/min"),
            ("2708-6", "Oxygen saturation", v.spo2, "%"),
            ("8480-6", "Systolic blood pressure", v.sbp, "mmHg"),
        ];
        for (loinc, name, val, unit) in vital_defs {
            if let Some(value) = val {
                entities.push(Entity {
                    kind: "Observation (Vital)".into(),
                    system: "http://loinc.org".into(),
                    code: loinc.into(),
                    display: format!("{name} {value}{unit}"),
                    conf: Some(0.99),
                });
            }
        }
    }

    for med in &ext.meds {
        entities.push(Entity {
            kind: "MedicationStatement".into(),
            system: "http://www.nlm.nih.gov/research/umls/rxnorm".into(),
            code: med.rxnorm.clone(),
            display: med.name.clone(),
            conf: Some(med.conf),
        });
    }

    entities
}

/// Walk FHIR Bundle entries and extract coded entities (fallback, no confidence scores).
#[allow(dead_code)]
pub fn flatten(bundle: &Bundle) -> Vec<Entity> {
    let mut entities = Vec::new();

    for entry in &bundle.entry {
        let res = &entry.resource;
        let resource_type = res["resourceType"].as_str().unwrap_or("");

        match resource_type {
            "Condition" => {
                if let Some(codings) = res["code"]["coding"].as_array() {
                    for coding in codings {
                        entities.push(Entity {
                            kind: "Condition".into(),
                            system: coding["system"].as_str().unwrap_or("").into(),
                            code: coding["code"].as_str().unwrap_or("").into(),
                            display: coding["display"].as_str().unwrap_or("").into(),
                            conf: None,
                        });
                    }
                }
            }
            "Observation" => {
                let category = res["category"]
                    .as_array()
                    .and_then(|cats| cats.first())
                    .and_then(|c| c["coding"].as_array())
                    .and_then(|cs| cs.first())
                    .and_then(|c| c["code"].as_str())
                    .unwrap_or("observation");

                let kind = match category {
                    "laboratory" => "Observation (Lab)",
                    "vital-signs" => "Observation (Vital)",
                    _ => "Observation",
                };

                if let Some(codings) = res["code"]["coding"].as_array() {
                    for coding in codings {
                        entities.push(Entity {
                            kind: kind.into(),
                            system: coding["system"].as_str().unwrap_or("").into(),
                            code: coding["code"].as_str().unwrap_or("").into(),
                            display: coding["display"].as_str().unwrap_or("").into(),
                            conf: None,
                        });
                    }
                }
            }
            "MedicationStatement" => {
                if let Some(codings) =
                    res["medicationCodeableConcept"]["coding"].as_array()
                {
                    for coding in codings {
                        entities.push(Entity {
                            kind: "MedicationStatement".into(),
                            system: coding["system"].as_str().unwrap_or("").into(),
                            code: coding["code"].as_str().unwrap_or("").into(),
                            display: coding["display"].as_str().unwrap_or("").into(),
                            conf: None,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    entities
}
