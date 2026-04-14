//! Compute field-level diffs between successive Extraction outputs.
//! Implements eCRims-005: determine what information was updated between eCRs.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::extraction::Extraction;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiffResult {
    pub has_changes: bool,
    pub conditions_added: Vec<CodeDiff>,
    pub conditions_removed: Vec<CodeDiff>,
    pub labs_added: Vec<CodeDiff>,
    pub labs_removed: Vec<CodeDiff>,
    pub labs_changed: Vec<ValueChange>,
    pub meds_added: Vec<CodeDiff>,
    pub meds_removed: Vec<CodeDiff>,
    pub vitals_changed: Vec<VitalChange>,
    pub patient_updates: Vec<FieldChange>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDiff {
    pub name: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueChange {
    pub name: String,
    pub code: String,
    pub old_value: String,
    pub new_value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalChange {
    pub name: String,
    pub old_value: Option<f64>,
    pub new_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldChange {
    pub field: String,
    pub old_value: String,
    pub new_value: String,
}

/// Compute diff between two Extractions (previous → current).
pub fn compute_extraction_diff(old: &Extraction, new: &Extraction) -> DiffResult {
    let mut result = DiffResult::default();

    // Conditions: compare by SNOMED code
    let old_snomeds: HashSet<&str> = old.conditions.iter().map(|c| c.snomed.as_str()).collect();
    let new_snomeds: HashSet<&str> = new.conditions.iter().map(|c| c.snomed.as_str()).collect();

    for c in &new.conditions {
        if !old_snomeds.contains(c.snomed.as_str()) {
            result.conditions_added.push(CodeDiff {
                name: c.name.clone(),
                code: c.snomed.clone(),
            });
        }
    }
    for c in &old.conditions {
        if !new_snomeds.contains(c.snomed.as_str()) {
            result.conditions_removed.push(CodeDiff {
                name: c.name.clone(),
                code: c.snomed.clone(),
            });
        }
    }

    // Labs: compare by LOINC code, detect result changes
    let old_labs: std::collections::HashMap<&str, &str> = old
        .labs
        .iter()
        .map(|l| (l.loinc.as_str(), l.result.as_str()))
        .collect();
    let new_labs: std::collections::HashMap<&str, &str> = new
        .labs
        .iter()
        .map(|l| (l.loinc.as_str(), l.result.as_str()))
        .collect();

    for l in &new.labs {
        match old_labs.get(l.loinc.as_str()) {
            None => {
                result.labs_added.push(CodeDiff {
                    name: l.name.clone(),
                    code: l.loinc.clone(),
                });
            }
            Some(&old_result) if old_result != l.result => {
                result.labs_changed.push(ValueChange {
                    name: l.name.clone(),
                    code: l.loinc.clone(),
                    old_value: old_result.to_string(),
                    new_value: l.result.clone(),
                });
            }
            _ => {}
        }
    }
    for l in &old.labs {
        if !new_labs.contains_key(l.loinc.as_str()) {
            result.labs_removed.push(CodeDiff {
                name: l.name.clone(),
                code: l.loinc.clone(),
            });
        }
    }

    // Meds: compare by RxNorm code
    let old_rxnorms: HashSet<&str> = old.meds.iter().map(|m| m.rxnorm.as_str()).collect();
    let new_rxnorms: HashSet<&str> = new.meds.iter().map(|m| m.rxnorm.as_str()).collect();

    for m in &new.meds {
        if !old_rxnorms.contains(m.rxnorm.as_str()) {
            result.meds_added.push(CodeDiff {
                name: m.name.clone(),
                code: m.rxnorm.clone(),
            });
        }
    }
    for m in &old.meds {
        if !new_rxnorms.contains(m.rxnorm.as_str()) {
            result.meds_removed.push(CodeDiff {
                name: m.name.clone(),
                code: m.rxnorm.clone(),
            });
        }
    }

    // Vitals: compare individual values
    fn vital_changed(name: &str, old_val: Option<f64>, new_val: Option<f64>) -> Option<VitalChange> {
        match (old_val, new_val) {
            (Some(o), Some(n)) if (o - n).abs() > 0.01 => Some(VitalChange {
                name: name.into(),
                old_value: Some(o),
                new_value: Some(n),
            }),
            (None, Some(n)) => Some(VitalChange {
                name: name.into(),
                old_value: None,
                new_value: Some(n),
            }),
            (Some(o), None) => Some(VitalChange {
                name: name.into(),
                old_value: Some(o),
                new_value: None,
            }),
            _ => None,
        }
    }

    let (ov, nv) = (
        old.vitals.as_ref().cloned().unwrap_or_default(),
        new.vitals.as_ref().cloned().unwrap_or_default(),
    );
    for vc in [
        vital_changed("Temperature", ov.temp, nv.temp),
        vital_changed("Heart Rate", ov.hr, nv.hr),
        vital_changed("Resp Rate", ov.rr, nv.rr),
        vital_changed("SpO2", ov.spo2, nv.spo2),
        vital_changed("Systolic BP", ov.sbp, nv.sbp),
    ] {
        if let Some(v) = vc {
            result.vitals_changed.push(v);
        }
    }

    // Patient demographic changes
    fn field_changed(field: &str, old: &str, new: &str) -> Option<FieldChange> {
        if old != new && !new.is_empty() {
            Some(FieldChange {
                field: field.into(),
                old_value: old.into(),
                new_value: new.into(),
            })
        } else {
            None
        }
    }

    for fc in [
        field_changed("Address", &old.patient.addr, &new.patient.addr),
        field_changed(
            "Phone",
            old.patient.phone.as_deref().unwrap_or(""),
            new.patient.phone.as_deref().unwrap_or(""),
        ),
        field_changed("Facility", &old.encounter.facility, &new.encounter.facility),
    ] {
        if let Some(f) = fc {
            result.patient_updates.push(f);
        }
    }

    // Build summary
    result.has_changes = !result.conditions_added.is_empty()
        || !result.conditions_removed.is_empty()
        || !result.labs_added.is_empty()
        || !result.labs_removed.is_empty()
        || !result.labs_changed.is_empty()
        || !result.meds_added.is_empty()
        || !result.meds_removed.is_empty()
        || !result.vitals_changed.is_empty()
        || !result.patient_updates.is_empty();

    let mut summary_parts = Vec::new();
    if !result.conditions_added.is_empty() {
        summary_parts.push(format!("+{} condition(s)", result.conditions_added.len()));
    }
    if !result.conditions_removed.is_empty() {
        summary_parts.push(format!("-{} condition(s)", result.conditions_removed.len()));
    }
    if !result.labs_added.is_empty() {
        summary_parts.push(format!("+{} lab(s)", result.labs_added.len()));
    }
    if !result.labs_changed.is_empty() {
        summary_parts.push(format!("{} lab result(s) changed", result.labs_changed.len()));
    }
    if !result.meds_added.is_empty() {
        summary_parts.push(format!("+{} med(s)", result.meds_added.len()));
    }
    if !result.meds_removed.is_empty() {
        summary_parts.push(format!("-{} med(s)", result.meds_removed.len()));
    }
    if !result.vitals_changed.is_empty() {
        summary_parts.push(format!("{} vital(s) changed", result.vitals_changed.len()));
    }
    if !result.patient_updates.is_empty() {
        summary_parts.push(format!("{} demographic update(s)", result.patient_updates.len()));
    }

    result.summary = if summary_parts.is_empty() {
        "No changes".into()
    } else {
        summary_parts.join("; ")
    };

    result
}
