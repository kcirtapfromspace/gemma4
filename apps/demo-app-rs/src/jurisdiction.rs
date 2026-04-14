//! Dynamic jurisdiction rules engine (eCRci-035).
//!
//! Rules are stored in DuckDB, loaded into memory at startup, and evaluated
//! per-case during ingest. Each rule matches on state code, condition SNOMED,
//! and minimum confidence threshold.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JurisdictionRule {
    pub rule_id: String,
    pub jurisdiction_name: String,
    pub state_codes: Vec<String>,
    pub condition_snomeds: Vec<String>,
    pub min_confidence: f64,
    pub priority: i32,
    pub active: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteResult {
    pub jurisdiction_name: String,
    pub is_out_of_state: bool,
    pub matched_rule_id: Option<String>,
}

/// Evaluate rules against patient state, condition, and confidence.
/// Rules should be pre-sorted by priority (descending).
pub fn evaluate(
    rules: &[JurisdictionRule],
    patient_state: &str,
    condition_snomed: &str,
    condition_confidence: f64,
) -> RouteResult {
    let state_upper = patient_state.to_uppercase();

    for rule in rules.iter().filter(|r| r.active) {
        // Check state match
        if !rule
            .state_codes
            .iter()
            .any(|s| s.eq_ignore_ascii_case(&state_upper))
        {
            continue;
        }
        // Check condition filter (empty = accept all conditions)
        if !rule.condition_snomeds.is_empty()
            && !rule.condition_snomeds.contains(&condition_snomed.to_string())
        {
            continue;
        }
        // Check confidence threshold
        if condition_confidence < rule.min_confidence {
            continue;
        }
        return RouteResult {
            jurisdiction_name: rule.jurisdiction_name.clone(),
            is_out_of_state: false,
            matched_rule_id: Some(rule.rule_id.clone()),
        };
    }

    RouteResult {
        jurisdiction_name: "Unknown".into(),
        is_out_of_state: true,
        matched_rule_id: None,
    }
}

/// Static fallback route (original 15-state map).
#[allow(dead_code)]
pub fn route_static(state: &str) -> (&'static str, bool) {
    match state.to_uppercase().as_str() {
        "CO" => ("Colorado DPHE", false),
        "IL" => ("Illinois DPH", false),
        "TX" => ("Texas DSHS", false),
        "AZ" => ("Arizona DHS", false),
        "WA" => ("Washington DOH", false),
        "GA" => ("Georgia DPH", false),
        "MA" => ("Massachusetts DPH", false),
        "FL" => ("Florida DOH", false),
        "OR" => ("Oregon OHA", false),
        "MN" => ("Minnesota DOH", false),
        "UT" => ("Utah DOH", false),
        "CA" => ("California CDPH", false),
        "NY" => ("New York DOH", false),
        "PA" => ("Pennsylvania DOH", false),
        "OH" => ("Ohio ODH", false),
        _ => ("Unknown", true),
    }
}

/// Generate default rules from the static map for database seeding.
pub fn default_rules() -> Vec<JurisdictionRule> {
    let now = chrono::Utc::now().to_rfc3339();
    let states = [
        ("CO", "Colorado DPHE"),
        ("IL", "Illinois DPH"),
        ("TX", "Texas DSHS"),
        ("AZ", "Arizona DHS"),
        ("WA", "Washington DOH"),
        ("GA", "Georgia DPH"),
        ("MA", "Massachusetts DPH"),
        ("FL", "Florida DOH"),
        ("OR", "Oregon OHA"),
        ("MN", "Minnesota DOH"),
        ("UT", "Utah DOH"),
        ("CA", "California CDPH"),
        ("NY", "New York DOH"),
        ("PA", "Pennsylvania DOH"),
        ("OH", "Ohio ODH"),
    ];
    states
        .iter()
        .enumerate()
        .map(|(_, (code, name))| JurisdictionRule {
            rule_id: format!("default-{}", code.to_lowercase()),
            jurisdiction_name: name.to_string(),
            state_codes: vec![code.to_string()],
            condition_snomeds: vec![],
            min_confidence: 0.0,
            priority: 0,
            active: true,
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .collect()
}
