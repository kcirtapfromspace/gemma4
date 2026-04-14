use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Multipart, Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json};
use serde::{Deserialize, Serialize};

use crate::case::{self, Case, QueueStatus};
use crate::config::{Config, LlmBackend};
use crate::diff;
use crate::entities;
use crate::extraction;
use crate::fhir;
use crate::hl7_output;
use crate::ingest;
use crate::jurisdiction;
use crate::llm::{self, LlmClient};
use crate::store::{CaseFilter, Store};

pub struct AppState {
    pub store: Arc<Store>,
    pub llm: Box<dyn LlmClient>,
    pub config: Config,
    pub started_at: Instant,
    pub rules: tokio::sync::RwLock<Vec<crate::jurisdiction::JurisdictionRule>>,
}

const HTML: &str = include_str!("../static/index.html");
const SAMPLE_EICR: &str = include_str!("../../../data/eicr-samples/sample_eicr_01.xml");

pub async fn index() -> Html<&'static str> {
    Html(HTML)
}

pub async fn sample_eicr() -> &'static str {
    SAMPLE_EICR
}

#[derive(Deserialize)]
pub struct ConvertRequest {
    /// Raw clinical data in any supported format (eICR XML, HL7 v2.5.1, FHIR R4 JSON).
    #[serde(alias = "eicr")]
    pub data: String,
    /// Optional format hint: "eicr", "hl7", "fhir". Auto-detected if omitted.
    pub format: Option<String>,
}

#[derive(Serialize)]
pub struct ConvertResponse {
    pub case_id: String,
    pub queue_status: String,
    pub jurisdiction: String,
    pub inference_ms: i64,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub tokens_per_second: Option<f64>,
    pub extraction: serde_json::Value,
    pub fhir_bundle: serde_json::Value,
    pub entities: Vec<crate::entities::Entity>,
    pub patient_name: String,
    pub condition_display: String,
    pub input_format: String,
    pub hl7_output: Option<String>,
    pub patient_hash: String,
    pub is_update: bool,
    pub diff_summary: Option<String>,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

fn err_json(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: msg.into(),
        }),
    )
}

/// Ingest one eICR through the LLM-first pipeline:
///   1. Format eICR XML → text prompt
///   2. Fine-tuned Gemma 4 → extraction JSON (with confidence scores)
///   3. Extraction drives everything: FHIR bundle, entities, dedup, routing
async fn ingest_one(
    state: &AppState,
    input: &str,
    format_hint: Option<&str>,
) -> Result<ConvertResponse, (StatusCode, Json<ErrorResponse>)> {
    // 1. Detect format and convert to prompt text
    let input_format = match format_hint {
        Some(hint) => ingest::parse_format_hint(hint),
        None => ingest::detect_format(input),
    };
    let prompt_text = ingest::format_for_prompt(input, input_format)
        .map_err(|e| err_json(StatusCode::BAD_REQUEST, e))?;

    // 2. Call the fine-tuned Gemma 4 model
    let start = Instant::now();
    let llm_result = state
        .llm
        .convert(llm::system_prompt(), &prompt_text)
        .await
        .map_err(|e| err_json(StatusCode::BAD_GATEWAY, format!("LLM error: {e}")))?;
    let inference_ms = start.elapsed().as_millis() as i64;

    let prompt_tokens = llm_result.prompt_tokens;
    let completion_tokens = llm_result.completion_tokens;
    let tokens_per_second = llm_result.tokens_per_second;
    let llm_raw = llm_result.content;

    // 3. Parse the model's extraction JSON — this is the source of truth
    let ext = extraction::parse(&llm_raw)
        .map_err(|e| err_json(StatusCode::UNPROCESSABLE_ENTITY, format!("Model output parse error: {e}")))?;

    let extraction_json_str = serde_json::to_string(&ext).unwrap_or_default();
    let extraction_val = serde_json::to_value(&ext).unwrap_or_default();

    // 4. Build FHIR Bundle from the model's extraction (not from regex)
    let fhir_json = fhir::build_bundle_from_extraction(&ext);

    // 5. Build entities from the model's extraction (with confidence scores)
    let entity_list = entities::from_extraction(&ext);

    // 6. Dedup hash from model's patient + primary condition
    let (given, family) = ext.patient.name_parts();
    let primary_snomed = ext.conditions.first().map(|c| c.snomed.as_str()).unwrap_or("");
    let primary_display = ext.conditions.first().map(|c| c.name.as_str()).unwrap_or("");
    let primary_loinc = ext.labs.first().map(|l| l.loinc.as_str()).unwrap_or("");
    let primary_lab_result = ext.labs.first().map(|l| l.result.as_str()).unwrap_or("");

    let hash = case::dedup_hash(family, given, &ext.patient.dob, primary_snomed);
    let is_dup = state.store.is_graylist(&hash).unwrap_or(false);

    // 6b. Patient hash + eCR diff (eCRims-001/005)
    let patient_hash_val = case::patient_hash(family, given, &ext.patient.dob);
    let case_id = uuid::Uuid::new_v4().to_string();
    let (diff_json_str, prev_case_id_val) = match state.store.find_previous_case(&patient_hash_val, &case_id) {
        Ok(Some(prev_case)) => {
            if let Ok(prev_ext) = extraction::parse_stored(&prev_case.extraction_json) {
                let d = diff::compute_extraction_diff(&prev_ext, &ext);
                (
                    serde_json::to_string(&d).unwrap_or_default(),
                    prev_case.case_id.clone(),
                )
            } else {
                (String::new(), prev_case.case_id.clone())
            }
        }
        _ => (String::new(), String::new()),
    };
    let is_update = !prev_case_id_val.is_empty();
    let diff_summary = if is_update && !diff_json_str.is_empty() {
        serde_json::from_str::<diff::DiffResult>(&diff_json_str)
            .ok()
            .map(|d| d.summary)
    } else {
        None
    };

    // 7. Jurisdiction routing via dynamic rules engine
    let primary_conf = ext.conditions.first().map(|c| c.conf).unwrap_or(0.0);
    let rules = state.rules.read().await;
    let route_result = jurisdiction::evaluate(&rules, &ext.jurisdiction, primary_snomed, primary_conf);
    drop(rules);
    let jurisdiction = route_result.jurisdiction_name;
    let is_out_of_state = route_result.is_out_of_state;

    // 8. Determine queue status
    let base_status = if ext.conditions.is_empty() {
        QueueStatus::Flagged
    } else {
        QueueStatus::Processed
    };

    let final_status = if is_dup {
        QueueStatus::Graylist
    } else if is_out_of_state {
        QueueStatus::OutOfState
    } else {
        base_status
    };

    let now = chrono::Utc::now().to_rfc3339();
    let (city, state_code, _) = ext.patient.addr_parts();

    let new_case = Case {
        case_id: case_id.clone(),
        patient_name: ext.patient.name.clone(),
        patient_dob: ext.patient.dob.clone(),
        patient_gender: ext.patient.sex.clone(),
        patient_state: state_code.to_string(),
        patient_city: city.to_string(),
        condition_snomed: primary_snomed.to_string(),
        condition_display: primary_display.to_string(),
        lab_loinc: primary_loinc.to_string(),
        lab_result: primary_lab_result.to_string(),
        jurisdiction: jurisdiction.to_string(),
        queue_status: final_status.to_string(),
        dedup_hash: hash,
        ingested_at: now.clone(),
        processed_at: if final_status == QueueStatus::Processed {
            Some(now)
        } else {
            None
        },
        inference_ms,
        prompt_tokens,
        completion_tokens,
        tokens_per_second,
        raw_eicr_xml: input.to_string(),
        extraction_json: extraction_json_str,
        fhir_bundle: serde_json::to_string(&fhir_json).unwrap_or_default(),
        entities_json: serde_json::to_string(&entity_list).unwrap_or_default(),
        input_format: input_format.as_str().to_string(),
        patient_hash: patient_hash_val.clone(),
        diff_json: diff_json_str,
        prev_case_id: prev_case_id_val,
    };

    state
        .store
        .insert_case(&new_case)
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;

    Ok(ConvertResponse {
        case_id,
        queue_status: final_status.to_string(),
        jurisdiction: jurisdiction.to_string(),
        inference_ms,
        prompt_tokens,
        completion_tokens,
        tokens_per_second,
        extraction: extraction_val,
        fhir_bundle: fhir_json,
        entities: entity_list,
        patient_name: ext.patient.name.clone(),
        condition_display: primary_display.to_string(),
        input_format: input_format.as_str().to_string(),
        hl7_output: Some(hl7_output::build_hl7_from_extraction(&ext)),
        patient_hash: patient_hash_val,
        is_update,
        diff_summary,
    })
}

pub async fn convert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ConvertRequest>,
) -> Result<Json<ConvertResponse>, (StatusCode, Json<ErrorResponse>)> {
    let resp = ingest_one(&state, &req.data, req.format.as_deref()).await?;
    Ok(Json(resp))
}

#[derive(Serialize)]
pub struct BatchResponse {
    pub results: Vec<BatchItem>,
}

#[derive(Serialize)]
pub struct BatchItem {
    pub filename: String,
    #[serde(flatten)]
    pub result: BatchItemResult,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum BatchItemResult {
    Ok(ConvertResponse),
    Err { error: String },
}

pub async fn convert_batch(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<BatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut results = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| err_json(StatusCode::BAD_REQUEST, format!("multipart error: {e}")))?
    {
        let filename = field.file_name().unwrap_or("unknown").to_string();
        let data = field
            .text()
            .await
            .map_err(|e| err_json(StatusCode::BAD_REQUEST, format!("read error: {e}")))?;

        let item = match ingest_one(&state, &data, None).await {
            Ok(resp) => BatchItem {
                filename,
                result: BatchItemResult::Ok(resp),
            },
            Err((_, json_err)) => BatchItem {
                filename,
                result: BatchItemResult::Err {
                    error: json_err.0.error,
                },
            },
        };
        results.push(item);
    }

    Ok(Json(BatchResponse { results }))
}

#[derive(Deserialize)]
pub struct CasesQuery {
    pub status: Option<String>,
    pub limit: Option<i64>,
}

pub async fn list_cases(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CasesQuery>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let filter = CaseFilter {
        status: query.status,
        limit: query.limit,
    };
    let cases = state
        .store
        .query_cases(&filter)
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;
    Ok(Json(cases))
}

pub async fn get_case(
    State(state): State<Arc<AppState>>,
    Path(case_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let c = state
        .store
        .get_case(&case_id)
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;
    match c {
        Some(case) => Ok(Json(case)),
        None => Err(err_json(StatusCode::NOT_FOUND, "case not found")),
    }
}

pub async fn get_stats(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let stats = state
        .store
        .stats()
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;
    Ok(Json(stats))
}

pub async fn patient_history(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let cases = state
        .store
        .query_patient_history(&hash)
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;
    Ok(Json(cases))
}

// --- Jurisdiction Rules CRUD ---

pub async fn list_rules(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<crate::jurisdiction::JurisdictionRule>> {
    let rules = state.rules.read().await;
    Json(rules.clone())
}

#[derive(Deserialize)]
pub struct CreateRuleRequest {
    pub jurisdiction_name: String,
    pub state_codes: Vec<String>,
    #[serde(default)]
    pub condition_snomeds: Vec<String>,
    #[serde(default)]
    pub min_confidence: f64,
    #[serde(default)]
    pub priority: i32,
    #[serde(default = "default_true")]
    pub active: bool,
}
fn default_true() -> bool { true }

pub async fn create_rule(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateRuleRequest>,
) -> Result<Json<crate::jurisdiction::JurisdictionRule>, (StatusCode, Json<ErrorResponse>)> {
    let now = chrono::Utc::now().to_rfc3339();
    let rule = crate::jurisdiction::JurisdictionRule {
        rule_id: uuid::Uuid::new_v4().to_string(),
        jurisdiction_name: body.jurisdiction_name,
        state_codes: body.state_codes,
        condition_snomeds: body.condition_snomeds,
        min_confidence: body.min_confidence,
        priority: body.priority,
        active: body.active,
        created_at: now.clone(),
        updated_at: now,
    };
    state
        .store
        .insert_rule(&rule)
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;
    let mut rules = state.rules.write().await;
    *rules = state
        .store
        .load_rules()
        .unwrap_or_default();
    Ok(Json(rule))
}

pub async fn delete_rule_handler(
    State(state): State<Arc<AppState>>,
    Path(rule_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let deleted = state
        .store
        .delete_rule(&rule_id)
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;
    if deleted {
        let mut rules = state.rules.write().await;
        *rules = state.store.load_rules().unwrap_or_default();
        Ok(Json(serde_json::json!({"ok": true})))
    } else {
        Err(err_json(StatusCode::NOT_FOUND, "rule not found"))
    }
}

#[derive(Serialize)]
pub struct SystemInfo {
    pub model_name: String,
    pub llm_backend: String,
    pub hardware: String,
    pub db_backend: String,
    pub uptime_seconds: u64,
}

pub async fn system_info(
    State(state): State<Arc<AppState>>,
) -> Json<SystemInfo> {
    Json(SystemInfo {
        model_name: state.config.model_name.clone(),
        llm_backend: match state.config.llm_backend {
            LlmBackend::LlamaServer => "llama-server".into(),
            LlmBackend::Ollama => "ollama".into(),
        },
        hardware: state.config.hardware_name.clone(),
        db_backend: "DuckDB + Arrow Flight".into(),
        uptime_seconds: state.started_at.elapsed().as_secs(),
    })
}

#[derive(Deserialize)]
pub struct PatchCase {
    pub queue_status: String,
}

pub async fn patch_case(
    State(state): State<Arc<AppState>>,
    Path(case_id): Path<String>,
    Json(body): Json<PatchCase>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    if case::QueueStatus::from_str(&body.queue_status).is_none() {
        return Err(err_json(
            StatusCode::BAD_REQUEST,
            format!("invalid status: {}", body.queue_status),
        ));
    }
    let updated = state
        .store
        .update_status(&case_id, &body.queue_status)
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {e}")))?;
    if updated {
        Ok(Json(serde_json::json!({"ok": true, "case_id": case_id, "queue_status": body.queue_status})))
    } else {
        Err(err_json(StatusCode::NOT_FOUND, "case not found"))
    }
}
