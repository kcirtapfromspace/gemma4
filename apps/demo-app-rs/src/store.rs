use std::sync::Mutex;

use anyhow::{Context, Result};
use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use duckdb::Connection;
use serde::Serialize;

use crate::case::Case;
use crate::jurisdiction::JurisdictionRule;

pub struct Store {
    conn: Mutex<Connection>,
}

#[derive(Debug, Serialize)]
pub struct Stats {
    pub total: i64,
    pub by_status: Vec<StatusCount>,
    pub by_condition_7d: Vec<ConditionCount>,
    pub inference_p50_ms: Option<f64>,
    pub inference_p95_ms: Option<f64>,
    pub avg_tokens_per_second: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct StatusCount {
    pub status: String,
    pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct ConditionCount {
    pub condition: String,
    pub count: i64,
}

#[derive(Debug, Default)]
pub struct CaseFilter {
    pub status: Option<String>,
    pub limit: Option<i64>,
}

impl Store {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path).context("failed to open DuckDB")?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cases (
                case_id          VARCHAR PRIMARY KEY,
                patient_name     VARCHAR,
                patient_dob      VARCHAR,
                patient_gender   VARCHAR,
                patient_state    VARCHAR,
                patient_city     VARCHAR,
                condition_snomed VARCHAR,
                condition_display VARCHAR,
                lab_loinc        VARCHAR,
                lab_result       VARCHAR,
                jurisdiction     VARCHAR,
                queue_status     VARCHAR,
                dedup_hash       VARCHAR,
                ingested_at      VARCHAR,
                processed_at     VARCHAR,
                inference_ms     BIGINT,
                raw_eicr_xml     TEXT,
                fhir_bundle      TEXT,
                entities         TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cases_dedup ON cases(dedup_hash, ingested_at);
            CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(queue_status);",
        )
        .context("migration failed")?;

        // Add token metric columns (ignore errors if they already exist)
        let _ = conn.execute_batch(
            "ALTER TABLE cases ADD COLUMN prompt_tokens BIGINT;
             ALTER TABLE cases ADD COLUMN completion_tokens BIGINT;
             ALTER TABLE cases ADD COLUMN tokens_per_second DOUBLE;
             ALTER TABLE cases ADD COLUMN extraction_json TEXT;
             ALTER TABLE cases ADD COLUMN input_format VARCHAR DEFAULT 'eicr';
             ALTER TABLE cases ADD COLUMN patient_hash VARCHAR;
             ALTER TABLE cases ADD COLUMN diff_json TEXT;
             ALTER TABLE cases ADD COLUMN prev_case_id VARCHAR;",
        );
        let _ = conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_cases_patient_hash ON cases(patient_hash, ingested_at);",
        );

        // Jurisdiction rules table
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS jurisdiction_rules (
                rule_id          VARCHAR PRIMARY KEY,
                jurisdiction_name VARCHAR NOT NULL,
                state_codes      VARCHAR NOT NULL,
                condition_snomeds VARCHAR NOT NULL DEFAULT '[]',
                min_confidence   DOUBLE NOT NULL DEFAULT 0.0,
                priority         INTEGER NOT NULL DEFAULT 0,
                active           BOOLEAN NOT NULL DEFAULT TRUE,
                created_at       VARCHAR NOT NULL,
                updated_at       VARCHAR NOT NULL
            );",
        )
        .context("jurisdiction_rules migration failed")?;

        Ok(())
    }

    /// Check if a dedup_hash was seen in the last 30 days.
    pub fn is_graylist(&self, hash: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT 1 FROM cases WHERE dedup_hash = ? AND CAST(ingested_at AS TIMESTAMP) > (CAST(now() AS TIMESTAMP) - INTERVAL 30 DAY) LIMIT 1",
        )?;
        let exists = stmt.query_row([hash], |_row| Ok(true)).unwrap_or(false);
        Ok(exists)
    }

    pub fn insert_case(&self, c: &Case) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO cases (case_id, patient_name, patient_dob, patient_gender,
                patient_state, patient_city, condition_snomed, condition_display,
                lab_loinc, lab_result, jurisdiction, queue_status, dedup_hash,
                ingested_at, processed_at, inference_ms,
                prompt_tokens, completion_tokens, tokens_per_second,
                raw_eicr_xml, extraction_json, fhir_bundle, entities, input_format,
                patient_hash, diff_json, prev_case_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                c.case_id,
                c.patient_name,
                c.patient_dob,
                c.patient_gender,
                c.patient_state,
                c.patient_city,
                c.condition_snomed,
                c.condition_display,
                c.lab_loinc,
                c.lab_result,
                c.jurisdiction,
                c.queue_status,
                c.dedup_hash,
                c.ingested_at,
                c.processed_at,
                c.inference_ms,
                c.prompt_tokens,
                c.completion_tokens,
                c.tokens_per_second,
                c.raw_eicr_xml,
                c.extraction_json,
                c.fhir_bundle,
                c.entities_json,
                c.input_format,
                c.patient_hash,
                c.diff_json,
                c.prev_case_id,
            ],
        )
        .context("insert_case failed")?;
        Ok(())
    }

    pub fn query_cases(&self, filter: &CaseFilter) -> Result<Vec<Case>> {
        let conn = self.conn.lock().unwrap();
        let limit = filter.limit.unwrap_or(100);

        let cols = "case_id, patient_name, patient_dob, patient_gender, patient_state,
            patient_city, condition_snomed, condition_display, lab_loinc, lab_result,
            jurisdiction, queue_status, dedup_hash, ingested_at, processed_at,
            inference_ms, prompt_tokens, completion_tokens, tokens_per_second,
            raw_eicr_xml, extraction_json, fhir_bundle, entities, input_format,
            patient_hash, diff_json, prev_case_id";

        let (sql, params): (String, Vec<Box<dyn duckdb::ToSql>>) = if let Some(status) = &filter.status {
            (
                format!("SELECT {cols} FROM cases WHERE queue_status = ? ORDER BY ingested_at DESC LIMIT ?"),
                vec![
                    Box::new(status.clone()) as Box<dyn duckdb::ToSql>,
                    Box::new(limit),
                ],
            )
        } else {
            (
                format!("SELECT {cols} FROM cases ORDER BY ingested_at DESC LIMIT ?"),
                vec![Box::new(limit) as Box<dyn duckdb::ToSql>],
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            row_to_case(row)
        })?;

        let mut cases = Vec::new();
        for row in rows {
            cases.push(row?);
        }
        Ok(cases)
    }

    pub fn get_case(&self, case_id: &str) -> Result<Option<Case>> {
        let conn = self.conn.lock().unwrap();
        let cols = "case_id, patient_name, patient_dob, patient_gender, patient_state,
            patient_city, condition_snomed, condition_display, lab_loinc, lab_result,
            jurisdiction, queue_status, dedup_hash, ingested_at, processed_at,
            inference_ms, prompt_tokens, completion_tokens, tokens_per_second,
            raw_eicr_xml, extraction_json, fhir_bundle, entities, input_format,
            patient_hash, diff_json, prev_case_id";
        let mut stmt = conn.prepare(&format!("SELECT {cols} FROM cases WHERE case_id = ?"))?;
        let result = stmt.query_row([case_id], |row| row_to_case(row)).ok();
        Ok(result)
    }

    pub fn update_status(&self, case_id: &str, new_status: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let updated = conn.execute(
            "UPDATE cases SET queue_status = ?, processed_at = current_timestamp WHERE case_id = ?",
            duckdb::params![new_status, case_id],
        )?;
        Ok(updated > 0)
    }

    /// Find the most recent case for a patient_hash, excluding a specific case_id.
    pub fn find_previous_case(&self, patient_hash_val: &str, exclude_case_id: &str) -> Result<Option<Case>> {
        let conn = self.conn.lock().unwrap();
        let cols = "case_id, patient_name, patient_dob, patient_gender, patient_state,
            patient_city, condition_snomed, condition_display, lab_loinc, lab_result,
            jurisdiction, queue_status, dedup_hash, ingested_at, processed_at,
            inference_ms, prompt_tokens, completion_tokens, tokens_per_second,
            raw_eicr_xml, extraction_json, fhir_bundle, entities, input_format,
            patient_hash, diff_json, prev_case_id";
        let mut stmt = conn.prepare(&format!(
            "SELECT {cols} FROM cases WHERE patient_hash = ? AND case_id != ? ORDER BY ingested_at DESC LIMIT 1"
        ))?;
        let result = stmt.query_row(
            duckdb::params![patient_hash_val, exclude_case_id],
            |row| row_to_case(row),
        ).ok();
        Ok(result)
    }

    /// Get all cases for a patient hash, ordered by ingested_at.
    pub fn query_patient_history(&self, patient_hash_val: &str) -> Result<Vec<Case>> {
        let conn = self.conn.lock().unwrap();
        let cols = "case_id, patient_name, patient_dob, patient_gender, patient_state,
            patient_city, condition_snomed, condition_display, lab_loinc, lab_result,
            jurisdiction, queue_status, dedup_hash, ingested_at, processed_at,
            inference_ms, prompt_tokens, completion_tokens, tokens_per_second,
            raw_eicr_xml, extraction_json, fhir_bundle, entities, input_format,
            patient_hash, diff_json, prev_case_id";
        let mut stmt = conn.prepare(&format!(
            "SELECT {cols} FROM cases WHERE patient_hash = ? ORDER BY ingested_at ASC"
        ))?;
        let rows = stmt.query_map([patient_hash_val], |row| row_to_case(row))?;
        let mut cases = Vec::new();
        for row in rows {
            cases.push(row?);
        }
        Ok(cases)
    }

    pub fn stats(&self) -> Result<Stats> {
        let conn = self.conn.lock().unwrap();

        // Total
        let total: i64 = conn.query_row("SELECT count(*) FROM cases", [], |r| r.get(0))?;

        // By status
        let mut stmt = conn.prepare("SELECT queue_status, count(*) FROM cases GROUP BY queue_status ORDER BY count(*) DESC")?;
        let by_status: Vec<StatusCount> = stmt
            .query_map([], |row| {
                Ok(StatusCount {
                    status: row.get(0)?,
                    count: row.get(1)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        // By condition last 7 days
        let mut stmt = conn.prepare(
            "SELECT condition_display, count(*) FROM cases
             WHERE CAST(ingested_at AS TIMESTAMP) > (CAST(now() AS TIMESTAMP) - INTERVAL 7 DAY) AND condition_display != ''
             GROUP BY condition_display ORDER BY count(*) DESC LIMIT 10",
        )?;
        let by_condition_7d: Vec<ConditionCount> = stmt
            .query_map([], |row| {
                Ok(ConditionCount {
                    condition: row.get(0)?,
                    count: row.get(1)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Inference percentiles
        let p50: Option<f64> = conn
            .query_row(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY inference_ms) FROM cases WHERE inference_ms > 0",
                [],
                |r| r.get(0),
            )
            .ok();
        let p95: Option<f64> = conn
            .query_row(
                "SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY inference_ms) FROM cases WHERE inference_ms > 0",
                [],
                |r| r.get(0),
            )
            .ok();

        // Average tokens per second
        let avg_tps: Option<f64> = conn
            .query_row(
                "SELECT AVG(tokens_per_second) FROM cases WHERE tokens_per_second > 0",
                [],
                |r| r.get(0),
            )
            .ok();

        Ok(Stats {
            total,
            by_status,
            by_condition_7d,
            inference_p50_ms: p50,
            inference_p95_ms: p95,
            avg_tokens_per_second: avg_tps,
        })
    }

    // --- Jurisdiction Rules CRUD ---

    pub fn load_rules(&self) -> Result<Vec<JurisdictionRule>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT rule_id, jurisdiction_name, state_codes, condition_snomeds, min_confidence, priority, active, created_at, updated_at FROM jurisdiction_rules ORDER BY priority DESC"
        )?;
        let rows = stmt.query_map([], |row| {
            let state_codes_json: String = row.get(2)?;
            let cond_snomeds_json: String = row.get(3)?;
            Ok(JurisdictionRule {
                rule_id: row.get(0)?,
                jurisdiction_name: row.get(1)?,
                state_codes: serde_json::from_str(&state_codes_json).unwrap_or_default(),
                condition_snomeds: serde_json::from_str(&cond_snomeds_json).unwrap_or_default(),
                min_confidence: row.get(4)?,
                priority: row.get(5)?,
                active: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        })?;
        let mut rules = Vec::new();
        for row in rows {
            rules.push(row?);
        }
        Ok(rules)
    }

    pub fn insert_rule(&self, rule: &JurisdictionRule) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO jurisdiction_rules (rule_id, jurisdiction_name, state_codes, condition_snomeds, min_confidence, priority, active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                rule.rule_id,
                rule.jurisdiction_name,
                serde_json::to_string(&rule.state_codes).unwrap_or_default(),
                serde_json::to_string(&rule.condition_snomeds).unwrap_or_default(),
                rule.min_confidence,
                rule.priority,
                rule.active,
                rule.created_at,
                rule.updated_at,
            ],
        )?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn update_rule(&self, rule: &JurisdictionRule) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let updated = conn.execute(
            "UPDATE jurisdiction_rules SET jurisdiction_name=?, state_codes=?, condition_snomeds=?, min_confidence=?, priority=?, active=?, updated_at=? WHERE rule_id=?",
            duckdb::params![
                rule.jurisdiction_name,
                serde_json::to_string(&rule.state_codes).unwrap_or_default(),
                serde_json::to_string(&rule.condition_snomeds).unwrap_or_default(),
                rule.min_confidence,
                rule.priority,
                rule.active,
                rule.updated_at,
                rule.rule_id,
            ],
        )?;
        Ok(updated > 0)
    }

    pub fn delete_rule(&self, rule_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let deleted = conn.execute(
            "DELETE FROM jurisdiction_rules WHERE rule_id = ?",
            [rule_id],
        )?;
        Ok(deleted > 0)
    }

    pub fn seed_default_rules(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT count(*) FROM jurisdiction_rules",
            [],
            |r| r.get(0),
        )?;
        if count > 0 {
            return Ok(());
        }
        drop(conn); // release lock before inserting
        for rule in crate::jurisdiction::default_rules() {
            self.insert_rule(&rule)?;
        }
        Ok(())
    }

    /// Materialize cases as Arrow RecordBatch for Flight export.
    pub fn cases_to_arrow(&self, status_filter: Option<&str>) -> Result<RecordBatch> {
        let conn = self.conn.lock().unwrap();

        let (sql, params): (String, Vec<Box<dyn duckdb::ToSql>>) = if let Some(status) = status_filter {
            (
                "SELECT case_id, patient_name, condition_display, queue_status, jurisdiction, ingested_at, inference_ms FROM cases WHERE queue_status = ? AND queue_status != 'Graylist' ORDER BY ingested_at DESC".into(),
                vec![Box::new(status.to_string()) as Box<dyn duckdb::ToSql>],
            )
        } else {
            (
                "SELECT case_id, patient_name, condition_display, queue_status, jurisdiction, ingested_at, inference_ms FROM cases WHERE queue_status != 'Graylist' ORDER BY ingested_at DESC".into(),
                vec![],
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let mut case_ids = Vec::new();
        let mut names = Vec::new();
        let mut conditions = Vec::new();
        let mut statuses = Vec::new();
        let mut jurisdictions = Vec::new();
        let mut ingested = Vec::new();
        let mut infer_ms = Vec::new();

        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
            ))
        })?;

        for row in rows {
            let (id, name, cond, status, juris, ing, ms) = row?;
            case_ids.push(id);
            names.push(name);
            conditions.push(cond);
            statuses.push(status);
            jurisdictions.push(juris);
            ingested.push(ing);
            infer_ms.push(ms);
        }

        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("case_id", DataType::Utf8, false),
            Field::new("patient_name", DataType::Utf8, false),
            Field::new("condition_display", DataType::Utf8, false),
            Field::new("queue_status", DataType::Utf8, false),
            Field::new("jurisdiction", DataType::Utf8, false),
            Field::new("ingested_at", DataType::Utf8, false),
            Field::new("inference_ms", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                std::sync::Arc::new(StringArray::from(case_ids)),
                std::sync::Arc::new(StringArray::from(names)),
                std::sync::Arc::new(StringArray::from(conditions)),
                std::sync::Arc::new(StringArray::from(statuses)),
                std::sync::Arc::new(StringArray::from(jurisdictions)),
                std::sync::Arc::new(StringArray::from(ingested)),
                std::sync::Arc::new(Int64Array::from(infer_ms)),
            ],
        )?;

        Ok(batch)
    }
}

/// Extract a Case from a DuckDB row with explicit column ordering.
fn row_to_case(row: &duckdb::Row) -> std::result::Result<Case, duckdb::Error> {
    Ok(Case {
        case_id: row.get::<_, String>(0)?,
        patient_name: row.get::<_, String>(1)?,
        patient_dob: row.get::<_, String>(2)?,
        patient_gender: row.get::<_, String>(3)?,
        patient_state: row.get::<_, String>(4)?,
        patient_city: row.get::<_, String>(5)?,
        condition_snomed: row.get::<_, String>(6)?,
        condition_display: row.get::<_, String>(7)?,
        lab_loinc: row.get::<_, String>(8)?,
        lab_result: row.get::<_, String>(9)?,
        jurisdiction: row.get::<_, String>(10)?,
        queue_status: row.get::<_, String>(11)?,
        dedup_hash: row.get::<_, String>(12)?,
        ingested_at: row.get::<_, String>(13)?,
        processed_at: row.get::<_, Option<String>>(14)?,
        inference_ms: row.get::<_, i64>(15)?,
        prompt_tokens: row.get::<_, Option<i64>>(16)?,
        completion_tokens: row.get::<_, Option<i64>>(17)?,
        tokens_per_second: row.get::<_, Option<f64>>(18)?,
        raw_eicr_xml: row.get::<_, String>(19)?,
        extraction_json: row.get::<_, String>(20).unwrap_or_default(),
        fhir_bundle: row.get::<_, String>(21)?,
        entities_json: row.get::<_, String>(22)?,
        input_format: row.get::<_, String>(23).unwrap_or_else(|_| "eicr".into()),
        patient_hash: row.get::<_, String>(24).unwrap_or_default(),
        diff_json: row.get::<_, String>(25).unwrap_or_default(),
        prev_case_id: row.get::<_, String>(26).unwrap_or_default(),
    })
}
