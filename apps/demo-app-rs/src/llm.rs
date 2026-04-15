use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::json;

use crate::config::{Config, LlmBackend};

const SYSTEM_PROMPT: &str = "Extract clinical entities from this eICR. Output compact JSON with: \
    patient, encounter, conditions (SNOMED), labs (LOINC), \
    meds (RxNorm), vitals. No summary. Valid JSON only.";

#[derive(Debug, Clone, Default, Serialize)]
pub struct LlmResponse {
    pub content: String,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub tokens_per_second: Option<f64>,
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn convert(&self, system: &str, user: &str) -> Result<LlmResponse>;
}

pub struct LlamaServer {
    base: String,
    model: String,
    http: reqwest::Client,
}

pub struct Ollama {
    base: String,
    model: String,
    http: reqwest::Client,
}

#[async_trait]
impl LlmClient for LlamaServer {
    async fn convert(&self, system: &str, user: &str) -> Result<LlmResponse> {
        let payload = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": false,
        });

        let resp = self
            .http
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&payload)
            .send()
            .await
            .context("llama-server unreachable")?;

        let body: serde_json::Value = resp.json().await.context("llama-server response parse")?;

        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .context("no content in llama-server response")?
            .to_string();

        let prompt_tokens = body["usage"]["prompt_tokens"].as_i64();
        let completion_tokens = body["usage"]["completion_tokens"].as_i64();

        // llama.cpp extension: timings.predicted_per_second
        let tokens_per_second = body["timings"]["predicted_per_second"]
            .as_f64()
            .or_else(|| {
                // Fallback: compute from usage if timing info absent
                let comp = completion_tokens? as f64;
                let total_ms = body["timings"]["predicted_ms"].as_f64()?;
                if total_ms > 0.0 {
                    Some(comp / (total_ms / 1000.0))
                } else {
                    None
                }
            });

        Ok(LlmResponse {
            content,
            prompt_tokens,
            completion_tokens,
            tokens_per_second,
        })
    }
}

#[async_trait]
impl LlmClient for Ollama {
    async fn convert(&self, system: &str, user: &str) -> Result<LlmResponse> {
        let payload = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": false,
            "options": {"temperature": 0.1, "num_ctx": 2048},
        });

        let resp = self
            .http
            .post(format!("{}/api/chat", self.base))
            .json(&payload)
            .send()
            .await
            .context("ollama unreachable")?;

        let body: serde_json::Value = resp.json().await.context("ollama response parse")?;

        let content = body["message"]["content"]
            .as_str()
            .context("no content in ollama response")?
            .to_string();

        let prompt_tokens = body["prompt_eval_count"].as_i64();
        let completion_tokens = body["eval_count"].as_i64();

        // Ollama reports durations in nanoseconds
        let tokens_per_second = completion_tokens.and_then(|count| {
            let eval_ns = body["eval_duration"].as_f64()?;
            if eval_ns > 0.0 {
                Some(count as f64 / (eval_ns / 1_000_000_000.0))
            } else {
                None
            }
        });

        Ok(LlmResponse {
            content,
            prompt_tokens,
            completion_tokens,
            tokens_per_second,
        })
    }
}

pub fn build_client(config: &Config) -> Box<dyn LlmClient> {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .expect("failed to build HTTP client");

    match config.llm_backend {
        LlmBackend::LlamaServer => Box::new(LlamaServer {
            base: config.llm_url.clone(),
            model: config.model_name.clone(),
            http,
        }),
        LlmBackend::Ollama => Box::new(Ollama {
            base: config.llm_url.clone(),
            model: config.model_name.clone(),
            http,
        }),
    }
}

pub fn system_prompt() -> &'static str {
    SYSTEM_PROMPT
}
