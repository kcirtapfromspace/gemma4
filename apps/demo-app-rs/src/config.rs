use std::env;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmBackend {
    LlamaServer,
    Ollama,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub llm_backend: LlmBackend,
    pub llm_url: String,
    pub model_name: String,
    pub db_path: String,
    pub http_port: u16,
    pub flight_port: u16,
    pub hardware_name: String,
}

impl Config {
    pub fn from_env() -> Self {
        let backend_str = env::var("LLM_BACKEND").unwrap_or_else(|_| "llama-server".into());
        let llm_backend = match backend_str.as_str() {
            "ollama" => LlmBackend::Ollama,
            _ => LlmBackend::LlamaServer,
        };

        let default_url = match llm_backend {
            LlmBackend::LlamaServer => "http://llama-server.gemma4.svc:8080",
            LlmBackend::Ollama => "http://ollama.gemma4.svc:11434",
        };

        Self {
            llm_backend,
            llm_url: env::var("LLM_URL").unwrap_or_else(|_| default_url.into()),
            model_name: env::var("MODEL_NAME").unwrap_or_else(|_| "gemma4-eicr-fhir".into()),
            db_path: env::var("DB_PATH").unwrap_or_else(|_| "cliniq.db".into()),
            http_port: env::var("HTTP_PORT")
                .or_else(|_| env::var("PORT"))
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            flight_port: env::var("FLIGHT_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8081),
            hardware_name: env::var("HARDWARE")
                .unwrap_or_else(|_| "Unknown Hardware".into()),
        }
    }
}
