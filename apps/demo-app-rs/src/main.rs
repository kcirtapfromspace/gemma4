mod case;
mod config;
mod dashboard;
mod eicr;
mod entities;
mod extraction;
mod fhir;
mod flight;
mod jurisdiction;
mod llm;
mod store;

use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use axum::routing::{get, patch, post};
use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tonic::transport::Server as TonicServer;
use tracing_subscriber::EnvFilter;

use crate::config::Config;
use crate::dashboard::AppState;
use crate::flight::CliniqFlightService;
use crate::store::Store;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("cliniq=info".parse()?))
        .init();

    let config = Config::from_env();
    tracing::info!(
        backend = ?config.llm_backend,
        llm_url = %config.llm_url,
        model = %config.model_name,
        db = %config.db_path,
        http_port = config.http_port,
        flight_port = config.flight_port,
        hardware = %config.hardware_name,
        "ClinIQ starting"
    );

    // Open DuckDB (single instance shared by HTTP + Flight)
    let db = Arc::new(Store::open(&config.db_path)?);

    // Build LLM client
    let llm_client = llm::build_client(&config);

    // Shared app state for axum
    let state = Arc::new(AppState {
        store: db.clone(),
        llm: llm_client,
        config: config.clone(),
        started_at: std::time::Instant::now(),
    });

    // Axum HTTP server
    let app = Router::new()
        .route("/", get(dashboard::index))
        .route("/api/sample", get(dashboard::sample_eicr))
        .route("/api/convert", post(dashboard::convert))
        .route("/api/convert/batch", post(dashboard::convert_batch))
        .route("/api/cases", get(dashboard::list_cases))
        .route("/api/cases/:id", get(dashboard::get_case))
        .route("/api/cases/:id", patch(dashboard::patch_case))
        .route("/api/stats", get(dashboard::get_stats))
        .route("/api/info", get(dashboard::system_info))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let http_addr = format!("0.0.0.0:{}", config.http_port);
    let flight_addr = format!("0.0.0.0:{}", config.flight_port).parse()?;

    tracing::info!("HTTP on {http_addr}, Flight gRPC on {flight_addr}");

    let http_listener = TcpListener::bind(&http_addr).await?;

    let flight_svc = CliniqFlightService {
        store: db.clone(),
    };

    // Run both servers concurrently
    tokio::try_join!(
        async {
            axum::serve(http_listener, app)
                .await
                .map_err(anyhow::Error::from)
        },
        async {
            TonicServer::builder()
                .add_service(FlightServiceServer::new(flight_svc))
                .serve(flight_addr)
                .await
                .map_err(anyhow::Error::from)
        },
    )?;

    Ok(())
}
