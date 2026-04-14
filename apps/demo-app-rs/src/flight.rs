use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::FlightService;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use futures::stream;
use futures::TryStreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::store::Store;

pub struct CliniqFlightService {
    pub store: Arc<Store>,
}

type BoxedFlightStream<T> = Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl FlightService for CliniqFlightService {
    type HandshakeStream = BoxedFlightStream<HandshakeResponse>;
    type ListFlightsStream = BoxedFlightStream<FlightInfo>;
    type DoGetStream = BoxedFlightStream<FlightData>;
    type DoPutStream = BoxedFlightStream<PutResult>;
    type DoExchangeStream = BoxedFlightStream<FlightData>;
    type DoActionStream = BoxedFlightStream<arrow_flight::Result>;
    type ListActionsStream = BoxedFlightStream<ActionType>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("handshake not needed"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }

    /// DoGet: stream cases as Arrow RecordBatch.
    /// Ticket bytes: "cases" or "cases?status=Processed"
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket_str = String::from_utf8(request.into_inner().ticket.to_vec())
            .map_err(|_| Status::invalid_argument("invalid ticket encoding"))?;

        let status_filter = if ticket_str.starts_with("cases?status=") {
            Some(ticket_str.strip_prefix("cases?status=").unwrap().to_string())
        } else if ticket_str == "cases" {
            None
        } else {
            return Err(Status::invalid_argument(format!(
                "unknown ticket: {ticket_str}"
            )));
        };

        let batch = self
            .store
            .cases_to_arrow(status_filter.as_deref())
            .map_err(|e| Status::internal(format!("arrow error: {e}")))?;

        let schema = batch.schema();
        let batches = vec![batch];
        let flight_stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream::iter(batches.into_iter().map(Ok)))
            .map_err(|e| Status::internal(format!("encode error: {e}")));

        Ok(Response::new(Box::pin(flight_stream)))
    }

    /// DoPut: accept inbound status updates.
    /// FlightDescriptor path: ["case-updates"]
    /// Each RecordBatch has columns: case_id (Utf8), new_status (Utf8)
    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut stream = request.into_inner();
        let mut updated = 0u64;

        // Collect all flight data and decode batches
        let mut flight_data_vec = Vec::new();
        while let Some(data) = stream
            .message()
            .await
            .map_err(|e| Status::internal(format!("stream error: {e}")))?
        {
            flight_data_vec.push(data);
        }

        // Decode: first message is schema, rest are batches
        if flight_data_vec.is_empty() {
            return Err(Status::invalid_argument("empty stream"));
        }

        // Use arrow_flight to decode
        let dictionaries_by_id = std::collections::HashMap::new();
        let schema_msg = &flight_data_vec[0];
        let _schema = arrow_flight::utils::flight_data_to_arrow_batch(
            schema_msg,
            Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("case_id", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new("new_status", arrow::datatypes::DataType::Utf8, false),
            ])),
            &dictionaries_by_id,
        );

        // Process each subsequent message as a batch
        let expected_schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("case_id", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("new_status", arrow::datatypes::DataType::Utf8, false),
        ]));

        for fd in flight_data_vec.iter().skip(1) {
            if let Ok(batch) = arrow_flight::utils::flight_data_to_arrow_batch(
                fd,
                expected_schema.clone(),
                &dictionaries_by_id,
            ) {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>();
                let statuses = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>();

                if let (Some(ids), Some(statuses)) = (ids, statuses) {
                    for i in 0..batch.num_rows() {
                        let case_id = ids.value(i);
                        let new_status = statuses.value(i);
                        if self.store.update_status(case_id, new_status).unwrap_or(false) {
                            updated += 1;
                        }
                    }
                }
            }
        }

        let result = PutResult {
            app_metadata: format!("{{\"updated\":{updated}}}").into(),
        };
        let out_stream = stream::once(async move { Ok(result) });
        Ok(Response::new(Box::pin(out_stream)))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions"))
    }
}
