use std::collections::HashMap;

use axum::extract::Query;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use qdrant_client::qdrant::{
    SearchParamsBuilder, SearchPointsBuilder, Value
};
use qdrant_client::Qdrant;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModel};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
struct RecordResponse {
    score: f32,
    payload: HashMap<String, Value>,
}

const COLLECTION_NAME: &str = "test";

async fn get_model() -> SentenceEmbeddingsModel {
    let model = tokio::task::spawn_blocking(|| {
        SentenceEmbeddingsBuilder::remote(
            rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL12V2,
        )
        .create_model()
    });

    let model = model.await.expect("Result of model");
    
    model.expect("No model")
}

async fn get_qdrant_client() -> Qdrant {
    // TODO cache this
    let client = Qdrant::from_url(&std::env::var("QDRANT_URL").unwrap_or("http://localhost:6334".to_owned()))
    .build();
    let client: Qdrant = client.expect("No client");
    client
}

#[derive(Debug, Serialize, Clone, Deserialize)]
struct Params {
    text: String
}

async fn search(
    Query(params): Query<Params>,
) -> impl IntoResponse {
    let client = get_qdrant_client().await;
    let model = get_model().await;
    let vector = model.encode(&[params.text]).unwrap().into_iter().next().unwrap();
    let request = SearchPointsBuilder::new(COLLECTION_NAME, vector, 10)
        .with_payload(true)
        .params(SearchParamsBuilder::default().exact(true));
    let result = client.search_points(request).await.unwrap();
    let mut response_data: Vec<RecordResponse> = vec![];
    let index = 0;
    for res in result.result.into_iter() {
        // TODO howto convert res.id
        let r = RecordResponse{score: res.score, payload: res.payload};
        response_data.insert(index, r);
    }
    Json(response_data)
}

#[tokio::main]
async fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let app = Router::new()
    .route("/search", get(search));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:9000")
        .await
        .unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}