use std::collections::HashMap;
use std::env;

use log::info;
use qdrant_client::qdrant::{
    PointStruct, UpsertPointsBuilder, Value, Vectors
};
use qdrant_client::{Qdrant, QdrantError};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record {
    payload: String,
}

#[tokio::main]
async fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let model = tokio::task::spawn_blocking(|| {
        SentenceEmbeddingsBuilder::remote(
        rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL12V2,
    )
    .create_model()
    });

    let model = model.await.unwrap();
    
    let model = model.unwrap();

    let client = Qdrant::from_url(&std::env::var("QDRANT_URL").unwrap_or("http://localhost:6334".to_owned()))
        .build();

    if client.is_err()  {
        info!("No client created");
        return;
    }
    let client = client.unwrap();
   
    let collections = client.list_collections().await;
    info!("{:?}", collections);

    let collection_name = "test";

    /*
    let _ = client.delete_collection(collection_name).await;
    client
        .create_collection(
            CreateCollectionBuilder::new(collection_name)
                .vectors_config(VectorParamsBuilder::new(384, Distance::Cosine))
                .quantization_config(ScalarQuantizationBuilder::default()),
        )
        .await
        .expect("failed to create collection");
    */

    // move to arg when built binary works
    let input_file = env::var("DATA_INPUT_FILENAME").unwrap_or("data/qdrant_test.csv".to_string());

    let mut rdr = csv::Reader::from_path(input_file).unwrap();
    
    let mut points: Vec<PointStruct> = vec![];
    let mut index = 0;
    for result in rdr.deserialize() {
        let record: Record = result.unwrap();
        
        let payload: HashMap<String, Value> = serde_json::from_str(&record.payload).unwrap();
        let payload_id_string = payload.get("id").unwrap().to_string();
        // SPECIAL case of data
        let payload_id = payload_id_string.trim_matches('"');
        let searchable_content = payload.get("searchable_content");
        if searchable_content.is_none() || payload_id.is_empty() {
            continue;
        }
        let searchable_content_string = searchable_content.unwrap().to_string();
        // SPECIAL broken json
        if searchable_content_string.is_empty() || searchable_content_string == "null" {
            continue;
        }
        let text = searchable_content_string;
        let embeddings: Vectors = model
            .encode(&[text])
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .into();

        let point = match payload_id.parse::<u64>() {
            Ok(id) => PointStruct::new(id, embeddings as Vectors, payload),
            Err(err) => {
                info!("err: {:?}", err);
                PointStruct::new(payload_id.to_string(), embeddings as Vectors, payload)
            }
        };

        points.insert(index, point);
        index += 1;
        if index > 100 {
            let batch = points.clone();
            index = 0;
            points.clear();
            
            let res: Result<qdrant_client::qdrant::PointsOperationResponse, QdrantError> = client
            .upsert_points(UpsertPointsBuilder::new(collection_name, batch))
            .await;
            info!("res: {:?}", res);
        }
    }

    // if any record left save it also
    if points.len() >  0 {
        let res: Result<qdrant_client::qdrant::PointsOperationResponse, QdrantError> = client
        .upsert_points(UpsertPointsBuilder::new(collection_name, points))
        .await;
        info!("res: {:?}", res);
    }
}