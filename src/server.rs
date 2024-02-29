use crate::algorithm::Algorithm;
use crate::handlers::AppState;
use crate::handlers::{handle_inference_step, handle_training_step};
use crate::model::Model;
use actix_web::{web, App, HttpServer};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter::Sum;
use std::sync::Arc;

// Starts an Actix web server with endpoints for inference and training steps.
pub async fn run_server<T, A>(address: &str, model: Model<T>, algorithm: A) -> std::io::Result<()>
where
    T: Float + Serialize + for<'de> Deserialize<'de> + 'static + Debug + Send + Sync + Sum,
    A: Algorithm<T> + 'static + Send + Sync,
{
    let shared_state = web::Data::new(AppState {
        model: Arc::new(model),
        algorithm: Arc::new(algorithm),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(shared_state.clone())
            .route("/inference", web::post().to(handle_inference_step::<T, A>))
            .route("/training", web::post().to(handle_training_step::<T, A>))
    })
    .bind(address)?
    .run()
    .await
}
