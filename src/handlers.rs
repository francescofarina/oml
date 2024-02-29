use crate::algorithm::Algorithm;
use crate::model::Model;
use actix_web::{web, HttpResponse, Responder};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::iter::Sum;
use std::sync::Arc;

/// Shared application state for use in Actix web server handlers.
///
/// Contains references to the model and algorithm that are used to perform
/// machine learning operations. Wrapped in an `Arc` to safely share across threads.
pub struct AppState<T, A>
where
    T: Float + Serialize + for<'de> Deserialize<'de> + Debug + Send + Sync + Sum + 'static,
    A: Algorithm<T> + 'static,
{
    pub model: Arc<Model<T>>,
    pub algorithm: Arc<A>,
}

/// Asynchronous handler for inference requests.
///
/// # Arguments
///
/// * `data` - Extracted application state including model and algorithm.
/// * `input` - JSON-parsed input value of type `T`.
///
/// # Returns
///
/// A responder that will result in an HTTP response indicating the outcome
/// of the inference operation.
pub async fn handle_inference_step<T, A>(
    data: web::Data<AppState<T, A>>,
    input: web::Json<T>,
) -> impl Responder
where
    T: Float + Serialize + for<'de> Deserialize<'de> + Debug + Send + Sync + Sum,
    A: Algorithm<T>,
{
    let model = data.model.clone(); // clone the Arc (not the model)
    let algorithm = data.algorithm.clone(); // clone the Arc (not the algo)

    match tokio::task::spawn_blocking(move || algorithm.inference_step(&model, *input)).await {
        Ok(response) => match response {
            Ok(result) => HttpResponse::Ok().json(result),
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
        },
        Err(e) => HttpResponse::InternalServerError().body(format!("Task failed: {:?}", e)),
    }
}

/// Asynchronous handler for training requests.
///
/// # Arguments
///
/// * `data` - Extracted application state including model and algorithm.
/// * `input` - JSON-parsed input value of type `T`.
///
/// # Returns
///
/// A responder that will result in an HTTP response indicating the outcome
/// of the training operation.
pub async fn handle_training_step<T, A>(
    data: web::Data<AppState<T, A>>,
    input: web::Json<T>,
) -> impl Responder
where
    T: Float + Serialize + for<'de> Deserialize<'de> + Debug + Send + Sync + Sum,
    A: Algorithm<T>,
{
    let model = data.model.clone(); // clone the Arc (not the model)
    let algorithm = data.algorithm.clone(); // clone the Arc (not the algo)

    match tokio::task::spawn_blocking(move || algorithm.training_step(&model, *input)).await {
        Ok(response) => match response {
            Ok(_) => HttpResponse::Ok().finish(),
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
        },
        Err(e) => HttpResponse::InternalServerError().body(format!("Task failed: {:?}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::{Algorithm, DummyAlgorithm};
    use crate::model::Model;
    use actix_web::{http, test, web, App};

    // Helper function to create app_state for the tests
    fn create_app_state<T, A>(model: Model<T>, algorithm: A) -> web::Data<AppState<T, A>>
    where
        T: Float + Serialize + for<'de> Deserialize<'de> + Debug + Send + Sync + Sum,
        A: Algorithm<T> + 'static,
    {
        let model_arc = Arc::new(model);
        let algorithm_arc = Arc::new(algorithm);
        web::Data::new(AppState {
            model: model_arc,
            algorithm: algorithm_arc,
        })
    }

    #[actix_rt::test]
    async fn test_handle_inference_step() {
        let model = Model::<f32>::with_parameters(vec![1.0, 2.0]);
        let algorithm = DummyAlgorithm {}; // Use your DummyAlgorithm for testing
        let app_state = create_app_state(model, algorithm);

        let mut app = test::init_service(App::new().app_data(app_state.clone()).route(
            "/inference",
            web::post().to(handle_inference_step::<f32, DummyAlgorithm>),
        ))
        .await;

        let req = test::TestRequest::post()
            .uri("/inference")
            .set_json(&3.5f32)
            .to_request();

        let resp = test::call_service(&mut app, req).await;
        assert_eq!(resp.status(), http::StatusCode::OK);

        let result: f32 = test::read_body_json(resp).await;
        assert_eq!(result, 10.5f32); // (1.0 * 3.5) + (2.0 * 3.5)
    }

    #[actix_rt::test]
    async fn test_handle_training_step() {
        let model = Model::<f32>::with_parameters(vec![1.0, 2.0]);
        let algorithm = DummyAlgorithm {}; // Use your DummyAlgorithm for testing
        let app_state = create_app_state(model, algorithm);

        let mut app = test::init_service(App::new().app_data(app_state.clone()).route(
            "/training",
            web::post().to(handle_training_step::<f32, DummyAlgorithm>),
        ))
        .await;

        let training_input = 1.1f32;
        let req = test::TestRequest::post()
            .uri("/training")
            .set_json(&training_input)
            .to_request();

        test::call_service(&mut app, req).await;

        // Unwrap the AppState to get the Model
        let model = &app_state.model;

        unsafe {
            // Inspect updated model state
            let updated_parameters = model.get_parameters().clone();

            // Ensure parameters have been updated correctly
            let expected_parameters: Vec<f32> = vec![1.0 * training_input, 2.0 * training_input];
            assert_eq!(updated_parameters, expected_parameters);
        }
    }
}
