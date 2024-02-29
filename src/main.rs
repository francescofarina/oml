use oml::algorithm::DummyAlgorithm;
use oml::model::Model;
use oml::server::run_server;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model: Model<f32> = Model::with_parameters(vec![1.0, 2.0]); // Create an instance of the Model for f32
    let algorithm = DummyAlgorithm;

    // Start the server and pass the server data to it
    run_server("127.0.0.1:8080", model, algorithm).await
}
