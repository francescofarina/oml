use crate::errors::ModelError;
use crate::model::Model;
use num_traits::Float;
use std::fmt::Debug;
use std::iter::Sum;
use std::{thread, time};

/// Defines the behavior for machine learning algorithms.
///
/// This trait should be implemented by any algorithm that can perform
/// training and inference steps on a given model.
pub trait Algorithm<T>: Send + Sync
where
    T: Float + Debug + Send + Sync + Sum,
{
    /// Performs a training step on the provided model with the given input `x`.
    ///
    /// # Arguments
    ///
    /// * `model` - A reference to the model on which the training step is performed.
    /// * `x` - The input value used for training.
    ///
    /// # Returns
    ///
    /// A result indicating whether the training step was successful or not.
    fn training_step(&self, model: &Model<T>, x: T) -> Result<(), ModelError>;

    /// Performs an inference step on the provided model with the given input `x`.
    ///
    /// # Arguments
    ///
    /// * `model` - A reference to the model on which the inference step is performed.
    /// * `x` - The input value used for inference.
    ///
    /// # Returns
    ///
    /// A result containing the inference output or an error.
    fn inference_step(&self, model: &Model<T>, x: T) -> Result<T, ModelError>;
}

/// A dummy algorithm used for demonstration purposes.
///
/// This implementation is solely for testing and should be replaced
/// with an actual algorithm implementation.
#[derive(Debug)]
pub struct DummyAlgorithm;

impl<T> Algorithm<T> for DummyAlgorithm
where
    T: Float + Debug + Send + Sync + Sum,
{
    fn training_step(&self, model: &Model<T>, x: T) -> Result<(), ModelError> {
        unsafe {
            let params = model.get_parameters_mut();
            thread::sleep(time::Duration::from_millis(5000)); // simulated delay
            params.iter_mut().for_each(|param| *param = *param * x);
            Ok(())
        }
    }

    fn inference_step(&self, model: &Model<T>, x: T) -> Result<T, ModelError> {
        unsafe {
            let params = model.get_parameters();
            thread::sleep(time::Duration::from_millis(500)); // simulated delay
            Ok(params.iter().map(|param| *param * x).sum())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Model;
    use super::*;
    use tokio::runtime; // Import runtime to execute async code in tests.

    // Helper function to initialize the Tokio runtime for tests.
    fn setup() -> runtime::Runtime {
        runtime::Runtime::new().unwrap()
    }

    #[test]
    fn test_inference_step() {
        let rt = setup();
        let model = Model::with_parameters(vec![1.0, 2.0, 3.0]);
        let algorithm = DummyAlgorithm;

        rt.block_on(async {
            let result = algorithm.inference_step(&model, 2.0);
            assert_eq!(result.unwrap(), 12.0); // 2*1 + 2*2 + 2*3
        });
    }

    #[test]
    fn test_training_step() {
        let rt = setup();
        let model = Model::with_parameters(vec![1.0, 2.0, 3.0]);
        let algorithm = DummyAlgorithm;
        let update_factor = 1.1; // Define the update factor for the test

        rt.block_on(async {
            algorithm.training_step(&model, update_factor).unwrap();
        });

        let params = unsafe { &*model.parameters.get() };
        let expected: Vec<f32> = vec![
            1.0 * update_factor,
            2.0 * update_factor,
            3.0 * update_factor,
        ];
        for (a, b) in params.iter().zip(expected.iter()) {
            assert!((a - b).abs() < f32::EPSILON); // Check if values are approximately equal.
        }
    }
}
