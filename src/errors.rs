use std::fmt;
use std::{error::Error, sync::PoisonError};

/// Error handler for the Model
#[derive(Debug)]
pub enum ModelError {
    LockError(String),
}

impl Error for ModelError {}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ModelError::LockError(ref err) => write!(f, "LockError: {}", err),
        }
    }
}
impl<T> From<PoisonError<T>> for ModelError {
    fn from(error: PoisonError<T>) -> Self {
        ModelError::LockError(error.to_string())
    }
}
