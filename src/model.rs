use num_traits::Float;
use std::cell::UnsafeCell;
use std::fmt::Debug;

// SAFETY: This type is marked `Sync` on the promise that it is only
// ever mutated (via calls to unsafe get_mut()) by one thread at a time.
// The user must ensure that when writing is occurring, no other writes
// are concurrently active. Reads can occur simultaneously without restriction,
// and this concurrent read behavior while writing is happening is acceptable
// per the external guarantees provided by the caller.
#[derive(Debug)]
pub struct SyncUnsafeCell<T>(UnsafeCell<T>);

unsafe impl<T> Sync for SyncUnsafeCell<T> where T: Send {}

impl<T> SyncUnsafeCell<T> {
    pub fn new(value: T) -> Self {
        SyncUnsafeCell(UnsafeCell::new(value))
    }

    // Allow immutable access.
    pub unsafe fn get(&self) -> &T {
        &*self.0.get()
    }

    // Allow mutable access. This is only safe if we can guarantee that no other
    // mutable references exist. This puts the burden of
    // guaranteeing no mutable aliasing on the caller.
    pub unsafe fn get_mut(&self) -> &mut T {
        &mut *self.0.get()
    }
}
/// A generic Model struct that holds a set of parameters.
#[derive(Debug)]
pub struct Model<T>
where
    T: Float + Debug + Send + Sync,
{
    pub parameters: SyncUnsafeCell<Vec<T>>,
}

impl<T> Model<T>
where
    T: Float + Debug + Send + Sync,
{
    /// Creates a new, empty Model.
    ///
    /// # Examples
    ///
    /// ```
    /// use oml::model::Model;
    ///
    /// let model: Model<f32> = Model::new();
    /// ```
    pub fn new() -> Self {
        Model {
            parameters: SyncUnsafeCell::new(Vec::new()),
        }
    }

    /// Creates a new Model with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - A vector of parameters to initialize the Model with.
    ///
    /// # Examples
    ///
    /// ```
    /// use oml::model::Model;
    ///
    /// let initial_params = vec![1.0, 2.0, 3.0];
    /// let model = Model::with_parameters(initial_params);
    /// ```
    pub fn with_parameters(params: Vec<T>) -> Self {
        Model {
            parameters: SyncUnsafeCell::new(params),
        }
    }

    /// Provides mutable access to the parameters.
    pub unsafe fn get_parameters_mut(&self) -> &mut Vec<T> {
        &mut *self.parameters.get_mut()
    }

    /// Provides immutable access to the parameters.
    pub unsafe fn get_parameters(&self) -> &Vec<T> {
        &*self.parameters.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let model: Model<f32> = Model::new();
        unsafe {
            let params = model.get_parameters();
            assert!(params.is_empty());
        }
    }

    #[test]
    fn test_new_from_parameters() {
        let model = Model::with_parameters(vec![1.0, 2.0, 3.0]);
        let expected = vec![1.0, 2.0, 3.0];
        unsafe {
            let params = model.get_parameters();
            for (a, b) in params.iter().zip(expected.iter()) {
                assert_eq!(a, b);
            }
        }
    }
}
