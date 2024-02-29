#[derive(Debug)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T: Copy + Clone> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        if shape.iter().product::<usize>() != data.len() {
            panic!("Data does not match tensor shape.");
        }
        Tensor { shape, data }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn get_data(&self) -> Vec<T> {
        self.data.clone()
    }
}
