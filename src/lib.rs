mod activation;
mod matrix;
mod arch;
mod utils;
mod tensor;

pub use matrix::Matrix;
pub use arch::Arch;
pub use tensor::Tensor;
pub use utils::{NNET, Result};
pub use activation::*;
