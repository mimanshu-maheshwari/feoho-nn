use std::{error, result};

/// Generic result type.
pub type Result<T> = result::Result<T, Box<dyn error::Error>>;

/// Neural Network Element Type.
pub type NNET = f64;
