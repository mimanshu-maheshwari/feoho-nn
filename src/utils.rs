use std::{result, error};

pub type Result<T> = result::Result<T, Box<dyn error::Error>>;
pub type NNET = f64;
