use crate::{ActivationFunction, Tensor};

pub struct Arch<A: ActivationFunction>{
    activation: std::marker::PhantomData<A>,
    model: Tensor,
    gradient: Tensor,
}

impl <A: ActivationFunction> Arch<A> {

    pub fn new(input_count: usize, arch_layers: &[usize]) -> Self {
        Self {
            activation: std::marker::PhantomData,
            model: Tensor::from(input_count, arch_layers),
            gradient: Tensor::from(input_count, arch_layers),
        }
    }
    
}

