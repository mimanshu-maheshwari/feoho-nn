use crate::ActivationFunction;

pub struct Arch<A: ActivationFunction> {
    activation: std::marker::PhantomData<A>,
}
