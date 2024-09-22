use crate::NNET;

pub trait ActivationFunction {
    fn activate(x: NNET) -> NNET;
    fn derivative(x: NNET) -> NNET;
}

/// Sigmoid Activation Function
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn activate(x: NNET) -> NNET {
        1.0 / (1.0 + (-x).exp())
    }
    fn derivative(x: NNET) -> NNET {
        let sigmoid_x = Self::activate(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }
}


/// ReLU
pub struct ReLU;
impl ActivationFunction for ReLU {
    fn activate(x: NNET) -> NNET {
        x.max(0.0)
    }
    fn derivative(x: NNET) -> NNET {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

/// Tanh
pub struct Tanh;
impl ActivationFunction for Tanh {
    fn activate(x: NNET) -> NNET {
        x.tanh()
    }
    fn derivative(x: NNET) -> NNET {
        1.0 - x.tanh().powi(2)
    }
}
/// Leaky ReLU
pub struct LeakyReLU;
impl ActivationFunction for LeakyReLU {
    fn activate(x: NNET) -> NNET {
        if x > 0.0 { x } else { 0.01 * x }
    }
    fn derivative(x: NNET) -> NNET {
        if x > 0.0 { 1.0 } else { 0.01 }
    }
}

/// Softplus
pub struct Softplus;
impl ActivationFunction for Softplus {
    fn activate(x: NNET) -> NNET {
        (1.0 + x.exp()).ln()
    }
    fn derivative(x: NNET) -> NNET {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Swish
pub struct Swish;
impl ActivationFunction for Swish {
    fn activate(x: NNET) -> NNET {
        x / (1.0 + (-x).exp())  // x * sigmoid(x)
    }
    fn derivative(x: NNET) -> NNET {
        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
        sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x)
    }
}
