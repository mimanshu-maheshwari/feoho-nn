use crate::{ActivationFunction, Matrix, Sigmoid, Tensor, NNET};
use std::mem;

pub struct Arch<A: ActivationFunction> {
    activation: std::marker::PhantomData<A>,
    model: Tensor,
    gradient: Tensor,
    input: Matrix,
    output: Matrix,
}

impl<A: ActivationFunction> Arch<A> {
    pub fn new(
        input_data: &[NNET],
        rows: usize,
        input_cols: usize,
        output_cols: usize,
        hidden_layers: &[usize],
    ) -> Self {
        assert_ne!(input_data.len(), 0);
        assert_ne!(rows, 0);
        assert_ne!(input_cols, 0);
        assert_ne!(output_cols, 0);

        // setup layers add the input size and output size
        let mut layers = hidden_layers.to_vec();
        layers.insert(0, input_cols);
        layers.push(output_cols);

        // copy input and output values.
        let input = Matrix::from(rows, input_cols, input_cols + output_cols, &input_data[..]);
        let output = Matrix::from(
            rows,
            output_cols,
            input_cols + output_cols,
            &input_data[input_cols..],
        );

        // create model
        let mut model = Tensor::from(&layers);
        model.randomize();

        // create gradient
        let mut gradient = Tensor::from(&layers);
        gradient.fill(0.0);


        // return Architecture for neural network
        Self {
            activation: std::marker::PhantomData,
            model,
            gradient,
            input,
            output,
        }
    }

    pub fn train(&mut self) {
        // copy row into model 
        let eps = 1e-1;
        let rate = 1e-1;
        println!("cost = {}", self.cost());
        self.finite_diff(eps);
        self.learn(rate);
        println!("cost = {}", self.cost());
    }

    pub fn forward(&mut self) {
        // create a default matrix to replace the value in matrix array
        let mut default_matrix = Matrix::default();
        for i in 0..self.model.count {

            // take the next layer and move default in its place
            let mut next_al_layer = mem::replace(self.model.get_ref_al_mut(i + 1), default_matrix);

            // forwarding logic
            next_al_layer.dot(self.model.get_ref_al(i), self.model.get_ref_wl(i));
            next_al_layer.add(self.model.get_ref_bl(i));
            next_al_layer.activate::<Sigmoid>();

            // move the next layer back to its place
            default_matrix = mem::replace(&mut self.model.al[i + 1], next_al_layer);
        }
    }

    fn cost(&mut self) -> NNET {
        let mut c = 0.0;
        let n = self.input.get_row_count();
        for i in 0..n {
            let x = self.input.get_row_ref(i);
            self.model.get_input_mut().copy_from_slice(x);
            self.forward();
            let q = self.output.get_col_count();
            let y = self.output.get_row_ref(i);
            for j in 0..q {
                let d = self.model.get_output().get_ref(0, j) - y[j];
                c = d * d;
            }
        }
        c / n as NNET
    }

    fn finite_diff(&mut self, eps: NNET) {
        let mut saved: NNET;
        let c: NNET = self.cost();
        for i in 0..self.model.count {
            // calculate for weights
            for j in 0..self.model.get_ref_wl(i).get_row_count() {
                for k in 0..self.model.get_ref_wl(i).get_col_count() {
                    // save the value as float calculation introduces error
                    saved = *self.model.get_ref_wl(i).get_ref(j, k);

                    // add epsilon value
                    *self.model.get_ref_wl_mut(i).get_ref_mut(j, k) += eps;

                    // save the calculated values in gradient
                    *self.gradient.get_ref_wl_mut(i).get_ref_mut(j, k) = (self.cost() - c) / eps; 

                    // return to the saved value.
                    *self.model.get_ref_wl_mut(i).get_ref_mut(j, k) = saved;
                }
            }

            // calculation for biases
            for j in 0..self.model.get_ref_bl(i).get_row_count() {
                for k in 0..self.model.get_ref_bl(i).get_col_count() {
                    // save the value as float calculation introduces error
                    saved = *self.model.get_ref_bl(i).get_ref(j, k);

                    // add epsilon value
                    *self.model.get_ref_bl_mut(i).get_ref_mut(j, k) += eps;

                    // save the calculated values in gradient
                    *self.gradient.get_ref_bl_mut(i).get_ref_mut(j, k) = (self.cost() - c) / eps; 

                    // return to the saved value.
                    *self.model.get_ref_bl_mut(i).get_ref_mut(j, k) = saved;
                }
            }
        }
        
    }

    fn learn(&mut self, rate: NNET) {
        for i in 0..self.model.count {

            // learn for weights
            for j in 0..self.model.get_ref_wl(i).get_row_count() {
                for k in 0..self.model.get_ref_wl(i).get_col_count() {
                    *self.model.get_ref_wl_mut(i).get_ref_mut(j, k) -= rate * self.gradient.get_ref_wl(i).get_ref(j, k);
                }
            }

            // learn for biases
            for j in 0..self.model.get_ref_bl(i).get_row_count() {
                for k in 0..self.model.get_ref_bl(i).get_col_count() {
                    *self.model.get_ref_bl_mut(i).get_ref_mut(j, k) -= rate * self.gradient.get_ref_bl(i).get_ref(j, k);
                }
            }
        }
    }

    pub fn print_gradient(&self) {
        println!("{}", self.gradient);
    }

    pub fn print_model(&self) {
        println!("{}", self.model);
    }

    pub fn print_input(&self) {
        println!("Input: {}", self.input);
    }

    pub fn print_output(&self) {
        println!("Input: {}", self.output);
    }
}
