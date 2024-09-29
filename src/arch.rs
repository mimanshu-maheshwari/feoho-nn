use crate::{activation, ActivationFunction, Matrix, Sigmoid, Tensor, NNET};

pub struct Arch<A: ActivationFunction>{
    activation: std::marker::PhantomData<A>,
    model: Tensor,
    gradient: Tensor,
    input: Matrix,
    output: Matrix,
}

impl <A: ActivationFunction> Arch<A> {

    pub fn new(input_data: &[NNET], rows: usize, input_cols: usize, output_cols: usize, hidden_layers: &[usize]) -> Self {
        let mut layers = hidden_layers.to_vec();
        layers.insert(0, input_cols);
        layers.push(output_cols);
        let mut model = Tensor::from(&layers);
        model.randomize();
        let mut gradient= Tensor::from(&layers);
        gradient.fill(0.0);
        let input = Matrix::from(rows, input_cols, input_cols + output_cols, &input_data[..]);
        let output = Matrix::from(rows, output_cols, input_cols + output_cols, &input_data[input_cols..]);
        Self {
            activation: std::marker::PhantomData,
            model, gradient,
            input, output,
        }
    }

    pub fn forward(&mut self) {
        for i in 0..self.model.count {
            let curr_al_layer = &self.model.get_ref_al(i);
            let next_al_layer = &mut self.model.get_ref_al_mut(i + 1);
            Matrix::dot(next_al_layer, curr_al_layer, &self.model.get_ref_wl(i));
            Matrix::add(next_al_layer, &self.model.get_ref_bl(i));
            next_al_layer.activate::<Sigmoid>();
        }
    }
    
    fn cost(&mut self){}
    fn finite_diff(&mut self) { }
    fn learn(&mut self) { }
    pub fn print_model(&self) { println!("{}", self.model);}
    pub fn print_input(&self) { println!("Input: {}", self.input);}
    pub fn print_output(&self) { println!("Input: {}", self.output);}
}

