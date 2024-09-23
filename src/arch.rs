use crate::{ActivationFunction, Matrix};

#[derive(Debug)]
pub struct Arch<A: ActivationFunction> {
    activation: std::marker::PhantomData<A>,
    count: usize, 
    wl: Vec<Matrix>,
    bl: Vec<Matrix>,
    al: Vec<Matrix>,
}

impl <A: ActivationFunction> Arch<A> {
    pub fn from(input_count: usize, arch_layers: &[usize]) -> Self {
        let count = arch_layers.len(); 
        assert_ne!(count, 0, "ERROR: Layer count should not be zero!");
        assert_ne!(input_count, 0, "ERROR: Input count should not be zero!");
        let mut wl: Vec<Matrix> = Vec::with_capacity(count);
        let mut bl: Vec<Matrix> = Vec::with_capacity(count);
        let mut al: Vec<Matrix> = Vec::with_capacity(count + 1);
        al.push(Matrix::zero(1, input_count));
        for i in 0..count {
            al.push(Matrix::zero(1, arch_layers[i]));
            bl.push(Matrix::zero(1, arch_layers[i]));
            wl.push(Matrix::zero(al[i].get_col_count(), arch_layers[i]));
        }

        Self {
            count, wl, bl, al, activation: std::marker::PhantomData,
        }
    }

    pub fn randomize(&mut self) -> &mut Self {
        for w in &mut self.wl {
            w.randomize();
        }
        for b in &mut self.bl {
            b.randomize();
        }
        for a in &mut self.al {
            a.randomize();
        }
        self
    }
}
