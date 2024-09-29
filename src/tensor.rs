use std::fmt;

use crate::{Matrix, NNET};

#[derive(Debug)]
pub struct Tensor { // <A: ActivationFunction> {
    // activation: std::marker::PhantomData<A>,
    pub (super) count: usize, 
    pub (super) wl: Vec<Matrix>,
    pub (super) bl: Vec<Matrix>,
    pub (super) al: Vec<Matrix>,
}

impl Tensor { //  <A: ActivationFunction> Tensor<A> {
    pub fn from(layers: &[usize]) -> Self {
        let count = layers.len() - 1; 
        assert_ne!(count, 0, "ERROR: Layer count should not be zero!");
        let mut wl: Vec<Matrix> = Vec::with_capacity(count);
        let mut bl: Vec<Matrix> = Vec::with_capacity(count);
        let mut al: Vec<Matrix> = Vec::with_capacity(count + 1);
        al.push(Matrix::zero(1, layers[0]));
        for i in 1..=count {
            al.push(Matrix::zero(1, layers[i]));
            bl.push(Matrix::zero(1, layers[i]));
            wl.push(Matrix::zero(al[i - 1].get_col_count(), layers[i]));
        }

        Self {
            count, wl, bl, al , // activation: std::marker::PhantomData,
        }
    }

    pub fn fill(&mut self, val: NNET) -> &mut Self {
        for w in &mut self.wl {
            w.fill(val);
        }
        for b in &mut self.bl {
            b.fill(val);
        }
        for a in &mut self.al {
            a.fill(val);
        }
        self
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

    pub fn get_ref_bl_mut(&mut self, index: usize) -> &mut Matrix {
        &mut self.bl[index]
    }
    pub fn get_ref_al_mut(&mut self, index: usize) -> &mut Matrix {
        &mut self.al[index]
    }
    pub fn get_ref_wl_mut(&mut self, index: usize) -> &mut Matrix {
        &mut self.wl[index]
    }

    pub fn get_ref_bl(&self, index: usize) -> &Matrix {
        &self.bl[index]
    }
    pub fn get_ref_al(&self, index: usize) -> &Matrix {
        &self.al[index]
    }
    pub fn get_ref_wl(&self, index: usize) -> &Matrix {
        &self.wl[index]
    }

}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.count {
            self.wl[i].print(format!("wl{}", i).as_str(), 4);
            self.bl[i].print(format!("bl{}", i).as_str(), 4);
        }
        writeln!(f, "]")
    }
}
