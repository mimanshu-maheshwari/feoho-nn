use std::{fmt, ops::Range};

use crate::{Matrix, NNET};

#[derive(Debug)]
pub struct Tensor {

    /// The number of Matrices present in each layer.
    /// Activation layer has count + 1 Matrices.
    pub(super) count: usize,

    /// ## Weight Layers: 
    pub(super) wl: Vec<Matrix>,

    /// ## Bias layers: 
    pub(super) bl: Vec<Matrix>,

    /// ## Activation layers:
    /// They are one more that the `count` variable.
    /// The first layer in al is the input
    /// Last layer is output
    pub(super) al: Vec<Matrix>,

}

impl Tensor {

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
            count,
            wl,
            bl,
            al, // activation: std::marker::PhantomData,
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

    pub fn randomize_range(&mut self, range: Range<NNET>) -> &mut Self {
        for w in &mut self.wl {
            w.random_range(range.clone());
        }
        for b in &mut self.bl {
            b.random_range(range.clone());
        }
        for a in &mut self.al {
            a.random_range(range.clone());
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

    // pub fn get_ref_bl_mut(&mut self, index: usize) -> &mut Matrix {
    //     &mut self.bl[index]
    // }

    // pub fn get_ref_al_mut(&mut self, index: usize) -> &mut Matrix {
    //     &mut self.al[index]
    // }

    // pub fn get_ref_wl_mut(&mut self, index: usize) -> &mut Matrix {
    //     &mut self.wl[index]
    // }

    // pub fn get_ref_bl(&self, index: usize) -> &Matrix {
    //     &self.bl[index]
    // }

    // pub fn get_ref_al(&self, index: usize) -> &Matrix {
    //     &self.al[index]
    // }

    // pub fn get_ref_wl(&self, index: usize) -> &Matrix {
    //     &self.wl[index]
    // }

    pub fn get_input_mut(&mut self) -> &mut Matrix {
        self.al.first_mut().unwrap()
    }

    pub fn get_output_mut(&mut self) -> &mut Matrix {
        self.al.last_mut().unwrap()
    }

    pub fn get_input(&self) -> &Matrix {
        self.al.first().unwrap()
    }

    pub fn get_output(&self) -> &Matrix {
        self.al.last().unwrap()
    }

}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        let padding = 2;
        for i in 0..self.count {
            self.wl[i].print(format!("wl{}", i).as_str(), padding);
            self.bl[i].print(format!("bl{}", i).as_str(), padding);
        }
        writeln!(f, "]")
    }
}
