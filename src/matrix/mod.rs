use std::{
    fmt::{Debug, Display},
    ops::{AddAssign, Range},
};

use rand::Rng;

use crate::{ActivationFunction, NNET};

#[derive(Debug, Default)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    stride: usize,
    data: Vec<NNET>,
}

#[allow(unused)]
impl Matrix {
    pub fn from(rows: usize, cols: usize, stride: usize, data: &[NNET]) -> Self {
        assert!(
            data.len() >= rows * cols,
            "ERROR:Size of data is not equal to given rows and cols."
        );
        let data = data.to_vec();
        Self {
            rows,
            cols,
            data,
            stride,
        }
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Self {
            rows,
            cols,
            data,
            stride: cols,
        }
    }

    pub fn identity(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];
        let min = usize::min(rows, cols);
        for i in 0..min {
            data[i * cols + i] = 1.0;
        }
        Self {
            rows,
            cols,
            data,
            stride: cols,
        }
    }

    pub fn random_range(&mut self, range: Range<NNET>) -> &mut Self {
        let mut rng = rand::thread_rng();
        for el in &mut self.data {
            *el = rng.gen_range(range.clone());
        }
        self
    }

    pub fn randomize(&mut self) -> &mut Self {
        let mut rng = rand::thread_rng();
        for el in &mut self.data {
            *el = rng.gen_range(0.0..1.0);
        }
        self
    }

    pub fn get_row_count(&self) -> usize {
        self.rows
    }

    pub fn get_col_count(&self) -> usize {
        self.cols
    }

    pub fn get_ref(&self, row: usize, col: usize) -> &NNET {
        assert!(
            row < self.rows,
            "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}",
            row,
            self.rows
        );
        assert!(
            col < self.cols,
            "ERROR: Given col {} is greater than or equal to number of cols available, i.e. {}",
            col,
            self.cols
        );
        // println!("\nindex: {}", (row * self.stride + col));
        &self.data[row * self.stride + col]
    }

    pub fn get_ref_mut(&mut self, row: usize, col: usize) -> &mut NNET {
        assert!(
            row < self.rows,
            "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}",
            row,
            self.rows
        );
        assert!(
            col < self.cols,
            "ERROR: Given col {} is greater than or equal to number of cols available, i.e. {}",
            col,
            self.cols
        );
        &mut self.data[row * self.stride + col]
    }

    pub fn get_row_ref(&self, row: usize) -> &[NNET] {
        assert!(
            row < self.rows,
            "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}",
            row,
            self.rows
        );
        let start = row * self.stride;
        &self.data[start..(start + self.cols)]
    }

    pub fn get_row_ref_mut(&mut self, row: usize) -> &mut [NNET] {
        assert!(
            row < self.rows,
            "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}",
            row,
            self.rows
        );
        let start = row * self.stride;
        &mut self.data[start..(start + self.cols)]
    }

    pub fn dot(&mut self, a: &Matrix, b: &Matrix) {
        assert_eq!(self.rows, a.rows);
        assert_eq!(self.cols, b.cols);
        assert_eq!(a.cols, b.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                *self.get_ref_mut(row, col) = 0.0;
                for k in 0..a.cols {
                    *self.get_ref_mut(row, col) += a.get_ref(row, k) * b.get_ref(k, col);
                }
            }
        }
    }

    pub fn _dot(dest: &mut Self, a: &Matrix, b: &Matrix) {
        assert_eq!(dest.rows, a.rows);
        assert_eq!(dest.cols, b.cols);
        assert_eq!(a.cols, b.rows);
        for row in 0..dest.rows {
            for col in 0..dest.cols {
                *dest.get_ref_mut(row, col) = 0.0;
                for k in 0..a.cols {
                    *dest.get_ref_mut(row, col) += a.get_ref(row, k) * b.get_ref(k, col);
                }
            }
        }
    }

    pub fn add(&mut self, src: &Matrix) {
        assert_eq!(self.rows, src.rows);
        assert_eq!(self.cols, src.cols);
        for (index, ele) in src.data.iter().enumerate() {
            self.data[index] += ele;
        }
    }

    pub fn _add(dest: &mut Matrix, src: &Matrix) {
        assert_eq!(dest.rows, src.rows);
        assert_eq!(dest.cols, src.cols);
        for (index, ele) in src.data.iter().enumerate() {
            dest.data[index] += ele;
        }
    }

    pub fn copy_from_slice(&mut self, src: &[NNET]) {
        assert_eq!(self.rows * self.cols, src.len());
        for (index, ele) in src.iter().enumerate() {
            self.data[index] = *ele;
        }
    }

    pub fn copy_from(&mut self, src: &Matrix) {
        assert_eq!(self.rows, src.rows);
        assert_eq!(self.cols, src.cols);
        for (index, ele) in src.data.iter().enumerate() {
            self.data[index] = *ele;
        }
    }

    pub fn get_data_ref(&self) -> &[NNET] {
        &self.data
    }

    pub fn get_data_ref_mut(&mut self) -> &mut [NNET] {
        &mut self.data
    }

    pub fn activate<Activation: ActivationFunction>(&mut self) {
        for x in &mut self.data {
            *x = Activation::activate(*x);
        }
    }
    pub fn fill(&mut self, val: NNET) {
        for x in &mut self.data {
            *x = val;
        }
    }
    pub fn print(&self, name: &str, padding: usize) {
        println!("{:-padding$} {}:", " ", name, padding = padding);
        for r in 0..self.rows {
            if r == 0 {
                print!("{:-padding$} ⌈", " ", padding = padding);
            } else if r == self.rows - 1 {
                print!("{:-padding$} ⌊", " ", padding = padding);
            } else {
                print!("{:-padding$} ∣", " ", padding = padding);
            }
            for c in 0..self.cols {
                // print!(f, " {:-18.16}", self.data[(r * self.cols) + c])?;
                print!(" {:-8.6}", self.get_ref(r, c));
            }
            if r == self.rows - 1 {
                print!(" ⌋");
            } else if r == 0 {
                print!(" ⌉");
            } else {
                print!(" ∣");
            }
            println!();
        }
        println!();
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        assert_eq!(
            self.cols, rhs.cols,
            "ERROR: Can't add matrix as cols size are different."
        );
        assert_eq!(
            self.rows, rhs.rows,
            "ERROR: Can't add matrix as rows size are different."
        );
        for (i, val) in rhs.data.iter().enumerate() {
            self.data[i] += val;
        }
    }
}

// impl Add<&Matrix> for &Matrix {
//     type Output = Matrix;
//     fn add(self, rhs: &Matrix) -> Self::Output {
//         assert_eq!(self.cols, rhs.cols, "ERROR: Can't add matrix as cols size are different.");
//         assert_eq!(self.rows, rhs.rows, "ERROR: Can't add matrix as rows size are different.");
//         let rows = self.rows;
//         let cols = self.cols;
//         let mut result = Matrix::zero(rows, cols);
//         //for row in 0..rows {
//         //    for col in 0..cols {
//         //        *result.get_ref_mut(row, col) = self.get_ref(row, col) + rhs.get_ref(row, col);
//         //    }
//         //}
//         for (i, (a, b)) in self.data.iter().zip(rhs.data.iter()).enumerate() {
//             result.data[i] = a + b;
//         }
//         result
//     }
//
// }
//
// impl Mul<&Matrix> for &Matrix {
//     type Output = Matrix;
//     fn mul(self, rhs: &Matrix) -> Self::Output {
//         assert_eq!(self.cols, rhs.rows, "ERROR: Can't multiply matrix as the inner size is not equal.");
//         let rows = self.rows;
//         let cols = rhs.cols;
//         let mid = self.cols;
//         let mut result = Matrix::zero(rows, cols);
//         for row in 0..rows {
//             for col in 0..cols {
//                 *result.get_ref_mut(row, col) = 0.0;
//                 for k in 0..mid {
//                     *result.get_ref_mut(row, col) += self.get_ref(row, k) * rhs.get_ref(k, col);
//                 }
//             }
//         }
//         result
//     }
// }

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "\n size: {}, row: {}, columns: {}",
            self.rows * self.cols,
            self.rows,
            self.cols
        )?;
        for r in 0..self.rows {
            if r == 0 {
                write!(f, " ⌈")?;
            } else if r == self.rows - 1 {
                write!(f, " ⌊")?;
            } else {
                write!(f, " ∣")?;
            }
            for c in 0..self.cols {
                // write!(f, " {:-18.16}", self.data[(r * self.cols) + c])?;
                write!(f, " {:-8.6}", self.get_ref(r, c))?;
            }
            if r == 0 {
                write!(f, " ⌉")?;
            } else if r == self.rows - 1 {
                write!(f, " ⌋")?;
            } else {
                write!(f, " ∣")?;
            }
            writeln!(f)?;
        }
        writeln!(f)
    }
}
