use std::{fmt::Display, ops::Mul};

use crate::NNET;

#[derive(Debug)]
pub struct Matrix {
    rows: usize, 
    cols: usize,
    data: Vec<NNET>,
}

#[allow(unused)]
impl Matrix {

    pub fn from(rows: usize,cols: usize, data: &[NNET]) -> Self {
        assert_eq!(data.len(), rows * cols, "ERROR:Size of data is not equal to given rows and cols.");
        let data = data.to_vec();
        Self {rows, cols, data}
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Self { rows, cols, data}
    }

    pub fn identity(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];
        let min = usize::min(rows,cols);
        for i in 0..min {
            data[i * cols + i] = 1.0;
        }
        Self { rows, cols, data}
    }

    fn get_row_count(&self) -> usize  {
        self.rows
    }

    fn get_col_count(&self) -> usize  {
        self.cols
    }

    fn get_ref(&self, row: usize, col: usize) -> &NNET {
        assert!(row < self.rows, "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}", row, self.rows);
        assert!(col < self.cols, "ERROR: Given col {} is greater than or equal to number of cols available, i.e. {}", col, self.cols);
        &self.data[row * self.cols + col]
    }

    fn get_ref_mut(&mut self, row: usize, col: usize) -> &mut NNET {
        assert!(row < self.rows, "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}", row, self.rows);
        assert!(col < self.cols, "ERROR: Given col {} is greater than or equal to number of cols available, i.e. {}", col, self.cols);
        &mut self.data[row * self.cols + col]
    }

    fn get_row_ref(&self, row: usize) -> &[NNET] {
        assert!(row < self.rows, "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}", row, self.rows);
        let start = row * self.cols;
        &self.data[start..(start + self.cols)]
    }

    fn get_row_ref_mut(&mut self, row: usize) -> &mut [NNET] {
        assert!(row < self.rows, "ERROR: Given row {} is greater than or equal to number of rows available, i.e. {}", row, self.rows);
        let start = row * self.cols;
        &mut self.data[start..(start + self.cols)]
    }

    pub fn get_data_ref(&self) -> &[NNET] {
        &self.data
    }

    pub fn get_data_ref_mut(&mut self) -> &mut [NNET] {
        &mut self.data
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "ERROR: Can't multiply matrix as the inner size is not equal.");
        let rows = self.rows;
        let cols = rhs.cols;
        let mid = self.cols;
        let mut result = Matrix::zero(rows, cols);
        for row in 0..rows {
            for col in 0..cols {
                *result.get_ref_mut(row, col) = 0.0;
                for k in 0..mid {
                    *result.get_ref_mut(row, col) += self.get_ref(row, k) * rhs.get_ref(k, col);
                }
            }
        }
        result
    }
}

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
                write!(f, " {}", self.data[(r * self.cols) + c])?;
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
        writeln!(f, "")
    }
}

