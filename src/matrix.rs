use std::fmt::Display;

use crate::NNET;

#[derive(Debug)]
pub struct Matrix {
    rows: usize, 
    cols: usize,
    data: Vec<NNET>,
}

impl Matrix {

    fn _zero(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Self { rows, cols, data}
    }

    fn _identity(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];
        let min = usize::min(rows,cols);
        for i in 0..min {
            data[i * cols + i] = 1.0;
        }
        Self { rows, cols, data}
    }

    fn _get_row_count(&self) -> usize  {
        self.rows
    }

    fn _get_col_count(&self) -> usize  {
        self.cols
    }

    fn _get_ref(&self, row: usize, col: usize) -> &NNET {
        assert!(row < self.rows, "ERROR: Given row {} is greater than number of rows available, i.e. {}", row, self.rows);
        assert!(col < self.cols, "ERROR: Given col {} is greater than number of cols available, i.e. {}", col, self.cols);
        &self.data[row * self.cols + col]
    }

    fn _get_ref_mut(&mut self, row: usize, col: usize) -> &mut NNET {
        assert!(row < self.rows, "ERROR: Given row {} is greater than number of rows available, i.e. {}", row, self.rows);
        assert!(col < self.cols, "ERROR: Given col {} is greater than number of cols available, i.e. {}", col, self.cols);
        &mut self.data[row * self.cols + col]
    }

    fn _get_row_ref(&self, row: usize) -> &[NNET] {
        assert!(row < self.rows, "ERROR: Given row {} is greater than number of rows available, i.e. {}", row, self.rows);
        let start = row * self.cols;
        &self.data[start..(start + self.cols)]
    }

    fn _get_row_ref_mut(&mut self, row: usize) -> &mut [NNET] {
        assert!(row < self.rows, "ERROR: Given row {} is greater than number of rows available, i.e. {}", row, self.rows);
        let start = row * self.cols;
        &mut self.data[start..(start + self.cols)]
    }

}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

