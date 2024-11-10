#[cfg(test)]
pub mod public {

    pub mod matrix {
        use feoho_nn::Matrix;


        #[test]
        fn zero_1() {
            let rows = 2; 
            let cols = 3; 
            let matrix = Matrix::zero(rows, cols);
            assert_eq!(matrix.get_row_count(), rows);
            assert_eq!(matrix.get_col_count(), cols);
            assert_eq!(matrix.get_data_ref().len(), rows * cols);
            for ele in matrix.get_data_ref() {
                assert_eq!(*ele, 0.0);
            }
        }

        #[test]
        fn zero_2() {
            let rows = 3; 
            let cols = 2; 
            let matrix = Matrix::zero(rows, cols);
            assert_eq!(matrix.get_row_count(), rows);
            assert_eq!(matrix.get_col_count(), cols);
            assert_eq!(matrix.get_data_ref().len(), rows * cols);
            for ele in matrix.get_data_ref() {
                assert_eq!(*ele, 0.0);
            }
        }

        #[test]
        fn from_and_get_row_1() {
            let data = [
                2.0, 3.0, 4.0,
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0,
                5.0, 6.0, 7.0,
                6.0, 7.0, 8.0,
            ];
            let rows = 5;
            let input_cols = 2; 
            let output_cols = 1;
            let input = Matrix::from(rows, input_cols, input_cols + output_cols, &data);
            let output = Matrix::from(rows, output_cols, input_cols + output_cols, &data[input_cols..]);


            assert_eq!(input.get_row_count(), rows);
            assert_eq!(input.get_col_count(), input_cols);
            for i in 0..rows {
                let input_row = input.get_row_ref(i);
                let start = i * (input_cols + output_cols);
                let expected_row = &data[start..(start + input_cols)];
                assert_eq!(expected_row.len(), input_row.len());
                for (a, e) in expected_row.iter().zip(input_row) {
                    assert_eq!(*a, *e);
                }
            }

            assert_eq!(output.get_row_count(), rows);
            assert_eq!(output.get_col_count(), output_cols);
            for i in 0..rows {
                let output_row = output.get_row_ref(i);
                let start = i * (input_cols + output_cols) + input_cols;
                let expected_row = &data[start..(start + output_cols)];
                assert_eq!(expected_row.len(), output_row.len());
                for (a, e) in expected_row.iter().zip(output_row) {
                    assert_eq!(*a, *e);
                }
            }
        }

        #[test]
        fn from_and_get_row_2() {
            let data = [
                2.0, 3.0, 4.0,
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0,
                5.0, 6.0, 7.0,
            ];
            let rows = 4;
            let input_cols = 2; 
            let output_cols = 1;
            let input = Matrix::from(rows, input_cols, input_cols + output_cols, &data);
            let output = Matrix::from(rows, output_cols, input_cols + output_cols, &data[input_cols..]);


            assert_eq!(input.get_row_count(), rows);
            assert_eq!(input.get_col_count(), input_cols);
            for i in 0..rows {
                let input_row = input.get_row_ref(i);
                let start = i * (input_cols + output_cols);
                let expected_row = &data[start..(start + input_cols)];
                assert_eq!(expected_row.len(), input_row.len());
                for (a, e) in expected_row.iter().zip(input_row) {
                    assert_eq!(*a, *e);
                }
            }

            assert_eq!(output.get_row_count(), rows);
            assert_eq!(output.get_col_count(), output_cols);
            for i in 0..rows {
                let output_row = output.get_row_ref(i);
                let start = i * (input_cols + output_cols) + input_cols;
                let expected_row = &data[start..(start + output_cols)];
                assert_eq!(expected_row.len(), output_row.len());
                for (a, e) in expected_row.iter().zip(output_row) {
                    assert_eq!(*a, *e);
                }
            }
        }

        #[test]
        fn get_ref_1(){
            let data = [
                2.0, 3.0, 4.0,
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0,
                5.0, 6.0, 7.0,
            ];
            let expected_input = [
                2.0, 3.0, 
                3.0, 4.0, 
                4.0, 5.0, 
                5.0, 6.0, 
            ];
            let expected_output = [
                4.0,
                5.0,
                6.0,
                7.0,
            ];
            let rows = 4; 
            let input_cols = 2; 
            let output_cols = 1;
            let actual_input = Matrix::from(rows, input_cols, input_cols + output_cols, &data);
            let actual_output = Matrix::from(rows, output_cols, input_cols + output_cols, &data[input_cols..]);
            for row in 0..rows {
                for col in 0..input_cols {
                    assert_eq!(
                        *actual_input.get_ref(row, col),
                        expected_input[row * input_cols + col]
                    );
                }
                for col in 0..output_cols {
                    assert_eq!(
                        *actual_output.get_ref(row, col),
                        expected_output[row * output_cols + col]
                    );
                }
            }
        }
        #[test]
        fn dot_1(){

            // setup
            let a_rows = 3;
            let a_cols = 2; 
            let b_rows = 2;
            let b_cols = 3;

            let a = Matrix::from(a_rows, a_cols, a_cols, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let b = Matrix::from(b_rows, b_cols, b_cols, &[1.0, 3.0, 2.0, 5.0, 7.0, 6.0]);

            let expected_output = Matrix::from(a_rows, b_cols, b_cols, &[11.0, 17.0, 14.0, 23.0, 37.0, 30.0, 35.0, 57.0, 46.0]);
            let mut actual_output = Matrix::zero(a_rows, b_cols);
            actual_output.randomize();

            // action
            actual_output.dot(&a, &b);

            // validation
            assert_eq!(expected_output.get_row_count(), actual_output.get_row_count());
            assert_eq!(expected_output.get_col_count(), actual_output.get_col_count());

            for (ac, ex) in actual_output.get_data_ref().iter().zip(expected_output.get_data_ref())  {
                assert_eq!(ac, ex);
            }
            
        }

        #[test]
        fn dot_2(){

            // setup
            let a_rows = 5;
            let a_cols = 8; 
            let b_rows = 8;
            let b_cols = 3;

            let a = Matrix::from(a_rows, a_cols, a_cols, &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0,
                9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0,
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 
                9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0,
            ]);
            let b = Matrix::from(b_rows, b_cols, b_cols, &[
                0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 0.0, 0.0, 9.0, 8.0, 7.0
            ]);

            let expected_output = Matrix::from(a_rows, b_cols, b_cols, &[
                216.0, 160.0, 161.0, 182.0, 218.0,
                217.0, 124.0, 112.0, 107.0, 243.0,
                193.0, 173.0, 105.0, 152.0, 142.0,
            ]);
            let mut actual_output = Matrix::zero(a_rows, b_cols);
            actual_output.randomize();

            // action
            actual_output.dot(&a, &b);

            // validation
            assert_eq!(expected_output.get_row_count(), actual_output.get_row_count());
            assert_eq!(expected_output.get_col_count(), actual_output.get_col_count());

            for (ac, ex) in actual_output.get_data_ref().iter().zip(expected_output.get_data_ref())  {
                assert_eq!(ac, ex);
            }
        }

        #[test]
        fn add_1(){

            // setup
            let a_rows = 3;
            let a_cols = 2; 

            let a = Matrix::from(a_rows, a_cols, a_cols, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

            let expected_output = Matrix::from(a_rows, a_cols, a_cols, &[6.0,6.0,6.0,6.0,6.0,6.0,]);
            let mut actual_output = Matrix::from(a_rows, a_cols, a_cols, &[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);

            // action
            actual_output.add(&a);

            // validation
            assert_eq!(expected_output.get_row_count(), actual_output.get_row_count());
            assert_eq!(expected_output.get_col_count(), actual_output.get_col_count());

            for (ac, ex) in actual_output.get_data_ref().iter().zip(expected_output.get_data_ref())  {
                assert_eq!(ac, ex);
            }
        }

    }
}
