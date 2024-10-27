#[cfg(test)]
pub mod public {

    pub mod matrix {
        use feoho_nn::Matrix;


        #[test]
        fn test__zero_success__1() {
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
        fn test__zero_success__2() {
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
        fn test__from_and_get_row_success__1() {
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
        fn test__from_and_get_row_success__2() {
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
        fn test__get_ref__1(){
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
        fn test__dot__1(){}

        #[test]
        fn test__dot__2(){}

        #[test]
        fn test__add__1(){}

        #[test]
        fn test__add__2(){}
    }
}
