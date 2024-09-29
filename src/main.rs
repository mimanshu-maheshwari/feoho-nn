use feoho_nn::{Matrix, Result, Sigmoid, Tensor};

fn main() -> Result<()> {

    let test_data = vec![
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    ];
    let test_mat_input = Matrix::from(4, 2, 3, &test_data[0..]);
    let test_mat_output = Matrix::from(4, 1, 3, &test_data[2..]);
    test_mat_input.print("test_mat_input", 4);
    test_mat_output.print("test_mat_output", 4);
    let arch_layers = vec![2, 1];
    let tensor = Tensor::from(2, &arch_layers);
    println!("Tensor: {tensor}");
    // println!("Input: {test_mat_input}");
    // println!("output: {test_mat_output}");

    
    Ok(())
}
