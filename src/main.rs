use feoho_nn::{Matrix, Result};

fn main() -> Result<()> {
    let test_data = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
    let test_data = Matrix::from(4, 2, &test_data);
    let identity = Matrix::identity(2, 4);
    let zero = Matrix::zero(3, 3);
    println!("Test data: {}", test_data);
    println!("Identity: {}", identity);
    println!("Zero: {}", zero);
    println!("test * identity: {}", &test_data * &identity);
    println!("Identity * test: {}", &identity * &test_data);

    Ok(())
}
