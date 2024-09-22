use feoho_nn::{Matrix, Result, Sigmoid};

fn main() -> Result<()> {

    let mut a = Matrix::zero(2, 2);
    a.fill(1.0);
    let mut b = Matrix::zero(2, 2);
    b.fill(1.0);
    a += &b;
    println!("{a}");
    Ok(())
}
