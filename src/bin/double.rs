use feoho_nn::{Matrix, Result, NNET};
use rand::Rng;

fn main() -> Result<()> {
    let test_data = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
    let test_data = Matrix::from(4, 2, 3, &test_data);
    let mut rng = rand::thread_rng();
    let mut w = rng.gen::<NNET>();
    let mut b = rng.gen::<NNET>();
    let eps = 1e-3;
    let rate = 1e-3;
    for _ in 0..100 {
        let dw = (cost(&test_data, w + eps, b) - cost(&test_data, w, b)) / eps;
        let db = (cost(&test_data, w, b + eps) - cost(&test_data, w, b)) / eps;
        w -= rate * dw;
        b -= rate * db;
        println!("cost: {}, w: {}, b: {}", cost(&test_data, w, b), w, b);
    }
    Ok(())
}

fn cost(test_data: &Matrix, w: NNET, b: NNET) -> NNET {
    // y = x * w + b;
    let mut cost = 0.0;
    for row in 0..test_data.get_row_count() {
        let row = test_data.get_row_ref(row);
        let x = row[0];
        let y = x * w + b;
        let d = y - row[1];
        cost += d * d;
        // println!("actual: {}, expected: {}", y, row[1]);
    }
    cost /= test_data.get_row_count() as NNET;
    cost
}
