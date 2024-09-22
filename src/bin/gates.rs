use feoho_nn::{Matrix, Result, Sigmoid, NNET, ActivationFunction};
use rand::Rng;

fn main() -> Result<()> {

    let xor_data = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        1.0, 1.0, 0.0,
    ];

    let _and_data = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
    ];

    let _or_data = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    ];

    let mut rng = rand::thread_rng();

    let test_data = Matrix::from(4, 3, &xor_data);
    let eps = 1e-2;
    let rate = 1e-1;

    let mut w1 = rng.gen_range(0.0..1.0);
    let mut w2 = rng.gen_range(0.0..1.0);
    let mut b = rng.gen_range(0.0..1.0);

    for _ in 0..100_000 {
        let dw1 = (cost(&test_data, w1 + eps, w2, b) - cost(&test_data, w1, w2, b)) / eps;
        let dw2 = (cost(&test_data, w1, w2 + eps, b) - cost(&test_data, w1, w2, b)) / eps;
        let db = (cost(&test_data, w1, w2, b + eps) - cost(&test_data, w1, w2, b)) / eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
        // println!("cost: {}", cost(&test_data, w1, w2, b));
    }
    println!("cost: {}", cost(&test_data, w1, w2, b));
    print_tests(w1, w2, b);
    Ok(())
}

fn print_tests(w1: NNET, w2: NNET, b: NNET) {
    for x in 0..2 {
        for y in 0..2 {
            let x = x as NNET;
            let y = y as NNET;
            println!("{} op {} = {}", x, y, Sigmoid::activate(x * w1 + y * w2 + b));
        }
    }
}

fn cost(test_data: &Matrix, w1: NNET, w2: NNET, b: NNET) -> NNET {
    // y = x1 * w1 + x2 * w2 + b;
    let mut cost = 0.0;
    for row in 0..test_data.get_row_count() {
        let row = test_data.get_row_ref(row);
        let x1 = row[0];
        let x2 = row[1];
        let y = Sigmoid::activate(x1 * w1 + x2 * w2 + b);
        let d = y - row[2];
        cost += d * d;
        // println!("actual: {}, expected: {}", y, row[1]);
    }
    cost /= test_data.get_row_count() as NNET;
    cost
}

/// will take model as input
/// and return gradiant
/// we will calculate the original cost
/// copy the weight and bias value so as we are working with float.
/// then we will add epsilon 
/// calculate finite difference and move this difference into gradiant 
/// and copy back the original value of weight
fn _finite_diff() {

}
/// will take model, gradient and rate as input and then 
/// subtract the gradient after multipling rate 
/// return the model
fn _learn() {}
