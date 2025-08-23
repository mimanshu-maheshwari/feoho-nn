fn main() {
    let input = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
    let mut w = rand::random_range(0.0..1.0);
    let mut b = rand::random_range(0.0..1.0);
    let iterations = 100000;
    let learning_rate = 0.001;

    for _ in 0..iterations {
        let (dw, db) = train(&w, &b, &input);
        w -= learning_rate * dw;
        b -= learning_rate * db;
        let c = cost(&w, &b, &input);
        println!("cost: {c:.6}, w={w:.4}, b={b:.4}");
        if c < 1e-6 {
            break;
        }
    }
}
// fn sigmoid(val: &f64) -> f64 {
//     1.0 / (1.0 + (-val).exp())
// }

fn train(w: &f64, b: &f64, input: &[[f64; 2]]) -> (f64, f64) {
    let mut dw = 0.0;
    let mut db = 0.0;
    for a in input {
        let x = a[0];
        let y = a[1];
        let out = w * x + b;
        dw += 2.0 * (out - y) * x;
        db += 2.0 * (out - y);
    }
    (dw / input.len() as f64, db / input.len() as f64)
}

fn cost(w: &f64, b: &f64, input: &[[f64; 2]]) -> f64 {
    let mut cost = 0.0;
    for a in input {
        let out = a[0] * w + b;
        cost += (out - a[1]) * (out - a[1]);
    }
    cost / input.len() as f64
}
