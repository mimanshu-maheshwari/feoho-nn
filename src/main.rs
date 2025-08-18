use rand::Rng;

fn main() {
    let input = [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ];
    // formula 
    let mut rng = rand::thread_rng();
    let mut w = rng.gen_range(0.0..1.0);
    let mut b = rng.gen_range(0.0..1.0);
    let mut cost = 0.0;

    let learning_rate = 0.01;
    let iterations = 50;

    for _ in 0..iterations {
        let cost = cost(&w, &b, &input);
    }

}

fn cost(w: &f64, b: &f64, input: &[[f64; 2]]) -> f64 {
    for a in input {
        let out = a[0] * w + b;
        cost += (a[1] - out) * (a[1] - out);
    }
    cost /= input.len() as f64;
    cost
}
