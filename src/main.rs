use feoho_nn::{Arch, Result, Sigmoid};

fn main() -> Result<()> {
    let test_data = vec![
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    ];
    let arch_layers = vec![2];
    let mut arch: Arch<Sigmoid> = Arch::new(&test_data, 4, 2, 1, &arch_layers);

    // arch.print_model();
    // arch.print_input();
    // arch.print_output();
    arch.train();

    Ok(())
}
