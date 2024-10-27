use feoho_nn::{Arch, Result, Sigmoid};

fn main() -> Result<()> {
    let test_data = [
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    ];
    let arch_layers = vec![1];
    let mut arch: Arch<Sigmoid> = Arch::new(&test_data, 4, 2, 1, &arch_layers);

    arch.print_model();
    arch.print_given_input();
    arch.print_given_output();
    arch.train();
    arch.print_model();
    arch._check_model();

    Ok(())
}
