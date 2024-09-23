use feoho_nn::{Arch, Result, Sigmoid};

fn main() -> Result<()> {

    let arch = vec![2, 1];
    let arch: Arch<Sigmoid> = Arch::from(2, &arch);
    
    Ok(())
}
