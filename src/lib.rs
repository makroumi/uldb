pub mod storage;
pub mod index;
pub mod tx;
pub mod query;

#[cfg(not(test))]
mod python;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(not(test))]
    python::register(m)?;
    #[cfg(test)]
    let _ = m;
    Ok(())
}
