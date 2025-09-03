use anyhow::Result;
use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::{Bound, *};
use rayon::prelude::*;
use std::ops::Sub;

static CURRENT_NUM_THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn calc_index<T>(value: T, xmin: T, norm: f64) -> usize
where
    T: Sub<Output = T> + Into<f64>,
{
    ((value - xmin).into() * norm) as usize
}

fn rh1(x: ArrayView1<f64>, bins: usize, range: (f64, f64)) -> Result<Array1<usize>> {
    let (xmin, xmax) = range;
    let norm = bins as f64 / (xmax - xmin);
    let chunk_size = (x.len() / CURRENT_NUM_THREADS.get().unwrap()).max(1000);
    Ok(x.as_slice()
        .ok_or(anyhow::anyhow!("Can't slice input array {x:?}"))?
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_hist = Array1::<usize>::zeros(bins);
            for &value in chunk {
                if value >= xmin && value < xmax {
                    let bin_index = calc_index(value, xmin, norm);
                    let bin_index = bin_index.min(bins - 1);
                    local_hist[bin_index] += 1;
                }
            }
            local_hist
        })
        .reduce(
            || Array1::<usize>::zeros(bins),
            |mut acc, local_hist| {
                acc += &local_hist;
                acc
            },
        ))
}

/// Fastest Histograms Routines in the South
///
/// This is a simple Python module with histogramming routines
/// accelerated with rayon.
#[pymodule]
mod fhrs {

    use super::*;

    /// Calculate a histogram
    #[pyfunction]
    #[pyo3(signature = (x, bins, range))]
    fn histogram<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        bins: usize,
        range: (f64, f64),
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let x = x.as_array();
        let h = py.detach(|| rh1(x, bins, range))?;
        Ok(h.into_pyarray(py))
    }

    #[pymodule_init]
    fn init(_: &Bound<'_, PyModule>) -> PyResult<()> {
        let nt = rayon::current_num_threads();
        CURRENT_NUM_THREADS.set(nt).unwrap();
        Ok(())
    }
}
