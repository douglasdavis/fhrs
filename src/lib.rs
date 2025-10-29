use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::{Bound, *};
use rayon::prelude::*;

static CURRENT_NUM_THREADS: std::sync::OnceLock<usize> =
    std::sync::OnceLock::new();

fn rh1(
    x: ArrayView1<f64>,
    bins: usize,
    range: (f64, f64),
) -> Result<Array1<usize>> {
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
                    let bin_index = ((value - xmin) * norm) as usize;
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

fn rh1v(x: ArrayView1<f64>, edges: ArrayView1<f64>) -> Result<Array1<usize>> {
    anyhow::ensure!(
        edges.len() >= 2,
        "edges array must have at least 2 elements"
    );

    let nbins = edges.len() - 1;
    let chunk_size = (x.len() / CURRENT_NUM_THREADS.get().unwrap()).max(1000);

    let edges_slice = edges
        .as_slice()
        .ok_or(anyhow::anyhow!("Can't slice edges array"))?;

    Ok(x.as_slice()
        .ok_or(anyhow::anyhow!("Can't slice input array {x:?}"))?
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_hist = Array1::<usize>::zeros(nbins);
            for &value in chunk {
                // binary search for bin index
                match edges_slice.binary_search_by(|edge| {
                    edge.partial_cmp(&value)
                        .unwrap_or(std::cmp::Ordering::Less)
                }) {
                    // Ok for exact index; use left edge
                    Ok(idx) => {
                        if idx < nbins {
                            local_hist[idx] += 1;
                        }
                    }
                    // Err for not found index, gives index of 1
                    // higher (see
                    // binary_search_by docs)
                    Err(idx) => {
                        // between bins; use
                        if idx > 0 && idx <= nbins {
                            local_hist[idx - 1] += 1;
                        }
                    }
                }
            }
            local_hist
        })
        .reduce(
            || Array1::<usize>::zeros(nbins),
            |mut acc, local_hist| {
                acc += &local_hist;
                acc
            },
        ))
}

fn rh1vw(
    x: ArrayView1<f64>,
    w: ArrayView1<f64>,
    edges: ArrayView1<f64>,
) -> Result<Array2<f64>> {
    anyhow::ensure!(
        x.len() == w.len(),
        "x and w arrays must have the same length"
    );
    anyhow::ensure!(
        edges.len() >= 2,
        "edges array must have at least 2 elements"
    );

    let nbins = edges.len() - 1;
    let chunk_size = (x.len() / CURRENT_NUM_THREADS.get().unwrap()).max(1000);

    let edges_slice = edges
        .as_slice()
        .ok_or(anyhow::anyhow!("Can't slice edges array"))?;
    let x_slice = x
        .as_slice()
        .ok_or(anyhow::anyhow!("Can't slice input array x"))?;
    let w_slice = w
        .as_slice()
        .ok_or(anyhow::anyhow!("Can't slice input array w"))?;

    // we're going to return a 2D array where the 0th dimension
    // is size 2; the first dimension is for the counts and
    // the second is for the variances.
    Ok(x_slice
        .par_chunks(chunk_size)
        .zip_eq(w_slice.par_chunks(chunk_size))
        .map(|(x_chunk, w_chunk)| {
            let mut local_hist = Array2::<f64>::zeros((nbins, 2));
            for (&value, &weight) in x_chunk.iter().zip(w_chunk.iter()) {
                match edges_slice.binary_search_by(|edge| {
                    edge.partial_cmp(&value)
                        .unwrap_or(std::cmp::Ordering::Less)
                }) {
                    Ok(idx) => {
                        if idx < nbins {
                            local_hist[[idx, 0]] += weight;
                            local_hist[[idx, 1]] += weight * weight;
                        }
                    }
                    Err(idx) => {
                        if idx > 0 && idx <= nbins {
                            local_hist[[idx - 1, 0]] += weight;
                            local_hist[[idx - 1, 1]] += weight * weight;
                        }
                    }
                }
            }
            local_hist
        })
        .reduce(
            || Array2::<f64>::zeros((nbins, 2)),
            |mut acc, local_hist| {
                acc += &local_hist;
                acc
            },
        ))
}

fn rh1w(
    x: ArrayView1<f64>,
    w: ArrayView1<f64>,
    bins: usize,
    range: (f64, f64),
) -> Result<Array2<f64>> {
    anyhow::ensure!(
        x.len() == w.len(),
        "x and w arrays must have the same length"
    );

    let (xmin, xmax) = range;
    let norm = bins as f64 / (xmax - xmin);
    let chunk_size = (x.len() / CURRENT_NUM_THREADS.get().unwrap()).max(1000);

    let x_slice = x
        .as_slice()
        .ok_or(anyhow::anyhow!("Can't slice input array x"))?;
    let w_slice = w
        .as_slice()
        .ok_or(anyhow::anyhow!("Can't slice input array w"))?;

    Ok(x_slice
        .par_chunks(chunk_size)
        .zip_eq(w_slice.par_chunks(chunk_size))
        .map(|(x_chunk, w_chunk)| {
            let mut local_hist = Array2::<f64>::zeros((bins, 2));
            for (&value, &weight) in x_chunk.iter().zip(w_chunk.iter()) {
                if value >= xmin && value < xmax {
                    let bin_index = ((value - xmin) * norm) as usize;
                    local_hist[[bin_index, 0]] += weight;
                    local_hist[[bin_index, 1]] += weight * weight;
                }
            }
            local_hist
        })
        .reduce(
            || Array2::<f64>::zeros((bins, 2)),
            |mut acc, local_hist| {
                acc += &local_hist;
                acc
            },
        ))
}

/// Fastest Histograms Routines in the South
///
/// This is a simple Python module with histogramming
/// routines accelerated with rayon.
#[pymodule]
mod fhrs {

    use super::*;

    /// Calculate a histogram with fixed-width bins and
    /// optional weights
    #[pyfunction]
    #[pyo3(signature = (x, bins, range, weights=None))]
    fn histogram_fixed<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        bins: usize,
        range: (f64, f64),
        weights: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Py<PyAny>> {
        let x = x.as_array();

        match weights {
            Some(w) => {
                let w = w.as_array();
                let h = py.detach(|| rh1w(x, w, bins, range))?;
                Ok(h.into_pyarray(py).into())
            }
            _ => {
                let h = py.detach(|| rh1(x, bins, range))?;
                Ok(h.into_pyarray(py).into())
            }
        }
    }

    /// Calculate a histogram with variable-width bins and
    /// optional weights
    #[pyfunction]
    #[pyo3(signature = (x, bins, weights=None))]
    fn histogram_variable<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        bins: PyReadonlyArray1<'py, f64>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Py<PyAny>> {
        let x = x.as_array();
        let edges = bins.as_array();

        match weights {
            Some(w) => {
                let w = w.as_array();
                let h = py.detach(|| rh1vw(x, w, edges))?;
                Ok(h.into_pyarray(py).into())
            }
            _ => {
                let h = py.detach(|| rh1v(x, edges))?;
                Ok(h.into_pyarray(py).into())
            }
        }
    }

    #[pymodule_init]
    fn init(_: &Bound<'_, PyModule>) -> PyResult<()> {
        let nt = rayon::current_num_threads();
        CURRENT_NUM_THREADS.set(nt).unwrap();
        Ok(())
    }
}
