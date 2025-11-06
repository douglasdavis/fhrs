use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::ToPrimitive;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::{Bound, *};
use rayon::prelude::*;

static CURRENT_NUM_THREADS: std::sync::OnceLock<usize> =
    std::sync::OnceLock::new();

fn rh1<T>(
    x: ArrayView1<T>,
    bins: usize,
    range: (f64, f64),
) -> Result<Array1<usize>>
where
    T: Copy + ToPrimitive + Send + Sync,
{
    let (xmin, xmax) = range;
    let norm = bins as f64 / (xmax - xmin);
    let chunk_size = (x.len() / CURRENT_NUM_THREADS.get().unwrap()).max(1000);
    Ok(x.as_slice()
        .ok_or(anyhow::anyhow!("Can't slice input array"))?
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_hist = Array1::<usize>::zeros(bins);
            for &value in chunk {
                let value_f64 =
                    value.to_f64().expect("numeric conversion failed");
                if value_f64 >= xmin && value_f64 < xmax {
                    let bin_index = ((value_f64 - xmin) * norm) as usize;
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

fn rh1w<T>(
    x: ArrayView1<T>,
    w: ArrayView1<f64>,
    bins: usize,
    range: (f64, f64),
) -> Result<Array2<f64>>
where
    T: Copy + ToPrimitive + Send + Sync,
{
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
                let value_f64 =
                    value.to_f64().expect("numeric conversion failed");
                if value_f64 >= xmin && value_f64 < xmax {
                    let bin_index = ((value_f64 - xmin) * norm) as usize;
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
#[pymodule(gil_used = false)]
mod fhrs {

    use numpy::PyArrayMethods;

    use super::*;

    macro_rules! dispatch_histogram_fixed {
        ($py:expr, $x:expr, $bins:expr, $range:expr, $weights:expr, $($dtype:ty),+) => {
            $(
                if let Ok(arr) = $x.cast::<numpy::PyArray1<$dtype>>() {
                    let x_readonly = arr.readonly();
                    let x_view = x_readonly.as_array();

                    return match $weights {
                        Some(w) => {
                            let w_view = w.as_array();
                            let h = $py.detach(|| rh1w(x_view, w_view, $bins, $range))?;
                            Ok(h.into_pyarray($py).into())
                        }
                        None => {
                            let h = $py.detach(|| rh1(x_view, $bins, $range))?;
                            Ok(h.into_pyarray($py).into())
                        }
                    };
                }
            )+
        };
    }

    /// Calculate a histogram with fixed-width bins and
    /// optional weights
    #[pyfunction]
    #[pyo3(signature = (x, bins, range, weights=None))]
    fn histogram_fixed<'py>(
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        bins: usize,
        range: (f64, f64),
        weights: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Py<PyAny>> {
        dispatch_histogram_fixed!(
            py, x, bins, range, weights, f32, f64, i32, i64, u32, u64
        );

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported dtype. Supported types: f32, f64, i32, i64, u32, u64",
        ))
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
