"""Fastest Histograms Routines in the South.

Histogramming routines accelerated with rayon.

Supported input dtypes: ``float32``, ``float64``, ``int32``, ``int64``,
``uint32``, ``uint64``.
"""

from typing import overload

import numpy as np
import numpy.typing as npt

@overload
def fixed(
    x: npt.ArrayLike,
    bins: int,
    range: tuple[float, float],
    weights: None = ...,
) -> npt.NDArray[np.intp]: ...
@overload
def fixed(
    x: npt.ArrayLike,
    bins: int,
    range: tuple[float, float],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def fixed(
    x: npt.ArrayLike,
    bins: int,
    range: tuple[float, float],
    weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.intp] | npt.NDArray[np.float64]:
    """Calculate a histogram with fixed-width bins and optional weights.

    Parameters
    ----------
    x
        Input data array. Supported dtypes: ``float32``, ``float64``,
        ``int32``, ``int64``, ``uint32``, ``uint64``.
    bins
        Number of equal-width bins.
    range
        ``(min, max)`` range of the histogram. Values outside the
        range are ignored.
    weights
        Optional array of weights, must be ``float64`` and the same
        length as *x*.

    Returns
    -------
    numpy.ndarray
        If *weights* is ``None``, a 1-D array of bin counts (shape
        ``(bins,)``).  If *weights* is provided, a 2-D array of shape
        ``(bins, 2)`` where column 0 contains weighted counts and
        column 1 contains the sum of squared weights per bin.

    Raises
    ------
    TypeError
        If the dtype of *x* is not supported.
    """
    ...

@overload
def variable(
    x: npt.ArrayLike,
    bins: npt.NDArray[np.float64],
    weights: None = ...,
) -> npt.NDArray[np.intp]: ...
@overload
def variable(
    x: npt.ArrayLike,
    bins: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def variable(
    x: npt.ArrayLike,
    bins: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.intp] | npt.NDArray[np.float64]:
    """Calculate a histogram with variable-width bins and optional weights.

    Parameters
    ----------
    x
        Input data array. Supported dtypes: ``float32``, ``float64``,
        ``int32``, ``int64``, ``uint32``, ``uint64``.
    bins
        Sorted array of bin edges (``float64``). Must have at least 2
        elements, defining ``len(bins) - 1`` bins.
    weights
        Optional array of weights, must be ``float64`` and the same
        length as *x*.

    Returns
    -------
    numpy.ndarray
        If *weights* is ``None``, a 1-D array of bin counts (shape
        ``(len(bins) - 1,)``).  If *weights* is provided, a 2-D array
        of shape ``(len(bins) - 1, 2)`` where column 0 contains
        weighted counts and column 1 contains the sum of squared
        weights per bin.

    Raises
    ------
    TypeError
        If the dtype of *x* is not supported.
    """
    ...
