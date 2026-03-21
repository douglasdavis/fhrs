import fhrs
import numpy as np
import pygram11
import pytest
from numpy.testing import assert_allclose, assert_equal

SUPPORTED_DTYPES = [np.float64, np.float32, np.int64, np.int32, np.uint64, np.uint32]
FLOAT_DTYPES = [np.float64, np.float32]
INT_DTYPES = [np.int64, np.int32, np.uint64, np.uint32]


# -- fixed-width, unweighted --------------------------------------------------


def test_1d_fixed():
    x = np.random.randn(2000)
    bins = 8
    range = (-2.8, 2.8)
    a = np.histogram(x, bins=bins, range=range)[0]
    b = fhrs.fixed(x, bins=bins, range=range)
    assert_equal(a, b)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_1d_fixed_dtype(dtype: np.dtype):
    rng = np.random.default_rng(42)
    if np.issubdtype(dtype, np.integer):
        x = rng.integers(-100, 100, size=5000).astype(dtype)
        range = (-100.0, 100.0)
    else:
        x = rng.standard_normal(5000).astype(dtype)
        range = (-3.0, 3.0)
    bins = 10
    expected = np.histogram(x.astype(np.float64), bins=bins, range=range)[0]
    result = fhrs.fixed(x, bins=bins, range=range)
    assert_equal(expected, result)


def test_1d_fixed_empty():
    x = np.array([], dtype=np.float64)
    result = fhrs.fixed(x, bins=5, range=(0.0, 1.0))
    assert_equal(result, np.zeros(5, dtype=np.uintp))


def test_1d_fixed_all_out_of_range():
    x = np.array([10.0, 20.0, 30.0])
    result = fhrs.fixed(x, bins=5, range=(0.0, 1.0))
    assert_equal(result, np.zeros(5, dtype=np.uintp))


def test_1d_fixed_single_element():
    x = np.array([0.5])
    result = fhrs.fixed(x, bins=10, range=(0.0, 1.0))
    expected = np.histogram(x, bins=10, range=(0.0, 1.0))[0]
    assert_equal(result, expected)


# -- fixed-width, weighted ----------------------------------------------------


def test_1d_fixed_weighted():
    x = np.random.randn(2000)
    w = np.ones_like(x) * 0.5
    bins = 8
    range = (-2.85, 2.85)
    a = np.histogram(x, bins=bins, range=range, weights=w)[0]
    b = fhrs.fixed(x, bins=bins, range=range, weights=w)
    assert_allclose(a, b[:, 0])
    c = pygram11.fix1d(x, bins=bins, range=range, weights=w)
    assert_allclose(c[1], np.sqrt(b[:, 1]))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_1d_fixed_weighted_dtype(dtype: np.dtype):
    rng = np.random.default_rng(99)
    if np.issubdtype(dtype, np.integer):
        x = rng.integers(-50, 50, size=3000).astype(dtype)
        range = (-50.0, 50.0)
    else:
        x = rng.standard_normal(3000).astype(dtype)
        range = (-3.0, 3.0)
    w = rng.uniform(0.1, 2.0, size=len(x))
    bins = 12
    expected = np.histogram(x.astype(np.float64), bins=bins, range=range, weights=w)[0]
    result = fhrs.fixed(x, bins=bins, range=range, weights=w)
    assert_allclose(expected, result[:, 0], atol=1e-10)


def test_1d_fixed_weighted_variance():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(5000)
    w = rng.uniform(0.5, 1.5, size=len(x))
    bins = 15
    range = (-3.0, 3.0)
    result = fhrs.fixed(x, bins=bins, range=range, weights=w)
    ref = pygram11.fix1d(x, bins=bins, range=range, weights=w)
    assert_allclose(ref[0], result[:, 0])
    assert_allclose(ref[1], np.sqrt(result[:, 1]))


# -- variable-width, unweighted -----------------------------------------------


def test_1d_variable():
    x = np.random.randn(2000)
    bins = np.array([-3.5, -3.0, -2.0, 0, 1.5, 2.5, 3.5])
    a = np.histogram(x, bins=bins)[0]
    b = fhrs.variable(x, bins=bins)
    assert_equal(a, b)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_1d_variable_dtype(dtype: np.dtype):
    rng = np.random.default_rng(42)
    edges = np.array([-100.0, -50.0, -10.0, 0.0, 10.0, 50.0, 100.0])
    if np.issubdtype(dtype, np.integer):
        x = rng.integers(-100, 100, size=5000).astype(dtype)
    else:
        x = (rng.standard_normal(5000) * 40).astype(dtype)
    expected = np.histogram(x.astype(np.float64), bins=edges)[0]
    result = fhrs.variable(x, bins=edges)
    assert_equal(expected, result)


@pytest.mark.parametrize("dtype", [np.uint64, np.uint32])
def test_1d_variable_unsigned_dtype(dtype: np.dtype):
    rng = np.random.default_rng(11)
    x = rng.integers(0, 200, size=4000).astype(dtype)
    edges = np.array([0.0, 25.0, 75.0, 150.0, 200.0])
    expected = np.histogram(x.astype(np.float64), bins=edges)[0]
    result = fhrs.variable(x, bins=edges)
    assert_equal(expected, result)


def test_1d_variable_empty():
    x = np.array([], dtype=np.float64)
    edges = np.array([0.0, 1.0, 2.0])
    result = fhrs.variable(x, bins=edges)
    assert_equal(result, np.zeros(2, dtype=np.uintp))


def test_1d_variable_all_out_of_range():
    x = np.array([10.0, 20.0, 30.0])
    edges = np.array([0.0, 1.0, 2.0])
    result = fhrs.variable(x, bins=edges)
    assert_equal(result, np.zeros(2, dtype=np.uintp))


def test_1d_variable_single_element():
    x = np.array([0.5])
    edges = np.array([0.0, 0.25, 0.75, 1.0])
    result = fhrs.variable(x, bins=edges)
    expected = np.histogram(x, bins=edges)[0]
    assert_equal(result, expected)


# -- variable-width, weighted -------------------------------------------------


def test_1d_variable_weighted():
    x = np.random.randn(2000)
    w = np.ones_like(x) * 0.5
    bins = np.array([-3.5, -3.0, -2.0, 0, 1.5, 2.5, 3.5])
    a = np.histogram(x, bins=bins, weights=w)[0]
    b = fhrs.variable(x, bins=bins, weights=w)
    assert_allclose(a, b[:, 0])
    c = pygram11.var1d(x, bins=bins, weights=w)
    assert_allclose(c[1], np.sqrt(b[:, 1]))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_1d_variable_weighted_dtype(dtype: np.dtype):
    rng = np.random.default_rng(77)
    edges = np.array([-100.0, -50.0, -10.0, 0.0, 10.0, 50.0, 100.0])
    if np.issubdtype(dtype, np.integer):
        x = rng.integers(-100, 100, size=4000).astype(dtype)
    else:
        x = (rng.standard_normal(4000) * 40).astype(dtype)
    w = rng.uniform(0.1, 2.0, size=len(x))
    expected = np.histogram(x.astype(np.float64), bins=edges, weights=w)[0]
    result = fhrs.variable(x, bins=edges, weights=w)
    assert_allclose(expected, result[:, 0], atol=1e-10)


def test_1d_variable_weighted_variance():
    rng = np.random.default_rng(13)
    x = rng.standard_normal(5000)
    w = rng.uniform(0.5, 1.5, size=len(x))
    edges = np.array([-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0])
    result = fhrs.variable(x, bins=edges, weights=w)
    ref = pygram11.var1d(x, bins=edges, weights=w)
    assert_allclose(ref[0], result[:, 0])
    assert_allclose(ref[1], np.sqrt(result[:, 1]))


# -- error handling ------------------------------------------------------------


def test_fixed_unsupported_dtype():
    x = np.array([1, 2, 3], dtype=np.int8)
    with pytest.raises(TypeError, match="Unsupported dtype"):
        fhrs.fixed(x, bins=5, range=(0.0, 5.0))


def test_variable_unsupported_dtype():
    x = np.array([1, 2, 3], dtype=np.int8)
    edges = np.array([0.0, 2.0, 4.0])
    with pytest.raises(TypeError, match="Unsupported dtype"):
        fhrs.variable(x, bins=edges)


def test_variable_too_few_edges():
    x = np.array([1.0, 2.0, 3.0])
    edges = np.array([0.0])
    with pytest.raises(Exception):
        fhrs.variable(x, bins=edges)


def test_fixed_weighted_length_mismatch():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 2.0])
    with pytest.raises(Exception):
        fhrs.fixed(x, bins=5, range=(0.0, 5.0), weights=w)


def test_variable_weighted_length_mismatch():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 2.0])
    edges = np.array([0.0, 2.0, 4.0])
    with pytest.raises(Exception):
        fhrs.variable(x, bins=edges, weights=w)


# -- large arrays --------------------------------------------------------------


def test_1d_fixed_large():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500_000)
    bins = 50
    range = (-4.0, 4.0)
    expected = np.histogram(x, bins=bins, range=range)[0]
    result = fhrs.fixed(x, bins=bins, range=range)
    assert_equal(expected, result)


def test_1d_variable_large():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500_000)
    edges = np.linspace(-4.0, 4.0, 51)
    expected = np.histogram(x, bins=edges)[0]
    result = fhrs.variable(x, bins=edges)
    assert_equal(expected, result)
