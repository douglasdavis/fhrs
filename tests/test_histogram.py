import fhrs
import numpy as np
import pygram11
from numpy.testing import assert_allclose, assert_equal


def test_1d_fixed():
    x = np.random.randn(2000)
    bins = 8
    range = (-2.8, 2.8)
    a = np.histogram(x, bins=bins, range=range)[0]
    b = fhrs.histogram_fixed(x, bins=bins, range=range)
    assert_equal(a, b)


def test_1d_fixed_weighted():
    x = np.random.randn(2000)
    w = np.ones_like(x) * 0.5
    bins = 8
    range = (-2.85, 2.85)
    a = np.histogram(x, bins=bins, range=range, weights=w)[0]
    b = fhrs.histogram_fixed(x, bins=bins, range=range, weights=w)
    assert_allclose(a, b[:, 0])
    c = pygram11.fix1d(x, bins=bins, range=range, weights=w)
    assert_allclose(c[1], np.sqrt(b[:, 1]))


def test_1d_variable():
    x = np.random.randn(2000)
    bins = np.array([-3.5, -3.0, -2.0, 0, 1.5, 2.5, 3.5])
    a = np.histogram(x, bins=bins)[0]
    b = fhrs.histogram_variable(x, bins=bins)
    assert_equal(a, b)


def test_1d_variable_weighted():
    x = np.random.randn(2000)
    w = np.ones_like(x) * 0.5
    bins = np.array([-3.5, -3.0, -2.0, 0, 1.5, 2.5, 3.5])
    a = np.histogram(x, bins=bins, weights=w)[0]
    b = fhrs.histogram_variable(x, bins=bins, weights=w)
    assert_allclose(a, b[:, 0])
    c = pygram11.var1d(x, bins=bins, weights=w)
    assert_allclose(c[1], np.sqrt(b[:, 1]))
