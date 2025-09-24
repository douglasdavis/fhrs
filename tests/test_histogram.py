import numpy as np
from numpy.testing import assert_allclose, assert_equal
import fhrs

def test_1d():
    x = np.random.randn(2000)
    bins = 8
    range = (-2.8, 2.8)
    a = np.histogram(x, bins=bins, range=range)[0]
    b = fhrs.histogram(x, bins=bins, range=range)
    assert_equal(a, b)


def test_1d_weighted():
    x = np.random.randn(2000)
    w = np.ones_like(x) * 0.5
    bins = 8
    range = (-2.85, 2.85)
    a = np.histogram(x, bins=bins, range=range, weights=w)[0]
    b = fhrs.histogram(x, bins=bins, range=range, weights=w)
    assert_allclose(a, b)
