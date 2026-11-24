# MIT License
#
# Copyright (C) IBM Corporation 2026
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Shared fixtures for the differentially private histogram tests.

The fixtures describe *binning geometry* (unequal-width edges and ranges) and input samples only.  Expected values in
the accompanying tests are never derived from the un-noised counts of these samples: every assertion is made against
the output of the private mechanism (counts produced by :class:`.GeometricTruncated` inside the histogram tools), or
against structural relationships (edge arrays, per-cell volumes and the normalisation rules) that hold for any private
realisation.
"""
import numpy as np

# --- 1-D geometry: deliberately unequal bin widths (1, 3 and 6) -------------------------------
EDGES_1D = np.array([0.0, 1.0, 4.0, 10.0])
RANGE_1D = (0.0, 10.0)
SAMPLE_1D = np.array([1, 2, 3, 4, 5, 6, 7, 8])
WIDTHS_1D = np.diff(EDGES_1D)

# --- 2-D geometry: unequal edges on both axes, differing per dimension ------------------------
EDGES_X = np.array([0.0, 2.0, 6.0, 10.0])
EDGES_Y = np.array([0.0, 5.0, 7.0, 10.0])
RANGE_2D = [(0.0, 10.0), (0.0, 10.0)]
SAMPLE_X = np.array([1, 2, 3, 4, 5])
SAMPLE_Y = np.array([5, 7, 1, 5, 9])
WIDTHS_X = np.diff(EDGES_X)
WIDTHS_Y = np.diff(EDGES_Y)

# Outer product of the axis widths gives the area of every (x, y) cell, shape (nx, ny).
CELL_AREAS_2D = WIDTHS_X[:, None] * WIDTHS_Y[None, :]

# --- 3-D geometry: unequal edges shared across three dimensions -------------------------------
EDGES_3D = np.array([0.0, 3.0, 7.0, 10.0])
RANGE_3D = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
SAMPLE_3D = np.random.RandomState(0).randint(0, 10, size=(20, 3))


def cell_volumes(edges):
    """Return the per-cell volume array for a histogram built with ``edges``.

    Parameters
    ----------
    edges : list of numpy.ndarray
        One monotonically increasing edge array per histogram dimension, as returned by the histogram tools.

    Returns
    -------
    numpy.ndarray
        Array broadcastable against the histogram, holding the product of the bin widths along every dimension.

    """
    dims = len(edges)
    volumes = np.ones(tuple(len(axis_edges) - 1 for axis_edges in edges))

    for axis, axis_edges in enumerate(edges):
        shape = np.ones(dims, dtype=int)
        shape[axis] = len(axis_edges) - 1
        volumes = volumes * np.diff(axis_edges).reshape(shape)

    return volumes


# Seeds for which an empty sample over the listed geometries yields an all-zero private histogram (every cell
# independently draws zero noise from GeometricTruncated).  These seeds were found by exhaustive search over seeds;
# the tests re-verify the all-zero precondition at runtime before exercising the degenerate normalisation path, so a
# change in the underlying random stream fails the precondition explicitly rather than producing a false negative.
ALL_ZERO_SEED_1D = 5
ALL_ZERO_SEED_2D = 31
ALL_ZERO_SEED_3D = 6633
