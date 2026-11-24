import numpy as np
from unittest import TestCase

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools.histograms import histogram2d, histogramdd
from diffprivlib.utils import BudgetError

from tests.tools._histogram_fixtures import (ALL_ZERO_SEED_2D, CELL_AREAS_2D, EDGES_X, EDGES_Y, RANGE_2D,
                                             SAMPLE_X, SAMPLE_Y, cell_volumes)


class TestHistogram2dDensityContract(TestCase):
    """Cross-cutting contracts for the 2-D wrapper and its histogramdd delegation."""

    def test_unequal_edges_preserved_on_both_axes(self):
        # Unequal-width edges supplied per dimension must be returned as-is for both axes.
        _, xedges, yedges = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=2, bins=(EDGES_X, EDGES_Y),
                                        range=RANGE_2D, random_state=0)

        self.assertTrue(np.array_equal(xedges, EDGES_X))
        self.assertTrue(np.array_equal(yedges, EDGES_Y))

    def test_counts_are_noisy_not_raw(self):
        # Private counts must differ from the un-noised histogram while staying on the non-negative integer support
        # of the truncated geometric mechanism.  Raw counts are only used to demonstrate the difference, never as an
        # expectation.
        raw_counts, _, _ = np.histogram2d(SAMPLE_X, SAMPLE_Y, bins=(EDGES_X, EDGES_Y), range=RANGE_2D)
        dp_counts, _, _ = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=0.1, bins=(EDGES_X, EDGES_Y),
                                      range=RANGE_2D, random_state=0)

        # histogramdd (which backs this wrapper) returns a float array, but every value is a non-negative,
        # integer-valued draw from the truncated geometric mechanism.
        self.assertTrue(np.issubdtype(dp_counts.dtype, np.floating))
        self.assertTrue((dp_counts >= 0).all())
        self.assertTrue(np.allclose(dp_counts, np.round(dp_counts)))
        self.assertTrue((dp_counts != raw_counts).any())

    def test_density_integrates_to_one_over_cell_areas(self):
        # The integral over the plane is one: sum(density * cell_area) == 1, where cell areas are the outer product
        # of the unequal axis widths (so a plain density sum is deliberately *not* expected to equal one).
        dp_density, xedges, yedges = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=2, bins=(EDGES_X, EDGES_Y),
                                                 range=RANGE_2D, density=True, random_state=7)
        areas = cell_volumes([xedges, yedges])

        self.assertAlmostEqual(float((dp_density * areas).sum()), 1.0)
        # Sanity check on the geometry itself: cells do not have unit area.
        self.assertFalse(np.allclose(areas, np.ones_like(areas)))

    def test_density_matches_noisy_counts_and_cell_areas(self):
        # Per-cell contract: density[j, k] == noisy_count[j, k] / cell_area[j, k] / total_noisy_count.  The reference
        # is reconstructed from the private counts under the same seed.
        dp_counts, _, _ = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=2, bins=(EDGES_X, EDGES_Y),
                                      range=RANGE_2D, density=False, random_state=7)
        dp_density, _, _ = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=2, bins=(EDGES_X, EDGES_Y),
                                       range=RANGE_2D, density=True, random_state=7)

        self.assertGreater(dp_counts.sum(), 0)
        expected_density = dp_counts / CELL_AREAS_2D / dp_counts.sum()
        self.assertTrue(np.allclose(dp_density, expected_density))

    def test_matches_histogramdd_with_same_edges_and_seed(self):
        # The wrapper delegates straight to histogramdd: identical edges and seed must yield identical counts,
        # densities and split edge arrays.
        counts_2d, xedges_2d, yedges_2d = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=2, bins=(EDGES_X, EDGES_Y),
                                                      range=RANGE_2D, random_state=7)
        counts_dd, edges_dd = histogramdd([SAMPLE_X, SAMPLE_Y], epsilon=2, bins=[EDGES_X, EDGES_Y],
                                          range=RANGE_2D, random_state=7)

        density_2d, _, _ = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=2, bins=(EDGES_X, EDGES_Y),
                                       range=RANGE_2D, density=True, random_state=7)
        density_dd, _ = histogramdd([SAMPLE_X, SAMPLE_Y], epsilon=2, bins=[EDGES_X, EDGES_Y],
                                    range=RANGE_2D, density=True, random_state=7)

        self.assertTrue(np.array_equal(counts_2d, counts_dd))
        self.assertTrue(np.allclose(density_2d, density_dd))
        self.assertTrue(np.array_equal(xedges_2d, edges_dd[0]))
        self.assertTrue(np.array_equal(yedges_2d, edges_dd[1]))

    def test_zero_total_density_is_finite_and_un_normalised(self):
        # All-zero private realisation on an empty sample: no valid density exists, so the output remains finite and
        # all zero rather than introducing NaN/inf through a zero division.
        empty = np.array([])
        zero_counts, _, _ = histogram2d(empty, empty, epsilon=1, bins=(EDGES_X, EDGES_Y), range=RANGE_2D,
                                        density=False, random_state=ALL_ZERO_SEED_2D)
        self.assertEqual(zero_counts.sum(), 0)

        dp_density, xedges, yedges = histogram2d(empty, empty, epsilon=1, bins=(EDGES_X, EDGES_Y),
                                                 range=RANGE_2D, density=True, random_state=ALL_ZERO_SEED_2D)

        self.assertTrue(np.isfinite(dp_density).all())
        self.assertTrue((dp_density == 0).all())
        self.assertTrue(np.array_equal(xedges, EDGES_X))
        self.assertTrue(np.array_equal(yedges, EDGES_Y))

    def test_budget_rejection_keeps_ledger_consistent(self):
        # Rejected spend leaves the ledger untouched; the remaining budget stays usable and is recorded correctly.
        accountant = BudgetAccountant(1.5, 0)
        histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=1, bins=(EDGES_X, EDGES_Y), range=RANGE_2D,
                    accountant=accountant)

        ledger_before = accountant.spent_budget
        total_before = accountant.total()

        with self.assertRaises(BudgetError):
            histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=1, bins=(EDGES_X, EDGES_Y), range=RANGE_2D,
                        accountant=accountant)

        self.assertEqual(accountant.spent_budget, ledger_before)
        self.assertEqual(accountant.total(), total_before)

        histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=0.5, bins=(EDGES_X, EDGES_Y), range=RANGE_2D,
                    accountant=accountant)
        self.assertEqual(len(accountant.spent_budget), 2)
        self.assertEqual(accountant.spent_budget[-1], (0.5, 0))

    def test_budget_rejection_does_not_consume_randomness(self):
        # Same-seed calls before and after a rejection must match when run against equally-funded accountants,
        # showing the rejection happens before any mechanism draw.
        accountant = BudgetAccountant(1.0, 0)
        first_counts, _, _ = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=1, bins=(EDGES_X, EDGES_Y),
                                         range=RANGE_2D, accountant=accountant, random_state=123)

        with self.assertRaises(BudgetError):
            histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=1, bins=(EDGES_X, EDGES_Y), range=RANGE_2D,
                        accountant=accountant, random_state=123)

        fresh_accountant = BudgetAccountant(1.0, 0)
        repeated_counts, _, _ = histogram2d(SAMPLE_X, SAMPLE_Y, epsilon=1, bins=(EDGES_X, EDGES_Y),
                                            range=RANGE_2D, accountant=fresh_accountant, random_state=123)

        self.assertTrue(np.array_equal(first_counts, repeated_counts))
