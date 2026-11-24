import numpy as np
from unittest import TestCase

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools.histograms import histogramdd
from diffprivlib.utils import BudgetError

from tests.tools._histogram_fixtures import (ALL_ZERO_SEED_1D, ALL_ZERO_SEED_3D, EDGES_1D, EDGES_3D, RANGE_1D,
                                             RANGE_3D, SAMPLE_1D, SAMPLE_3D, cell_volumes)


class TestHistogramddDensityContract(TestCase):
    """Multidimensional contracts for histogramdd: unequal geometry, per-cell volume normalisation and the ledger."""

    def test_unequal_edges_preserved_per_dimension(self):
        # Explicit edge sequences along every dimension are returned unchanged, including their unequal widths.
        _, edges = histogramdd(SAMPLE_3D, epsilon=2, bins=[EDGES_3D] * 3, range=RANGE_3D, random_state=0)

        self.assertEqual(len(edges), 3)
        for axis_edges in edges:
            self.assertTrue(np.array_equal(axis_edges, EDGES_3D))

    def test_density_integrates_to_one_over_cell_volumes_3d(self):
        # In three dimensions the normaliser is the per-cell volume.  The integral over the whole space is one:
        # sum(density * volume) == 1, while the plain density sum differs from one because no cell has unit volume.
        dp_density, edges = histogramdd(SAMPLE_3D, epsilon=2, bins=[EDGES_3D] * 3, range=RANGE_3D,
                                        density=True, random_state=11)
        volumes = cell_volumes(edges)

        self.assertEqual(volumes.shape, dp_density.shape)
        self.assertAlmostEqual(float((dp_density * volumes).sum()), 1.0)
        self.assertFalse(np.allclose(volumes, np.ones_like(volumes)))

    def test_density_matches_noisy_counts_and_cell_volumes(self):
        # Per-cell contract in 3-D: density == noisy_counts / cell_volume / total_noisy_count, rebuilt from the
        # private counts under the same seed rather than from any un-noised histogram.
        dp_counts, edges = histogramdd(SAMPLE_3D, epsilon=2, bins=[EDGES_3D] * 3, range=RANGE_3D,
                                       density=False, random_state=11)
        dp_density, _ = histogramdd(SAMPLE_3D, epsilon=2, bins=[EDGES_3D] * 3, range=RANGE_3D,
                                    density=True, random_state=11)

        self.assertGreater(dp_counts.sum(), 0)
        expected_density = dp_counts / cell_volumes(edges) / dp_counts.sum()
        self.assertTrue(np.allclose(dp_density, expected_density))

    def test_counts_are_noisy_not_raw_3d(self):
        # The multidimensional output is a private realisation: non-negative counts that do not reproduce the raw
        # histogram cell-for-cell.
        raw_counts, _ = np.histogramdd(SAMPLE_3D, bins=[EDGES_3D] * 3, range=RANGE_3D)
        dp_counts, _ = histogramdd(SAMPLE_3D, epsilon=0.1, bins=[EDGES_3D] * 3, range=RANGE_3D, random_state=0)

        # histogramdd returns a float array regardless of density, but every value is a non-negative, integer-valued
        # draw from the truncated geometric mechanism.
        self.assertTrue(np.issubdtype(dp_counts.dtype, np.floating))
        self.assertTrue((dp_counts >= 0).all())
        self.assertTrue(np.allclose(dp_counts, np.round(dp_counts)))
        self.assertTrue((dp_counts != raw_counts).any())

    def test_1d_input_density_uses_bin_widths(self):
        # histogramdd on a single dimension obeys the same volume rule, where the volume is the bin width.
        dp_density, edges = histogramdd(SAMPLE_1D, epsilon=2, bins=[EDGES_1D], range=[RANGE_1D],
                                        density=True, random_state=42)
        widths = np.diff(edges[0])

        self.assertAlmostEqual(float((dp_density * widths).sum()), 1.0)

    def test_zero_total_density_is_finite_and_un_normalised_3d(self):
        # An all-zero private realisation (reached naturally on an empty sample) must produce a finite, all-zero
        # density instead of an illegal zero-total normalisation, and keep the requested geometry.
        empty = np.empty((0, 3))
        zero_counts, _ = histogramdd(empty, epsilon=1, bins=[EDGES_3D] * 3, range=RANGE_3D,
                                     density=False, random_state=ALL_ZERO_SEED_3D)
        self.assertEqual(zero_counts.sum(), 0)

        dp_density, edges = histogramdd(empty, epsilon=1, bins=[EDGES_3D] * 3, range=RANGE_3D,
                                        density=True, random_state=ALL_ZERO_SEED_3D)

        self.assertTrue(np.isfinite(dp_density).all())
        self.assertTrue((dp_density == 0).all())
        self.assertEqual(dp_density.shape, zero_counts.shape)
        for axis_edges in edges:
            self.assertTrue(np.array_equal(axis_edges, EDGES_3D))

    def test_zero_total_density_is_finite_1d(self):
        # The same degenerate branch must be safe in the single-dimensional case.
        empty = np.array([])
        zero_counts, _ = histogramdd(empty, epsilon=1, bins=[EDGES_1D], range=[RANGE_1D],
                                     density=False, random_state=ALL_ZERO_SEED_1D)
        self.assertEqual(zero_counts.sum(), 0)

        dp_density, _ = histogramdd(empty, epsilon=1, bins=[EDGES_1D], range=[RANGE_1D],
                                    density=True, random_state=ALL_ZERO_SEED_1D)

        self.assertTrue(np.isfinite(dp_density).all())
        self.assertTrue((dp_density == 0).all())

    def test_budget_rejection_keeps_ledger_consistent(self):
        # A rejected multidimensional query must not spend budget; a later smaller query still succeeds and the
        # ledger records exactly the successful spends.
        accountant = BudgetAccountant(1.5, 0)
        histogramdd(SAMPLE_3D, epsilon=1, bins=[EDGES_3D] * 3, range=RANGE_3D, accountant=accountant)

        ledger_before = accountant.spent_budget
        total_before = accountant.total()

        with self.assertRaises(BudgetError):
            histogramdd(SAMPLE_3D, epsilon=1, bins=[EDGES_3D] * 3, range=RANGE_3D, accountant=accountant)

        self.assertEqual(accountant.spent_budget, ledger_before)
        self.assertEqual(accountant.total(), total_before)

        histogramdd(SAMPLE_3D, epsilon=0.5, bins=[EDGES_3D] * 3, range=RANGE_3D, accountant=accountant)
        self.assertEqual(len(accountant.spent_budget), 2)
        self.assertEqual(accountant.spent_budget[-1], (0.5, 0))
        self.assertEqual(accountant.total(), (1.5, 0))

    def test_budget_rejection_does_not_consume_randomness(self):
        # The check precedes the mechanism draws, so a rejected call between two same-seed calls does not alter the
        # reproducible output.
        accountant = BudgetAccountant(1.0, 0)
        first_counts, _ = histogramdd(SAMPLE_3D, epsilon=1, bins=[EDGES_3D] * 3, range=RANGE_3D,
                                      accountant=accountant, random_state=123)

        with self.assertRaises(BudgetError):
            histogramdd(SAMPLE_3D, epsilon=1, bins=[EDGES_3D] * 3, range=RANGE_3D,
                        accountant=accountant, random_state=123)

        fresh_accountant = BudgetAccountant(1.0, 0)
        repeated_counts, _ = histogramdd(SAMPLE_3D, epsilon=1, bins=[EDGES_3D] * 3, range=RANGE_3D,
                                         accountant=fresh_accountant, random_state=123)

        self.assertTrue(np.array_equal(first_counts, repeated_counts))
