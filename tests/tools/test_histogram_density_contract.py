import numpy as np
from unittest import TestCase

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools.histograms import histogram, histogramdd
from diffprivlib.utils import BudgetError

from tests.tools._histogram_fixtures import (ALL_ZERO_SEED_1D, EDGES_1D, RANGE_1D, SAMPLE_1D, WIDTHS_1D)


class TestHistogramDensityContract(TestCase):
    """Cross-cutting contracts for the 1-D wrapper, from unequal bin geometry through density output."""

    def test_unequal_edges_preserved(self):
        # Explicit, unequal-width edges must be returned untouched (no rebinning onto equal widths).
        _, bin_edges = histogram(SAMPLE_1D, epsilon=2, bins=EDGES_1D, range=RANGE_1D, random_state=0)

        self.assertTrue(np.array_equal(bin_edges, EDGES_1D))
        self.assertTrue(np.array_equal(np.diff(bin_edges), WIDTHS_1D))

    def test_counts_are_noisy_not_raw(self):
        # The returned counts are a private realisation: they must not equal the un-noised counts cell-for-cell,
        # and every cell must carry a non-negative integer (the support of the truncated geometric mechanism).
        raw_counts, _ = np.histogram(SAMPLE_1D, bins=EDGES_1D, range=RANGE_1D)
        dp_counts, _ = histogram(SAMPLE_1D, epsilon=0.1, bins=EDGES_1D, range=RANGE_1D, random_state=0)

        self.assertTrue(np.issubdtype(dp_counts.dtype, np.integer))
        self.assertTrue((dp_counts >= 0).all())
        self.assertTrue((dp_counts != raw_counts).any())

    def test_density_integrates_to_one_over_unequal_widths(self):
        # With a positive private total, the density is normalised so that the *integral* over the range is one:
        # sum(density * width) == 1 even though no bin has unit width.
        dp_density, bin_edges = histogram(SAMPLE_1D, epsilon=2, bins=EDGES_1D, range=RANGE_1D,
                                          density=True, random_state=42)
        widths = np.diff(bin_edges)

        self.assertAlmostEqual(float((dp_density * widths).sum()), 1.0)

    def test_density_matches_noisy_counts_and_cell_widths(self):
        # Per-cell contract: density[i] == noisy_count[i] / width[i] / total_noisy_count.  The expectation is rebuilt
        # from the private counts (obtained separately with the same seed), never from the raw sample counts.
        dp_counts, _ = histogram(SAMPLE_1D, epsilon=2, bins=EDGES_1D, range=RANGE_1D,
                                 density=False, random_state=42)
        dp_density, _ = histogram(SAMPLE_1D, epsilon=2, bins=EDGES_1D, range=RANGE_1D,
                                  density=True, random_state=42)

        self.assertGreater(dp_counts.sum(), 0)
        expected_density = dp_counts / WIDTHS_1D / dp_counts.sum()
        self.assertTrue(np.allclose(dp_density, expected_density))

    def test_zero_total_density_is_finite_and_un_normalised(self):
        # Degenerate case: the private mechanism can return zero in every cell (here reached naturally on an empty
        # sample).  No valid density exists, so the output must stay finite and all zero instead of dividing by zero.
        empty = np.array([])
        zero_counts, _ = histogram(empty, epsilon=1, bins=EDGES_1D, range=RANGE_1D,
                                   density=False, random_state=ALL_ZERO_SEED_1D)
        self.assertEqual(zero_counts.sum(), 0)

        dp_density, bin_edges = histogram(empty, epsilon=1, bins=EDGES_1D, range=RANGE_1D,
                                          density=True, random_state=ALL_ZERO_SEED_1D)

        self.assertTrue(np.isfinite(dp_density).all())
        self.assertTrue((dp_density == 0).all())
        self.assertTrue(np.array_equal(bin_edges, EDGES_1D))

    def test_zero_total_density_deterministic_across_calls(self):
        # The degenerate output must be stable across repeated calls with the same seed (no NaN drift or random
        # fallback normalisation).
        empty = np.array([])
        outputs = [histogram(empty, epsilon=1, bins=EDGES_1D, range=RANGE_1D, density=True,
                             random_state=ALL_ZERO_SEED_1D)[0] for _ in range(3)]

        self.assertTrue(all(np.array_equal(outputs[0], out) for out in outputs))

    def test_matches_histogramdd_with_same_edges_and_seed(self):
        # Cross-interface contract: the 1-D wrapper and histogramdd apply the same mechanism cell-by-cell, so equal
        # edges, range and seed produce identical counts, densities and edges.
        counts_1d, edges_1d = histogram(SAMPLE_1D, epsilon=2, bins=EDGES_1D, range=RANGE_1D, random_state=42)
        counts_dd, edges_dd = histogramdd(SAMPLE_1D, epsilon=2, bins=[EDGES_1D], range=[RANGE_1D], random_state=42)

        density_1d, _ = histogram(SAMPLE_1D, epsilon=2, bins=EDGES_1D, range=RANGE_1D,
                                  density=True, random_state=42)
        density_dd, _ = histogramdd(SAMPLE_1D, epsilon=2, bins=[EDGES_1D], range=[RANGE_1D],
                                    density=True, random_state=42)

        self.assertTrue(np.array_equal(counts_1d, counts_dd))
        self.assertTrue(np.allclose(density_1d, density_dd))
        self.assertTrue(np.array_equal(edges_1d, edges_dd[0]))

    def test_budget_rejection_keeps_ledger_consistent(self):
        # A rejected call must not mutate the ledger, and the remaining budget must still be spendable afterwards.
        accountant = BudgetAccountant(1.5, 0)
        histogram(SAMPLE_1D, epsilon=1, bins=EDGES_1D, range=RANGE_1D, accountant=accountant)

        ledger_before = accountant.spent_budget
        total_before = accountant.total()

        with self.assertRaises(BudgetError):
            histogram(SAMPLE_1D, epsilon=1, bins=EDGES_1D, range=RANGE_1D, accountant=accountant)

        self.assertEqual(accountant.spent_budget, ledger_before)
        self.assertEqual(accountant.total(), total_before)

        # Subsequent, in-budget calls still succeed and are recorded exactly once.
        histogram(SAMPLE_1D, epsilon=0.5, bins=EDGES_1D, range=RANGE_1D, accountant=accountant)
        self.assertEqual(len(accountant.spent_budget), 2)
        self.assertEqual(accountant.spent_budget[-1], (0.5, 0))

    def test_budget_rejection_precedes_noise_and_is_reproducible(self):
        # Because the budget check runs before any randomness is consumed, a same-seed call after a rejection remains
        # reproducible against a fresh accountant.
        accountant = BudgetAccountant(1.0, 0)
        first_counts, _ = histogram(SAMPLE_1D, epsilon=1, bins=EDGES_1D, range=RANGE_1D,
                                    accountant=accountant, random_state=123)

        with self.assertRaises(BudgetError):
            histogram(SAMPLE_1D, epsilon=1, bins=EDGES_1D, range=RANGE_1D,
                      accountant=accountant, random_state=123)

        fresh_accountant = BudgetAccountant(1.0, 0)
        repeated_counts, _ = histogram(SAMPLE_1D, epsilon=1, bins=EDGES_1D, range=RANGE_1D,
                                       accountant=fresh_accountant, random_state=123)

        self.assertTrue(np.array_equal(first_counts, repeated_counts))
