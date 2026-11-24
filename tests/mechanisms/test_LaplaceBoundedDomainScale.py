"""Deterministic tests for the scale and output semantics of the bounded-domain (finite-domain) Laplace mechanism.

The scale of :class:`.LaplaceBoundedDomain` is not the naive Laplace scale ``sensitivity / epsilon``: truncating
ordinary Laplace samples to the domain does not yield a valid mechanism.  These tests pin down, without relying on
random sampling, how the scale responds to changes in the privacy parameters, the sensitivity and the domain width,
and verify analytically that released outputs respect the privacy bound for adjacent inputs and place no extra
probability mass on the boundary points.
"""
import numpy as np
from unittest import TestCase, mock

from diffprivlib.mechanisms import Laplace, LaplaceBoundedDomain


def _delta_c(scale, sensitivity, diam):
    """The normaliser ratio bound Delta_c of [HABM20], as used by the mechanism."""
    if scale == 0:
        return 2.0
    return (2 - np.exp(-sensitivity / scale) - np.exp(-(diam - sensitivity) / scale)) \
        / (1 - np.exp(-diam / scale))


def _normalizer(value, scale, lower, upper):
    """Z(value): the mass of Laplace(value, scale) falling inside [lower, upper]."""
    return 1 - (np.exp(-(value - lower) / scale) + np.exp(-(upper - value) / scale)) / 2


def _density(output, value, mech):
    """Analytic density of the mechanism output at ``output`` for true input ``value``."""
    return np.exp(-np.abs(output - value) / mech._scale) / (2 * mech._scale * mech._normalizer(value))


def _make_mech(epsilon, delta, sensitivity, lower, upper, **kwargs):
    mech = LaplaceBoundedDomain(epsilon=epsilon, delta=delta, sensitivity=sensitivity,
                                lower=lower, upper=upper, **kwargs)
    mech._scale = mech._find_scale()
    mech._normalizer = lambda value: _normalizer(value, mech._scale, mech.lower, mech.upper)
    return mech


class TestLaplaceBoundedDomainScale(TestCase):
    def setup_method(self, method):
        self.mech = LaplaceBoundedDomain

    def teardown_method(self, method):
        del self.mech

    # ---------- scale semantics ----------

    def test_scale_is_not_truncated_laplace_scale(self):
        # Truncating Laplace(sensitivity / epsilon) samples to the domain is not DP; the scale must be inflated
        mech = _make_mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=10)
        naive_scale = 1.0

        self.assertGreater(mech._scale, naive_scale)
        self.assertNotAlmostEqual(mech._scale, naive_scale, places=5)

    def test_scale_equals_laplace_scale_when_domain_matches_sensitivity(self):
        # When the domain width equals the sensitivity, Delta_c = 1 and the bounded scale reduces to the naive one
        mech = _make_mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        self.assertAlmostEqual(mech._scale, 1.0, places=12)

        mech = _make_mech(epsilon=2.5, delta=0, sensitivity=0.5, lower=-1, upper=-0.5)
        self.assertAlmostEqual(mech._scale, 0.2, places=12)

    def test_scale_satisfies_fixed_point(self):
        # The scale must solve b = sensitivity / (epsilon - log(Delta_c(b)) - log(1 - delta))  [HABM20]
        for epsilon, delta, sensitivity, lower, upper in [
                (1.0, 0.0, 1.0, 0.0, 10.0),
                (2.5, 0.0, 0.5, -1.0, 1.0),
                (0.5, 0.2, 2.0, 0.0, 5.0),
                (1.0, 0.1, 1.0, 0.0, 1.0),
                (3.0, 0.5, 1.5, -3.0, 7.0)]:
            mech = _make_mech(epsilon=epsilon, delta=delta, sensitivity=sensitivity, lower=lower, upper=upper)
            diam = upper - lower
            fixed_point = sensitivity / (epsilon - np.log(_delta_c(mech._scale, sensitivity, diam))
                                         - np.log(1 - delta))
            self.assertAlmostEqual(mech._scale, fixed_point, places=12)

    def test_scale_decreases_as_epsilon_increases(self):
        scales = [_make_mech(epsilon=eps, delta=0, sensitivity=1, lower=0, upper=10)._scale
                  for eps in (0.5, 1.0, 2.0, 4.0)]
        for weaker, stronger in zip(scales, scales[1:]):
            self.assertGreater(weaker, stronger)

    def test_scale_increases_as_sensitivity_increases(self):
        scales = [_make_mech(epsilon=1, delta=0, sensitivity=sens, lower=0, upper=10)._scale
                  for sens in (0.5, 1.0, 2.0, 4.0)]
        for smaller, larger in zip(scales, scales[1:]):
            self.assertLess(smaller, larger)

    def test_scale_increases_as_domain_widens(self):
        mech_narrow = _make_mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=2)
        mech_wide = _make_mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=20)

        self.assertLess(mech_narrow._scale, mech_wide._scale)
        # A wider domain buys a smaller effective epsilon for the same budget
        self.assertGreater(mech_narrow.effective_epsilon(), mech_wide.effective_epsilon())

    def test_scale_decreases_as_delta_increases(self):
        scales = [_make_mech(epsilon=1, delta=delta, sensitivity=1, lower=0, upper=10)._scale
                  for delta in (0.0, 0.1, 0.3)]
        for smaller_delta, larger_delta in zip(scales, scales[1:]):
            self.assertGreater(smaller_delta, larger_delta)

    def test_effective_epsilon_matches_sensitivity_over_scale(self):
        for upper in (1.0, 2.0, 10.0):
            mech = _make_mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=upper)
            self.assertAlmostEqual(mech.effective_epsilon(), mech.sensitivity / mech._scale, places=12)
            self.assertLessEqual(mech.effective_epsilon(), 1.0)

    # ---------- privacy bound for released outputs ----------

    def test_privacy_bound_pure_dp(self):
        # For delta = 0, every output in the domain must satisfy the pure-DP density bound for adjacent inputs
        for epsilon, sensitivity, lower, upper in [(1.0, 1.0, 0.0, 10.0),
                                                   (0.5, 2.0, 0.0, 5.0),
                                                   (3.0, 0.5, -2.0, 2.0)]:
            mech = _make_mech(epsilon=epsilon, delta=0, sensitivity=sensitivity, lower=lower, upper=upper)
            outputs = np.linspace(lower, upper, 2001)
            max_ratio = 0.0

            for value in np.linspace(lower, upper - sensitivity, 41):
                neighbour = value + sensitivity
                ratio = _density(outputs, value, mech) / _density(outputs, neighbour, mech)
                max_ratio = max(max_ratio, ratio.max())

            self.assertLessEqual(max_ratio, np.exp(epsilon) * (1 + 1e-9))
            # The bound is attained (up to grid resolution): the scale is not wastefully large
            self.assertGreaterEqual(max_ratio, np.exp(epsilon) * (1 - 1e-6))

    def test_privacy_bound_approx_dp(self):
        # For delta > 0, the output mass where the epsilon-bound is violated must not exceed delta
        for epsilon, delta, sensitivity, lower, upper in [(1.0, 0.1, 1.0, 0.0, 10.0),
                                                          (0.5, 0.2, 2.0, 0.0, 5.0),
                                                          (2.0, 0.05, 1.0, 0.0, 3.0)]:
            mech = _make_mech(epsilon=epsilon, delta=delta, sensitivity=sensitivity, lower=lower, upper=upper)
            outputs = np.linspace(lower, upper, 20001)
            step = (upper - lower) / (len(outputs) - 1)

            for value in np.linspace(lower, upper - sensitivity, 21):
                neighbour = value + sensitivity
                violation = np.maximum(_density(outputs, value, mech)
                                       - np.exp(epsilon) * _density(outputs, neighbour, mech), 0.0)
                # Trapezoidal integration of the violating density mass
                mass = np.sum((violation[:-1] + violation[1:]) / 2) * step
                self.assertLessEqual(mass, delta + 1e-6)

    # ---------- no spurious probability mass on the boundary ----------

    def test_density_integrates_to_one(self):
        # The continuous density must account for all output mass: no atoms hidden at the boundary points
        for epsilon, sensitivity, lower, upper in [(1.0, 1.0, 0.0, 10.0),
                                                   (0.5, 2.0, 0.0, 5.0)]:
            mech = _make_mech(epsilon=epsilon, delta=0, sensitivity=sensitivity, lower=lower, upper=upper)
            outputs = np.linspace(lower, upper, 100001)
            step = (upper - lower) / (len(outputs) - 1)

            for value in (lower, (lower + upper) / 2, upper):
                density = _density(outputs, value, mech)
                total = np.sum((density[:-1] + density[1:]) / 2) * step
                self.assertAlmostEqual(total, 1.0, places=8)

    def test_out_of_domain_draws_are_rejected_not_clamped(self):
        # A sample outside the domain must be discarded and redrawn; clamping it would pile mass on the boundary
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1, random_state=0)

        with mock.patch.object(Laplace, "_laplace_sampler",
                               side_effect=[np.array([1000.0]), np.array([0.0])]) as sampler:
            output = mech.randomise(0.3)

        self.assertEqual(sampler.call_count, 2)
        self.assertEqual(output, 0.3)
        self.assertNotIn(output, (mech.lower, mech.upper))

    def test_outputs_never_land_on_boundary(self):
        # With a huge scale most draws fall outside the domain; rejection sampling must still never emit a boundary
        # point, whereas truncation would concentrate mass there
        mech = self.mech(epsilon=0.1, delta=0, sensitivity=1, lower=0, upper=1, random_state=1)
        self.assertGreater(mech._find_scale(), 5)  # ensure most raw draws are rejected

        outputs = np.array([mech.randomise(0.5) for _ in range(2000)])

        self.assertTrue(np.all(outputs > 0))
        self.assertTrue(np.all(outputs < 1))

    # ---------- output semantics ----------

    def test_input_outside_domain_is_clamped_before_noising(self):
        mech_at_bound = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1, random_state=7)
        mech_outside = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1, random_state=7)

        self.assertEqual([mech_outside.randomise(5.0) for _ in range(10)],
                         [mech_at_bound.randomise(1.0) for _ in range(10)])

        mech_at_bound = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1, random_state=7)
        mech_outside = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1, random_state=7)

        self.assertEqual([mech_outside.randomise(-3.0) for _ in range(10)],
                         [mech_at_bound.randomise(0.0) for _ in range(10)])

    def test_outputs_within_domain(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=-2, upper=3, random_state=0)

        for value in (-2.0, -0.5, 1.0, 3.0):
            outputs = [mech.randomise(value) for _ in range(100)]
            self.assertTrue(all(-2 <= output <= 3 for output in outputs))
