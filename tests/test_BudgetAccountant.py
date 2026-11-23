from unittest import TestCase

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import Budget, BudgetError


class TestBudgetAccountant(TestCase):
    def tearDown(self):
        BudgetAccountant.pop_default()

    @staticmethod
    def sample_model(epsilon=1.0, accountant=None):
        accountant = BudgetAccountant.load_default(accountant)
        accountant.check(epsilon, 0.0)

        accountant.spend(epsilon, 0.0)

    @staticmethod
    def sample_model2(epsilon=1.0, accountant=None):
        accountant = BudgetAccountant.load_default(accountant)
        accountant.check(epsilon, 0.0)

        accountant.spend(epsilon, 0.0)

    def test_init(self):
        acc = BudgetAccountant()
        self.assertEqual(acc.epsilon, float("inf"))
        self.assertEqual(acc.delta, 1)

    def test_init_epsilon(self):
        acc = BudgetAccountant(1, 0)
        self.assertEqual(acc.epsilon, 1.0)
        self.assertEqual(acc.delta, 0.0)

    def test_init_delta(self):
        acc = BudgetAccountant(0, 0.5)
        self.assertEqual(acc.epsilon, 0)
        self.assertAlmostEqual(acc.delta, 0.5, places=5)

    def test_init_neg_eps(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(-1)

    def test_init_neg_del(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(1, -0.5)

    def test_init_large_del(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(1, 1.5)

    def test_init_zero_budget(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(0, 0)

    def test_init_scalar_spent(self):
        with self.assertRaises(TypeError):
            BudgetAccountant(spent_budget=2)

    def test_init_non_list_spent(self):
        with self.assertRaises(TypeError):
            BudgetAccountant(spent_budget=(1, 0))

    def test_init_small_tuple_spent(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(spent_budget=[(1,)])

    def test_init_large_tuple_spent(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(spent_budget=[(1, 0, 2)])

    def test_init_large_slack(self):
        with self.assertRaises(ValueError):
            BudgetAccountant(1, 1e-2, 1e-1)

    def test_change_large_slack(self):
        acc = BudgetAccountant(1, 0.2, 0)
        acc.spend(0.1, 0.1)
        acc.spend(0.1, 0.1)

        with self.assertRaises(BudgetError):
            acc.slack = 0.2

    def test_total_large_slack(self):
        acc = BudgetAccountant(1, 0.5)
        with self.assertRaises(ValueError):
            acc.total(slack=0.75)

    def test_init_exceed_budget(self):
        with self.assertRaises(BudgetError):
            BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0), (0.5, 0)])

    def test_spent_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        with self.assertRaises(BudgetError):
            acc.check(0.1, 0)

    def test_get_spent_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        spent_budget = acc.spent_budget

        self.assertIsInstance(spent_budget, list)
        self.assertEqual(2, len(spent_budget))

    def test_change_spent_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        with self.assertRaises(AttributeError):
            acc.spent_budget = [(1, 0)]

        with self.assertRaises(AttributeError):
            del acc.spent_budget

        acc.spent_budget.append((1, 0))
        self.assertEqual(2, len(acc))
        self.assertEqual(acc.spent_budget, acc._BudgetAccountant__spent_budget)
        self.assertIsNot(acc.spent_budget, acc._BudgetAccountant__spent_budget)

    def test_change_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        with self.assertRaises(AttributeError):
            acc.epsilon = 2

        with self.assertRaises(AttributeError):
            del acc.epsilon

        with self.assertRaises(AttributeError):
            acc.delta = 0.1

        with self.assertRaises(AttributeError):
            del acc.delta

    def test_get_budget(self):
        acc = BudgetAccountant(1, 0, spent_budget=[(0.5, 0), (0.5, 0)])

        self.assertEqual(1, acc.epsilon)
        self.assertEqual(0, acc.delta)

    def test_remaining_budget_epsilon(self):
        acc = BudgetAccountant(1, 0)
        eps, delt = acc.remaining()
        self.assertAlmostEqual(eps, 1.0)
        self.assertEqual(delt, 0)

        acc = BudgetAccountant(1, 0)
        eps, delt = acc.remaining(10)
        self.assertAlmostEqual(eps, 0.1)
        self.assertEqual(delt, 0)

    def test_remaining_budget_epsilon_slack(self):
        acc = BudgetAccountant(1, 1e-15, slack=1e-15)
        eps, delt = acc.remaining(100)
        self.assertGreaterEqual(eps, 0.01)
        self.assertEqual(delt, 0)

    def test_remaining_budget_delta(self):
        acc = BudgetAccountant(1, 1e-2)
        eps, delt = acc.remaining(100)
        self.assertGreaterEqual(delt, 1e-4)
        self.assertLessEqual(delt, 1e-3)

        acc = BudgetAccountant(1, 1e-2, slack=1e-5)
        eps, delt = acc.remaining(100)
        self.assertGreaterEqual(eps, 0.01)
        self.assertGreaterEqual(delt, 1e-4)
        self.assertLessEqual(delt, 1e-3)

    def test_remaining_budget_zero_delta(self):
        acc = BudgetAccountant(1, 1e-2, 1e-2)
        _, delt = acc.remaining(100)
        self.assertEqual(0.0, delt)

    def test_remaining_budget_k(self):
        acc = BudgetAccountant(1, 1e-2, 1e-3)

        with self.assertRaises(ValueError):
            acc.remaining(0)

        with self.assertRaises(TypeError):
            acc.remaining(1.0)

    def test_remaining_budget_inf(self):
        acc = BudgetAccountant()
        self.assertEqual((float("inf"), 1.0), acc.remaining())
        self.assertEqual((float("inf"), 1.0), acc.remaining(100))

        acc.spend(float("inf"), 1)
        self.assertEqual((float("inf"), 1.0), acc.remaining())
        self.assertEqual((float("inf"), 1.0), acc.remaining(100))

    def test_spend(self):
        acc = BudgetAccountant()
        acc.spend(1, 0)
        self.assertEqual(acc.total(), (1, 0))

        acc.spend(1, 0.5)
        self.assertEqual(acc.total(), (2, 0.5))

        acc.spend(5, 0)
        self.assertEqual(acc.total(), (7, 0.5))

    def test_spend_errors(self):
        acc = BudgetAccountant()

        with self.assertRaises(ValueError):
            acc.spend(0, 0)

        with self.assertRaises(ValueError):
            acc.spend(-1, 0)

        with self.assertRaises(ValueError):
            acc.spend(1, -1)

        with self.assertRaises(ValueError):
            acc.spend(1, 2)

    def test_spend_exceed(self):
        acc = BudgetAccountant(5, 0)
        acc.spend(3, 0)

        with self.assertRaises(BudgetError):
            acc.spend(3, 0)

        with self.assertRaises(BudgetError):
            acc.spend(0, 1e-5)

    def test_inf_spend(self):
        acc = BudgetAccountant()
        acc.spend(float("inf"), 1)
        self.assertEqual((float("inf"), 1), acc.total())
        self.assertEqual((float("inf"), 1), acc.remaining())
        self.assertEqual((float("inf"), 1), acc.remaining(100))
        self.assertTrue(acc.check(float("inf"), 1))

    def test_remaining_budget_positive_vals(self):
        acc = BudgetAccountant(1, 1e-2, 1e-5, [(0.01, 0), (0.01, 0), (0.01, 0)])
        eps, delt = acc.remaining(50)
        self.assertGreaterEqual(eps, 0)
        self.assertGreaterEqual(delt, 0)

    def test_remaining_budget_implementation(self):
        acc = BudgetAccountant(1, 1e-2, 1e-5, [(0.01, 0), (0.01, 0), (0.01, 0)])
        k = 50

        eps, delt = acc.remaining(k)

        for i in range(k-1):
            acc.spend(eps, delt)

        remaining_eps, remaining_delt = acc.remaining()

        self.assertAlmostEqual(remaining_eps, eps)
        self.assertAlmostEqual(remaining_delt, delt)

    def test_remaining_budget_implementation2(self):
        acc = BudgetAccountant(1, 1e-2, 1e-5)
        k = 50

        eps, delt = acc.remaining(k)

        for i in range(k//2):
            acc.spend(eps, delt)

        eps, delt = acc.remaining(k)

        for i in range(k-1):
            acc.spend(eps, delt)

        remaining_eps, remaining_delt = acc.remaining()

        self.assertAlmostEqual(remaining_eps, eps)
        self.assertAlmostEqual(remaining_delt, delt)

    def test_load_wrong_type(self):
        with self.assertRaises(TypeError):
            BudgetAccountant.load_default(0)

        with self.assertRaises(TypeError):
            BudgetAccountant.load_default([1, 2, 3])

        with self.assertRaises(TypeError):
            BudgetAccountant.load_default("BudgetAccountant")

    def test_small_epsilon(self):
        acc = BudgetAccountant(1)

        with self.assertRaises(ValueError):
            acc.spend(1e-16, 0)

    def test_set_default(self):
        acc = BudgetAccountant()
        acc.set_default()

        self.assertIs(BudgetAccountant._default, acc)

    def test_pop_default(self):
        acc = BudgetAccountant().set_default()
        acc2 = BudgetAccountant.pop_default()

        self.assertIs(acc, acc2)

    def test_default(self):
        # Specify accountant as arg
        acc1 = BudgetAccountant(1.5, 0.0)
        self.sample_model(accountant=acc1)
        self.assertEqual((1.0, 0.0), acc1.total())

        # Use default accountant without one being set
        self.sample_model()
        acc2 = BudgetAccountant.pop_default()
        self.assertIsNot(acc1, acc2)
        self.assertEqual(float("inf"), acc2.epsilon)
        self.assertEqual((1.0, 0.0), acc2.total())

        # Set accountant as default
        acc3 = BudgetAccountant(2.0, 0.0).set_default()
        self.sample_model(epsilon=1.5)
        self.assertEqual((1.5, 0), acc3.total())
        self.assertEqual(2.0, acc3.epsilon)
        self.assertIsNot(acc3, acc2)
        self.assertIsNot(acc3, acc1)

        # Check default is same as what we set it
        acc4 = BudgetAccountant.pop_default()
        self.assertIs(acc3, acc4)
        self.assertEqual((1.0, 0.0), acc2.total())

        # Run again in 2 different functions without setting a default
        self.sample_model()
        self.sample_model2()
        acc5 = BudgetAccountant.pop_default()
        self.assertIsNot(acc5, acc2)
        self.assertIsNot(acc5, acc3)
        self.assertEqual((2.0, 0), acc5.total())

    def test_correct_composition(self):
        epsilons = [2**-9] * 700
        slack = 2**-25

        acc = BudgetAccountant(slack=slack)

        for epsilon in epsilons:
            acc.spend(epsilon, 0)

        spent_epsilon, spent_delta = acc.total()

        self.assertAlmostEqual(spent_epsilon, 0.27832280615743646366122002955588987576913442137093, places=14)
        self.assertEqual(spent_delta, slack)

    def test_with_statement(self):
        acc = BudgetAccountant()
        acc2 = BudgetAccountant()

        with acc:
            self.sample_model(1)
            self.sample_model(2)

        self.assertIsNone(BudgetAccountant._default)
        self.assertEqual((3, 0), acc.total())

        acc.set_default()

        with acc2:
            self.sample_model(5)
            self.sample_model(1, accountant=acc)

        self.assertEqual((4, 0), acc.total())
        self.assertEqual((5, 0), acc2.total())
        self.assertIs(BudgetAccountant._default, acc)

    def test_with_statement_errors(self):
        with BudgetAccountant(1.5) as acc:
            self.assertIsInstance(acc, BudgetAccountant)

            self.sample_model(1)

            with self.assertRaises(BudgetError):
                self.sample_model(1)

        with self.assertRaises(BudgetError):
            with BudgetAccountant(1):
                self.sample_model(2)

    def test_budget_output(self):
        acc = BudgetAccountant()
        self.assertIsInstance(acc.total(), Budget)
        self.assertIsInstance(acc.remaining(), Budget)
        self.assertIsInstance(acc.remaining(5), Budget)

        acc.spend(1, 0.5)
        self.assertIsInstance(acc.total(), Budget)
        self.assertIsInstance(acc.remaining(), Budget)
        self.assertIsInstance(acc.remaining(5), Budget)

        acc.slack = 0.5
        self.assertIsInstance(acc.total(), Budget)
        self.assertIsInstance(acc.remaining(), Budget)
        self.assertIsInstance(acc.remaining(5), Budget)

    def test_len(self):
        acc = BudgetAccountant()
        self.assertEqual(0, len(acc))

        acc.spend(1, 0)
        acc.spend(1, 0)
        self.assertEqual(2, len(acc))

        acc.spend(1, 0)
        self.assertEqual(3, len(acc))

    def test_repr(self):
        acc = BudgetAccountant()

        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertEqual("BudgetAccountant()", repr(acc))

        acc = BudgetAccountant(epsilon=1, delta=0.01, slack=0.01)
        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertIn("epsilon", repr(acc))
        self.assertIn("delta", repr(acc))
        self.assertIn("slack", repr(acc))
        self.assertNotIn("spent_budget", repr(acc))

        acc = BudgetAccountant(spent_budget=[(1, 0), (0, 1)])
        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertNotIn("epsilon", repr(acc))
        self.assertNotIn("delta", repr(acc))
        self.assertNotIn("slack", repr(acc))
        self.assertIn("spent_budget", repr(acc))

        acc = BudgetAccountant(spent_budget=[(1., 0.)] * 10 + [(5., 0.5)])
        self.assertIn("BudgetAccountant(", repr(acc))
        self.assertIn("...", repr(acc))
        self.assertNotIn("5", repr(acc))
        self.assertNotIn("5", acc.__repr__(10))
        self.assertIn("5", acc.__repr__(11))

    # ------------------------------------------------------------------
    # Regression contract: subsequent-query allocation.
    #
    # The remaining budget for further queries is a composition contract
    # coupling the composition upper bound, the k-query plan, the default
    # accountant context and the state left behind by a failed allocation.
    # A legal plan must consume the available budget in full, an over-limit
    # plan must be rejected without mutating recorded spend, the next call
    # after leaving a context must not inherit a broken budget, and boundary
    # results across numeric scales must remain explainable.
    # ------------------------------------------------------------------

    def test_remaining_contract_combines_prior_spend_and_slack(self):
        # remaining() is not a naive subtraction from the ceiling: the plan
        # must account for every recorded spend and for the slack used in the
        # composition upper bound.
        acc = BudgetAccountant(1.0, 0.0, spent_budget=[(0.3, 0.0)])
        eps, delt = acc.remaining()
        self.assertAlmostEqual(eps, 0.7)
        self.assertEqual(delt, 0.0)

        # Slack relaxes the composition upper bound, so the same k-query plan
        # can legitimately spend more epsilon per query than a slack-free one.
        acc_no_slack = BudgetAccountant(1.0, 1e-4, slack=0.0)
        acc_slack = BudgetAccountant(1.0, 1e-4, slack=1e-6)
        eps_no_slack, _ = acc_no_slack.remaining(100)
        eps_slack, _ = acc_slack.remaining(100)
        self.assertGreater(eps_slack, eps_no_slack)

        # The plan always returns a Budget, never a bare tuple.
        self.assertIsInstance(acc.remaining(4), Budget)

    def test_remaining_plan_monotone_in_k(self):
        # Splitting the same ceiling over more queries can only shrink the
        # per-query epsilon; this makes the boundary value explainable.
        acc = BudgetAccountant(1.0, 1e-3)
        per_query = [acc.remaining(k)[0] for k in (1, 2, 5, 10)]
        for larger, smaller in zip(per_query, per_query[1:]):
            self.assertGreater(larger, smaller)

        # For pure DP the bisection reproduces the analytic ceiling / k.
        acc = BudgetAccountant(1.0, 0.0)
        for k in (2, 3, 5, 10):
            self.assertAlmostEqual(acc.remaining(k)[0], 1.0 / k)

    def test_k_query_plan_fully_consumes_pure_dp_budget(self):
        # A legal k-query plan returned by remaining() must consume the whole
        # available budget, across several k values and numeric scales.
        for epsilon_ceiling in (1e-6, 1.0, 1e6):
            for k in (1, 2, 5, 10):
                acc = BudgetAccountant(epsilon_ceiling, 0.0)
                plan_epsilon, plan_delta = acc.remaining(k)

                for _ in range(k):
                    acc.spend(plan_epsilon, plan_delta)

                total_epsilon, total_delta = acc.total()
                self.assertAlmostEqual(total_epsilon, epsilon_ceiling)
                self.assertEqual(total_delta, 0.0)
                self.assertEqual(len(acc), k)

                # The ceiling is exactly exhausted: a further identical query
                # is rejected.
                with self.assertRaises(BudgetError):
                    acc.spend(plan_epsilon, plan_delta)

    def test_k_query_plan_with_prior_spend_consumes_remaining_budget(self):
        # The plan must be computed against the *remaining* budget, i.e. the
        # ceiling minus the composition of the spends already recorded.
        acc = BudgetAccountant(1.0, 0.0, spent_budget=[(0.5, 0.0)])
        plan_epsilon, plan_delta = acc.remaining(2)
        self.assertAlmostEqual(plan_epsilon, 0.25)

        for _ in range(2):
            acc.spend(plan_epsilon, plan_delta)

        total_epsilon, total_delta = acc.total()
        self.assertAlmostEqual(total_epsilon, 1.0)
        self.assertEqual(total_delta, 0.0)

        with self.assertRaises(BudgetError):
            acc.spend(plan_epsilon, plan_delta)

    def test_replan_after_partial_plan_consumption(self):
        # Re-querying remaining() after part of a plan is spent yields a
        # follow-up plan whose execution still lands exactly on the ceiling.
        acc = BudgetAccountant(1.0, 0.0)
        first_epsilon, _ = acc.remaining(10)

        for _ in range(4):
            acc.spend(first_epsilon, 0.0)

        next_epsilon, next_delta = acc.remaining(6)
        for _ in range(6):
            acc.spend(next_epsilon, next_delta)

        self.assertAlmostEqual(acc.total()[0], 1.0)
        self.assertEqual(acc.total()[1], 0.0)
        self.assertEqual(len(acc), 10)

    def test_slack_k_query_plan_respects_ceiling_and_uses_slack(self):
        # With slack, the advanced-composition plan spends more epsilon per
        # query than the naive division while the composed total stays within
        # the (epsilon, delta) ceiling and charges the slack in delta.
        k = 10
        acc = BudgetAccountant(1.0, 1e-3, slack=1e-6)
        plan_epsilon, _ = acc.remaining(k)
        self.assertGreater(plan_epsilon, 1.0 / k)

        for _ in range(k):
            acc.spend(plan_epsilon, 0.0)

        total_epsilon, total_delta = acc.total()
        self.assertLessEqual(total_epsilon, 1.0)
        self.assertGreaterEqual(total_delta, acc.slack)
        self.assertLessEqual(total_delta, acc.delta)
        # The advanced composition has consumed essentially all the ceiling.
        self.assertAlmostEqual(total_epsilon, 1.0)

    def test_delta_plan_consumes_delta_ceiling(self):
        # Spending only the delta component of the k-query plan (with zero
        # epsilon) composes up to, but never beyond, the delta ceiling.
        k = 5
        acc = BudgetAccountant(1.0, 1e-2)
        _, plan_delta = acc.remaining(k)

        for _ in range(k):
            acc.spend(0.0, plan_delta)

        self.assertAlmostEqual(acc.total()[1], 1e-2)
        self.assertLessEqual(acc.total()[1], acc.delta)

    def test_rejected_spend_preserves_accountant_state(self):
        # An over-limit allocation must be atomic: check() runs before the
        # spend is recorded, so the rejected plan changes neither the spend
        # list, the composed total nor the accountant length.
        acc = BudgetAccountant(1.0, 1e-3, spent_budget=[(0.6, 1e-4)])
        spent_before = acc.spent_budget
        total_before = acc.total()

        with self.assertRaises(BudgetError):
            acc.spend(0.5, 0.0)

        self.assertEqual(acc.spent_budget, spent_before)
        self.assertEqual(acc.total(), total_before)
        self.assertEqual(len(acc), 1)

        # An over-delta plan is rejected with the same atomicity guarantee.
        with self.assertRaises(BudgetError):
            acc.spend(0.1, 1e-2)

        self.assertEqual(acc.spent_budget, spent_before)
        self.assertEqual(acc.total(), total_before)

        # The accountant remains fully usable for a legal follow-up plan.
        self.assertTrue(acc.check(0.4, 5e-4))
        acc.spend(0.4, 5e-4)
        self.assertEqual(len(acc), 2)
        self.assertLessEqual(acc.total(), Budget(acc.epsilon, acc.delta))

    def test_failed_check_does_not_record_spend(self):
        # check() is a pure predicate: neither a passing nor a failing check
        # may leave a trace in the spend history.
        acc = BudgetAccountant(1.0, 0.0)
        self.assertTrue(acc.check(0.4, 0.0))
        self.assertEqual(len(acc), 0)

        with self.assertRaises(BudgetError):
            acc.check(2.0, 0.0)
        self.assertEqual(len(acc), 0)
        self.assertEqual(acc.total(), (0.0, 0.0))

    def test_exact_remaining_plan_accepted_after_rejection(self):
        # After a rejected over-limit plan, the previously computed remaining
        # budget is still the maximal legal plan and must be accepted.
        acc = BudgetAccountant(1.0, 0.0, spent_budget=[(0.6, 0.0)])
        remaining_epsilon, remaining_delta = acc.remaining()

        with self.assertRaises(BudgetError):
            acc.spend(remaining_epsilon * 1.5, remaining_delta)

        self.assertEqual(len(acc), 1)
        acc.spend(remaining_epsilon, remaining_delta)
        self.assertAlmostEqual(acc.total()[0], 1.0)

        with self.assertRaises(BudgetError):
            acc.spend(1e-12, 0.0)

    def test_context_restores_previous_default_after_failure(self):
        # A BudgetError raised inside a context must not strand the failed
        # accountant as the default: the previously active default is restored
        # on exit, and the outer accountant's spend is untouched.
        outer = BudgetAccountant(3.0, 0.0).set_default()
        inner = BudgetAccountant(0.5, 0.0)

        with self.assertRaises(BudgetError):
            with inner:
                self.assertIs(BudgetAccountant._default, inner)
                inner.spend(1.0, 0.0)

        self.assertIs(BudgetAccountant._default, outer)
        self.assertEqual(len(inner), 0)
        self.assertEqual(outer.total(), (0.0, 0.0))
        self.assertFalse(hasattr(inner, "old_default"))

        # The restored default keeps serving subsequent queries on its own
        # budget instead of inheriting the inner accountant's broken state.
        outer.spend(1.0, 0.0)
        self.assertEqual(outer.total(), (1.0, 0.0))

    def test_nested_contexts_restore_default_chain_after_failure(self):
        outer = BudgetAccountant(3.0, 0.0).set_default()
        middle = BudgetAccountant(2.0, 0.0)

        with middle:
            self.assertIs(BudgetAccountant._default, middle)
            inner = BudgetAccountant(0.5, 0.0)

            with self.assertRaises(BudgetError):
                with inner:
                    inner.spend(1.0, 0.0)

            # The innermost failure restores the middle accountant, which is
            # still usable with its own intact budget.
            self.assertIs(BudgetAccountant._default, middle)
            self.assertEqual(len(inner), 0)
            middle.spend(1.0, 0.0)
            self.assertEqual(middle.total(), (1.0, 0.0))

        self.assertIs(BudgetAccountant._default, outer)
        self.assertEqual(outer.total(), (0.0, 0.0))

    def test_context_without_previous_default_clears_default_on_failure(self):
        # With no pre-existing default, exiting a failed context must leave no
        # default behind rather than leaking the failed accountant.
        BudgetAccountant.pop_default()
        failed = BudgetAccountant(1.0, 0.0)

        with self.assertRaises(BudgetError):
            with failed:
                failed.spend(2.0, 0.0)

        self.assertIsNone(BudgetAccountant._default)
        self.assertEqual(len(failed), 0)

        # The next default lookup creates a fresh, unbounded accountant: the
        # failed budget is not inherited.
        fresh = BudgetAccountant.load_default(None)
        self.assertIsNot(fresh, failed)
        self.assertEqual(fresh.epsilon, float("inf"))
        self.assertEqual(len(fresh), 0)
        fresh.spend(5.0, 0.0)
        self.assertEqual(fresh.total(), (5.0, 0.0))

    def test_successful_context_exit_restores_and_clears_default(self):
        # Successful exits obey the same restoration contract as failed ones.
        BudgetAccountant.pop_default()
        with BudgetAccountant(1.0, 0.0) as acc:
            acc.spend(0.5, 0.0)
        self.assertIsNone(BudgetAccountant._default)
        self.assertEqual(acc.total(), (0.5, 0.0))

        outer = BudgetAccountant(2.0, 0.0).set_default()
        with BudgetAccountant(1.0, 0.0) as inner:
            inner.spend(0.5, 0.0)
        self.assertIs(BudgetAccountant._default, outer)

    def test_boundary_plans_across_numeric_scales(self):
        # The k-query boundary must stay explainable at very small and very
        # large epsilon scales: per-query epsilon equals ceiling / k and the
        # executed plan lands back on the ceiling.
        for epsilon_ceiling in (1e-8, 1e-6, 1.0, 1e3, 1e6):
            k = 10
            acc = BudgetAccountant(epsilon_ceiling, 0.0)
            plan_epsilon, plan_delta = acc.remaining(k)

            self.assertAlmostEqual(plan_epsilon * k, epsilon_ceiling)
            self.assertGreater(plan_epsilon, 0.0)

            for _ in range(k):
                acc.spend(plan_epsilon, plan_delta)

            self.assertAlmostEqual(acc.total()[0], epsilon_ceiling)
            self.assertLessEqual(acc.total(), Budget(acc.epsilon, acc.delta))

    def test_floating_point_boundary_is_conservative_and_explainable(self):
        # With many tiny repeated spends, floating-point accumulation can make
        # the literal last spend overshoot the composition computed in bulk.
        # The contract is conservative safety: the accountant never exceeds its
        # ceiling, the rejection leaves prior spends intact, and a slightly
        # smaller plan still exhausts the budget without error.
        k = 25
        acc = BudgetAccountant(1.0, 0.0)
        plan_epsilon, _ = acc.remaining(k)

        for _ in range(k - 1):
            acc.spend(plan_epsilon, 0.0)

        recorded_before_final = len(acc)
        try:
            acc.spend(plan_epsilon, 0.0)
        except BudgetError:
            # Conservative rejection: nothing was appended by the failure.
            self.assertEqual(len(acc), recorded_before_final)

        # Every recorded composition remains within the ceiling, and the
        # unspent head-room is itself explainable (one per-query unit at most).
        self.assertLessEqual(acc.total(), Budget(acc.epsilon, acc.delta))
        self.assertGreaterEqual(acc.remaining(1)[0], 0.0)
        self.assertLess(acc.remaining(1)[0], plan_epsilon * 1.5)

        # A marginally reduced final plan is legal and consumes the head-room.
        head_room, _ = acc.remaining(1)
        acc.spend(head_room, 0.0)
        self.assertAlmostEqual(acc.total()[0], 1.0)
        self.assertEqual(len(acc), k)
