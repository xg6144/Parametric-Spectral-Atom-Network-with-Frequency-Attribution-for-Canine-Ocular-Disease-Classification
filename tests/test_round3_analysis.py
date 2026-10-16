import unittest

import numpy as np

from experiments.round3_analysis import GroupBootstrap, crossed, holm, observed_f1, tail
from experiments.statistics import weighted_macro_f1


class Round3AnalysisTests(unittest.TestCase):
    def test_vectorized_bootstrap_matches_weighted_reference(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 5, 400)
        pred = np.where(rng.random(400) < .7, y, rng.integers(0, 5, 400))
        groups = rng.integers(0, 120, 400)
        boot = GroupBootstrap(groups, b=50, seed=1)
        observed, distribution = boot.macro_f1(y, pred, 5)
        self.assertAlmostEqual(observed, weighted_macro_f1(y, pred, np.ones(len(y)), 5), places=12)
        for i in range(5):
            weights = boot.counts[i][boot.codes]
            self.assertAlmostEqual(distribution[i], weighted_macro_f1(y, pred, weights, 5), places=5)

    def test_observed_f1_subset_of_labels(self):
        y = np.array([0, 1, 2, 2]); pred = np.array([0, 2, 2, 1])
        full = observed_f1(y, pred, 3)
        subset = observed_f1(y, pred, 3, [1, 2])
        self.assertAlmostEqual(full, (1 + 0 + .5) / 3)
        self.assertAlmostEqual(subset, (0 + .5) / 2)

    def test_tail_probability_floor_and_symmetry(self):
        self.assertAlmostEqual(tail(np.ones(9999)), 2 / 10000)
        self.assertAlmostEqual(tail(np.r_[-np.ones(50), np.ones(50)]), 1.)

    def test_holm_is_monotone_and_bounded(self):
        adjusted = holm([.01, .04, .03, .2])
        np.testing.assert_allclose(adjusted, [.04, .09, .09, .2])
        self.assertTrue(np.all(holm([.9, .8]) <= 1))

    def test_crossed_summary_counts_positive_seeds(self):
        rows = [dict(comparison='a', seed=s, delta_macro_f1=d, _dist=np.full(200, d)) for s, d in [(42, .01), (43, .02), (44, .03)]]
        summary = crossed(rows, 'a')
        self.assertEqual(summary['seeds_positive'], 3)
        self.assertAlmostEqual(summary['mean_delta'], .02)
        self.assertGreater(summary['crossed_ci_low'], 0)


if __name__ == '__main__':
    unittest.main()
