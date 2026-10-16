import unittest

import numpy as np
from sklearn.metrics import f1_score

from experiments.statistics import factorial_scores, weighted_macro_f1


class FollowupStatisticsTests(unittest.TestCase):
    def test_weighted_f1_matches_reference_with_absent_classes_and_zero_weights(self):
        rng = np.random.default_rng(17)
        y = rng.integers(0, 5, 73)
        pred = rng.integers(0, 9, 73)
        weights = rng.integers(0, 4, 73)
        expected = f1_score(y, pred, labels=range(11), average='macro',
                            zero_division=0, sample_weight=weights)
        self.assertAlmostEqual(weighted_macro_f1(y, pred, weights), expected, places=14)

    def test_factorial_interaction_is_difference_of_differences(self):
        # Known additive effects have zero interaction; interaction changes one cell.
        values = np.asarray([[.5, .6, .7, .8], [.5, .6, .7, .85]])
        result = factorial_scores(values)
        np.testing.assert_allclose(result[0], [.2, .2, .1, .1, .2, .1, 0], atol=1e-14)
        self.assertAlmostEqual(result[1, -1], .05)


if __name__ == '__main__':
    unittest.main()
