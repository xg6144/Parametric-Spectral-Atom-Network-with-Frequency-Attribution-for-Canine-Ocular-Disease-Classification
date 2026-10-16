import unittest

import numpy as np

from evaluation_metrics import (
    cluster_bootstrap_ci,
    compute_metric_bundle,
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood,
    paired_cluster_bootstrap,
)


class EvaluationMetricTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        self.perfect = np.eye(3)[self.labels]
        self.groups = np.asarray(["a", "a", "b", "c", "d", "e"])
        self.classes = ["A", "B", "C"]

    def test_perfect_predictions_have_zero_calibration_losses(self):
        self.assertAlmostEqual(
            expected_calibration_error(self.labels, self.perfect),
            0.0,
        )
        self.assertAlmostEqual(multiclass_brier_score(self.labels, self.perfect), 0.0)
        self.assertAlmostEqual(negative_log_likelihood(self.labels, self.perfect), 0.0)

    def test_bundle_contains_requested_minority_metrics(self):
        probabilities = self.perfect * 0.8 + 0.2 / 3.0
        bundle = compute_metric_bundle(
            self.labels,
            probabilities,
            self.classes,
            n_bins=5,
        )
        self.assertIn("ece", bundle["metrics"])
        self.assertIn("brier_score", bundle["metrics"])
        self.assertIn("macro_pr_auc", bundle["metrics"])
        self.assertEqual(len(bundle["per_class"]), 3)
        for row in bundle["per_class"]:
            self.assertIn("specificity", row)
            self.assertIn("sensitivity", row)
            self.assertIn("pr_auc", row)

    def test_group_bootstrap_is_reproducible(self):
        probabilities = self.perfect * 0.8 + 0.2 / 3.0
        first = cluster_bootstrap_ci(
            self.labels,
            probabilities,
            self.groups,
            iterations=20,
            seed=7,
            metric_names=("macro_f1", "ece", "brier_score"),
            n_bins=5,
        )
        second = cluster_bootstrap_ci(
            self.labels,
            probabilities,
            self.groups,
            iterations=20,
            seed=7,
            metric_names=("macro_f1", "ece", "brier_score"),
            n_bins=5,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["groups"], 5)

    def test_paired_bootstrap_reports_a_minus_b(self):
        good = self.perfect * 0.8 + 0.2 / 3.0
        bad = np.roll(good, shift=1, axis=1)
        result = paired_cluster_bootstrap(
            self.labels,
            good,
            bad,
            self.groups,
            iterations=20,
            seed=11,
            metric_names=("macro_f1", "brier_score"),
            n_bins=5,
        )
        self.assertGreater(
            result["metrics"]["macro_f1"]["observed_difference_a_minus_b"],
            0.0,
        )
        self.assertLess(
            result["metrics"]["brier_score"]["observed_difference_a_minus_b"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
