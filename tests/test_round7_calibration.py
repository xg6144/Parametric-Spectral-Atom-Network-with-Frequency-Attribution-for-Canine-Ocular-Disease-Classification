"""Numerical checks for the reviewer-driven calibration experiment."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import softmax

from experiments import round7_calibration as cal
from evaluation_metrics import compute_metric_bundle


class CalibrationExperimentTests(unittest.TestCase):
    def test_vector_recovers_class_dependent_miscalibration_on_unseen_samples(self):
        rng = np.random.default_rng(91023)
        truth = rng.normal(size=(12000, 4))
        probabilities = softmax(truth, axis=1)
        labels = (rng.random(len(truth))[:, None] > probabilities.cumsum(1)).sum(1)
        observed = (truth - np.array([1., -.7, .4, -.5])) / np.array([.5, 1.6, .8, 1.2])
        train, test = observed[:8000], observed[8000:]
        fit_y, test_y = labels[:8000], labels[8000:]
        temperature = cal.fit_temperature(train, fit_y)
        vector = cal.fit_vector(train, fit_y, .001)
        scalar_loss = cal.nll(test / temperature, test_y).mean()
        vector_loss = cal.nll(cal.transform(test, vector), test_y).mean()
        self.assertLess(vector_loss, scalar_loss - .08)
        self.assertTrue(np.all(np.array(vector['scale']) > 0))
        np.testing.assert_allclose(softmax(cal.transform(test, vector), axis=1).sum(1), 1., atol=1e-12)

    def test_positive_temperature_preserves_decisions(self):
        logits = np.array([[12., -3., 2.], [-2., 9., 1.], [.2, .1, .3], [2., 1., 0.]])
        labels = np.array([0, 1, 1, 2])
        temperature = cal.fit_temperature(logits, labels)
        self.assertGreater(temperature, 0.)
        np.testing.assert_array_equal(logits.argmax(1), (logits / temperature).argmax(1))

    def test_group_bootstrap_keeps_duplicate_rows_paired(self):
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        frame = pd.DataFrame({'path': [str(i) for i in range(8)],
                              'group_id': np.repeat(['a', 'b', 'c', 'd'], 2), 'label': labels})
        scalar = np.tile([.7, .3], (8, 1))
        vector = np.array([[.9, .1] if y == 0 else [.1, .9] for y in labels])
        saved = {}
        for seed in cal.SEEDS:
            saved[('psan', seed)] = dict(frame=frame, probabilities={'temperature': scalar, 'vector': vector},
                metrics={method: compute_metric_bundle(labels, probabilities, ['zero', 'one'])['metrics']
                         for method, probabilities in [('temperature', scalar), ('vector', vector)]})
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(cal, 'OUT', Path(temporary)):
                contrasts = cal.bootstrap('psan', saved)
        f1 = next(row for row in contrasts if row['metric'] == 'macro_f1')
        expected = saved[('psan', 42)]['metrics']['vector']['macro_f1'] - saved[('psan', 42)]['metrics']['temperature']['macro_f1']
        self.assertAlmostEqual(f1['mean_delta'], expected)
        self.assertGreater(f1['mean_delta'], 0.)
        self.assertTrue(all(np.isfinite(row['ci_low']) and np.isfinite(row['ci_high']) for row in contrasts))

    def test_class_ece_matches_known_perfect_probabilities(self):
        labels = np.array([0, 1, 2, 0])
        probabilities = np.eye(3)[labels]
        np.testing.assert_allclose(cal.class_ece(labels, probabilities), [0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
