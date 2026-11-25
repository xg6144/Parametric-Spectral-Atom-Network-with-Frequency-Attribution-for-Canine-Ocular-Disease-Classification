"""Numerical and design checks for the major-revision diagnostic/intervention."""
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.round7_prior import effective_full, rank_residual, row_projection_residual, spectrum_metrics
from experiments.round7_support import choose_support


class Round7Checks(unittest.TestCase):
    def test_effective_full_matches_real_fft_operation(self):
        g = torch.Generator().manual_seed(4)
        for width in [7, 8]:
            x = torch.randn(2, 8, width, dtype=torch.float64, generator=g)
            shape = (8, width // 2 + 1)
            k = torch.complex(torch.randn(shape, dtype=torch.float64, generator=g),
                              torch.randn(shape, dtype=torch.float64, generator=g))
            actual = torch.fft.irfft2(torch.fft.rfft2(x, norm='ortho') * k, s=(8, width), norm='ortho')
            expanded = torch.fft.ifft2(torch.fft.fft2(x, norm='ortho') * effective_full(k, width), norm='ortho')
            self.assertTrue(torch.allclose(actual, expanded.real, atol=1e-12))
            self.assertLess(float(expanded.imag.abs().max()), 1e-12)

    def test_exact_basis_span_and_svd_bound(self):
        g = torch.Generator().manual_seed(9)
        basis = torch.randn(5, 17, dtype=torch.float64, generator=g)
        target = torch.randn(9, 5, dtype=torch.float64, generator=g) @ basis
        residual, rank = row_projection_residual(target, basis)
        self.assertEqual(rank, 5)
        self.assertLess(residual, 1e-25)
        self.assertLess(rank_residual(target, 5), 1e-25)
        random_target = torch.randn(9, 17, dtype=torch.float64, generator=g)
        residual, _ = row_projection_residual(random_target, basis)
        self.assertGreaterEqual(residual + 1e-12, rank_residual(random_target, 5))

    def test_constant_and_alternating_frequency_summaries(self):
        constant = spectrum_metrics(torch.ones(2, 8, 8, 3))
        np.testing.assert_allclose(constant['dc_fraction'], 1)
        alternating = torch.tensor([1., -1.] * 4).view(1, 1, 8, 1).expand(2, 8, 8, 3)
        stats = spectrum_metrics(alternating)
        np.testing.assert_allclose(stats['dc_fraction'], 0, atol=1e-12)
        np.testing.assert_allclose(stats['centroid_non_dc'], 1, atol=1e-12)
        np.testing.assert_allclose(stats['low_non_dc_fraction'], 0, atol=1e-12)

    def test_supports_nested_and_whole_group(self):
        records = []
        for label, count in [(0, 30), (1, 12), (2, 3)]:
            for group in range(count):
                records.extend([dict(label=label, group_id=f'{label}/{group}')] * (1 + group % 3))
        frame = pd.DataFrame(records)
        supports = [choose_support(frame, f, 42) for f in [10, 25, 100]]
        self.assertTrue(set(supports[0]) <= set(supports[1]) <= set(supports[2]))
        self.assertEqual(set(supports[2]), set(range(len(frame))))
        full_counts = np.bincount(frame.label)
        for ids in supports:
            selected = frame.iloc[ids]
            group_ids = selected.group_id.unique()
            self.assertEqual(set(ids), set(np.flatnonzero(frame.group_id.isin(group_ids))))
            counts = np.bincount(selected.label, minlength=3)
            self.assertTrue((counts > 0).all())
            weights = np.sqrt(full_counts)[selected.label] / counts[selected.label]
            masses = np.bincount(selected.label, weights=weights, minlength=3)
            np.testing.assert_allclose(masses, np.sqrt(full_counts))

    def test_inconsistent_group_label_rejected(self):
        frame = pd.DataFrame(dict(group_id=['same', 'same'], label=[0, 1]))
        with self.assertRaises(ValueError): choose_support(frame, 10, 42)


if __name__ == '__main__': unittest.main()
