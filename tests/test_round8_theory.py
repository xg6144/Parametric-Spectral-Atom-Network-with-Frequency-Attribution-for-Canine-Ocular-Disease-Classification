"""Counterexamples and bounds for real-FFT filter statements in the revision.

These checks exercise the implemented FFT projection without changing any
training source or reproducing an implementation-shaped assertion alone.
"""
import itertools
import unittest

import numpy as np
import torch

from experiments.round7_prior import effective_full


class HermitianProjectionTheoryTests(unittest.TestCase):
    def test_projected_stored_matrix_has_doubled_complex_span_bound(self):
        generator = torch.Generator().manual_seed(20261109)
        for height, width, channels, atoms in [(5, 7, 9, 2), (6, 8, 9, 2), (4, 2, 7, 3)]:
            bins = height * (width // 2 + 1)
            weights = torch.complex(
                torch.randn(channels, atoms, generator=generator, dtype=torch.float64),
                torch.randn(channels, atoms, generator=generator, dtype=torch.float64))
            basis = torch.complex(
                torch.randn(atoms, bins, generator=generator, dtype=torch.float64),
                torch.randn(atoms, bins, generator=generator, dtype=torch.float64))
            stored = (weights @ basis).reshape(channels, height, width // 2 + 1)
            projected = effective_full(stored, width)[..., :width // 2 + 1].reshape(channels, bins)
            diagonal = torch.zeros(bins, dtype=torch.complex128)
            boundary_columns = [0, width // 2] if width % 2 == 0 else [0]
            boundary_indices = []
            permutation = []
            for u in range(height):
                for v in range(width // 2 + 1):
                    permutation.append(((-u) % height) * (width // 2 + 1) + v)
                    if v in boundary_columns:
                        diagonal[u * (width // 2 + 1) + v] = 1
                        boundary_indices.append(u * (width // 2 + 1) + v)
            matrix = stored.reshape(channels, bins)
            formula = matrix * (1 - diagonal / 2) + matrix.conj()[:, permutation] * diagonal / 2
            torch.testing.assert_close(projected, formula, atol=1e-12, rtol=1e-12)
            self.assertLessEqual(int(torch.linalg.matrix_rank(matrix)), atoms)
            self.assertLessEqual(int(torch.linalg.matrix_rank(projected)), min(channels, bins, 2 * atoms))
            self.assertLessEqual(int(torch.linalg.matrix_rank(projected[:, boundary_indices])),
                                 min(channels, len(boundary_indices), 2 * atoms))

    def test_rank_can_increase_from_one_to_two_after_projection(self):
        weights = torch.tensor([[1.], [1j]], dtype=torch.complex128)
        basis = torch.ones(1, 2, 3, dtype=torch.complex128)
        basis[0, 0, 0] = 1 + 1j
        stored = (weights @ basis.flatten(1)).reshape(2, 2, 3)
        projected = effective_full(stored, 4)[..., :3].flatten(1)
        self.assertEqual(int(torch.linalg.matrix_rank(stored.flatten(1))), 1)
        self.assertEqual(int(torch.linalg.matrix_rank(projected)), 2)

    def test_nonzero_stored_dc_can_produce_a_singular_real_operator(self):
        height = width = 4
        stored = torch.ones(height, width // 2 + 1, dtype=torch.complex128)
        stored[0, 0] = 1j
        self.assertTrue(torch.all(stored.abs() > 0))
        effective = effective_full(stored, width)
        self.assertEqual(effective[0, 0], 0)
        constants = torch.ones(height, width, dtype=torch.float64)
        output = torch.fft.irfft2(torch.fft.rfft2(constants, norm='ortho') * stored,
                                 s=(height, width), norm='ortho')
        torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-12, rtol=0)
        inputs = torch.eye(height * width, dtype=torch.float64).reshape(-1, height, width)
        outputs = torch.fft.irfft2(torch.fft.rfft2(inputs, norm='ortho') * stored,
                                  s=(height, width), norm='ortho')
        operator = outputs.reshape(height * width, height * width).T
        self.assertEqual(int(torch.linalg.matrix_rank(operator)), height * width - 1)
        restored = stored.clone(); restored[0, 0] = 1 + 1j
        full = torch.fft.irfft2(torch.fft.rfft2(inputs, norm='ortho') * restored,
                               s=(height, width), norm='ortho').reshape(height * width, height * width).T
        self.assertEqual(int(torch.linalg.matrix_rank(full)), height * width)


class AtomDropoutExpectationTests(unittest.TestCase):
    def test_unconditional_and_nonempty_mask_conditional_expectations(self):
        for atoms in [1, 2, 4, 5]:
            masks = np.asarray(list(itertools.product([0, 1], repeat=atoms)), dtype=float)
            survivors = masks.sum(axis=1)
            multiplier = np.divide(atoms * masks, survivors[:, None],
                                   out=np.zeros_like(masks), where=survivors[:, None] > 0)
            for probability in [0., .2, .7]:
                mass = (1 - probability) ** survivors * probability ** (atoms - survivors)
                expectation = mass @ multiplier
                np.testing.assert_allclose(expectation, 1 - probability ** atoms, atol=1e-14)
                np.testing.assert_allclose(expectation / (1 - probability ** atoms), 1., atol=1e-14)
                specific_survival = (mass * multiplier[:, 0]).sum() / (1 - probability)
                self.assertAlmostEqual(specific_survival, (1 - probability ** atoms) / (1 - probability))
                if atoms > 1 and probability > 0:
                    self.assertGreater(specific_survival, 1.)


if __name__ == '__main__':
    unittest.main()
