import unittest

import numpy as np
import torch

from experiments.round7_prior import effective_full, rank_residual, row_projection_residual, spectrum_metrics


class PriorDiagnosticsTests(unittest.TestCase):
    def test_effective_filter_reproduces_irfft_for_arbitrary_complex_filter(self):
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(3, 8, 8, dtype=torch.float64, generator=generator)
        k = torch.complex(torch.randn(3, 8, 5, dtype=torch.float64, generator=generator),
                          torch.randn(3, 8, 5, dtype=torch.float64, generator=generator))
        expected = torch.fft.irfft2(torch.fft.rfft2(x, norm='ortho')*k, s=(8, 8), norm='ortho')
        expanded = effective_full(k, 8)
        actual = torch.fft.ifft2(torch.fft.fft2(x, norm='ortho')*expanded, norm='ortho')
        torch.testing.assert_close(actual.real, expected, atol=1e-12, rtol=1e-12)
        self.assertLess(float(actual.imag.abs().max()), 1e-12)

    def test_projection_recovers_known_row_span_and_respects_rank_bound(self):
        generator = torch.Generator().manual_seed(19)
        basis = torch.randn(5, 30, dtype=torch.float64, generator=generator).to(torch.complex128)
        coefficients = torch.complex(torch.randn(12, 5, dtype=torch.float64, generator=generator),
                                     torch.randn(12, 5, dtype=torch.float64, generator=generator))
        exact = coefficients@basis
        residual, rank = row_projection_residual(exact, basis)
        self.assertEqual(rank, 5)
        self.assertLess(residual, 1e-24)
        noisy = exact + torch.randn(12, 30, dtype=torch.float64, generator=generator)
        residual, _ = row_projection_residual(noisy, basis)
        self.assertGreaterEqual(residual+1e-12, rank_residual(noisy, 5))

    def test_spectral_summary_known_constant_and_single_frequency(self):
        constant = torch.ones(1, 8, 8, 2, dtype=torch.float64)
        summary = spectrum_metrics(constant)
        np.testing.assert_allclose(summary['dc_fraction'], [1.], atol=1e-12)
        np.testing.assert_allclose(summary['entropy_non_dc'], [0.], atol=1e-12)
        wave = torch.cos(2*torch.pi*torch.arange(8, dtype=torch.float64)/8)
        summary = spectrum_metrics(wave[None, :, None, None].expand(1, 8, 8, 1))
        np.testing.assert_allclose(summary['dc_fraction'], [0.], atol=1e-12)
        np.testing.assert_allclose(summary['centroid_non_dc'], [.25], atol=1e-12)
        np.testing.assert_allclose(summary['low_non_dc_fraction'], [1.], atol=1e-12)
        np.testing.assert_allclose(summary['entropy_non_dc'], [np.log(2)/np.log(63)], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
