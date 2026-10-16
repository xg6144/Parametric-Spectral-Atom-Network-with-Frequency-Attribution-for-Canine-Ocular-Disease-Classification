import copy
import unittest

import torch

from models.psan import (PSANConfig,PSANFilter,PSANNet,build_rfft_freq_grid,
                         frequency_attribution)


class RevisionPSANTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_odd_and_even_frequency_bins(self):
        for h,w in [(14,14),(7,7),(5,9)]:
            u,v=build_rfft_freq_grid(h,w)
            torch.testing.assert_close(u[:,0],2*torch.pi*torch.fft.fftfreq(h))
            torch.testing.assert_close(v[0],2*torch.pi*torch.fft.rfftfreq(w))

    def test_exact_decomposition_with_shared_dropout_mask(self):
        f=PSANFilter(PSANConfig(8,7,7,M=4,atom_dropout=.2)).train()
        x=torch.randn(2,7,7,8)
        for mask in [torch.ones(4),torch.tensor([2.,0.,2.,0.]),torch.zeros(4)]:
            torch.testing.assert_close(f(x,mask),f.per_atom_contribution(x,mask).sum(1),atol=1e-6,rtol=1e-5)

    def test_cache_invalidates_on_state_train_and_parameter_changes(self):
        f=PSANFilter(PSANConfig(8,7,7,M=4)).eval()
        x=torch.randn(2,7,7,8)
        with torch.no_grad():
            reference=f(x);f.cache_filter('checkpoint-test');torch.testing.assert_close(f(x),reference)
            f.w_real.add_(.1);changed=f(x);self.assertIsNone(f._cached_filter)
            self.assertGreater(float((reference-changed).abs().max()),1e-5)
            f.cache_filter();sd=copy.deepcopy(f.state_dict());f.load_state_dict(sd)
            self.assertIsNone(f._cached_filter)
            f.cache_filter();f.train();self.assertIsNone(f._cached_filter)

    def test_attribution_matches_small_mask_directional_derivative(self):
        model=PSANNet(img_size=32,patch_size=16,embed_dim=8,depth=2,
                      num_classes=3,drop_path_rate=0,
                      psan_kwargs=dict(M=4,atom_dropout=0)).double().eval()
        x=torch.randn(1,3,32,32,dtype=torch.float64)
        attr,_=frequency_attribution(model,x,1,block_idx=0)
        f=model.blocks[0].filter
        original=f.forward
        eps=1e-4
        scores=[]
        try:
            with torch.no_grad():
                for sign in [1,-1]:
                    mask=torch.ones(4,dtype=torch.float64);mask[0]+=sign*eps
                    f.forward=lambda inp,mask=mask: original(inp,atom_mask=mask)
                    scores.append(model(x)[0,1])
        finally:f.forward=original
        numerical=(scores[0]-scores[1])/(2*eps)
        torch.testing.assert_close(attr[0,0],numerical,atol=1e-8,rtol=1e-4)

    def test_resolution_change_rebuilds_fft_grid_and_cache(self):
        model=PSANNet(img_size=64,embed_dim=8,depth=1,num_classes=3,
                      psan_kwargs=dict(M=4)).eval()
        with torch.no_grad():
            model.cache_filters();model.resize_input(112)
            self.assertIsNone(model.blocks[0].filter._cached_filter)
            self.assertEqual(model(torch.randn(1,3,112,112)).shape,(1,3))
            torch.testing.assert_close(model.blocks[0].filter.V[0],2*torch.pi*torch.fft.rfftfreq(7))


if __name__=='__main__':unittest.main()
