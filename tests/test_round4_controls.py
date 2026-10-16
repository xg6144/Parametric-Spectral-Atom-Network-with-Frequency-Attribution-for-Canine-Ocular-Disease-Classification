import copy
import unittest

import torch

from experiments.common import build_model, seed_all
from experiments.round4 import NO_DECAY, cv_split, outer_folds
from models.lowrank import LowRankFilter, freeze_atom_geometry, to_low_rank
from train.train_base import build_param_groups


def psan_ad0(seed=42):
    seed_all(seed)
    return build_model('psan', classes=11, size=224, atom='gabor', dropout=0., atoms=16, init='morlet')


class Round4ControlTests(unittest.TestCase):
    def test_lowrank16_starts_as_the_same_network(self):
        reference = psan_ad0().eval()
        control = to_low_rank(copy.deepcopy(reference), 16).eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            self.assertTrue(torch.allclose(reference(x), control(x), atol=1e-5, rtol=1e-5))
        block = control.blocks[0].filter
        self.assertIsInstance(block, LowRankFilter)
        self.assertEqual(sum(p.numel() for p in block.parameters()), 2 * 192 * 16 + 2 * 16 * 14 * 8)

    def test_lowrank10_keeps_backbone_and_matches_budget(self):
        reference = psan_ad0()
        control = to_low_rank(psan_ad0(), 10)
        self.assertEqual(sum(p.numel() for p in control.blocks[3].filter.parameters()), 6080)
        reference_state = {k: v for k, v in reference.state_dict().items() if '.filter.' not in k}
        control_state = {k: v for k, v in control.state_dict().items() if '.filter.' not in k}
        self.assertEqual(reference_state.keys(), control_state.keys())
        for key in reference_state:
            self.assertTrue(torch.equal(reference_state[key], control_state[key]), key)

    def test_lowrank_filter_is_rank_bounded(self):
        control = to_low_rank(psan_ad0(), 10)
        with torch.no_grad():
            K = control.blocks[0].filter.build_filter().reshape(192, -1)
        self.assertLessEqual(int(torch.linalg.matrix_rank(K)), 10)

    def test_fixed_atoms_receive_no_updates(self):
        model = psan_ad0()
        frozen = freeze_atom_geometry(model)
        self.assertEqual(frozen, 12 * 128)
        groups = build_param_groups(model, 0.05, NO_DECAY)
        optimized = {id(p) for g in groups for p in g['params']}
        block = model.blocks[0].filter
        self.assertNotIn(id(block.mu), optimized)
        self.assertIn(id(block.w_real), optimized)
        before_mu, before_w = block.mu.detach().clone(), block.w_real.detach().clone()
        optimizer = torch.optim.AdamW(groups, lr=1e-2)
        model.train()
        loss = model(torch.randn(2, 3, 224, 224)).square().mean()
        loss.backward(); optimizer.step()
        self.assertTrue(torch.equal(before_mu, block.mu))
        self.assertFalse(torch.equal(before_w, block.w_real))

    def test_basis_receives_no_weight_decay(self):
        model = to_low_rank(psan_ad0(), 16)
        groups = build_param_groups(model, 0.05, NO_DECAY)
        decayed = {id(p) for g in groups if g['weight_decay'] > 0 for p in g['params']}
        block = model.blocks[0].filter
        for parameter in [block.basis_real, block.basis_imag, block.w_real, block.w_imag]:
            self.assertNotIn(id(parameter), decayed)

    def test_cv_folds_are_group_disjoint_and_cover_all_images(self):
        folds = outer_folds()
        self.assertEqual(sorted(folds), [1, 2, 3, 4, 5])
        frame = cv_split(3)
        self.assertEqual(set(frame.split), {'train', 'val', 'test'})
        self.assertTrue((frame[frame.split == 'test'].fold == 3).all())
        self.assertTrue((frame[frame.split == 'val'].fold == 4).all())


if __name__ == '__main__':
    unittest.main()
