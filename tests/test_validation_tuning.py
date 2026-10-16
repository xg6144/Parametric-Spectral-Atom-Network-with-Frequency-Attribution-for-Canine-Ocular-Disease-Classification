import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import validation_tuning as tuning
from experiments.common import sha256


class ValidationSelectionTests(unittest.TestCase):
    def make_trials(self, root):
        protocol_path = root / 'protocol.json'
        protocol_path.write_text('{}')
        protocol = {'seed': 42, 'objective': 'validation only',
                    'candidates': [{'id': 'c0'}, {'id': 'c1'}, {'id': 'c2'}]}
        paths = []
        for i, score in enumerate([.1, .6, .4]):
            path = root / 'results/tuning/trials' / f'canine_psan_c{i}_s42_screening/summary.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(stage='screening', best_validation_macro_f1=score,
                             protocol_sha256=sha256(protocol_path), tag=f'c{i}')))
            paths.append(path)
        return protocol_path, protocol, paths

    def test_selection_uses_validation_and_rejects_test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path, protocol, paths = self.make_trials(root)
            with patch.object(tuning, 'ROOT', root), patch.object(tuning, 'PROTOCOL', protocol_path):
                selected = tuning.select(protocol, 'psan')
                self.assertEqual(selected['candidate']['id'], 'c1')
                self.assertFalse(selected['test_used'])
                bad = json.loads(paths[0].read_text()); bad['test'] = {'macro_f1': 1.}
                paths[0].write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'test result'):
                    tuning.select(protocol, 'psan')

    def test_selection_rejects_a_different_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path, protocol, paths = self.make_trials(root)
            bad = json.loads(paths[1].read_text()); bad['protocol_sha256'] = 'stale'
            paths[1].write_text(json.dumps(bad))
            with patch.object(tuning, 'ROOT', root), patch.object(tuning, 'PROTOCOL', protocol_path):
                with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                    tuning.select(protocol, 'psan')


if __name__ == '__main__':
    unittest.main()
