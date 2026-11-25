"""Verify every canonical v3 inference cell and index its exact content hash."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import ROOT, sha256, write_json


def main():
    out = ROOT / 'results/round7/efficiency_v3'
    protocol_path = ROOT / 'configs/round7_efficiency_v3_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for path, expected in protocol['source_code_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    cells = []
    for mode in protocol['modes']:
        for batch in protocol['batches']:
            for replicate in protocol['replicates']:
                path = out / 'cells' / f'{mode}_b{batch}_r{replicate}.json'
                record = json.loads(path.read_text())
                assert (record['mode'], record['batch'], record['replicate']) == (mode, batch, replicate)
                assert record['protocol_sha256'] == sha256(protocol_path)
                assert record['source_code_sha256'] == protocol['source_code_sha256']['experiments/round7_efficiency.py']
                assert record['gpu_uuid'] == protocol['gpu']['uuid']
                assert record['physical_gpu_index'] == 1
                assert record['warmup'] == 20 and record['iterations'] == 100
                assert record['precision'] == 'float32'
                assert record['matmul_allow_tf32'] is False and record['cudnn_allow_tf32'] is True
                arm = 'dense' if mode == 'dense' else 'psan'
                assert record['checkpoint_sha256'] == protocol['checkpoints'][arm]['sha256']
                values = np.asarray(record['latency_ms_values'])
                assert len(values) == 100 and np.all(values > 0) and np.all(np.isfinite(values))
                for key, q in [('q25', 25), ('median', 50), ('q75', 75)]:
                    np.testing.assert_allclose(record[f'latency_ms_{key}'], np.percentile(values, q), atol=1e-12)
                assert record['peak_allocated_bytes'] >= record['allocated_before_bytes']
                assert record['peak_reserved_bytes'] >= record['peak_allocated_bytes']
                assert record['cache_bytes'] == (2064384 if mode == 'psan_cached' else 0)
                assert record['cache_tensors'] == (12 if mode == 'psan_cached' else 0)
                if mode == 'psan_cached':
                    assert record['synthetic_equivalence']['argmax_equal'] is True
                    assert record['synthetic_equivalence']['max_logit_abs_difference'] <= 1e-5
                    assert len(record['cache_construction']['values_ms']) == 20
                cells.append((path, record))
    assert len(cells) == 27
    assert len({record['process_id'] for _, record in cells}) == 27
    summary = pd.read_csv(out / 'summary.csv')
    for _, row in summary.iterrows():
        medians = [record['latency_ms_median'] for _, record in cells
                   if record['mode'] == row['mode'] and record['batch'] == row['batch']]
        np.testing.assert_allclose(row['latency_ms_median_of_process_medians'], np.median(medians), atol=1e-12)
    previous = protocol['historical_full_test_equality']
    assert sha256(ROOT / previous['source']) == previous['sha256']
    write_json(out / 'verification.json', {
        'status': 'passed', 'canonical_protocol': str(protocol_path.relative_to(ROOT)),
        'protocol_sha256': sha256(protocol_path), 'audit_source_sha256': sha256(__file__),
        'verified_process_isolated_cells': 27, 'unique_worker_processes': 27,
        'synthetic_equivalence_cells': 9,
        'cell_sha256': {str(path.relative_to(ROOT)): sha256(path) for path, _ in cells},
        'summary_sha256': sha256(out / 'summary.csv'),
        'comparisons_sha256': sha256(out / 'matched_comparisons.csv'),
        'provenance_sha256': sha256(out / 'provenance.json'),
        'excluded_versions': ['v1 source mismatch', 'v2 matched-dense loading failure'],
        'execution_note': 'V3 resumed after one fresh worker failed during CUDA initialization. The 24 completed cells were retained, and only the 3 missing cells were run on resume; no failed timing entered aggregation.',
    })
    print('Verified all 27 v3 cells, 27 unique worker processes and 9 synthetic cache equivalence checks.', flush=True)


if __name__ == '__main__':
    main()
