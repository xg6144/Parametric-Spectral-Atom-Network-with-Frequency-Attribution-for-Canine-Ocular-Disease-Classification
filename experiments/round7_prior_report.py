"""Summarize all fixed diagnostics without selecting a favorable metric/domain."""
import json

import pandas as pd

from .common import ROOT, sha256, write_json
from .round7_prior import OUT, PROTOCOL


def run():
    provenance = json.loads((OUT / 'provenance.json').read_text())
    if sha256(PROTOCOL) != provenance['protocol_sha256']:
        raise ValueError('Prior protocol changed.')
    for name, digest in provenance['output_sha256'].items():
        if sha256(OUT / name) != digest: raise ValueError(f'Diagnostic output changed: {name}')
    filters = pd.read_csv(OUT / 'filter_per_seed.csv')
    filters['operator_adaptation_gain'] = (filters.initialized_operator_projection_residual
                                           - filters.learned_operator_projection_residual)
    columns = [c for c in filters if c not in ['dataset', 'seed']]
    summary = filters.groupby('dataset')[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / 'filter_summary.csv', index=False)
    tokens = pd.read_csv(OUT / 'token_per_seed_block.csv')
    summary = tokens.groupby(['dataset', 'arm', 'block'])[
        ['dc_fraction', 'low_non_dc_fraction', 'centroid_non_dc', 'entropy_non_dc']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / 'token_summary.csv', index=False)
    raw = [json.loads(p.read_text()) for p in sorted(OUT.glob('*_input_summary.json'))]
    pd.DataFrame(raw).to_csv(OUT / 'input_summary.csv', index=False)
    text = '''# Interpretation of the frozen validation-spectrum/filter-affinity diagnostics

These are post-hoc descriptive analyses, specified with all submitted performance results known. They use only validation images and existing validation-selected checkpoints, with the same128-pixel/8×8 grid in all four domains. They do not provide external canine or clinical validation.

All metrics and all domains/seeds/blocks are retained. Input and token summaries give equal weight to each class after taking a fixed random sample of up to64 images per class. They do not estimate the natural acquisition or disease distribution. The correctedDerma validation sample is smaller because several classes have fewer than64 available images.

The numerical feature/filter differences do not support a simple rule that low-frequency or smoother datasets uniquely favor PSAN. Non-DC raw-image spectral centroids are similar for canine,Blood andDerma, while learned dense-filter roughness and optimal rank16 residual do not order domains in the same way as the submitted PSAN–dense accuracy findings. The lack of statistical significance on MedMNIST+ remains distinct from equivalence.

The learned PSAN bank reduces the exact dense-operator projection residual relative to the initialized bank more on canine than in the three other domains (canine .06485; Blood .00283; Derma .00868; OCT .03085; descriptive means over three seeds, differences in squared relative residual). However, the absolute projection residual is lower onDerma than oncanine. This is an affinity diagnostic for separately trained architectures, not a performance bound, evidence that their channel representations are equivalent, or a raw-data cause. The optimal complex rank16 and real-operator rank32 residuals are separate unconstrained approximation references, not guarantees about the learned atom bank.

The strongest warranted conclusion is that the bank adapts differently across these training domains and that the current simple descriptors do not identify a reliable moderator. A controlled intervention on the number of unique training groups at fixed optimization exposure is evaluated separately; its complete results must be reported regardless of direction. No regression or significance test over only four domains is used here.
'''
    (OUT / 'interpretation.md').write_text(text)
    write_json(OUT / 'report_provenance.json', dict(
        report_source_sha256=sha256(ROOT / 'experiments/round7_prior_report.py'),
        analysis_provenance_sha256=sha256(OUT / 'provenance.json'),
        output_sha256={name: sha256(OUT / name) for name in
                       ['filter_summary.csv', 'token_summary.csv', 'input_summary.csv', 'interpretation.md']}))


if __name__ == '__main__': run()
