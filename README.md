# PSAN: Parametric Spectral Atom Network

Code and results for the accepted *Applied Sciences* manuscript *PSAN: Parametric Spectral Atom Network with
Frequency Attribution for Canine Ocular Disease Classification* by Dongbeen Jeon, Youngrok Kim, Kyoungyee Kim
and Heecheol Kim.

PSAN combines shared learnable Gaussian or Gabor-like atoms to build complex spectral filters. The atom bank
bounds the complex rank of the stored channel–frequency matrix and yields an exact decomposition of the filter
output into atom contributions. The manuscript evaluates PSAN on an audited canine ocular benchmark and three
MedMNIST+ datasets. The reported architectural comparisons use models trained from scratch.

This release includes the scientific code, frozen protocols, aggregate tables and run records for the journal
revision, including the final controls in Round 9. It also provides aggregate results for the original comparisons.
The previously published cross-validation and diagnostic artifacts remain available.

## Verify the release

```bash
python tools/verify_release.py
```

The script uses NumPy, pandas and scikit-learn. It checks the public file manifest, the frozen protocol source
hashes and the Macro F1 values of the prediction files present in the repository. It also checks fold assignment
when a local audited split is present. Unit tests for the experiments use the class folders in `data_raw/`. The
cross-validation split test also reads the locally reconstructed `results/revision_split.csv`.

## Files

| Path | Contents |
|---|---|
| `models/` | PSAN, GFNet, AFNO and the matched spectral filter controls |
| `experiments/` | Training, validation selection, evaluation and analyses |
| `configs/` | Frozen protocols, run settings and source digests |
| `environment/` | Recorded package versions and precision settings |
| `results/ARTICLE_SOURCES.csv` | Public result files used to populate manuscript numbers and their source hashes |
| `results/round7/`, `results/round8/`, `results/round9/` | Revision results, run records and provenance |
| `results/round3/tables/` through `results/round6/tables/` and `results/tables/` | Aggregate results for the earlier comparisons |
| `MANIFEST.csv` | SHA-256 and size of each public file other than the manifest itself |

## Environment

The reported runs used Python 3.13.13, PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.26, NumPy 2.4.4,
pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0 and medmnist 3.0.2. See `environment/versions.json` and
`environment/precision_protocol.json` for the recorded environment, including single precision and the TF32
settings. The conda and pip snapshots are in `environment/conda-explicit.txt` and `environment/pip-freeze.txt`.

## Data

**Canine ocular photographs.** Obtain the Companion Animal Ocular Disease Data from [AI-Hub](https://aihub.or.kr/)
(dataset 562) under its terms of access. Place the images and their JSON records in `data_raw/<class>/` using the
original filenames. The experiment modules read class names from these directories. With the dataset in place,
run the audit locally in a separate working copy:

```bash
python run_experiments.py --phase audit
```

The audit uses the earlier group assignments in `cv_splits/stratified_group_5fold.csv` and writes the audited
104,037-image split and its metadata-bearing training manifest locally. Compare the SHA-256 of the rebuilt
`results/revision_split.csv` with `revision_manifest_sha256` in the **released** `results/corrected_integrity.json`
before treating the reconstruction as exact. If you already have the audited `results/corrected_sgkf5.csv` locally,
`python tools/restore_split_manifest.py --check-images` rebuilds the training manifest and verifies the recorded
manifest and image hashes.

**MedMNIST+.** Download the 128-pixel BloodMNIST, OCTMNIST and DermaMNIST files from the
[MedMNIST+ record](https://zenodo.org/records/10519652) into `results/external_data/round3_128/`. Obtain the corrected
DermaMNIST mapping from [Corrected-Skin-Image-Datasets](https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets)
using the source and filenames in `results/external_data/derma_mapping/SOURCE.txt`. Run
`python -m experiments.derma_corrected` to prepare the corrected 64-pixel mapping. Then run
`python -m experiments.round3 prepare` to verify the official 128-pixel archives and apply the lesion-disjoint
mapping to DermaMNIST.

A full rerun generates per-image predictions and checkpoints from locally prepared data. The revision results
in this release are aggregate records, with checksums identifying the local analysis inputs. AI-Hub images and
JSON records remain available through AI-Hub's authorized access process.

## Stages

| Stage | Code | Protocol | Experiments |
|---|---|---|---|
| Round 2 | `run_experiments.py`, `experiments/training.py`, `validation_tuning.py`, `tuning_results.py`, `queue_ofat.py`, `round2_queue.py`, `outer_cv.py`, `analysis.py`, `statistics.py`, `robustness.py`, `faithfulness.py`, `selected_efficiency.py` | `configs/revision.json`, `round2_protocol.json`, `round2_statistics.json`, `validation_tuning.json`, `outer_cv.json` | audit and split; PSAN, GFNet, AFNO and the matched dense control under a shared preset; factorial and one-factor ablations; validation-only tuning; outer cross-validation; calibration, robustness, attribution and efficiency |
| Round 3 | `experiments/round3.py`, `round3_analysis.py`, `round3_figures.py` | `configs/round3_protocol.json` | validation-selected dense control and PSAN without atom dropout; further seeds of the selected settings; atom-count replication; MedMNIST+ benchmarks |
| Round 4 | `experiments/round4.py`, `round4_analysis.py`, `round4_attribution.py`, `round4_figures.py` | `configs/round4_protocol.json` | free low-rank and fixed-atom controls; five-fold cross-validation; 100-epoch runs; seed stability of attributions |
| Round 5 | `experiments/round5.py`, `round5_analysis.py` | `configs/round5_protocol.json` | DeiT-Tiny, ConvNeXt-Atto, FastViT-T8, Swin-Tiny and SHViT-S1 trained from scratch |
| Round 6 | `experiments/round6.py`, `round6_analysis.py`, `round6_figures.py`, `round6_posthoc.py`, `round6_factorial.py`, `round6_robustness.py` | `configs/round6_protocol.json`, `round6_posthoc_protocol.json`, `round6_factorial_protocol.json` | nested cross-validation; training without Normal images; 100-epoch runs at three seeds; 128-pixel input; post hoc checks of the disease-only comparison; robustness of the matched dense control |
| Round 7 | `experiments/round7_calibration.py`, `round7_dropout.py`, `round7_efficiency.py`, `round7_efficiency_audit.py`, `round7_prior.py`, `round7_prior_report.py`, `round7_support.py`, `round7_support_analysis.py`, `round7_support_sensitivity.py` | `configs/round7_calibration_protocol.json`, `round7_dropout_protocol.json`, `round7_efficiency_v3_protocol.json`, `round7_prior_protocol.json`, `round7_support_protocol.json`, `round7_support_sensitivity_protocol.json` | vector recalibration with validation-only selection; the two missing 100-epoch seeds without atom dropout; matched inference timing in fresh processes; validation-spectrum and filter diagnostics; training-group support (10%, 25%, 100%) on BloodMNIST and corrected DermaMNIST; sensitivity to test rasters that also occur in training or validation |
| Round 8 | `experiments/round8_external_sensitivity.py` | `configs/round8_external_sensitivity_protocol.json` | the same raster sensitivity for the validation-selected BloodMNIST models |
| Round 9 | `experiments/round9.py`, `round9_analysis.py`, `round9_timing.py` | `configs/round9_protocol.json`, `round9_timing_protocol.json` | post hoc controls: matched dense filter with weight decay; BloodMNIST and corrected DermaMNIST runs matched to the optimizer steps of a 30-epoch canine run; repetition of the inference timing on an idle host |

The released code is byte-identical to the code that produced the results. From round 3 on, the protocol of each
stage records the SHA-256 of its training sources, and the `freeze` step refuses to proceed if a source differs. The
round-2 runs used the frozen sources in `revision_history/20261013_round2_training_source/`. They differ from the
released files only in `experiments/common.py`, where the timm model factory of the released version does not pass a
stochastic-depth rate to SHViT, which does not accept one. No released round-2 run uses that factory.


Use a separate working copy to rerun experiments. Move the released run directories for the runs you intend to
repeat to a backup location because training uses `summary.json` as a completion marker. The module docstrings
give the commands and inputs for each protocol. Use the distributed frozen protocols for the reported settings.

For example, after preparing the datasets and reproducing the earlier comparison runs, the Round 9 training
and analysis are:

```bash
python -m experiments.round9 worker --exp wd-screen --device cuda:0
python -m experiments.round9 select
python -m experiments.round9 worker --exp wd-final --device cuda:0
python -m experiments.round9 worker --exp steps --device cuda:0
python -m experiments.round9 report
```

The reported paired differences use 10,000 bootstrap resamples. Resampling uses visual groups for canine,
lesions for corrected DermaMNIST and images otherwise. Multi-seed contrasts resample both groups and seeds.
Confidence intervals are unadjusted. The p values use the correction and comparison family in each frozen protocol.


## Journal revision (rounds 7–9)

The revision analyses were specified after earlier results were known. Their protocols identify the fixed
configurations and statistical comparisons.

| Analysis | Tables |
|---|---|
| Spectral diagnostics and training-group support | [Prior](results/round7/prior), [support](results/round7/support/tables) |
| Atom dropout and class-wise recalibration | [Dropout](results/round7/dropout/tables), [calibration](results/round7/calibration) |
| Official split raster sensitivity | [Fixed-recipe training support](results/round7/support_sensitivity/tables), [validation-selected BloodMNIST](results/round8/external_sensitivity/tables) |
| Final dense-filter weight-decay and step-matched external controls | [Round 9 contrasts and run summaries](results/round9/tables) |
| Matched inference latency and memory | [Original timing session](results/round7/efficiency_v3/summary.csv), [final idle-host session](results/round9/timing/summary.csv) |

Run records include configurations, learning curves, summaries and available validation and test metrics.
The timing tables summarize repeated process measurements. Their provenance records the hardware conditions
and the separately timed cache construction.

## Citation

Jeon, D., Kim, Y., Kim, K., and Kim, H. *PSAN: Parametric Spectral Atom Network with Frequency Attribution for
Canine Ocular Disease Classification*. Applied Sciences (accepted manuscript).
