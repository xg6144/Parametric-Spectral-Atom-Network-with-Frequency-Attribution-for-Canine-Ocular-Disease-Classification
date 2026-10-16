# PSAN: Parametric Spectral Atom Network

Code and results for the article *PSAN: Parametric Spectral Atom Network with Frequency Attribution for Canine Ocular
Disease Classification* by Dongbeen Jeon, Youngrok Kim, Kyoungyee Kim and Heecheol Kim.

PSAN is a frequency-domain token mixer whose filters combine a small bank of shared, learnable Gaussian or Gabor-like
spectral atoms. This bounds the filter rank and decomposes every filter exactly into atom contributions. The article
evaluates PSAN on an audited corpus of canine ocular photographs and on three MedMNIST+ datasets. All models are
trained from scratch.

This repository contains the code that produced the reported results, the frozen protocols and configurations, the
split manifests with image hashes, and the per-image predictions, run records and result tables. The images are not
included (see [Data](#data)).

## Contents

| Path | Contents |
|---|---|
| `models/` | PSAN (`psan.py`), the GFNet and AFNO implementations, and the free low-rank and fixed-atom controls (`lowrank.py`) |
| `experiments/` | training, tuning, evaluation, statistics and figure code of every stage (see [Stages](#stages)) |
| `run_experiments.py` | entry point of the round-2 phases |
| `train/train_base.py`, `data/`, `utils/`, `eval/eval_psan.py`, `evaluation_metrics.py`, `analyze_predictions.py`, `model_resources.py` | training loop, metrics and helpers used by the experiment code |
| `deduplicate_dataset.py`, `prepare_stratified_group_folds.py` | duplicate audit and grouped fold assignment |
| `cv_splits/` | fold assignment of an earlier audit, an input of `experiments/integrity.py` |
| `configs/` | protocols and configurations of every stage, with the SHA-256 of the training sources |
| `revision_history/20261013_round2_training_source/` | frozen sources of the round-2 runs (see [Stages](#stages)) |
| `environment/` | Python and package versions and the numerical-precision settings |
| `results/` | manifests, run records, per-image predictions, result tables and analysis provenance (see [Released results](#released-results)) |
| `eval_results/` | stored test predictions of the earlier, unaudited study (re-analysis table of the Supplementary Materials) |
| `tests/` | unit tests |
| `tools/` | `verify_release.py` (checks the released files) and `restore_split_manifest.py` (rebuilds the full split manifest from the dataset) |
| `reports/` | empty; several round-2 phases write Markdown summaries here |
| `MANIFEST.csv` | every released file with its SHA-256, its source and its changes relative to that source |

## Environment

The experiments ran on Linux with CUDA GPUs, using Python 3.13.13, PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.26,
NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0 and medmnist 3.0.2, in single precision with TF32 matrix
multiplication disabled (`environment/versions.json`, `environment/precision_protocol.json`).
`environment/conda-explicit.txt` lists the conda packages and `environment/pip-freeze.txt` the pip packages of that
environment, for example:

```bash
conda create -n psan --file environment/conda-explicit.txt
conda activate psan
pip install -r environment/pip-freeze.txt
```

## Data

**Canine ocular photographs.** The images come from the companion-animal ocular disease dataset of the National
Information Society Agency (AI-Hub, Republic of Korea; <https://aihub.or.kr>, dataset no. 562) and are available from
AI-Hub under its terms of use. The code expects every image with its JSON record in `data_raw/<class>/`, one folder per
class (`Blepharitis`, `Cataract`, `Conjunctivitis`, `Entropion`, `Epiphora`, `Eyelid_tumor`,
`Non_ulcerative_keratitis`, `Normal`, `Nuclear_sclerosis`, `Pigmentary_keratitis`, `Ulcerative_keratitis`), with the
file names given in the `path` column of the manifests, for example
`data_raw/Cataract/crop_D0_03035855-60a5-11ec-8402-0a7404972c70.jpg` and its
`data_raw/Cataract/crop_D0_03035855-60a5-11ec-8402-0a7404972c70.json`. The experiment modules read the class names
from these folders.

The training and analysis code reads the split from `results/revision_split.csv`, which also holds per-image metadata
taken from the JSON records (capture date, device, file name, breed, sex, age and eye side). Because these come from
the dataset, that file is not distributed here. Rebuild it from your copy of the dataset with

```bash
python tools/restore_split_manifest.py --check-images
```

The script checks the rebuilt file against the SHA-256 recorded in `results/corrected_integrity.json`, which the code
also checks before it uses the split. With `--check-images`, it also checks every retained image against its SHA-256.

**MedMNIST+.** The 128-pixel BloodMNIST, OCTMNIST and DermaMNIST files (`bloodmnist_128.npz`, `octmnist_128.npz`,
`dermamnist_128.npz`) come from the MedMNIST+ record <https://zenodo.org/records/10519652> and go in
`results/external_data/round3_128/`. The corrected, lesion-disjoint DermaMNIST split uses the mapping published by its
authors (<https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets>). Put the files `dermamnist_split_info.csv`,
`combined_metadata_original.csv` and `combined_metadata_corrected-HAM10000_corrected.csv` from the folder named in
`results/external_data/derma_mapping/SOURCE.txt` into that folder. `experiments/derma_corrected.py` applies the mapping
to the 64-pixel DermaMNIST release, which it downloads with the `medmnist` package, and writes the corrected split
(`results/external_data/dermamnist_c64_{train,val,test}_identity.csv`). `python -m experiments.round3 prepare` then
checks the published MD5 digests of the 128-pixel files and applies the corrected split to them. BloodMNIST and
OCTMNIST are licensed under CC BY 4.0, and DermaMNIST under CC BY-NC 4.0.

## Stages

The study was run in stages ("rounds"). Round 2 writes its runs and analyses to `results/training`, `results/tuning`
and `results/tables`. The later stages write to `results/round3`, `round4`, `round5`, `round6`, `round6_posthoc`,
`round6_factorial` and `round6_robustness`.

| Stage | Code | Protocol | Experiments |
|---|---|---|---|
| Round 2 | `run_experiments.py`, `experiments/training.py`, `validation_tuning.py`, `tuning_results.py`, `queue_ofat.py`, `round2_queue.py`, `outer_cv.py`, `analysis.py`, `statistics.py`, `robustness.py`, `faithfulness.py`, `selected_efficiency.py` | `configs/revision.json`, `round2_protocol.json`, `round2_statistics.json`, `validation_tuning.json`, `outer_cv.json` | audit and split; PSAN, GFNet, AFNO and the matched dense control under a shared preset; factorial and one-factor ablations; validation-only tuning; outer cross-validation; calibration, robustness, attribution and efficiency |
| Round 3 | `experiments/round3.py`, `round3_analysis.py`, `round3_figures.py` | `configs/round3_protocol.json` | validation-selected dense control and PSAN without atom dropout; further seeds of the selected settings; atom-count replication; MedMNIST+ benchmarks |
| Round 4 | `experiments/round4.py`, `round4_analysis.py`, `round4_attribution.py`, `round4_figures.py` | `configs/round4_protocol.json` | free low-rank and fixed-atom controls; five-fold cross-validation; 100-epoch runs; seed stability of attributions |
| Round 5 | `experiments/round5.py`, `round5_analysis.py` | `configs/round5_protocol.json` | DeiT-Tiny, ConvNeXt-Atto, FastViT-T8, Swin-Tiny and SHViT-S1 trained from scratch |
| Round 6 | `experiments/round6.py`, `round6_analysis.py`, `round6_figures.py`, `round6_posthoc.py`, `round6_factorial.py`, `round6_robustness.py` | `configs/round6_protocol.json`, `round6_posthoc_protocol.json`, `round6_factorial_protocol.json` | nested cross-validation; training without Normal images; 100-epoch runs at three seeds; 128-pixel input; post hoc checks of the disease-only comparison; robustness of the matched dense control |

The released code is byte-identical to the code that produced the results. From round 3 on, the protocol of each
stage records the SHA-256 of its training sources, and the `freeze` step refuses to proceed if a source differs. The
round-2 runs used the frozen sources in `revision_history/20261013_round2_training_source/`. They differ from the
released files only in `experiments/common.py`, where the timm model factory of the released version does not pass a
stochastic-depth rate to SHViT, which does not accept one. No released round-2 run uses that factory.

The code does not retrain a run whose folder already contains a `summary.json`, so move the released run folders away
before retraining them. The module docstrings give the full commands, for example:

```bash
# Round 2: replay in a new workspace with the frozen round-2 sources
python -m experiments.reproduction_workspace --destination ../psan_round2_replay
cd ../psan_round2_replay
python run_experiments.py --phase multiseed --models psan gfnet afno --seeds 42 43 44 45 46 --train --resume --device cuda:0

# Rounds 3-6: freeze the protocol, run one worker per GPU, then the analysis
python -m experiments.round4 freeze
python -m experiments.round4 worker --device cuda:0
python -m experiments.round4_analysis
```

The paired comparisons use percentile-bootstrap intervals with B = 10,000 resamples of evaluation groups: visual
groups for the canine test set, lesions for DermaMNIST and images otherwise. Where several seeds are pooled, the
resampling is crossed with the seeds. The p values are Bonferroni-adjusted within the pre-declared round-2 families
and Holm-adjusted within the later families. The intervals are not adjusted for multiplicity.

## Released results

**Split and audit.**

- `results/corrected_sgkf5.csv` is the split manifest. It has one row per retained image (104,037), with its path,
  class, label, visual group (`group_id`), fold and SHA-256. Fold 1 is the test set, fold 2 the validation set, and
  folds 3–5 the training set. The cross-validation analyses rotate the folds (`configs/outer_cv.json`,
  `experiments/round4.py`).
- `results/corrected_dedup_manifest.csv.gz` is the deduplication audit, with one row per raw image (198,874). It lists
  the file and pixel hashes, the perceptual hashes, the decision with its reason, the duplicate group and the visual
  group.
- `results/reconstructed_dedup/near_duplicate_pairs.csv` lists every perceptual-hash candidate pair with its
  verification measures.
- `cv_splits/stratified_group_5fold.csv` is the fold assignment of an earlier audit. Its groups are merged into the
  visual groups by `experiments/integrity.py`.

**Runs.** Every run folder (`results/<stage>/.../<run>/`) contains the following files:

- `config.json`: the full configuration.
- `summary.json`: the selected epoch, the validation score, and the hashes of the checkpoint, the configuration and
  the split.
- `epochs.csv`: the training curve.
- `val_metrics.json` and `test_metrics.json`: the metrics.
- `test_predictions.csv.gz`: one row per test image with its path, label, visual group, predicted label, and the
  probability and logit of every class.

Runs used by the calibration analyses also include `val_predictions.csv.gz`. The out-of-fold predictions of the outer
cross-validation are in `results/cross_validation/`. Checkpoints are not included.

**Tables and provenance.** The analysis outputs are in `results/round3/tables`–`results/round6/tables`,
`results/round6_factorial/tables`, `results/round6_robustness` and `results/tables` (round 2), with
`analysis_provenance.json` and the validation selections of every stage. `results/ARTICLE_SOURCES.csv` lists the files
from which the numbers in the article and its Supplementary Materials were generated, with the SHA-256 of each file
as used. The remaining numbers come from the run records. `MANIFEST.csv` lists every released file with its SHA-256
and its changes relative to the original file.

The per-image files are gzip-compressed. The code reads uncompressed files, so decompress them before running an
analysis:

```bash
find results eval_results -name '*.csv.gz' -exec gunzip -k {} +
```

**Not included.** The following files are not included:

- the images and the per-image metadata from their JSON records. The metadata columns were removed from every
  released per-image file.
- checkpoints, caches, logs and figures. Several figures show dataset photographs.
- runs of the 64-pixel MedMNIST+ benchmarks, which the article does not report.
- runs of ImageNet-pretrained backbones from round 2, because the article compares models trained from scratch.

The rows of these backbones were also removed from `results/tables/calibration.csv`, `device_performance.csv`,
`fresh_aggregate.csv` and `validation_tuning_trials.csv` and from `results/round3/tables/main_contrasts.csv` and
`disease_per_run.csv`. The pre-declared round-2 family of 18 primary contrasts included five comparisons with these
backbones, so the Bonferroni-adjusted p values in `main_contrasts.csv` still use the family size of 18. The SHA-256
recorded for `results/revision_split.csv` in the configurations and run summaries refers to the full manifest that
`tools/restore_split_manifest.py` rebuilds.

## Verification

```bash
python tools/verify_release.py
```

This script needs neither the images nor a GPU. It checks three things:

- the source files whose SHA-256 the protocols record are unchanged;
- the Macro F1 recomputed from the per-image predictions of every run equals the value stored with the run;
- the split manifest assigns each image, image hash and visual group to a single fold.

The unit tests (`python -m unittest discover -s tests`) need the class folders in `data_raw/`. One of them also needs
the rebuilt `results/revision_split.csv`.

## Citation

Jeon, D.; Kim, Y.; Kim, K.; Kim, H. PSAN: Parametric Spectral Atom Network with Frequency Attribution for Canine
Ocular Disease Classification.
