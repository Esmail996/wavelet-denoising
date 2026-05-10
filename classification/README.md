# Multifrequenz feature extraction and classification

Six Python files implement the post-TOF stage of the pipeline:

- `roi_preprocessing.py` — shared ROI helpers: per-channel envelope re-alignment and per-trial energy normalisation
- `features_family1.py` — classical SWT/WPT statistical features (~55/channel × 6 channels = ~330/trial)
- `features_family15.py` — cepstral coefficients + cross-channel amplitude ratios (~93/trial)
- `features_family2.py` — Wavelet Scattering Transform via Kymatio (~900-1500/trial)
- `extract_features_all_trials.py` — driver that walks denoised pickles, reads the calibrated TOF CSV, builds per-(Mic, Tx) ROIs, runs all three families, writes a single feature CSV
- `classify_features.py` — runs F1 / F15 / F2 plus their hybrids through Logistic Regression / Random Forest / LightGBM / RBF-SVM with StratifiedGroupKFold cross-validation grouped by (distance, angle)
- `diagnose_classifier_results.py` — reads `summary.csv` + per-cell + confusion outputs, computes permutation importance on the best combo, and writes a ranked `improvement_proposals.txt`

## Setup

```bash
pip install numpy scipy pandas scikit-learn pywavelets kymatio lightgbm
```

`kymatio` and `lightgbm` are optional. If absent, F2 falls back to a numpy-only scattering implementation and LightGBM is skipped from the classifier roster.

## Step 1 — extract features

```bash
python extract_features_all_trials.py \
    --denoised-dir "Multifrequenz Dataset/Multifrequenz_denoised_branchA_sym6" \
    --tof-csv      "outputs/tof_matched/multifrequenz_denoised_branchA_sym6.csv" \
    --output       "outputs/features/features_all.csv" \
    --use-kymatio
```

Options:

- `--include-mic3` — include Mic3 channels (degraded; off by default per pipeline decision)
- `--skip-f2` — skip Family 2 (use this if Kymatio not installed; F1+F1.5 run is much faster)
- `--skip-f15` — skip Family 1.5
- `--no-realign` — disable per-channel envelope re-alignment within ROI
- `--no-normalise` — disable per-trial ROI energy normalisation
- `--max-files N` — smoke test with the first N pickles
- `--roi-n 1024` — Family 2 ROI length in samples (must be ≥ 2^J for kymatio)
- `--roi-half-us-f1 200` — Family 1 / Family 1.5 ROI half-width in µs

Default preprocessing (recommended): both `realign` and `normalise` ON. This removes residual TOF mis-alignment within the ROI and the 1/R² distance amplitude effect, so features encode echo SHAPE only.

Expected output: 7500 rows × ~1500 columns CSV (roughly 1.2 GB).

## Step 2 — run classification

```bash
python classify_features.py \
    --features-csv outputs/features/features_all.csv \
    --output-dir   outputs/classification \
    --n-splits     5 \
    --add-distance-angle
```

Outputs in `--output-dir`:

- `summary.csv` — one row per (family × classifier), sorted by mean accuracy
- `per_cell_<family>_<clf>.csv` — accuracy heatmap per (distance, angle)
- `confusion_<family>_<clf>.csv` — aggregated confusion matrix across folds

Family combinations evaluated by default:

- F1, F15, F2 (each alone)
- F1+F15, F1+F2, F15+F2 (pairwise hybrids)
- F1+F15+F2_full_hybrid (everything)
- All of the above with `_with_da` suffix when `--add-distance-angle` is on

## Step 3 — diagnose results

```bash
python diagnose_classifier_results.py \
    --features-csv outputs/features/features_all.csv \
    --classification-dir outputs/classification \
    --output-dir outputs/diagnostics
```

Produces:

- `feature_importance_<best>.csv` — permutation importance ranked, top features identified by family
- `improvement_proposals.txt` — ranked, evidence-based improvement plan based on:
  * family ranking (which features matter)
  * classifier ranking (which model fits best)
  * per-cell weakness (which (distance, angle) cells fail)
  * confusion patterns (which material pairs get confused)
  * top-feature distribution across families
  * impact of distance/angle as features

## Cross-validation protocol

`StratifiedGroupKFold` groups by `(distance_cm, angle_deg)` so that each fold holds out *entire (distance, angle) configurations* rather than random trials. With 25 unique cells × 3 classes and 5 folds, each fold has 5 cells = 1500 trials. This is the protocol that gives examiner-defensible numbers.

## Two variants per family/classifier (when `--add-distance-angle`)

1. **Without distance/angle** — clean material-only test. Answers "can we classify material from echo shape alone?"
2. **With distance/angle in features** — deployment-realistic test. Answers "given that the system measures distance/angle online, how good is classification?"

The first is the cleaner scientific claim. The second is the more practically useful number. Both should be reported.

## Compute & memory notes

- Family 1 + 1.5 extraction: ~0.15 s per trial on a single CPU core. Full 7500 trials: ~20 min.
- Family 2 with Kymatio (CPU): ~0.5 s per trial. Full dataset: ~1 hour.
- With kymatio + GPU: ~5x faster.
- Classification: each fold trains in seconds for LR/SVM, ~30s for LightGBM with default depth.
- Diagnostics permutation importance: a few minutes on the best fold.

## When you have results

Send back the contents of `outputs/classification/summary.csv` and `outputs/diagnostics/improvement_proposals.txt`. From those two files, I can give you concrete next-step guidance.

