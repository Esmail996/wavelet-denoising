# Ultrasonic Pipeline: Denoising -> TOF -> Triangulation -> Classification

This repository now contains a full end-to-end pipeline for the Multifrequenz
dataset, with dedicated runners for each stage.

## Pipeline overview

1. Denoising
  1. Input: raw pickles from Multifrequenz Dataset/Multifrequenz
  2. Output: denoised pickles + denoising summary CSV

2. TOF + calibration
  1. Input: denoised dataset
  2. Output: peaks CSV, empirical delay CSV, structural delay CSV,
    calibration report CSV, calibrated TOF CSV

3. Triangulation
  1. Input: calibrated TOF CSV
  2. Output: per-trial triangulation results CSV

4. Classification
  1. Input: denoised dataset + TOF peaks CSV
  2. Output: extracted feature CSV, confusion CSVs, summary CSV

## Environment setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional dependency for Family-2 scattering features:

```bash
pip install kymatio
```

## Denoising runner

Runner location:
[denoising/runners/run_global_denoising.py](denoising/runners/run_global_denoising.py)

What it does:
1. Runs Path A (SWT) or Path B (WPT) denoising over all pickle files.
2. Preserves original payload structure.
3. Writes denoising_summary.csv in output folder.

Basic usage:

```bash
python -m denoising.runners.run_global_denoising --path A
python -m denoising.runners.run_global_denoising --path B
```

Main options:
1. Required
  1. --path {A,B}
2. I/O
  1. --input-dir
  2. --output-dir
3. Shared preprocessing
  1. --preprocessing {bandpassed,detrended_only}
  2. --bp-center-hz
  3. --bp-bw-hz
  4. --bp-method {iir,fir}
  5. --bp-iir-order
4. Path A specific
  1. --a-wavelet
  2. --a-level
  3. --a-noise-level
  4. --a-use-fixed-sigma
5. Path B specific
  1. --b-wavelet
  2. --b-level
  3. --b-carrier-attenuation
  4. --b-noise-packet-idx

## TOF global runner

Runner location:
[tof/run_tof_global_pipeline.py](tof/run_tof_global_pipeline.py)

What it does:
1. Stage 1: TOF estimation (Path A and Path B)
2. Stage 2: fit and save empirical and structural delays
3. Stage 3: apply calibration using saved delay CSV files

Basic usage:

```bash
python -m tof.run_tof_global_pipeline \
  --data-root "Multifrequenz Dataset/Multifrequenz_denoised_branchA_bayes_garrote" \
  --name "run1" \
  --picker max
```

Main options:
1. --data-root
2. --out-dir
3. --name
4. --fs-hz
5. --gate-cm
6. --picker {max,nearest}
7. --max-files

Outputs for run name run1:
1. outputs/tof_matched/run1_peaks.csv
2. outputs/tof_matched/calibration/tau_estimates_run1.csv
3. outputs/tof_matched/calibration/structural_params_run1.csv
4. outputs/tof_matched/calibration/calibration_report_run1.csv
5. outputs/tof_matched/run1_calibrated.csv

## Triangulation runner

Runner location:
[triangulation/triangulation.py](triangulation/triangulation.py)

What it does:
1. Reads Path-B TOF rows from calibrated TOF CSV.
2. Runs multi-start triangulation per trial group.
3. Writes per-trial x/y estimates and error metrics.

Basic usage:

```bash
python -m triangulation.triangulation \
  --input-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_calibrated.csv" \
  --output-csv "outputs/triangulation/path_b_calibrated_results.csv"
```

Main options:
1. --input-csv
2. --output-csv
3. --no-calibration
4. --max-groups

## Classification runner

Runner location:
[classification/classify.py](classification/classify.py)

What it does:
1. Extracts Family-1 and Family-2 features from denoised dataset using TOF peaks.
2. Runs 4 classifiers with Leave-One-(distance,angle)-Cell-Out CV.
3. Supports normal and no_leak feature modes.

Classifiers:
1. LR-L2
2. RandomForest
3. RBF-SVM
4. LightGBM

Feature modes:
1. normal
2. no_leak
3. both

No-leak exclusions:
1. Mic3_* columns
2. *_cwt_peakamp*
3. *_wpt_snr_db*
4. *_wpt_neighbour_leak*
5. *_cwt_late_to_peak*

Basic usage:

```bash
python -m classification.classify \
  --denoised-dir "Multifrequenz Dataset/Multifrequenz_denoised_branchA_bayes_garrote" \
  --peaks-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv" \
  --output-dir "outputs/classification_end_to_end" \
  --mode both
```

Main options:
1. --denoised-dir
2. --peaks-csv
3. --output-dir
4. --mode {normal,no_leak,both}
5. --trees
6. --seed
7. --trials
8. --all-trials
9. --mics
10. --before
11. --after
12. --cache
13. --save-oof

## Recommended end-to-end order

1. Run denoising on raw dataset.
2. Run TOF global pipeline on denoised dataset.
3. Run triangulation on calibrated TOF CSV.
4. Run classification with same denoised dataset + peaks CSV.

## Notes

1. Use the virtual environment Python when running commands on Windows:
  c:/Users/49162/Desktop/wavelet_denoising-master/wavelet_denoising-master/venv/Scripts/python.exe
2. For quick smoke tests, use smaller subsets:
  1. TOF: --max-files
  2. Triangulation: --max-groups
  3. Classification: --trials


