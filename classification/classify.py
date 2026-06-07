#!/usr/bin/env python3
"""End-to-end classification runner: feature extraction + model benchmarking.

This script is the main classification CLI for the denoised Multifrequenz dataset.
It performs two stages in one command:

1) Feature extraction from denoised pickles
   - Uses TOF Path-B peak indices to center a fixed ROI per mic/trial.
   - Extracts Family-1 features via features_family.extract_for_trial.
   - Extracts Family-2 scattering features via features_family2.extract_features2_for_trial.
   - Caches the extracted table as CSV for reuse.

2) Classification benchmark
   - Runs 4 classifiers: LR-L2, RandomForest, RBF-SVM, LightGBM.
   - Validates with Leave-One-(distance, angle)-Cell-Out CV (LOGO on grouped cells).
   - Supports feature modes:
       normal: keep all extracted features.
       no_leak: drop leakage-prone groups listed below.
       both: run normal and no_leak in one execution.

Leakage-prone exclusions in no_leak mode
---------------------------------------
- Any column starting with "Mic3_" (defective channel).
- Any column containing "_cwt_peakamp".
- Any column containing "_wpt_snr_db".
- Any column containing "_wpt_neighbour_leak".
- Any column containing "_cwt_late_to_peak".

Input expectations
------------------
- Denoised dataset structure:
    <denoised_dir>/<Category>/<distance>cm_<angle>Grad.pickle
- Peaks CSV from TOF stage with at least:
    relative_path, category, distance_cm, angle_deg, mic, trial, peak_idx_path_b

Outputs
-------
Under --output-dir:
- features CSV (cache; configurable via --cache)
- summary CSV with one row per (mode, classifier)
- confusion matrices per (mode, classifier)
- optional OOF predictions per (mode, classifier)

Examples
--------
Normal mode (default):
  python -m classification.classify \
    --denoised-dir "Multifrequenz Dataset/Multifrequenz_denoised_branchA_bayes_garrote" \
    --peaks-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv" \
    --output-dir "outputs/classification_end_to_end"

No-leak mode:
  python -m classification.classify \
    --denoised-dir "Multifrequenz Dataset/Multifrequenz_denoised_branchA_bayes_garrote" \
    --peaks-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv" \
    --output-dir "outputs/classification_end_to_end_noleak" \
    --mode no_leak

Run both modes:
  python -m classification.classify \
    --denoised-dir "Multifrequenz Dataset/Multifrequenz_denoised_branchA_bayes_garrote" \
    --peaks-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv" \
    --output-dir "outputs/classification_end_to_end_both" \
    --mode both
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CLASSIFICATION_DIR = PROJECT_ROOT / "classification"
if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_DIR))

import features_family as F1
import features_family2 as F2


NAME_RE = re.compile(r"(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad", re.IGNORECASE)
META = ["category", "distance_cm", "angle_deg", "file", "trial"]
CLASSES = ["Box", "Dose", "Glas"]


def make_classifiers(seed: int, trees: int, class_weight: str = "balanced"):
    """Return the 4 required classifiers used in the benchmark."""
    from lightgbm import LGBMClassifier

    return {
        "LR-L2": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=trees,
            max_depth=None,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        ),
        "RBF-SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=10.0, gamma="scale", class_weight=class_weight, random_state=seed)),
            ]
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        ),
    }


def parse_dist_angle(filename: str) -> tuple[int | None, int | None]:
    m = NAME_RE.search(filename)
    if not m:
        return None, None
    return int(m.group("dist")), int(m.group("ang"))


def build_peak_lookup(peaks_csv: Path, mics: list[str]) -> dict[tuple[str, str, int], int]:
    """Build (relative_path, mic, trial) -> median Path-B peak index lookup."""
    pk = pd.read_csv(peaks_csv)
    required = {"relative_path", "mic", "trial", "peak_idx_path_b"}
    missing = sorted(required.difference(pk.columns))
    if missing:
        raise ValueError(f"peaks CSV missing required columns: {missing}")

    pk = pk[pk["mic"].isin(mics)]
    return (
        pk.groupby(["relative_path", "mic", "trial"])["peak_idx_path_b"]
        .median()
        .astype(int)
        .to_dict()
    )


def enumerate_files(peaks_csv: Path) -> list[tuple[str, str, int, int]]:
    """Return (relative_path, category, distance_cm, angle_deg) rows from peaks CSV."""
    pk = pd.read_csv(peaks_csv)
    required = {"relative_path", "category", "distance_cm", "angle_deg"}
    missing = sorted(required.difference(pk.columns))
    if missing:
        raise ValueError(f"peaks CSV missing required columns: {missing}")

    f = pk[["relative_path", "category", "distance_cm", "angle_deg"]].drop_duplicates()
    return list(f.itertuples(index=False, name=None))


def _read_trial_frame(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"Unsupported pickle payload in {path}: expected DataFrame, got {type(obj)}")
    return obj


def extract_all(
    denoised_dir: Path,
    peaks_csv: Path,
    mics: list[str],
    trials: int,
    before: int,
    after: int,
    cache: Path,
    all_trials: bool = False,
) -> pd.DataFrame:
    """Extract Family-1 + Family-2 features for all available files/trials."""
    if cache.exists():
        print(f"[cache] loading features from {cache}")
        return pd.read_csv(cache)

    look = build_peak_lookup(peaks_csv, mics)
    files = enumerate_files(peaks_csv)
    trial_txt = "all available" if all_trials else str(trials)
    print(
        f"[extract] {len(files)} files x {trial_txt} trials x {len(mics)} mics, "
        f"window [-{before}, +{after}]"
    )

    rows: list[dict] = []
    for i, (rel_path, category, dist_cm, angle_deg) in enumerate(files, start=1):
        if str(category).strip() not in CLASSES:
            continue

        full_path = denoised_dir / rel_path
        if not full_path.exists():
            continue

        df = _read_trial_frame(full_path)
        if any(m not in df.columns for m in mics):
            continue

        n_avail = len(df)
        n_use = n_avail if all_trials else min(trials, n_avail)

        for tr in range(n_use):
            try:
                signals = {m: np.asarray(df[m].iloc[tr], dtype=float) for m in mics}
                peaks = {m: int(look[(rel_path, m, tr)]) for m in mics}
            except KeyError:
                continue

            f1 = F1.extract_for_trial(signals, peaks=peaks, before=before, after=after, mics=mics)
            f2 = F2.extract_features2_for_trial(signals, peaks=peaks, before=before, after=after, mics=mics)

            row = {
                "category": str(category).strip(),
                "distance_cm": int(dist_cm),
                "angle_deg": int(angle_deg),
                "file": rel_path,
                "trial": int(tr),
            }
            row.update(f1)
            row.update(f2)
            rows.append(row)

        print(f"[extract]   {i}/{len(files)} files done ({n_use} trials, {len(rows)} rows total)", flush=True)

    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"[cache] wrote {cache} ({out.shape[0]} rows, {out.shape[1] - len(META)} features)")
    return out


def _no_leak_reason(col: str) -> str | None:
    if col.startswith("Mic3_"):
        return "mic3_defective_channel"
    if "_cwt_peakamp" in col:
        return "absolute_amplitude_cwt_peakamp"
    if "_wpt_snr_db" in col:
        return "off_carrier_snr_wpt_snr_db"
    if "_wpt_neighbour_leak" in col:
        return "neighbour_leak_wpt"
    if "_cwt_late_to_peak" in col:
        return "tail_late_energy_cwt_late_to_peak"
    return None


def select_features(df: pd.DataFrame, mode: str) -> list[str]:
    feats = [c for c in df.columns if c not in META]
    if mode == "normal":
        return feats

    kept: list[str] = []
    dropped: dict[str, int] = {}
    for c in feats:
        reason = _no_leak_reason(c)
        if reason is None:
            kept.append(c)
            continue
        dropped[reason] = dropped.get(reason, 0) + 1

    print(f"[mode=no_leak] dropped {len(feats) - len(kept)} columns, kept {len(kept)}")
    for reason, cnt in sorted(dropped.items()):
        print(f"  - {reason}: {cnt}")
    return kept


def clean_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0.0)


def build_leave_one_cell_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-One-(distance,angle)-Cell-Out CV splits."""
    groups = df["distance_cm"].astype(str) + "_" + df["angle_deg"].astype(str)
    logo = LeaveOneGroupOut()
    idx = np.arange(len(df))
    return list(logo.split(idx, df["category"].values, groups.values))


def crossval_predictions(clf, xdf: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        c = clone(clf)
        c.fit(xdf.iloc[tr], y[tr])
        pred[te] = c.predict(xdf.iloc[te])
    return pred


def run_mode(
    df: pd.DataFrame,
    mode: str,
    trees: int,
    seed: int,
    output_dir: Path,
    save_oof: bool,
) -> pd.DataFrame:
    cols = select_features(df, mode)
    if not cols:
        raise ValueError(f"No feature columns left in mode '{mode}'")

    xdf = clean_frame(df, cols)
    y = df["category"].values
    folds = build_leave_one_cell_folds(df)
    print(f"[{mode}] folds: {len(folds)}")

    clfs = make_classifiers(seed=seed, trees=trees)
    summary_rows: list[dict] = []

    for name, clf in clfs.items():
        print(f"[{mode}] training {name} ...")
        pred = crossval_predictions(clf, xdf, y, folds)
        acc = accuracy_score(y, pred)
        bacc = balanced_accuracy_score(y, pred)
        mf1 = f1_score(y, pred, average="macro")

        summary_rows.append(
            {
                "mode": mode,
                "classifier": name,
                "n_features": len(cols),
                "n_rows": len(df),
                "n_folds": len(folds),
                "mean_acc": float(acc),
                "mean_balanced_acc": float(bacc),
                "mean_macro_f1": float(mf1),
            }
        )

        cm = confusion_matrix(y, pred, labels=CLASSES)
        cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        cm_df.to_csv(output_dir / f"confusion_{mode}_{name}.csv")

        if save_oof:
            oof = df[["category", "distance_cm", "angle_deg", "file", "trial"]].copy()
            oof["pred"] = pred
            oof.to_csv(output_dir / f"oof_{mode}_{name}.csv", index=False)

    return pd.DataFrame(summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract Family-1/Family-2 features from denoised data and run leave-one-cell classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--denoised-dir", type=Path, required=True)
    ap.add_argument("--peaks-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--mode", choices=["normal", "no_leak", "both"], default="normal")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--trials", type=int, default=30, help="Trials per file if --all-trials is not set.")
    ap.add_argument("--all-trials", action="store_true")
    ap.add_argument("--mics", nargs="+", default=["Mic1", "Mic2", "Mic3"])
    ap.add_argument("--before", type=int, default=F1.WIN_BEFORE)
    ap.add_argument("--after", type=int, default=F1.WIN_AFTER)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--save-oof", action="store_true", help="Save out-of-fold prediction CSVs.")

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = args.cache if args.cache is not None else (args.output_dir / "features_family1_family2.csv")

    df = extract_all(
        denoised_dir=args.denoised_dir,
        peaks_csv=args.peaks_csv,
        mics=list(args.mics),
        trials=args.trials,
        before=args.before,
        after=args.after,
        cache=cache,
        all_trials=args.all_trials,
    )
    print(f"[features] rows={len(df)} cols={df.shape[1]}")

    modes = ["normal", "no_leak"] if args.mode == "both" else [args.mode]
    all_summary: list[pd.DataFrame] = []
    for mode in modes:
        s = run_mode(
            df=df,
            mode=mode,
            trees=args.trees,
            seed=args.seed,
            output_dir=args.output_dir,
            save_oof=args.save_oof,
        )
        all_summary.append(s)

    summary = pd.concat(all_summary, ignore_index=True)
    summary = summary.sort_values(["mode", "mean_acc"], ascending=[True, False]).reset_index(drop=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
