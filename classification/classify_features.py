"""
classify_features.py — material classification benchmark.

For each of three feature families (F1, F2, F3) and four classifiers (LR,
RF, LightGBM, RBF-SVM), runs StratifiedGroupKFold cross-validation grouped
by (distance_cm, angle_deg) so that the classifier never sees the same
(distance, angle) configuration in train and test.

Two variants per (family, classifier) combo:
    - "without distance/angle in features": clean material-only test
    - "with distance/angle in features"   : informative-of-deployment test

Reports:
    - per-fold accuracy + macro F1
    - aggregated confusion matrix
    - per-(distance, angle) accuracy heatmap
    - permutation importance for the top features

USAGE (your machine):
    python classify_features.py \
        --features-csv outputs/features_all.csv \
        --output-dir   outputs/classification_results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             balanced_accuracy_score, classification_report)
from sklearn.pipeline import Pipeline


META_COLS = ("category", "distance_cm", "angle_deg", "trial", "file")
LABEL_COL = "category"


def _no_leak_exclusion_reason(col: str) -> str | None:
    """Return exclusion reason for no_leak mode, or None if feature is allowed."""
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


def apply_feature_mode(
    feat_cols: Sequence[str],
    mode: str,
) -> tuple[list[str], dict[str, int]]:
    """Filter columns based on requested feature mode.

    mode="normal": keep all feature columns.
    mode="no_leak": drop leakage-prone columns listed in _no_leak_exclusion_reason.
    """
    if mode == "normal":
        return list(feat_cols), {}

    kept: list[str] = []
    dropped_by_reason: dict[str, int] = {}
    for c in feat_cols:
        reason = _no_leak_exclusion_reason(c)
        if reason is None:
            kept.append(c)
            continue
        dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
    return kept, dropped_by_reason


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise input schema to expected meta column names.

    Supports both legacy columns (category/distance_cm/angle_deg) and
    extraction-only columns (object/distance/angle).
    """
    out = df.copy()
    rename_map = {}
    if "category" not in out.columns and "object" in out.columns:
        rename_map["object"] = "category"
    if "distance_cm" not in out.columns and "distance" in out.columns:
        rename_map["distance"] = "distance_cm"
    if "angle_deg" not in out.columns and "angle" in out.columns:
        rename_map["angle"] = "angle_deg"
    if rename_map:
        out = out.rename(columns=rename_map)

    required = {"category", "distance_cm", "angle_deg", "trial"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns after schema normalisation: {sorted(missing)}")
    return out


def is_family15_col(c: str) -> bool:
    """Family-1.5 columns: cepstral coeffs (_cep_NN) and cross-channel ratios."""
    if "_cep_" in c:
        return True
    if "_dlog_energy" in c or "_dlog_peak" in c:
        return True
    if c == "signed_amp_ratio":
        return True
    return False


def is_family2_col(c: str) -> bool:
    """Family-2 columns end in _S followed by digits."""
    parts = c.rsplit("_", 1)
    return len(parts) == 2 and parts[1].startswith("S") and parts[1][1:].isdigit()


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (family1_cols, family15_cols, family2_cols)."""
    f1, f15, f2 = [], [], []
    for c in df.columns:
        if c in META_COLS:
            continue
        if is_family15_col(c):
            f15.append(c)
        elif is_family2_col(c):
            f2.append(c)
        else:
            f1.append(c)
    return f1, f15, f2


def make_classifiers(class_weight="balanced", skip_lightgbm: bool = False):
    """Return dict of name → sklearn-compatible estimator (uninstantiated pipeline)."""
    out = {
        "LR-L2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.3, max_iter=2000,
                                       class_weight=class_weight)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=18, min_samples_leaf=4, max_features="sqrt",
            class_weight=class_weight, n_jobs=-1, random_state=0,
        ),
        "RBF-SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=3.0, gamma="scale",
                        class_weight=class_weight, random_state=0)),
        ]),
    }
    # LightGBM if available
    if not skip_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            out["LightGBM"] = LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                min_child_samples=40, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=1,
                reg_alpha=0.1, reg_lambda=1.0,
                class_weight=class_weight, n_jobs=1, random_state=0, verbosity=-1,
            )
        except ImportError:
            print("  (LightGBM not installed — skipping)")
    return out


def filter_classifiers(classifiers: dict[str, Any], only: Sequence[str] | None) -> dict[str, Any]:
    if not only:
        return classifiers
    keep = [name.strip() for name in only if name.strip()]
    missing = [k for k in keep if k not in classifiers]
    if missing:
        raise ValueError(f"Unknown classifier(s): {missing}. Available: {list(classifiers.keys())}")
    return {k: classifiers[k] for k in keep}


def evaluate_fold(clf, X_train, y_train, X_test, y_test):
    # Ensure X_train and X_test have feature names (convert numpy to DataFrame if needed)
    # This prevents LightGBMClassifier warnings about missing feature names
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "y_true": y_test, "y_pred": y_pred,
        "acc": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
    }


def run_cv(df: pd.DataFrame, feat_cols: Sequence[str], clf_name: str,
           clf, n_splits: int = 5, seed: int = 0, cv_mode: str = "sgkf",
           k_best: int = 0) -> dict:
    """Run StratifiedGroupKFold by (distance, angle). Each fold holds out
    a set of (distance, angle) cells across all 3 objects."""
    df = df.dropna(subset=list(feat_cols) + [LABEL_COL]).reset_index(drop=True)
    X = df[list(feat_cols)].values.astype(np.float32)
    y = df[LABEL_COL].values
    # Group: distinct (distance, angle) cells. There are 25.
    groups = df["distance_cm"].astype(str) + "_" + df["angle_deg"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    logo = LeaveOneGroupOut()
    fold_results = []
    all_true, all_pred, all_groups = [], [], []
    if cv_mode == "leave-one-cell":
        split_iter = logo.split(X, y, groups)
    else:
        split_iter = sgkf.split(X, y, groups)

    n_features_eff = len(feat_cols)
    for fold, (tr_idx, te_idx) in enumerate(split_iter):
        # Important: fit-transform scaler within fold (no leakage)
        from sklearn.base import clone
        c = clone(clf)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Optional fold-safe feature selection for high-dimensional sets.
        if k_best and 0 < int(k_best) < X_tr.shape[1]:
            selector = SelectKBest(score_func=f_classif, k=int(k_best))
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_te = selector.transform(X_te)
            n_features_eff = X_tr.shape[1]

        res = evaluate_fold(c, X_tr, y_tr, X_te, y_te)
        res["fold"] = fold
        fold_results.append(res)
        all_true.extend(res["y_true"].tolist())
        all_pred.extend(res["y_pred"].tolist())
        all_groups.extend(groups.iloc[te_idx].tolist())

    agg = {
        "clf": clf_name,
        "n_features": int(n_features_eff),
        "n_trials": len(df),
        "mean_acc": float(np.mean([r["acc"] for r in fold_results])),
        "std_acc": float(np.std([r["acc"] for r in fold_results])),
        "mean_macro_f1": float(np.mean([r["macro_f1"] for r in fold_results])),
        "mean_balanced_acc": float(np.mean([r["balanced_acc"] for r in fold_results])),
        "fold_results": fold_results,
        "all_true": all_true,
        "all_pred": all_pred,
        "all_groups": all_groups,
    }
    return agg


def per_cell_accuracy(all_true, all_pred, all_groups) -> pd.DataFrame:
    """Per (distance, angle) accuracy."""
    df = pd.DataFrame({"true": all_true, "pred": all_pred, "grp": all_groups})
    df["correct"] = (df["true"] == df["pred"]).astype(int)
    parts = df["grp"].str.split("_", expand=True)
    df["distance_cm"] = parts[0].astype(int)
    df["angle_deg"] = parts[1].astype(int)
    return df.groupby(["distance_cm", "angle_deg"])["correct"].mean().unstack()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv", choices=["sgkf", "leave-one-cell"], default="leave-one-cell",
                        help="Cross-validation mode. 'leave-one-cell' holds out exactly one (distance,angle) cell per fold.")
    parser.add_argument("--skip-lightgbm", action="store_true",
                        help="Skip LightGBM classifier.")
    parser.add_argument("--only-classifiers", nargs="+", default=None,
                        help="Optional subset of classifiers to run, e.g. --only-classifiers LightGBM")
    parser.add_argument("--family-mode", choices=["all", "full"], default="all",
                        help="Run all family combinations or only the full available feature set.")
    parser.add_argument("--k-best", type=int, default=256,
                        help="If >0, keep top-k ANOVA features per training fold (no leakage).")
    parser.add_argument("--add-distance-angle", action="store_true",
                        help="Include distance_cm + angle_deg as features")
    parser.add_argument(
        "--mode",
        choices=["normal", "no_leak"],
        default="normal",
        help="Feature mode: normal keeps all columns, no_leak drops leakage-prone features.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    df = normalise_schema(df)
    print(f"  {len(df)} rows × {df.shape[1]} cols")
    print(f"  classes: {df[LABEL_COL].value_counts().to_dict()}")

    f1_cols, f15_cols, f2_cols = split_feature_columns(df)
    print(f"  Family 1   features: {len(f1_cols)}")
    print(f"  Family 1.5 features: {len(f15_cols)}")
    print(f"  Family 2   features: {len(f2_cols)}")

    families = {
        "F1": f1_cols,
        "F15": f15_cols,
        "F2": f2_cols,
        "F1+F15": f1_cols + f15_cols,
        "F1+F2": f1_cols + f2_cols,
        "F15+F2": f15_cols + f2_cols,
        "F1+F15+F2_full_hybrid": f1_cols + f15_cols + f2_cols,
    }
    if args.add_distance_angle:
        new_families = {}
        for k, cols in families.items():
            new_families[k + "_with_da"] = cols + ["distance_cm", "angle_deg"]
        families.update(new_families)

    if args.family_mode == "full":
        families = {"FULL": f1_cols + f15_cols + f2_cols}

    # Apply feature mode after family composition so no_leak filtering is consistent
    # across single families and combined families.
    if args.mode == "no_leak":
        filtered_families: dict[str, list[str]] = {}
        total_dropped = 0
        total_kept = 0
        aggregate_reasons: dict[str, int] = {}
        for fam_name, cols in families.items():
            kept_cols, dropped_by_reason = apply_feature_mode(cols, mode=args.mode)
            filtered_families[fam_name] = kept_cols
            total_kept += len(kept_cols)
            total_dropped += len(cols) - len(kept_cols)
            for reason, cnt in dropped_by_reason.items():
                aggregate_reasons[reason] = aggregate_reasons.get(reason, 0) + cnt

        families = filtered_families
        print("\n[mode=no_leak] leakage-prone features dropped")
        print(f"  kept columns across family definitions: {total_kept}")
        print(f"  dropped columns across family definitions: {total_dropped}")
        for reason, cnt in sorted(aggregate_reasons.items()):
            print(f"  - {reason}: {cnt}")

    classifiers = make_classifiers(skip_lightgbm=args.skip_lightgbm)
    classifiers = filter_classifiers(classifiers, args.only_classifiers)

    summary_rows = []
    all_per_cell = {}
    all_confusion = {}

    for fam_name, cols in families.items():
        if not cols:
            continue
        print(f"\n=== Family {fam_name} ({len(cols)} features) ===")
        for clf_name, clf in classifiers.items():
            print(f"  Training {clf_name}...")
            try:
                res = run_cv(df, cols, clf_name, clf,
                             n_splits=args.n_splits, seed=args.seed, cv_mode=args.cv,
                             k_best=args.k_best)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
            row = {
                "family": fam_name, "classifier": clf_name,
                "n_features": res["n_features"],
                "mean_acc": res["mean_acc"],
                "std_acc": res["std_acc"],
                "mean_macro_f1": res["mean_macro_f1"],
                "mean_balanced_acc": res["mean_balanced_acc"],
            }
            summary_rows.append(row)
            print(f"    acc={res['mean_acc']:.3f} ± {res['std_acc']:.3f}, "
                  f"macro F1={res['mean_macro_f1']:.3f}")

            # Per-cell heatmap
            heat = per_cell_accuracy(res["all_true"], res["all_pred"], res["all_groups"])
            all_per_cell[(fam_name, clf_name)] = heat
            heat.to_csv(args.output_dir / f"per_cell_{fam_name}_{clf_name}.csv")

            # Confusion matrix
            classes = sorted(set(res["all_true"]))
            cm = confusion_matrix(res["all_true"], res["all_pred"], labels=classes)
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            all_confusion[(fam_name, clf_name)] = cm_df
            cm_df.to_csv(args.output_dir / f"confusion_{fam_name}_{clf_name}.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("mean_acc", ascending=False).reset_index(drop=True)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)

    print("\n=== SUMMARY (sorted by mean accuracy) ===")
    print(summary_df.round(4).to_string(index=False))

    # Pretty-print best per-cell heatmap
    if summary_rows:
        best = summary_rows[0]
        for r in summary_rows:
            if r["mean_acc"] > best["mean_acc"]:
                best = r
        bf = best["family"]; bc = best["classifier"]
        print(f"\nBest combo: {bf} + {bc}")
        print(f"Per-cell accuracy heatmap (rows=distance_cm, cols=angle_deg):")
        print(all_per_cell[(bf, bc)].round(2))
        print(f"\nConfusion matrix:")
        print(all_confusion[(bf, bc)])
    print(f"\nSaved all results to {args.output_dir}")


if __name__ == "__main__":
    main()
