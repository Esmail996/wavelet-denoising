"""
Investigate seed-to-seed misclassification identity overlap for the baseline
LightGBM setup.

This script uses the same feature split and model specs as classify_features.py
for the F1 baseline and reports whether misclassified trial identities are
stable or seed-sensitive.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from classify_features import split_feature_columns


def run_seed(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_seed: int,
    model_seed: int,
    n_splits: int,
) -> pd.DataFrame:
    """Run CV for one model seed and return per-trial predictions."""
    x = df[feature_cols].astype(np.float32)
    y = df["category"].values
    groups = df["distance_cm"].astype(str) + "_" + df["angle_deg"].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)

    clf = LGBMClassifier(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        class_weight="balanced",
        n_jobs=-1,
        random_state=model_seed,
        verbosity=-1,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
    )

    rows: list[pd.DataFrame] = []
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(x, y, groups)):
        c = clone(clf)
        c.fit(x.iloc[tr_idx], y[tr_idx])
        y_pred = c.predict(x.iloc[te_idx])

        out = df.iloc[te_idx][["category", "distance_cm", "angle_deg", "trial"]].copy()
        out["fold"] = fold
        out["model_seed"] = model_seed
        out["y_pred"] = y_pred
        out["correct"] = (out["category"] == out["y_pred"]).astype(int)
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def error_key_set(pred_df: pd.DataFrame) -> set[tuple[str, int, int, int]]:
    """Return set of misclassified trial identities."""
    err = pred_df[pred_df["correct"] == 0]
    return set(
        zip(
            err["category"].astype(str),
            err["distance_cm"].astype(int),
            err["angle_deg"].astype(int),
            err["trial"].astype(int),
        )
    )


def pairwise_overlap_report(
    seed_sets: dict[int, set[tuple[str, int, int, int]]],
    reference_seed: int,
) -> pd.DataFrame:
    """Compute overlap against a reference seed."""
    ref = seed_sets[reference_seed]
    rows = []
    for s, s_set in seed_sets.items():
        overlap = ref & s_set
        rows.append(
            {
                "reference_seed": reference_seed,
                "seed": s,
                "n_errors_ref": len(ref),
                "n_errors_seed": len(s_set),
                "n_overlap": len(overlap),
                "overlap_pct_of_ref": 100.0 * len(overlap) / max(len(ref), 1),
                "jaccard_pct": 100.0 * len(overlap) / max(len(ref | s_set), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def parse_int_list(items: Iterable[str]) -> list[int]:
    return [int(x) for x in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate seed-wise misclassification identity overlap.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("outputs/features/features_stability_check.csv"),
        help="Feature CSV used for baseline classification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classifier_investigate"),
        help="Directory for diagnostic outputs.",
    )
    parser.add_argument(
        "--model-seeds",
        nargs="+",
        default=["0", "42"],
        help="Model random_state values to compare.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Fixed StratifiedGroupKFold split seed.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--reference-seed",
        type=int,
        default=0,
        help="Seed used as reference for overlap percentages.",
    )
    args = parser.parse_args()

    model_seeds = parse_int_list(args.model_seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    f1_cols, _, _ = split_feature_columns(df)

    preds_by_seed: dict[int, pd.DataFrame] = {}
    errors_by_seed: dict[int, set[tuple[str, int, int, int]]] = {}

    for seed in model_seeds:
        pred_df = run_seed(
            df=df,
            feature_cols=f1_cols,
            split_seed=args.split_seed,
            model_seed=seed,
            n_splits=args.n_splits,
        )
        preds_by_seed[seed] = pred_df
        errors_by_seed[seed] = error_key_set(pred_df)

        pred_df.to_csv(args.output_dir / f"predictions_seed{seed}.csv", index=False)
        pred_df[pred_df["correct"] == 0].to_csv(
            args.output_dir / f"misclassified_seed{seed}.csv", index=False
        )

    if args.reference_seed not in errors_by_seed:
        raise ValueError(f"reference seed {args.reference_seed} not in --model-seeds")

    overlap_df = pairwise_overlap_report(errors_by_seed, args.reference_seed)
    overlap_df.to_csv(args.output_dir / "overlap_report.csv", index=False)

    print("\nSeed-wise error counts:")
    for seed in sorted(errors_by_seed):
        print(f"seed {seed}: {len(errors_by_seed[seed])} errors")

    print("\nOverlap vs reference seed:")
    print(overlap_df.round(2).to_string(index=False))

    # Convenience print aligned with your suggested snippet for seed 0 vs 42.
    if 0 in errors_by_seed and 42 in errors_by_seed:
        err0 = errors_by_seed[0]
        err42 = errors_by_seed[42]
        overlap = err0 & err42
        print("\nDirect 0 vs 42 summary:")
        print(f"seed 0: {len(err0)} errors")
        print(f"seed 42: {len(err42)} errors")
        print(f"overlap: {len(overlap)} ({100.0 * len(overlap) / max(len(err0), 1):.0f}%)")

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
