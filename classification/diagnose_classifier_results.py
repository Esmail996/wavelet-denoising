"""
diagnose_classifier_results.py — turn classify_features.py outputs into analysis.

Run this after classify_features.py finishes. Produces:

    - feature_importance_<best>.csv      :  permutation importance ranking
    - per_class_f1_<best>.csv            :  P/R/F1 per class
    - improvement_proposals.txt          :  ranked list of likely improvements
                                              based on the observed error pattern

USAGE:
    python diagnose_classifier_results.py \
        --features-csv outputs/features/features_all.csv \
        --classification-dir outputs/classification \
        --output-dir outputs/diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


META_COLS = ("category", "distance_cm", "angle_deg", "trial")
LABEL_COL = "category"


def load_summary(classification_dir: Path) -> pd.DataFrame:
    p = classification_dir / "summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"summary.csv not found at {p}")
    return pd.read_csv(p)


def load_per_cell(classification_dir: Path, family: str, clf: str) -> pd.DataFrame:
    p = classification_dir / f"per_cell_{family}_{clf}.csv"
    return pd.read_csv(p, index_col=0)


def load_confusion(classification_dir: Path, family: str, clf: str) -> pd.DataFrame:
    p = classification_dir / f"confusion_{family}_{clf}.csv"
    return pd.read_csv(p, index_col=0)


def feature_columns_for_family(df: pd.DataFrame, family: str) -> list[str]:
    """Return the column subset for a given family token from summary."""
    f1, f15, f2 = [], [], []
    for c in df.columns:
        if c in META_COLS:
            continue
        if "_cep_" in c or "_dlog_energy" in c or "_dlog_peak" in c or c == "signed_amp_ratio":
            f15.append(c)
        elif c.split("_")[-1].startswith("S") and c.split("_")[-1][1:].isdigit():
            f2.append(c)
        else:
            f1.append(c)
    if family == "F1": return f1
    if family == "F15": return f15
    if family == "F2": return f2
    if family == "F1+F15": return f1 + f15
    if family == "F1+F2": return f1 + f2
    if family == "F15+F2": return f15 + f2
    if family == "F1+F15+F2_full_hybrid": return f1 + f15 + f2
    if family.endswith("_with_da"):
        base = family[:-len("_with_da")]
        return feature_columns_for_family(df, base) + ["distance_cm", "angle_deg"]
    raise ValueError(f"Unknown family token: {family}")


def compute_permutation_importance(
    df: pd.DataFrame, family: str, clf_name: str, n_repeats: int = 5,
) -> pd.DataFrame:
    cols = feature_columns_for_family(df, family)
    X = df[cols].values.astype(np.float32)
    y = df[LABEL_COL].values
    groups = df["distance_cm"].astype(str) + "_" + df["angle_deg"].astype(str)

    if clf_name == "LR-L2":
        clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    elif clf_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    elif clf_name == "RBF-SVM":
        clf = Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=10.0, gamma="scale", random_state=0))])
    elif clf_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ValueError("clf LightGBM requested for diagnostics but lightgbm is not installed") from e
        clf = LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=0,
            verbosity=-1,
        )
    else:
        raise ValueError(f"clf {clf_name} not supported for diagnostics")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    tr_idx, te_idx = next(sgkf.split(X, y, groups))
    clf.fit(X[tr_idx], y[tr_idx])
    pi = permutation_importance(clf, X[te_idx], y[te_idx],
                                n_repeats=n_repeats, random_state=0, n_jobs=-1)
    out = pd.DataFrame({
        "feature": cols,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return out


def write_proposals(
    summary: pd.DataFrame,
    best_per_cell: pd.DataFrame,
    best_confusion: pd.DataFrame,
    importance: pd.DataFrame,
    out_path: Path,
) -> None:
    """Build a ranked, evidence-based list of improvement proposals."""
    proposals: list[tuple[float, str]] = []  # (priority, message)
    lines = []
    lines.append("=" * 72)
    lines.append("Classifier diagnostics — improvement proposals")
    lines.append("=" * 72)

    # 1. Family ranking
    lines.append("\n[1] Family ranking (mean accuracy across classifiers):")
    by_family = summary.groupby("family")["mean_acc"].mean().sort_values(ascending=False)
    for f, a in by_family.items():
        lines.append(f"    {f:<28} {a:.3f}")
    if by_family.iloc[0] - by_family.iloc[-1] > 0.05:
        worst, best = by_family.index[-1], by_family.index[0]
        proposals.append((10, f"Family '{worst}' lags '{best}' by "
                              f"{by_family.iloc[0]-by_family.iloc[-1]:.1%}. "
                              f"Drop or reweight features in {worst} that don't survive into the hybrid."))

    # 2. Classifier ranking
    lines.append("\n[2] Classifier ranking (mean accuracy across families):")
    by_clf = summary.groupby("classifier")["mean_acc"].mean().sort_values(ascending=False)
    for c, a in by_clf.items():
        lines.append(f"    {c:<28} {a:.3f}")

    # 3. Per-cell breakdown
    lines.append("\n[3] Per-(distance, angle) accuracy heatmap (best combo):")
    lines.append(best_per_cell.round(2).to_string())
    weakest = best_per_cell.stack().nsmallest(3)
    lines.append("\nWeakest cells:")
    for (d, a), v in weakest.items():
        lines.append(f"    distance={d}, angle={a}°: {v:.2f}")
    if weakest.iloc[0] < 0.5:
        d, a = weakest.index[0]
        proposals.append((9, f"Cell (distance={d}, angle={a}°) accuracy = "
                              f"{weakest.iloc[0]:.2f}. Inspect ROI alignment + SNR there. "
                              f"Consider a per-distance feature normalisation or a bespoke "
                              f"classifier for that range."))

    # 4. Confusion patterns
    lines.append("\n[4] Aggregated confusion matrix (best combo):")
    lines.append(best_confusion.to_string())
    cm = best_confusion.values
    classes = best_confusion.index.tolist()
    n_classes = len(classes)
    off_diag = cm - np.diag(np.diag(cm))
    if off_diag.sum() > 0:
        i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
        true_class, pred_class = classes[i], classes[j]
        rate = off_diag[i, j] / cm[i].sum()
        lines.append(f"\nDominant confusion: true {true_class} → predicted {pred_class} "
                     f"({rate:.1%} of {true_class} trials)")
        if rate > 0.20:
            proposals.append((8, f"{true_class} is misclassified as {pred_class} on {rate:.0%} of trials. "
                                  f"Engineer a feature that discriminates these two materials specifically — "
                                  f"e.g. cepstral coefficients capturing surface roughness, "
                                  f"or per-Tx amplitude ratios that probe frequency-dependent absorption."))

    # 5. Top features
    lines.append("\n[5] Top 15 features by permutation importance (best combo):")
    for _, r in importance.head(15).iterrows():
        lines.append(f"    {r['feature']:<60} {r['importance_mean']:+.4f} ± {r['importance_std']:.4f}")
    # Identify which family the top features belong to
    fam_counts = {"F1": 0, "F15": 0, "F2": 0}
    for f in importance.head(20)["feature"].tolist():
        if "_cep_" in f or "_dlog_" in f or f == "signed_amp_ratio":
            fam_counts["F15"] += 1
        elif f.split("_")[-1].startswith("S") and f.split("_")[-1][1:].isdigit():
            fam_counts["F2"] += 1
        else:
            fam_counts["F1"] += 1
    lines.append(f"\nFamily share of top-20 features: {fam_counts}")
    leading = max(fam_counts, key=fam_counts.get)
    if fam_counts[leading] >= 12:
        proposals.append((7, f"Top features are dominated by {leading}. "
                              f"Consider reducing the other families or applying within-family "
                              f"feature selection (mutual information, recursive feature elimination)."))

    # 6. Distance/angle in features
    has_da = any("_with_da" in f for f in summary.family)
    if has_da:
        sans = summary[~summary.family.str.endswith("_with_da")].mean_acc.mean()
        with_da = summary[summary.family.str.endswith("_with_da")].mean_acc.mean()
        diff = with_da - sans
        lines.append(f"\n[6] Distance/angle as features: avg accuracy +{diff:.3f} when added.")
        if diff > 0.05:
            proposals.append((6, f"Including distance/angle as features improves accuracy by {diff:.1%}. "
                                  f"This indicates the 'pure material' features are still amplitude- or "
                                  f"timing-coupled to distance. Add per-distance normalisation or train "
                                  f"separate classifiers per distance bin."))
        else:
            lines.append("    (Difference small — features generalise well across distance/angle.)")

    # Render proposals
    proposals.sort(key=lambda x: -x[0])
    lines.append("\n" + "=" * 72)
    lines.append("Improvement proposals (ranked by expected impact)")
    lines.append("=" * 72)
    if not proposals:
        lines.append("Results look clean — no specific weakness identified.")
    else:
        for i, (prio, msg) in enumerate(proposals, 1):
            lines.append(f"\n[{i}] (priority {prio}/10) {msg}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Permutation importance repeats")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    print(f"  {len(df)} rows × {df.shape[1]} cols")

    summary = load_summary(args.classification_dir)
    print(f"\nSummary table:\n{summary.round(3).to_string(index=False)}")
    best = summary.iloc[0]
    print(f"\nBest combo: {best['family']} + {best['classifier']} -> "
          f"{best['mean_acc']:.3f} +/- {best['std_acc']:.3f}")

    per_cell = load_per_cell(args.classification_dir, best["family"], best["classifier"])
    confusion = load_confusion(args.classification_dir, best["family"], best["classifier"])

    print("\nComputing permutation importance (this may take a few minutes)...")
    importance = compute_permutation_importance(
        df, best["family"], best["classifier"], n_repeats=args.n_repeats,
    )
    importance.to_csv(args.output_dir / f"feature_importance_{best['family']}_{best['classifier']}.csv",
                      index=False)

    write_proposals(
        summary, per_cell, confusion, importance,
        args.output_dir / "improvement_proposals.txt",
    )
    print(f"\nDiagnostics written to {args.output_dir}")
    print(f"  - feature_importance_{best['family']}_{best['classifier']}.csv")
    print(f"  - improvement_proposals.txt")


if __name__ == "__main__":
    main()
