"""
Comparative study: Path A (matched filter) vs Path B (Morlet CWT)
Breakdowns: per-distance, per-angle, per-mic, per-category, combined.
Uses the calibrated distance columns from the runner output.
"""
import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("outputs/tof_matched/multifrequenz_denoised_branchA_sym6.csv")
df = pd.read_csv(csv_path)

df["err_a"] = df["est_dist_path_a_cal_cm"] - df["distance_cm"]
df["err_b"] = df["est_dist_path_b_cal_cm"] - df["distance_cm"]

SEP = "=" * 110

def stats(series):
    return {
        "MAE":  series.abs().mean(),
        "bias": series.mean(),
        "std":  series.std(),
        "p50":  series.abs().quantile(0.50),
        "p90":  series.abs().quantile(0.90),
        "≤1cm": (series.abs() <= 1.0).mean() * 100,
        "≤2cm": (series.abs() <= 2.0).mean() * 100,
    }

def table(groups, key_label, rows_fn):
    hdr = f"{key_label:<18} {'MAE_A':>6} {'MAE_B':>6} {'bias_A':>7} {'bias_B':>7} {'std_A':>6} {'std_B':>6} {'p90_A':>6} {'p90_B':>6} {'≤2cm_A':>7} {'≤2cm_B':>7} {'winner':>7}"
    print(hdr)
    print("-" * 110)
    for label, g in groups:
        sa = stats(g["err_a"])
        sb = stats(g["err_b"])
        winner = "B" if sb["MAE"] < sa["MAE"] else "A"
        print(f"{str(label):<18} {sa['MAE']:>6.3f} {sb['MAE']:>6.3f} "
              f"{sa['bias']:>7.3f} {sb['bias']:>7.3f} "
              f"{sa['std']:>6.3f} {sb['std']:>6.3f} "
              f"{sa['p90']:>6.3f} {sb['p90']:>6.3f} "
              f"{sa['≤2cm']:>6.1f}% {sb['≤2cm']:>6.1f}%  {'  →'+winner:>5}")
    print()

# ──────────────────────────────────────────────────────────────────────────────
print(SEP)
print("OVERALL")
print(SEP)
sa = stats(df["err_a"])
sb = stats(df["err_b"])
print(f"{'Metric':<12}  Path A (BP)   Path B (CWT)")
print("-" * 40)
for k in ["MAE", "bias", "std", "p50", "p90", "≤1cm", "≤2cm"]:
    unit = "%" if "≤" in k else " cm"
    print(f"  {k:<10} {sa[k]:>8.3f}{unit}  {sb[k]:>8.3f}{unit}")

# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("PER DISTANCE")
print(SEP)
table(df.groupby("distance_cm"), "distance_cm", None)

# ──────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("PER ANGLE")
print(SEP)
table(df.groupby("angle_deg"), "angle_deg", None)

# ──────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("PER MIC")
print(SEP)
table(df.groupby("mic"), "mic", None)

# ──────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("PER TX")
print(SEP)
table(df.groupby("tx"), "tx", None)

# ──────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("PER CATEGORY (object type)")
print(SEP)
table(df.groupby("category"), "category", None)

# ──────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("PER MIC × DISTANCE (Path B MAE, calibrated)")
print(SEP)
pivot = df.groupby(["mic", "distance_cm"]).apply(lambda g: g["err_b"].abs().mean()).unstack()
print(pivot.round(3).to_string())

# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("PER MIC × DISTANCE (Path A MAE, calibrated)")
print(SEP)
pivot_a = df.groupby(["mic", "distance_cm"]).apply(lambda g: g["err_a"].abs().mean()).unstack()
print(pivot_a.round(3).to_string())

# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("PATH A vs PATH B: AGREEMENT (|est_a - est_b| per row)")
print(SEP)
df["a_b_diff"] = (df["est_dist_path_a_cal_cm"] - df["est_dist_path_b_cal_cm"]).abs()
print(f"  Mean |A-B|:   {df['a_b_diff'].mean():.3f} cm")
print(f"  Median |A-B|: {df['a_b_diff'].median():.3f} cm")
print(f"  p90 |A-B|:    {df['a_b_diff'].quantile(0.9):.3f} cm")
print(f"  Max |A-B|:    {df['a_b_diff'].max():.3f} cm")
print(f"  |A-B| ≤ 1cm:  {(df['a_b_diff'] <= 1.0).mean()*100:.1f}%")
print(f"  |A-B| ≤ 2cm:  {(df['a_b_diff'] <= 2.0).mean()*100:.1f}%")

print(f"\nPer-mic agreement:")
for mic, g in df.groupby("mic"):
    d = g["a_b_diff"]
    print(f"  {mic}: mean={d.mean():.3f} cm  p90={d.quantile(0.9):.3f} cm")

print(f"\n{SEP}")
print("SUMMARY: WINNER BY CATEGORY")
print(SEP)
for grp_col in ["distance_cm", "angle_deg", "mic", "tx", "category"]:
    wins_b = sum(1 for _, g in df.groupby(grp_col) if g["err_b"].abs().mean() < g["err_a"].abs().mean())
    total = df[grp_col].nunique()
    print(f"  {grp_col:<15}: Path B wins {wins_b}/{total} groups")
