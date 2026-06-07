from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from TOF_estimation import PCB, SOUND_SPEED_M_S
from calibration import GT_OFFSET_MM, tof_to_distance_m


METHODS = {
    "path_a": {"label": "Path A (BP)", "tof_col": "tof_path_a_s", "prefix": "path_a"},
    "path_b": {"label": "Path B", "tof_col": "tof_path_b_s", "prefix": "path_b"},
}

DEFAULT_INPUT_CSV = Path("outputs") / "tof_matched" / "multifrequenz_denoised_branchA_db6.csv"
DEFAULT_TAU_CSV = Path("outputs") / "tof_matched" / "calibration" / "tau_estimates_db6.csv"
DEFAULT_REPORT_CSV = Path("outputs") / "tof_matched" / "calibration" / "calibration_report_db6.csv"
DEFAULT_CALIBRATED_CSV = Path("outputs") / "tof_matched" / "multifrequenz_denoised_branchA_db6_calibrated.csv"


def compute_distance_cm(frame: pd.DataFrame, tof_col: str, tau_map: dict[tuple[str, str], float] | None = None) -> np.ndarray:
    tau_map = tau_map or {}
    return np.array(
        [
            100.0
            * tof_to_distance_m(
                getattr(r, tof_col) - float(tau_map.get((r.tx, r.mic), 0.0)),
                r.mic,
                r.tx,
                PCB,
                SOUND_SPEED_M_S,
            )
            for r in frame.itertuples(index=False)
        ],
        dtype=np.float64,
    )


def fit_objective_median(frame: pd.DataFrame, tof_col: str, tau_s: float) -> float:
    est_cm = compute_distance_cm(frame, tof_col, {(r.tx, r.mic): tau_s for r in frame[["tx", "mic"]].itertuples(index=False)})
    err_cm = est_cm - frame["distance_cm_corrected"].to_numpy(dtype=np.float64)
    return float(np.median(err_cm))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate tau separately from runner output and write separate tau/report/calibrated artifacts.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Peak-only ToF CSV from run_tof_estimation.py")
    parser.add_argument("--tau-csv", type=Path, default=DEFAULT_TAU_CSV, help="Output CSV for fitted tau constants")
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV, help="Output CSV for evaluation summary")
    parser.add_argument("--calibrated-csv", type=Path, default=DEFAULT_CALIBRATED_CSV, help="Output CSV with added uncal/cal distance columns")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = pd.read_csv(args.input_csv)
    # Apply engineer-documented housing-to-acoustic correction to ground truth.
    df["distance_cm_corrected"] = df["distance_cm"] + (GT_OFFSET_MM / 10.0)

    required = {"distance_cm", "mic", "tx", "tof_path_a_s", "tof_path_b_s"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    print("=" * 100)
    print("STEP 1: Fit delay per (Tx, Mic, Method)")
    print("=" * 100)

    tau_rows: list[dict] = []
    report_rows: list[dict] = []

    tx_values = sorted(df["tx"].unique())
    mic_values = sorted(df["mic"].unique())

    for method_key, cfg in METHODS.items():
        tof_col = cfg["tof_col"]
        label = cfg["label"]
        print(f"\n{label}")
        print("-" * 100)
        print(f'{"Tx":>5} {"Mic":>5} {"tau_us":>10} {"MAE_cm":>8} {"std_cm":>8}')
        print("-" * 100)

        for tx_name in tx_values:
            for mic_name in mic_values:
                subset = df[(df["tx"] == tx_name) & (df["mic"] == mic_name)]
                if subset.empty:
                    continue

                tau_s = brentq(lambda tau: fit_objective_median(subset, tof_col, tau), -200e-6, 200e-6)
                est_cal_cm = compute_distance_cm(subset, tof_col, {(tx_name, mic_name): tau_s})
                err_cal_cm = est_cal_cm - subset["distance_cm_corrected"].to_numpy(dtype=np.float64)

                tau_rows.append(
                    {
                        "method": method_key,
                        "method_label": label,
                        "tx": tx_name,
                        "mic": mic_name,
                        "tau_s": float(tau_s),
                        "tau_us": float(tau_s * 1e6),
                        "mae_cal_cm": float(np.mean(np.abs(err_cal_cm))),
                        "std_cal_cm": float(np.std(err_cal_cm, ddof=1)),
                    }
                )

                print(f"{tx_name:>5} {mic_name:>5} {tau_s*1e6:>10.3f} {np.mean(np.abs(err_cal_cm)):>8.3f} {np.std(err_cal_cm, ddof=1):>8.3f}")

    tau_df = pd.DataFrame(tau_rows)

    print("\n" + "=" * 100)
    print("STEP 2: Evaluate and report (separate output)")
    print("=" * 100)

    out_df = df.copy()
    for method_key, cfg in METHODS.items():
        tof_col = cfg["tof_col"]
        prefix = cfg["prefix"]
        label = cfg["label"]

        tau_subset = tau_df[tau_df["method"] == method_key]
        tau_map = {(r.tx, r.mic): float(r.tau_s) for r in tau_subset.itertuples(index=False)}

        est_uncal_col = f"est_dist_{prefix}_cm"
        est_cal_col = f"est_dist_{prefix}_cal_cm"
        err_uncal_col = f"err_{prefix}_uncal_cm"
        err_cal_col = f"err_{prefix}_cal_cm"

        out_df[est_uncal_col] = compute_distance_cm(out_df, tof_col)
        out_df[est_cal_col] = compute_distance_cm(out_df, tof_col, tau_map)
        out_df[err_uncal_col] = out_df[est_uncal_col] - out_df["distance_cm_corrected"]
        out_df[err_cal_col] = out_df[est_cal_col] - out_df["distance_cm_corrected"]

        print(f"\n{label}")
        print("=" * 100)

        mae_uncal = float(out_df[err_uncal_col].abs().mean())
        mae_cal = float(out_df[err_cal_col].abs().mean())
        print("\nFULL DATASET (all mics):")
        print(f"  Uncalibrated: MAE={mae_uncal:.3f} cm")
        print(f"  Calibrated:   MAE={mae_cal:.3f} cm  improvement={mae_uncal - mae_cal:+.3f} cm")

        report_rows.append(
            {
                "method": method_key,
                "scope": "overall",
                "group": "all",
                "mae_uncal_cm": mae_uncal,
                "mae_cal_cm": mae_cal,
                "improvement_cm": mae_uncal - mae_cal,
                "bias_cal_cm": float(out_df[err_cal_col].mean()),
                "std_cal_cm": float(out_df[err_cal_col].std()),
                "pct_le_2cm_cal": float((out_df[err_cal_col].abs() <= 2.0).mean() * 100.0),
            }
        )

        for mic_name in mic_values:
            g = out_df[out_df["mic"] == mic_name]
            mae_u = float(g[err_uncal_col].abs().mean())
            mae_c = float(g[err_cal_col].abs().mean())
            print(f"\n{mic_name}:")
            print(f"  Uncalibrated: MAE={mae_u:.3f} cm")
            print(f"  Calibrated:   MAE={mae_c:.3f} cm  improvement={mae_u - mae_c:+.3f} cm")

            report_rows.append(
                {
                    "method": method_key,
                    "scope": "mic",
                    "group": mic_name,
                    "mae_uncal_cm": mae_u,
                    "mae_cal_cm": mae_c,
                    "improvement_cm": mae_u - mae_c,
                    "bias_cal_cm": float(g[err_cal_col].mean()),
                    "std_cal_cm": float(g[err_cal_col].std()),
                    "pct_le_2cm_cal": float((g[err_cal_col].abs() <= 2.0).mean() * 100.0),
                }
            )

        print("\nPER-DISTANCE (full dataset):")
        print(f'{"distance":>8} {"uncal_mae":>10} {"cal_mae":>8} {"cal_bias":>9} {"≤2cm%":>8}')
        print("-" * 60)
        for distance_cm in sorted(out_df["distance_cm"].unique()):
            g = out_df[out_df["distance_cm"] == distance_cm]
            mae_u = float(g[err_uncal_col].abs().mean())
            mae_c = float(g[err_cal_col].abs().mean())
            bias = float(g[err_cal_col].mean())
            pct = float((g[err_cal_col].abs() <= 2.0).mean() * 100.0)
            print(f"{int(distance_cm):>8} {mae_u:>10.3f} {mae_c:>8.3f} {bias:>9.3f} {pct:>7.1f}%")

            report_rows.append(
                {
                    "method": method_key,
                    "scope": "distance",
                    "group": int(distance_cm),
                    "mae_uncal_cm": mae_u,
                    "mae_cal_cm": mae_c,
                    "improvement_cm": mae_u - mae_c,
                    "bias_cal_cm": bias,
                    "std_cal_cm": float(g[err_cal_col].std()),
                    "pct_le_2cm_cal": pct,
                }
            )

    report_df = pd.DataFrame(report_rows)

    args.tau_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    args.calibrated_csv.parent.mkdir(parents=True, exist_ok=True)

    tau_df.to_csv(args.tau_csv, index=False)
    report_df.to_csv(args.report_csv, index=False)
    out_df.to_csv(args.calibrated_csv, index=False)

    print("\n" + "=" * 100)
    print("STEP 3: Save separate outputs")
    print("=" * 100)
    print(f"Tau estimates: {args.tau_csv}")
    print(f"Report table:  {args.report_csv}")
    print(f"Calibrated CSV:{args.calibrated_csv}")


if __name__ == "__main__":
    main()



"""
python -m tof.evaluate_tof_calibration `
  --input-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv" `
  --tau-csv "outputs/tof_matched/calibration/tau_estimates_bayes_garrote.csv" `
  --report-csv "outputs/tof_matched/calibration/calibration_report_bayes_garrote.csv" `
  --calibrated-csv "outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_calibrated.csv"
"""