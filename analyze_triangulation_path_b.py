from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from triangulation import triangulate_from_path_b_rows


DEFAULT_INPUT_CSV = Path("outputs") / "tof_matched" / "multifrequenz_denoised_branchA_sym6.csv"
DEFAULT_OUTPUT_CSV = Path("outputs") / "triangulation" / "path_b_calibrated_results.csv"


def _angle_diff_deg(est_deg: float, true_deg: float) -> float:
    return float((est_deg - true_deg + 180.0) % 360.0 - 180.0)


def _aggregate_trials_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse trial-level rows to one mean row per (object, distance, angle, mic, tx)."""
    required = {"category", "distance_cm", "angle_deg", "trial", "mic", "tx", "tof_path_b_s", "amp_path_b"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns for trial aggregation: {missing}")

    grouped = (
        df.groupby(["category", "distance_cm", "angle_deg", "mic", "tx"], as_index=False)
        .agg(
            tof_path_b_s=("tof_path_b_s", "mean"),
            amp_path_b=("amp_path_b", "mean"),
            n_trials=("trial", "nunique"),
        )
    )
    return grouped


def run(input_csv: Path, output_csv: Path, calibrate: bool = True, aggregate_trials_mean: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required = {"category", "distance_cm", "angle_deg", "mic", "tx", "tof_path_b_s", "amp_path_b"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in input CSV: {missing}")

    if aggregate_trials_mean:
        work = _aggregate_trials_mean(df)
        group_cols = ["category", "distance_cm", "angle_deg"]
    else:
        required_trial = {"file", "relative_path", "trial"}
        missing_trial = sorted(required_trial.difference(df.columns))
        if missing_trial:
            raise KeyError(f"Missing required columns for per-trial mode: {missing_trial}")
        work = df
        group_cols = ["category", "file", "relative_path", "distance_cm", "angle_deg", "trial"]

    rows = []

    for group_key, g in work.groupby(group_cols, dropna=False, sort=False):
        if aggregate_trials_mean:
            category, distance_cm, angle_deg = group_key
            file_name = "<mean_over_trials>"
            rel_path = "<mean_over_trials>"
            trial = -1
            n_trials = int(g["n_trials"].min()) if "n_trials" in g.columns else np.nan
        else:
            category, file_name, rel_path, distance_cm, angle_deg, trial = group_key
            n_trials = 1

        try:
            R, cov, n_iter, theta_deg = triangulate_from_path_b_rows(g, calibrate=calibrate)
        except Exception as exc:
            rows.append(
                {
                    "category": category,
                    "file": file_name,
                    "relative_path": rel_path,
                    "distance_cm": float(distance_cm),
                    "angle_deg": float(angle_deg),
                    "trial": int(trial),
                    "n_trials": int(n_trials) if np.isfinite(n_trials) else np.nan,
                    "ok": False,
                    "error": str(exc),
                }
            )
            continue

        d_true_m = float(distance_cm) * 1e-2
        a_true_rad = np.deg2rad(float(angle_deg))
        x_true_m = d_true_m * np.sin(a_true_rad)
        y_true_m = d_true_m * np.cos(a_true_rad)

        x_est_m = float(R[0])
        y_est_m = float(R[1])
        d_est_m = float(np.hypot(x_est_m, y_est_m))
        a_est_deg = float(np.rad2deg(np.arctan2(x_est_m, y_est_m)))

        err_xy_m = float(np.hypot(x_est_m - x_true_m, y_est_m - y_true_m))
        err_d_m = d_est_m - d_true_m
        err_a_deg = _angle_diff_deg(a_est_deg, float(angle_deg))

        rows.append(
            {
                "category": category,
                "file": file_name,
                "relative_path": rel_path,
                "distance_cm": float(distance_cm),
                "angle_deg": float(angle_deg),
                "trial": int(trial),
                "n_trials": int(n_trials) if np.isfinite(n_trials) else np.nan,
                "ok": True,
                "x_true_m": x_true_m,
                "y_true_m": y_true_m,
                "x_est_m": x_est_m,
                "y_est_m": y_est_m,
                "dist_true_m": d_true_m,
                "dist_est_m": d_est_m,
                "angle_est_deg": a_est_deg,
                "xy_err_cm": err_xy_m * 100.0,
                "dist_err_cm": err_d_m * 100.0,
                "angle_err_deg": err_a_deg,
                "theta_est_deg": float(theta_deg),
                "n_iter": int(n_iter),
                "cov_xx": float(cov[0, 0]),
                "cov_xy": float(cov[0, 1]),
                "cov_yy": float(cov[1, 1]),
                "cov_trace": float(np.trace(cov)),
            }
        )

    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def summarize(result: pd.DataFrame) -> None:
    total = len(result)
    ok = int(result["ok"].sum()) if "ok" in result.columns else 0
    failed = total - ok
    print(f"Total groups: {total}")
    print(f"Solved: {ok}")
    print(f"Failed: {failed}")

    if ok == 0:
        return

    good = result[result["ok"]].copy()

    print("\nOverall error summary (Path B calibrated)")
    print(good[["xy_err_cm", "dist_err_cm", "angle_err_deg"]].describe(percentiles=[0.5, 0.9, 0.95]).round(3))

    by_dist = (
        good.groupby("distance_cm")[["xy_err_cm", "dist_err_cm", "angle_err_deg"]]
        .agg(["mean", "median", "std"])
        .round(3)
    )
    print("\nBy distance")
    print(by_dist)

    by_cat = (
        good.groupby("category")[["xy_err_cm", "dist_err_cm", "angle_err_deg"]]
        .agg(["mean", "median", "std"])
        .round(3)
    )
    print("\nBy category")
    print(by_cat)



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch triangulation using Path B ToFs after calibration and error analysis.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Input ToF CSV containing Path B columns.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV for per-group triangulation results.")
    parser.add_argument("--no-calibration", action="store_true", help="Disable Path B delay calibration before triangulation.")
    parser.add_argument(
        "--aggregate-trials-mean",
        action="store_true",
        help="Average Path B TOF/amp over trials for each (category, distance, angle, mic, tx), then triangulate one result per (category, distance, angle).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        calibrate=not args.no_calibration,
        aggregate_trials_mean=args.aggregate_trials_mean,
    )
    print(f"Saved results to {args.output_csv}")
    summarize(result)


if __name__ == "__main__":
    main()
