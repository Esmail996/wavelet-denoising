"""Global TOF pipeline runner.

Purpose
-------
Single entrypoint for the complete TOF workflow on a denoised dataset.
It replaces multi-step manual execution of separate scripts.

What this runner does
---------------------
Stage 1: TOF estimation
    - Runs Path A and Path B peak picking over all dataset trials.
    - Produces one peaks CSV containing raw TOF/peak columns.

Stage 2: Delay fitting and persistence
    - Fits empirical delays per (method, tx, mic).
    - Fits structural delay parameters per method.
    - Saves both delay models to CSV files.

Stage 3: Calibration using saved delay files
    - Reloads saved empirical/structural delay files from disk.
    - Applies calibration to produce calibrated distance/error columns.
    - Saves calibrated per-row output and summary report.

Important behavior
------------------
- Calibration is intentionally applied from the saved CSV artifacts, not from
    hardcoded constants in the TOF modules.
- This makes runs reproducible and auditable by artifact files alone.

Main outputs
------------
For run name <name> under --out-dir (default outputs/tof_matched):
    - <name>_peaks.csv
    - calibration/tau_estimates_<name>.csv
    - calibration/structural_params_<name>.csv
    - calibration/calibration_report_<name>.csv
    - <name>_calibrated.csv

CLI summary
-----------
    --data-root PATH   Denoised dataset root with .pickle files.
    --out-dir PATH     Output root for all generated CSV artifacts.
    --name STR         Run name prefix for outputs (defaults to data-root folder name).
    --fs-hz FLOAT      Sampling rate forwarded to TOF estimation.
    --gate-cm FLOAT    Fallback gate half-width for missing STFT windows.
    --picker {max,nearest}
                                         Peak picker mode for TOF estimation.
    --max-files N      Optional cap on files for smoke tests.

Examples
--------
python -m tof.run_tof_global_pipeline

python -m tof.run_tof_global_pipeline \
    --data-root "Multifrequenz Dataset/Multifrequenz_denoised_branchA_bayes_garrote" \
    --name "compare_delays_current" \
    --picker max
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

# Make imports robust for both:
#   python -m tof.run_tof_global_pipeline
#   python tof/run_tof_global_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOF_DIR = PROJECT_ROOT / "tof"
if str(TOF_DIR) not in sys.path:
    sys.path.insert(0, str(TOF_DIR))

from calibration import GT_OFFSET_MM, tof_to_distance_structural
from evaluate_tof_calibration import METHODS, compute_distance_cm, fit_objective_median
from TOF_estimation import CARRIERS_HZ, PCB, SOUND_SPEED_M_S
from run_tof_estimation import FS_HZ, GATE_CM, run_all


DEFAULT_DATA_ROOT = Path("Multifrequenz Dataset") / "Multifrequenz_denoised_branchA_sym6"
DEFAULT_OUT_DIR = Path("outputs") / "tof_matched"


def _safe_name_from_path(path: Path) -> str:
    return path.name.replace(" ", "_")


def _path_length_m(distance_m: float, mic_name: str, tx_name: str) -> float:
    tx_xy = np.asarray(PCB[tx_name], dtype=np.float64)
    mic_xy = np.asarray(PCB[mic_name], dtype=np.float64)
    reflector = np.array([0.0, float(distance_m)], dtype=np.float64)
    return float(np.linalg.norm(tx_xy - reflector) + np.linalg.norm(reflector - mic_xy))


def _resonator_group_delay_s(f_hz: np.ndarray, f0_hz: float, q: float) -> np.ndarray:
    w = 2.0 * np.pi * f_hz
    w0 = 2.0 * np.pi * float(f0_hz)
    numerator = (w0 / float(q)) * (w0**2 + w**2)
    denominator = (w0**2 - w**2) ** 2 + (w * w0 / float(q)) ** 2
    return numerator / denominator


def _fit_structural(df: pd.DataFrame, tof_col: str) -> tuple[dict[str, float], least_squares]:
    distance_corr_m = (df["distance_cm"].to_numpy(dtype=np.float64) + (GT_OFFSET_MM / 10.0)) / 100.0
    mic = df["mic"].to_numpy(dtype=object)
    tx = df["tx"].to_numpy(dtype=object)
    tof_meas = df[tof_col].to_numpy(dtype=np.float64)

    tof_geom = np.array(
        [_path_length_m(d, m, t) / SOUND_SPEED_M_S for d, m, t in zip(distance_corr_m, mic, tx)],
        dtype=np.float64,
    )
    tau_obs = tof_meas - tof_geom

    f_tx = np.array([float(CARRIERS_HZ[t]) for t in tx], dtype=np.float64)
    mic2_mask = mic == "Mic2"
    mic3_mask = mic == "Mic3"

    x0 = np.array([2.760e-6, 0.093, 39_000.0, 2.5, -7.683e-6, 55.767e-6], dtype=np.float64)
    lower = np.array([-200e-6, -1.0, 30_000.0, 0.4, -200e-6, -200e-6], dtype=np.float64)
    upper = np.array([200e-6, 1.0, 80_000.0, 20.0, 200e-6, 200e-6], dtype=np.float64)

    def residuals(x: np.ndarray) -> np.ndarray:
        tau_drive_s, a, f0_hz, q, delta_mic2_s, delta_mic3_s = x
        tau_pred = tau_drive_s + a * _resonator_group_delay_s(f_tx, f0_hz=f0_hz, q=q)
        tau_pred = tau_pred + mic2_mask.astype(np.float64) * delta_mic2_s
        tau_pred = tau_pred + mic3_mask.astype(np.float64) * delta_mic3_s
        return tau_obs - tau_pred

    fit = least_squares(residuals, x0=x0, bounds=(lower, upper), loss="soft_l1", f_scale=5e-6, max_nfev=400)
    tau_drive_s, a, f0_hz, q, delta_mic2_s, delta_mic3_s = fit.x
    params = {
        "tau_drive_s": float(tau_drive_s),
        "a": float(a),
        "f0_hz": float(f0_hz),
        "Q": float(q),
        "delta_Mic1_s": 0.0,
        "delta_Mic2_s": float(delta_mic2_s),
        "delta_Mic3_s": float(delta_mic3_s),
    }
    return params, fit


def _run_calibration_pipeline(
    input_csv: Path,
    tau_csv: Path,
    structural_csv: Path,
    report_csv: Path,
    calibrated_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not input_csv.exists() or input_csv.stat().st_size == 0:
        raise ValueError(
            f"TOF peaks CSV is empty or missing: {input_csv}. "
            "No rows were produced by estimation, so calibration cannot be fitted."
        )

    df = pd.read_csv(input_csv)
    required = {"distance_cm", "mic", "tx", "tof_path_a_s", "tof_path_b_s"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df.copy()
    df["distance_cm_corrected"] = df["distance_cm"] + (GT_OFFSET_MM / 10.0)

    tx_values = sorted(df["tx"].unique())
    mic_values = sorted(df["mic"].unique())

    tau_rows: list[dict] = []
    structural_rows: list[dict] = []
    report_rows: list[dict] = []

    for method_key, cfg in METHODS.items():
        tof_col = cfg["tof_col"]

        # Fit empirical tau per (tx, mic).
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
                        "method_label": cfg["label"],
                        "tx": tx_name,
                        "mic": mic_name,
                        "tau_s": float(tau_s),
                        "tau_us": float(tau_s * 1e6),
                        "mae_cal_cm": float(np.mean(np.abs(err_cal_cm))),
                        "std_cal_cm": float(np.std(err_cal_cm, ddof=1)),
                    }
                )

        params, fit = _fit_structural(df, tof_col)
        structural_rows.append(
            {
                "method": method_key,
                **params,
                "cost": float(fit.cost),
                "nfev": int(fit.nfev),
                "status": int(fit.status),
            }
        )

    tau_df = pd.DataFrame(tau_rows)
    structural_df = pd.DataFrame(structural_rows)

    tau_csv.parent.mkdir(parents=True, exist_ok=True)
    structural_csv.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    calibrated_csv.parent.mkdir(parents=True, exist_ok=True)

    tau_df.to_csv(tau_csv, index=False)
    structural_df.to_csv(structural_csv, index=False)

    # IMPORTANT: Load calibration from saved files (not hardcoded constants).
    tau_saved = pd.read_csv(tau_csv)
    structural_saved = pd.read_csv(structural_csv)
    out_df = df.copy()

    for method_key, cfg in METHODS.items():
        tof_col = cfg["tof_col"]
        prefix = cfg["prefix"]

        tau_subset = tau_saved[tau_saved["method"] == method_key]
        tau_map = {(r.tx, r.mic): float(r.tau_s) for r in tau_subset.itertuples(index=False)}

        est_uncal_col = f"est_dist_{prefix}_cm"
        est_emp_col = f"est_dist_{prefix}_cal_cm"
        err_uncal_col = f"err_{prefix}_uncal_cm"
        err_emp_col = f"err_{prefix}_cal_cm"

        out_df[est_uncal_col] = compute_distance_cm(out_df, tof_col)
        out_df[est_emp_col] = compute_distance_cm(out_df, tof_col, tau_map)
        out_df[err_uncal_col] = out_df[est_uncal_col] - out_df["distance_cm_corrected"]
        out_df[err_emp_col] = out_df[est_emp_col] - out_df["distance_cm_corrected"]

        struct_row = structural_saved[structural_saved["method"] == method_key]
        if struct_row.empty:
            raise ValueError(f"No structural params saved for method '{method_key}'")

        sr = struct_row.iloc[0]
        struct_params = {
            "tau_drive_s": float(sr["tau_drive_s"]),
            "a": float(sr["a"]),
            "f0_hz": float(sr["f0_hz"]),
            "Q": float(sr["Q"]),
            "delta_Mic1_s": float(sr["delta_Mic1_s"]),
            "delta_Mic2_s": float(sr["delta_Mic2_s"]),
            "delta_Mic3_s": float(sr["delta_Mic3_s"]),
        }

        est_struct_col = f"est_dist_{prefix}_struct_cal_cm"
        err_struct_col = f"err_{prefix}_struct_cal_cm"
        out_df[est_struct_col] = np.array(
            [
                100.0 * tof_to_distance_structural(getattr(r, tof_col), r.mic, r.tx, params=struct_params)
                for r in out_df.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        out_df[err_struct_col] = out_df[est_struct_col] - out_df["distance_cm_corrected"]

        report_rows.append(
            {
                "method": method_key,
                "calibration_model": "empirical",
                "scope": "overall",
                "group": "all",
                "mae_uncal_cm": float(out_df[err_uncal_col].abs().mean()),
                "mae_cal_cm": float(out_df[err_emp_col].abs().mean()),
                "improvement_cm": float(out_df[err_uncal_col].abs().mean() - out_df[err_emp_col].abs().mean()),
                "bias_cal_cm": float(out_df[err_emp_col].mean()),
                "std_cal_cm": float(out_df[err_emp_col].std()),
                "pct_le_2cm_cal": float((out_df[err_emp_col].abs() <= 2.0).mean() * 100.0),
            }
        )
        report_rows.append(
            {
                "method": method_key,
                "calibration_model": "structural",
                "scope": "overall",
                "group": "all",
                "mae_uncal_cm": float(out_df[err_uncal_col].abs().mean()),
                "mae_cal_cm": float(out_df[err_struct_col].abs().mean()),
                "improvement_cm": float(out_df[err_uncal_col].abs().mean() - out_df[err_struct_col].abs().mean()),
                "bias_cal_cm": float(out_df[err_struct_col].mean()),
                "std_cal_cm": float(out_df[err_struct_col].std()),
                "pct_le_2cm_cal": float((out_df[err_struct_col].abs() <= 2.0).mean() * 100.0),
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_csv, index=False)
    out_df.to_csv(calibrated_csv, index=False)

    return tau_df, structural_df, report_df, out_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ToF estimation and fit/save empirical + structural calibrations in one command."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root folder containing denoised .pickle files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root for tof and calibration artifacts.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name used in filenames. Default: data-root folder name.",
    )

    parser.add_argument("--fs-hz", type=float, default=FS_HZ, help="Sampling rate in Hz.")
    parser.add_argument("--gate-cm", type=float, default=GATE_CM, help="Fallback gate half-width in cm.")
    parser.add_argument("--picker", type=str, default="max", choices=("max", "nearest"), help="Peak picker in search window.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    run_name = args.name if args.name else _safe_name_from_path(args.data_root)
    tof_csv = args.out_dir / f"{run_name}_peaks.csv"
    cal_dir = args.out_dir / "calibration"
    tau_csv = cal_dir / f"tau_estimates_{run_name}.csv"
    structural_csv = cal_dir / f"structural_params_{run_name}.csv"
    report_csv = cal_dir / f"calibration_report_{run_name}.csv"
    calibrated_csv = args.out_dir / f"{run_name}_calibrated.csv"

    print("=" * 100)
    print("STEP 1/3: ToF estimation (Path A + Path B)")
    print("=" * 100)
    tof_df = run_all(
        data_root=args.data_root,
        out_csv=tof_csv,
        fs_hz=float(args.fs_hz),
        gate_cm=float(args.gate_cm),
        picker=str(args.picker),
        max_files=args.max_files,
    )
    print(f"Wrote {len(tof_df)} rows to {tof_csv}")
    if tof_df.empty:
        raise ValueError(
            "TOF estimation produced zero rows. "
            "Check --data-root and filename pattern '<distance>cm_<angle>Grad' before calibration."
        )

    print("\n" + "=" * 100)
    print("STEP 2/3: Fit and save empirical + structural delays")
    print("=" * 100)
    tau_df, structural_df, report_df, out_df = _run_calibration_pipeline(
        input_csv=tof_csv,
        tau_csv=tau_csv,
        structural_csv=structural_csv,
        report_csv=report_csv,
        calibrated_csv=calibrated_csv,
    )

    print("\n" + "=" * 100)
    print("STEP 3/3: Calibration applied from saved delay files")
    print("=" * 100)
    print(f"Tau estimates:   {tau_csv} ({len(tau_df)} rows)")
    print(f"Structural fit:  {structural_csv} ({len(structural_df)} rows)")
    print(f"Calib report:    {report_csv} ({len(report_df)} rows)")
    print(f"Calibrated data: {calibrated_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
