from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from TOF_estimation import CARRIERS_HZ, PCB, SOUND_SPEED_M_S
from calibration import GT_OFFSET_MM, tof_to_distance_structural


DEFAULT_INPUT_CSV = Path("outputs") / "tof_matched" / "multifrequenz_denoised_branchA_bayes_garrote_peaks.csv"
DEFAULT_OUT_CSV = Path("outputs") / "tof_matched" / "calibration" / "structural_refit_bayes_garrote.csv"


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


def _structural_mae_cm(df: pd.DataFrame, tof_col: str, params: dict[str, float]) -> tuple[float, float]:
    est_cm = np.array(
        [100.0 * tof_to_distance_structural(getattr(r, tof_col), r.mic, r.tx, params=params) for r in df.itertuples(index=False)],
        dtype=np.float64,
    )
    gt_cm = df["distance_cm"].to_numpy(dtype=np.float64) + (GT_OFFSET_MM / 10.0)
    err = est_cm - gt_cm
    all_mic_mae = float(np.mean(np.abs(err)))

    m12 = df[df["mic"].isin(["Mic1", "Mic2"])].copy()
    est_m12_cm = np.array(
        [100.0 * tof_to_distance_structural(getattr(r, tof_col), r.mic, r.tx, params=params) for r in m12.itertuples(index=False)],
        dtype=np.float64,
    )
    gt_m12_cm = m12["distance_cm"].to_numpy(dtype=np.float64) + (GT_OFFSET_MM / 10.0)
    m12_mae = float(np.mean(np.abs(est_m12_cm - gt_m12_cm)))
    return all_mic_mae, m12_mae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refit structural calibration params on a peaks CSV.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"distance_cm", "mic", "tx", "tof_path_a_s", "tof_path_b_s"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    rows: list[dict] = []
    for method, tof_col in (("path_a", "tof_path_a_s"), ("path_b", "tof_path_b_s")):
        params, fit = _fit_structural(df, tof_col)
        all_mic_mae_cm, m12_mae_cm = _structural_mae_cm(df, tof_col, params)
        rows.append(
            {
                "method": method,
                **params,
                "cost": float(fit.cost),
                "nfev": int(fit.nfev),
                "status": int(fit.status),
                "all_mic_mae_cm": all_mic_mae_cm,
                "mic12_mae_cm": m12_mae_cm,
            }
        )

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved structural refit table: {args.out_csv}")


if __name__ == "__main__":
    main()
