"""
Test: compare mean-zeroing vs median-zeroing tau fit for each (Tx, Mic, Method).
Prints tau difference and resulting MAE for both strategies side by side.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from TOF_estimation import PCB, SOUND_SPEED_M_S, tof_to_distance_m

csv_path = Path("outputs/tof_matched/multifrequenz_denoised_branchA_sym6.csv")
df = pd.read_csv(csv_path)

methods = {"Path B": "tof_path_b_s", "Path A (BP)": "tof_path_a_s"}


def compute_errors(frame, tof_col, tau_s):
    errs = np.array([
        100.0 * tof_to_distance_m(getattr(r, tof_col) - tau_s, r.mic, r.tx, PCB, SOUND_SPEED_M_S)
        - r.distance_cm
        for r in frame.itertuples(index=False)
    ])
    return errs


for method_name, tof_col in methods.items():
    print(f"\n{'='*100}")
    print(f"{method_name}  —  mean-zero vs median-zero tau fit")
    print(f"{'='*100}")
    print(f"{'Tx':>5} {'Mic':>5}  {'tau_mean_us':>12} {'tau_med_us':>11} {'Δtau_us':>9}  "
          f"{'MAE_mean':>9} {'MAE_med':>8}  {'ΔMAE':>7}  {'MedAE_mean':>11} {'MedAE_med':>10}")
    print("-" * 100)

    for tx_name in sorted(df["tx"].unique()):
        for mic_name in ["Mic1", "Mic2", "Mic3"]:
            subset = df[(df["tx"] == tx_name) & (df["mic"] == mic_name)]

            # Mean-zeroing fit (current behaviour)
            tau_mean = brentq(
                lambda tau: compute_errors(subset, tof_col, tau).mean(),
                -200e-6, 200e-6,
            )

            # Median-zeroing fit (proposed)
            tau_med = brentq(
                lambda tau: np.median(compute_errors(subset, tof_col, tau)),
                -200e-6, 200e-6,
            )

            errs_mean = compute_errors(subset, tof_col, tau_mean)
            errs_med  = compute_errors(subset, tof_col, tau_med)

            mae_mean   = np.abs(errs_mean).mean()
            mae_med    = np.abs(errs_med).mean()
            medae_mean = np.median(np.abs(errs_mean))
            medae_med  = np.median(np.abs(errs_med))
            delta_tau  = (tau_med - tau_mean) * 1e6
            delta_mae  = mae_med - mae_mean  # negative = median fit is better

            print(
                f"{tx_name:>5} {mic_name:>5}  "
                f"{tau_mean*1e6:>12.3f} {tau_med*1e6:>11.3f} {delta_tau:>9.3f}  "
                f"{mae_mean:>9.3f} {mae_med:>8.3f}  {delta_mae:>+7.3f}  "
                f"{medae_mean:>11.3f} {medae_med:>10.3f}"
            )
