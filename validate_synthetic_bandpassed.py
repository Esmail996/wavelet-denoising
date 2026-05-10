"""
Synthetic ground-truth validation with preprocessing-aligned bandpass.

Generates Hanning-windowed ultrasonic bursts at 40 kHz with AWGN at
configurable SNRs, applies the same 40 kHz bandpass used in the pipeline,
runs the adapted (envelope + SWT) wavelet selection, and reports
selection consistency and TOF error.
"""

import numpy as np
import pandas as pd

from preprocess import bandpass
from wavelet_choice import optimal_wavelets, wavespace

FS = 2_000_000.0
F_CARRIER = 40_000.0
N_SAMPLES = 10_000
BURST_LEN_SAMPLES = 800  # ~400 us
N_TRIALS = 100
SNR_LIST_DB = [5, 10, 20, 30]

# Match run_wavelet_choice_multifrequenz.py 40 kHz band config.
BP_METHOD = "iir"
BP_ORDER = 8
BP_BW_HZ = 4_000.0


def make_burst(burst_center: int, snr_db: float, rng: np.random.Generator):
    t = np.arange(N_SAMPLES) / FS
    sig = np.zeros(N_SAMPLES)
    idx = np.arange(
        burst_center - BURST_LEN_SAMPLES // 2,
        burst_center + BURST_LEN_SAMPLES // 2,
    )
    window = np.hanning(len(idx))
    sig[idx] = window * np.sin(2 * np.pi * F_CARRIER * t[idx])

    signal_power = np.mean(sig[idx] ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), N_SAMPLES)
    return sig + noise, burst_center


def estimate_tof_from_raw(x):
    return int(np.argmax(np.abs(x)))


def run():
    rng = np.random.default_rng(42)
    wave_family = wavespace()
    rows = []

    for snr_db in SNR_LIST_DB:
        for trial in range(N_TRIALS):
            burst_center_true = int(rng.integers(2000, N_SAMPLES - 2000))
            x, tof_true = make_burst(burst_center_true, snr_db, rng)

            x_bp = bandpass(
                x,
                fs=FS,
                f0=F_CARRIER,
                bw=BP_BW_HZ,
                method=BP_METHOD,
                order=BP_ORDER,
            )

            results, _ = optimal_wavelets(x_bp, wave_family, nw=5)
            if not results:
                continue

            best = results[0]
            tof_estimated = estimate_tof_from_raw(x_bp)
            tof_error = tof_estimated - tof_true

            rows.append(
                {
                    "snr_db": int(snr_db),
                    "trial": int(trial),
                    "best_wavelet": best["wavelet"],
                    "kappa": int(best["kappa"]),
                    "mu_sc": float(best["mu_sc"]),
                    "mu_sc_hoyer": float(best.get("mu_sc_hoyer", np.nan)),
                    "tof_error_samples": int(tof_error),
                    "tof_error_us": float(tof_error / FS * 1e6),
                    "bp_method": BP_METHOD,
                    "bp_order": BP_ORDER,
                    "bp_bw_hz": BP_BW_HZ,
                    "envelope_used": False,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv("validation_synthetic_bandpassed_results.csv", index=False)

    print("\n=== Selection consistency per SNR (bandpassed) ===")
    for snr in SNR_LIST_DB:
        sub = df[df["snr_db"] == snr]
        if sub.empty:
            print(f"SNR={snr:2d}dB: no rows")
            continue
        top_wavelet = sub["best_wavelet"].mode().iloc[0]
        consistency = (sub["best_wavelet"] == top_wavelet).mean()
        kappa_mode = sub["kappa"].mode().iloc[0]
        print(
            f"SNR={snr:2d}dB: top wavelet={top_wavelet} "
            f"({consistency:.0%} of trials), modal kappa={kappa_mode}"
        )

    print("\n=== TOF error per SNR (bandpassed) ===")
    print(df.groupby("snr_db")["tof_error_us"].describe()[["mean", "std", "50%"]].round(3))


if __name__ == "__main__":
    run()
