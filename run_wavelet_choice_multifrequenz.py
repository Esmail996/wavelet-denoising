import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt

from bandpass import bandpass
from wavelet_choice import optimal_wavelets, wavespace


NAME_RE = re.compile(r"(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad", re.IGNORECASE)

# Fixed per-band filter configs selected from bandpass grid search.
BANDPASS_CFG_BY_CENTER_HZ = {
    40_000: {"method": "iir", "order": 8, "bw_hz": 4_000.0},
    50_000: {"method": "iir", "order": 1, "bw_hz": 7_000.0},
    60_000: {"method": "iir", "order": 1, "bw_hz": 7_000.0},
}


def parse_dist_angle(filename: str) -> tuple[int | None, int | None]:
    match = NAME_RE.search(filename)
    if not match:
        return None, None
    return int(match.group("dist")), int(match.group("ang"))


def iter_pickle_files(data_root: Path):
    for category_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for fp in sorted(category_dir.glob("*.pickle")):
            yield category_dir.name, fp


def extract_mic_trials(obj, mic_col: str):
    if isinstance(obj, pd.DataFrame):
        col_lut = {c.lower(): c for c in obj.columns}
        if mic_col not in col_lut:
            raise ValueError(
                f"Expected column '{mic_col}' in DataFrame. Got columns: {list(obj.columns)}"
            )
        col = col_lut[mic_col]
        return [np.asarray(obj.iloc[i][col], dtype=float) for i in range(len(obj))]

    arr = np.asarray(obj, dtype=object)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Unsupported pickle array shape: {arr.shape}")

    # Dataset convention used in this project: rows are mics [Mic1, Mic2, Mic3].
    # mic2 means row index 1.
    mic_idx = {"mic1": 0, "mic2": 1, "mic3": 2}[mic_col]
    if mic_idx >= arr.shape[0]:
        raise ValueError(f"Array does not contain {mic_col}. Shape: {arr.shape}")

    return [np.asarray(arr[mic_idx, j], dtype=float) for j in range(arr.shape[1])]


def preprocess_signal_like_preprocessing_py(
    x: np.ndarray,
    fs_hz: float,
    bw_hz: float,
    bp_method: str,
    iir_order: int,
    fir_numtaps: int,
    fir_window: str,
    preprocess_mode: str = "bandpass",
):
    x = np.asarray(x, dtype=float)
    dc_removed = x - np.mean(x)
    detrended = detrend(dc_removed)

    nominal_bands = [40_000, 50_000, 60_000]

    if preprocess_mode == "raw":
        # Bypass bandpass: apply the same detrended signal to all three band slots.
        return {nom: detrended.copy() for nom in nominal_bands}

    if preprocess_mode == "bp25_75_sos1":
        # Wideband pass (25-75 kHz) using first-order Butterworth SOS filter.
        low = 25_000.0 / (fs_hz / 2.0)
        high = 75_000.0 / (fs_hz / 2.0)
        sos = butter(1, [low, high], btype="band", output="sos")
        wide = sosfiltfilt(sos, detrended)
        return {nom: wide.copy() for nom in nominal_bands}

    band_signals = {
        nom: bandpass(
            detrended, fs_hz,
            float(nom),
            bw=BANDPASS_CFG_BY_CENTER_HZ[nom]["bw_hz"],
            method=BANDPASS_CFG_BY_CENTER_HZ[nom]["method"],
            order=BANDPASS_CFG_BY_CENTER_HZ[nom]["order"],
            numtaps=fir_numtaps,
            window=fir_window,
        )
        for nom in nominal_bands
    }
    return band_signals


def pick_best_result(results, min_kappa: int):
    """
    Select best wavelet with a minimum decomposition level constraint.
        Preference order:
            1) Highest mu_sc among candidates with kappa >= min_kappa
            2) Otherwise highest mu_sc overall
    """
    eligible = [r for r in results if int(r["kappa"]) >= min_kappa]
    if eligible:
        best = eligible[0]
        selected_kappa = int(best["kappa"])
        return best, selected_kappa, True

    best = results[0]
    selected_kappa = int(best["kappa"])
    return best, selected_kappa, False


def run_all(
    data_root: Path,
    out_csv: Path,
    fs_hz: float = 2_000_000.0,
    bw_hz: float = 5_000.0,
    bp_method: str = "iir",
    iir_order: int = 2,
    fir_numtaps: int = 71,
    fir_window: str = "hamming",
    top_n: int = 5,
    min_kappa: int = 4,
    max_files: int | None = None,
    max_trials: int | None = None,
    preprocess_mode: str = "bandpass",
):
    if preprocess_mode not in ("bandpass", "raw", "bp25_75_sos1"):
        raise ValueError(
            f"Unknown preprocess_mode: {preprocess_mode!r}. "
            "Use 'bandpass', 'raw', or 'bp25_75_sos1'."
        )

    wave_family = wavespace()
    rows = []

    file_counter = 0
    for category, fp in iter_pickle_files(data_root):
        if max_files is not None and file_counter >= max_files:
            break
        file_counter += 1

        dist_cm, angle_deg = parse_dist_angle(fp.name)
        obj = pd.read_pickle(fp)
        mic2_trials = extract_mic_trials(obj, mic_col="mic2")
        if max_trials is not None:
            mic2_trials = mic2_trials[:max_trials]

        for meas_idx, signal in enumerate(mic2_trials):
            band_signals = preprocess_signal_like_preprocessing_py(
                signal,
                fs_hz=fs_hz,
                bw_hz=bw_hz,
                bp_method=bp_method,
                iir_order=iir_order,
                fir_numtaps=fir_numtaps,
                fir_window=fir_window,
                preprocess_mode=preprocess_mode,
            )

            for center_hz, sig_bp in band_signals.items():
                if preprocess_mode == "bandpass":
                    cfg = BANDPASS_CFG_BY_CENTER_HZ[int(center_hz)]
                    cfg_id = f"{cfg['method']}_o{int(cfg['order'])}_bw{int(round(cfg['bw_hz']))}"
                    bp_method_row = str(cfg["method"])
                    iir_order_row = int(cfg["order"])
                    bw_hz_row = float(cfg["bw_hz"])
                elif preprocess_mode == "bp25_75_sos1":
                    cfg_id = "sos_o1_bp25_75k"
                    bp_method_row = "iir_sos"
                    iir_order_row = 1
                    bw_hz_row = 25_000.0
                else:
                    cfg_id = "raw_no_bandpass"
                    bp_method_row = "none"
                    iir_order_row = 0
                    bw_hz_row = 0.0

                # Evaluate all candidate wavelets so min_kappa filtering is exact,
                # not limited by an early top-N truncation.
                results, _ = optimal_wavelets(sig_bp, wave_family, nw=len(wave_family))
                if not results:
                    continue

                best, selected_kappa, min_kappa_satisfied = pick_best_result(results, min_kappa=min_kappa)
                rows.append(
                    {
                        "category": category,
                        "file": fp.name,
                        "distance_cm": dist_cm,
                        "angle_deg": angle_deg,
                        "meas_idx": int(meas_idx),
                        "mic": "Mic2",
                        "band_center_hz": int(center_hz),
                        "best_wavelet": best["wavelet"],
                        "best_kappa_decomposition_level": int(selected_kappa),
                        "min_kappa_target": int(min_kappa),
                        "min_kappa_satisfied": bool(min_kappa_satisfied),
                        "best_effective_level": int(best["effective_level"]),
                        "best_mu_sc": float(best["mu_sc"]),
                        "bp_cfg_id": str(cfg_id),
                        "bp_method": str(bp_method_row),
                        "iir_order": int(iir_order_row),
                        "bw_hz": float(bw_hz_row),
                        "preprocess_mode": str(preprocess_mode),
                    }
                )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run preprocessing (DC removal + detrend + bandpass) and wavelet choice "
            "for all Mic2 trials in the full Multifrequenz dataset."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"Multifrequenz Dataset\Multifrequenz_mean25",
        
        help="Root folder containing Box, Dose, Glas, referenz.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/wavelet_choice/multifrequenz_mic2mean_best_wavelets.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--fs-hz",
        type=float,
        default=2_000_000.0,
        help="Sampling rate in Hz. Keep default to match preprocessing.py.",
    )
    parser.add_argument(
        "--bw-hz",
        type=float,
        default=5_000.0,
        help="Bandpass half-bandwidth in Hz.",
    )
    parser.add_argument(
        "--bp-method",
        type=str,
        choices=["fir", "iir"],
        default="iir",
        help="Bandpass method.",
    )
    parser.add_argument(
        "--iir-order",
        type=int,
        default=5,
        help="IIR filter order.",
    )
    parser.add_argument(
        "--fir-numtaps",
        type=int,
        default=120,
        help="FIR number of taps.",
    )
    parser.add_argument(
        "--fir-window",
        type=str,
        default="hamming",
        help="FIR window type.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top-N wavelets to evaluate internally. CSV stores only the best one.",
    )
    parser.add_argument(
        "--min-kappa",
        type=int,
        default=4,
        help="Minimum decomposition level in output (set 4 or 5 as needed).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file limit for quick test runs.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional per-file trial limit for quick test runs.",
    )
    parser.add_argument(
        "--preprocess-mode",
        type=str,
           choices=["bandpass", "raw", "bp25_75_sos1"],
        default="bandpass",
        help="'bandpass' (default): DC removal + detrend + bandpass per band. "
               "'raw': DC removal + detrend only, same signal fed to all band slots. "
               "'bp25_75_sos1': first-order SOS bandpass from 25 to 75 kHz, then reused across band slots.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    df = run_all(
        data_root=Path(args.data_root),
        out_csv=Path(args.out_csv),
        fs_hz=float(args.fs_hz),
        bw_hz=float(args.bw_hz),
        bp_method=str(args.bp_method),
        iir_order=int(args.iir_order),
        fir_numtaps=int(args.fir_numtaps),
        fir_window=str(args.fir_window),
        top_n=int(args.top_n),
        min_kappa=int(args.min_kappa),
        max_files=args.max_files,
        max_trials=args.max_trials,
        preprocess_mode=str(args.preprocess_mode),
    )
    print(f"Rows written: {len(df)}")
    print(f"CSV: {Path(args.out_csv)}")
