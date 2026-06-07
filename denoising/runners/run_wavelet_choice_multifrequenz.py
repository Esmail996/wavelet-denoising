import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from denoising.preprocess import bandpass
from denoising.wavelet_choice import optimal_wavelets, wavespace


NAME_RE = re.compile(r"(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad", re.IGNORECASE)

# Fixed per-band filter configs selected from bandpass grid search.
BANDPASS_CFG_BY_CENTER_HZ = {
    40_000: {"method": "iir", "order": 2, "bw_hz": 6_000.0},
    50_000: {"method": "iir", "order": 2, "bw_hz": 6_000.0},
    60_000: {"method": "iir", "order": 2, "bw_hz": 6_000.0},
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
    fir_window,
    fir_beta: float = 6.0,
    preprocess_mode: str = "bandpass",
    remove_dc: bool = True,
):
    x = np.asarray(x, dtype=float)
    if remove_dc:
        x_for_bandpass = detrend(x - np.mean(x))
    else:
        x_for_bandpass = x.copy()

    nominal_bands = [40_000, 50_000, 60_000]

    if preprocess_mode == "raw":
        # Bypass bandpass.
        return {nom: x_for_bandpass.copy() for nom in nominal_bands}

    if preprocess_mode == "bp25_75":
        # Wideband frontend: 25-75 kHz, method selected by --bp-method.
        x_wb = np.asarray(x_for_bandpass, dtype=np.float64)
        fir_window_cfg = ("kaiser", float(fir_beta)) if str(fir_window).lower() == "kaiser" else fir_window
        wide = bandpass(
            x_wb,
            fs_hz,
            50_000.0,
            bw=25_000.0,
            method=str(bp_method),
            order=iir_order,
            numtaps=fir_numtaps,
            window=fir_window_cfg,
        )
        return {nom: wide.copy() for nom in nominal_bands}

    band_signals = {
        nom: bandpass(
            x_for_bandpass, fs_hz,
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
    fir_beta: float = 6.0,
    top_n: int = 5,
    min_kappa: int = 4,
    max_files: int | None = None,
    max_trials: int | None = None,
    preprocess_mode: str = "bandpass",
    mic: str = "mic1",
    gate_start_sample: int = 0,
    remove_dc: bool = True,
):
    if preprocess_mode not in ("bandpass", "raw", "bp25_75"):
        raise ValueError(
            f"Unknown preprocess_mode: {preprocess_mode!r}. "
            "Use 'bandpass', 'raw', or 'bp25_75'."
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
        mic2_trials = extract_mic_trials(obj, mic_col=mic)
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
                fir_beta=fir_beta,
                preprocess_mode=preprocess_mode,
                remove_dc=remove_dc,
            )

            if gate_start_sample > 0:
                n_gate = int(gate_start_sample)
                for key in list(band_signals.keys()):
                    sig = np.asarray(band_signals[key], dtype=float).copy()
                    sig[:n_gate] = 0.0
                    band_signals[key] = sig

            # For wideband/raw all band slots are identical — only process once.
            if preprocess_mode != "bandpass":
                first_key = next(iter(band_signals))
                band_signals = {first_key: band_signals[first_key]}

            for center_hz, sig_bp in band_signals.items():
                if preprocess_mode == "bandpass":
                    cfg = BANDPASS_CFG_BY_CENTER_HZ[int(center_hz)]
                    cfg_id = f"{cfg['method']}_o{int(cfg['order'])}_bw{int(round(cfg['bw_hz']))}"
                    bp_method_row = str(cfg["method"])
                    iir_order_row = int(cfg["order"])
                    bw_hz_row = float(cfg["bw_hz"])
                    band_center_hz_row = int(center_hz)
                elif preprocess_mode == "bp25_75":
                    cfg_id = f"{bp_method}_o{int(iir_order)}_bp25_75k"
                    bp_method_row = str(bp_method)
                    iir_order_row = int(iir_order)
                    bw_hz_row = 25_000.0
                    band_center_hz_row = None
                else:
                    cfg_id = "raw_no_bandpass"
                    bp_method_row = "none"
                    iir_order_row = 0
                    bw_hz_row = 0.0
                    band_center_hz_row = None

                # Evaluate all candidate wavelets so min_kappa filtering is exact,
                # not limited by an early top-N truncation.
                results, _ = optimal_wavelets(sig_bp, wave_family, nw=len(wave_family))
                if not results:
                    continue

                best, selected_kappa, min_kappa_satisfied = pick_best_result(results, min_kappa=min_kappa)
                top_candidates = results[: max(1, int(top_n))]
                top_fields = {}
                for rank in range(1, int(top_n) + 1):
                    if rank <= len(top_candidates):
                        candidate = top_candidates[rank - 1]
                        top_fields[f"top{rank}_wavelet"] = str(candidate["wavelet"])
                        top_fields[f"top{rank}_mu_sc"] = float(candidate["mu_sc"])
                        top_fields[f"top{rank}_kappa"] = int(candidate["kappa"])
                    else:
                        top_fields[f"top{rank}_wavelet"] = ""
                        top_fields[f"top{rank}_mu_sc"] = np.nan
                        top_fields[f"top{rank}_kappa"] = np.nan

                rows.append(
                    {
                        "category": category,
                        "file": fp.name,
                        "distance_cm": dist_cm,
                        "angle_deg": angle_deg,
                        "meas_idx": int(meas_idx),
                        "mic": mic.capitalize(),
                        "band_center_hz": band_center_hz_row,
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
                        "dc_removed": bool(remove_dc),
                        **top_fields,
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
        default=2,
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
        "--fir-beta",
        type=float,
        default=6.0,
        help="Beta for Kaiser FIR window (used when --fir-window kaiser).",
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
        "--mic",
        type=str,
        choices=["mic1", "mic2", "mic3"],
        default="mic1",
        help="Which microphone channel to process (mic1, mic2, or mic3).",
    )
    parser.add_argument(
        "--preprocess-mode",
        type=str,
                choices=["bandpass", "raw", "bp25_75"],
        default="bandpass",
        help="'bandpass' (default): DC removal + detrend + bandpass per band. "
               "'raw': DC removal + detrend only, same signal fed to all band slots. "
                             "'bp25_75': Butterworth SOS frontend from 25 to 75 kHz with mean removal and --iir-order.",
    )
    parser.add_argument(
        "--gate-start-sample",
        type=int,
        default=0,
        help="Set first N samples to zero after preprocessing (e.g. 2000).",
    )
    parser.add_argument(
        "--no-dc-remove",
        action="store_true",
        help="Disable DC removal before preprocessing.",
    )
    return parser


def main() -> None:
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
        fir_beta=float(args.fir_beta),
        top_n=int(args.top_n),
        min_kappa=int(args.min_kappa),
        max_files=args.max_files,
        max_trials=args.max_trials,
        preprocess_mode=str(args.preprocess_mode),
        mic=str(args.mic),
        gate_start_sample=int(args.gate_start_sample),
        remove_dc=not bool(args.no_dc_remove),
    )
    print(f"Rows written: {len(df)}")
    print(f"CSV: {Path(args.out_csv)}")


if __name__ == "__main__":
    main()

# Run command used for full Multifrequenz dataset (wideband 25-75 kHz, IIR SOS order 6):
# python.exe -m denoising.runners.run_wavelet_choice_multifrequenz --data-root "Multifrequenz Dataset\Multifrequenz" --preprocess-mode bp25_75 --bp-method iir --iir-order 2 --mic mic2 --out-csv "outputs/wavelet_choice/multifrequenz_full_bp25_75_iir_o2_mic2.csv"

#python.exe -m denoising.runners.run_wavelet_choice_multifrequenz --data-root "Multifrequenz Dataset\Multifrequenz" --preprocess-mode raw --mic mic2 --gate-start-sample 0 --out-csv "outputs/wavelet_choice/multifrequenz_raw_nogate_mic2.csv"