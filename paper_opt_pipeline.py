"""
paper_opt_pipeline.py

One-file pipeline: choose wavelet + decomposition level using the Ni et al. (Scientific Reports 2024)
multi-index fusion OPT approach, but adapted to your ultrasonic setup with:
- reference recording (noise-only) + measurement recording (signal+noise)
- band-pass around 40/50/60 kHz
- wavelet denoising (fixed-threshold + hard thresholding by default, per paper)
- candidate-coupled SNR scoring: for each (wavelet, level), apply the SAME
    denoiser to measurement and reference and score:
        20*log10( RMS(denoised_measurement_eval_segment) / RMS(denoised_reference_eval_segment_aggregate) )
  where the evaluation segment is selected by SNR_SCOPE ("roi" or "whole_signal")
- OPT multi-index fusion using candidate-coupled evaluation-segment RMS-ratio SNR
    and full-signal smoothness r to pick wavelet and level

USAGE
-----
1) Edit REF_DIR, SIG_DATA_DIR and SIG_CLASSES at the top of this file.
2) Run:
   python paper_opt_pipeline.py

Outputs (per band: 40k / 50k / 60k)
-------
- per_row_wavelet_level.csv          (file traceability + distance_cm, roi_start/end,
                                     snr_db_peak, snr_db_eval_rms_ratio, RMS-QC diagnostics per row)
- problematic_measurements.csv       (rows where evaluation-segment signal RMS is not above aggregated reference RMS)
- summary_by_band_wavelet_level.csv  (primary summary from all rows)
- summary_by_band_wavelet_level_qc_filtered.csv (QC-only summary where evaluation-segment signal RMS > aggregated reference RMS)
- wavelet_rank_<band>.csv            (step-1 wavelet ranking across levels)
- chosen_wavelet_opt_curve_<band>.csv / .png (step-2 OPT over levels for fixed chosen wavelet)
- chosen_wavelet_opt_change_rate_<band>.png
- best_wavelet_level_per_band.csv    (consolidated final results with level_inflection and level_maxopt)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend

from roi_windows import ROI_WINDOWS, parse_distance_cm


# =============================================================================
# USER INPUTS
# =============================================================================
_HERE = Path(__file__).resolve().parent

REF_DIR = _HERE / "Multifrequenz Dataset/Multifrequenz_mean25/referenz"
SIG_DATA_DIR = _HERE / "Multifrequenz Dataset/Multifrequenz_mean25"  # contains subdirs Box, Dose, Glas
SIG_CLASSES  = ["Box", "Dose", "Glas"]  # subdirectories to include

MIC_COL = "Mic2"           # "Mic1" / "Mic2" / "Mic3"
MAX_ROWS_REF = None        # e.g. 30 for quick run; None = all
MAX_ROWS_SIG = None        # rows per pickle file; None = all

# Keep default output path for the existing bandpass workflow.
PREPROCESS_MODE = "bp50k_bw25k"  # supported: "bandpass", "raw", "bp50k_bw25k"
if PREPROCESS_MODE == "bandpass":
    OUT_DIR = _HERE / "outputs_paper_opt"
elif PREPROCESS_MODE == "raw":
    OUT_DIR = _HERE / "outputs_paper_opt_raw"
elif PREPROCESS_MODE == "bp50k_bw25k":
    OUT_DIR = _HERE / "outputs_paper_opt_bp50k_bw25k"
else:
    raise ValueError(f"Unknown PREPROCESS_MODE: {PREPROCESS_MODE}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stable reference denominator settings for candidate-coupled scoring
REF_SUBSET_SIZE = 16
REF_SUBSET_AGG = "median"   # supported: "median", "mean"
REF_RMS_FLOOR_FRAC = 0.10   # floor denoised reference RMS by 10% of raw reference RMS aggregate


# =============================================================================
# DEBUG CONTROLS
# Set DEBUG_SINGLE_FILE_ONLY=True to run one file and print detailed SNR terms.
# =============================================================================
DEBUG_SINGLE_FILE_ONLY = False
DEBUG_FILE_RELATIVE = "Box/25cm_-10Grad.pickle"  # relative to SIG_DATA_DIR; None => first discovered file
DEBUG_MAX_ROWS_SIG = 1      # limit trials for debug run
DEBUG_LOG_LIMIT = 60        # max number of per-evaluation debug lines to print
DEBUG_WAVELET_FILTER = None # e.g. "bior1.1" to reduce output
DEBUG_LEVEL_FILTER = None   # e.g. 2 to reduce output
DEBUG_BAND_FILTER = None    # e.g. ["40k"] to reduce output


# =============================================================================
# SIGNAL / PREPROCESS PARAMETERS
# =============================================================================
FS_HZ   = 2_000_000.0
F0S = (40_000.0, 50_000.0, 60_000.0)

# Per-band IIR configs selected from grid search.
# cfg ids: 40k -> iir_o8_bw4000, 50k -> iir_o1_bw7000, 60k -> iir_o1_bw7000
BANDPASS_CFG_BY_BAND = {
    "40k": {"f0": F0S[0], "order": 5, "bw": 5000.0},
    "50k": {"f0": F0S[1], "order": 5, "bw": 5000.0},
    "60k": {"f0": F0S[2], "order": 5, "bw": 5000.0},
}

# Wideband mode around 50 kHz with 25 kHz half-bandwidth (25-75 kHz).
BANDPASS_CFG_WIDEBAND_50K = {"f0": 50_000.0, "order": 1, "bw": 25_000.0}

# Wavelet grid — matches wavespace() in wavelet_choice.py
WAVELETS = [
    # Biorthogonal
    "bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6",
    # Coiflet
    "coif1", "coif2", "coif3", "coif4", "coif5",
    # Daubechies
    "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11",
    # Reverse biorthogonal
    "rbio1.3", "rbio1.5", "rbio2.2", "rbio2.4", "rbio2.6", "rbio2.8",
    # Symlet
    "sym2", "sym3", "sym4", "sym5", "sym6", "sym7",
    # DMEY
    "dmey",
]
LEVELS_TO_TEST = [3, 4, 5, 6, 7, 8]


# =============================================================================
# Paper-like denoising choices (can change)
# =============================================================================
THRESH_RULE = "fixed"   # "fixed" (paper eq. 12) or "universal" (sigma*sqrt(2logN))
THR_MODE    = "hard"    # "hard" (paper) or "soft" (classic)
INVERT_SMOOTHNESS_FOR_OPT = False  # recommended: smaller smoothness should improve OPT
# SNR evaluation scope:
#   "roi"          -> use only the echo ROI
#   "whole_signal" -> use the full bandpassed signal
SNR_SCOPE = "whole_signal"


# =============================================================================
# Utilities
# =============================================================================

def bandpass(x: np.ndarray, fs: float, f0: float, bw: float, order: int) -> np.ndarray:
    """Butterworth bandpass centered at f0 with +/- bw."""
    x = np.asarray(x, float)
    low = (f0 - bw) / (fs / 2.0)
    high = (f0 + bw) / (fs / 2.0)
    low = max(low, 1e-6)
    high = min(high, 0.999999)
    if low >= high:
        raise ValueError(f"Invalid band: low={low}, high={high} for fs={fs}, f0={f0}, bw={bw}")
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def robust_sigma_from_time_domain(x: np.ndarray) -> float:
    """Robust noise sigma estimate from time-domain via MAD."""
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad / 0.6745) if mad > 0 else float(np.std(x))


def fixed_threshold_paper(sigma: float, N: int) -> float:
    """Paper eq. (12): th = sigma * (0.3936 + 0.10829*log2(N)), for N>32 else 0."""
    if N <= 32:
        return 0.0
    return float(sigma * (0.3936 + 0.10829 * math.log2(N)))


def universal_threshold(sigma: float, N: int) -> float:
    """Classic universal threshold: sigma*sqrt(2*ln(N))."""
    if N <= 1:
        return 0.0
    return float(sigma * math.sqrt(2.0 * math.log(N)))


def wavelet_denoise(x: np.ndarray, wavelet: str, level: int,
                    sigma_noise: float, thr_rule: str = "fixed",
                    thr_mode: str = "hard") -> np.ndarray:
    """
    Wavelet threshold denoising with an externally supplied noise sigma (from reference).
    Threshold is applied to ALL detail coeffs at all scales.
    """
    x = np.asarray(x, float)
    if x.size < 8:
        return x.copy()

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode="symmetric")
    N = x.size

    if thr_rule == "fixed":
        thr = fixed_threshold_paper(sigma_noise, N)
    elif thr_rule == "universal":
        thr = universal_threshold(sigma_noise, N)
    else:
        raise ValueError(f"Unknown thr_rule: {thr_rule}")

    # threshold only details, keep approximation
    cA = coeffs[0]
    details = [pywt.threshold(cD, value=thr, mode=thr_mode) for cD in coeffs[1:]]
    y = pywt.waverec([cA] + details, wavelet=wavelet, mode="symmetric")
    # match length
    return y[:N]


def snr_db_roi_peak(roi: np.ndarray, ref_peak: float, eps: float = 1e-12) -> float:
    """Peak amplitude ratio in dB: 20*log10(max_abs(roi) / ref_peak)."""
    sig_peak = float(np.max(np.abs(roi))) if roi.size else 0.0
    return float(20.0 * math.log10((sig_peak + eps) / (ref_peak + eps)))


def segment_rms(x: np.ndarray) -> float:
    """RMS feature for the selected evaluation segment."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def get_snr_segment(x: np.ndarray, roi_start: int, roi_end: int) -> np.ndarray:
    """Return the segment used for SNR evaluation."""
    if SNR_SCOPE == "roi":
        return x[roi_start:roi_end]
    if SNR_SCOPE == "whole_signal":
        return x
    raise ValueError(f"Unknown SNR_SCOPE: {SNR_SCOPE}")


def snr_db_ratio(signal_feature: float, noise_feature: float, eps: float = 1e-12) -> float:
    """Generic dB ratio helper for amplitude-like features (RMS ratio)."""
    return float(20.0 * math.log10((signal_feature + eps) / (noise_feature + eps)))


def deterministic_subset_indices(total_count: int, subset_size: int) -> List[int]:
    """Return exactly k deterministic, evenly spaced unique indices over [0, total_count)."""
    if total_count <= 0:
        return []

    k = min(max(1, int(subset_size)), int(total_count))
    if k == total_count:
        return list(range(total_count))

    # Midpoint sampling across k equal bins guarantees uniqueness when k <= total_count
    idx = ((np.arange(k) + 0.5) * total_count / k).astype(int)
    idx = np.clip(idx, 0, total_count - 1)

    # Safety check
    idx = np.unique(idx)
    if len(idx) != k:
        # deterministic fallback
        idx = np.arange(total_count)
        idx = idx[np.linspace(0, total_count - 1, num=k, dtype=int)]
        idx = np.unique(idx)

    return [int(i) for i in idx]


def smoothness_r_paper(den: np.ndarray, orig: np.ndarray, eps: float = 1e-12) -> float:
    """
    Paper description: ratio of squared differential values between denoised and original.
    Smaller is better. (We implement: sum(diff(den)^2)/sum(diff(orig)^2))
    """
    den = np.asarray(den, float)
    orig = np.asarray(orig, float)
    if den.size < 3 or orig.size < 3:
        return float("nan")
    d_den = np.diff(den)
    d_org = np.diff(orig)
    num = float(np.sum(d_den * d_den))
    deno = float(np.sum(d_org * d_org)) + eps
    return num / deno


def normalize_01(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, float)
    mn = np.nanmin(v)
    mx = np.nanmax(v)
    return (v - mn) / (mx - mn + eps)


def cv(v: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(v, float)
    return float(np.nanstd(v) / (np.nanmean(v) + eps))


def opt_curve_from_snr_r(
    levels: List[int],
    snr_vals: np.ndarray,
    r_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    OPT over levels using normalized SNR and normalized smoothness.

    If INVERT_SMOOTHNESS_FOR_OPT is True, smaller smoothness values are treated
    as better by inverting the normalized smoothness term.
    """
    levels_arr = np.asarray(levels, dtype=int)
    snr_vals = np.asarray(snr_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)

    Psnr = normalize_01(snr_vals)

    Pr_raw = normalize_01(r_vals)
    Pr = 1.0 - Pr_raw if INVERT_SMOOTHNESS_FOR_OPT else Pr_raw

    CVsnr = cv(Psnr)
    CVr = cv(Pr)

    WPr = CVr / (CVr + CVsnr + 1e-12)
    Wsnr = CVsnr / (CVr + CVsnr + 1e-12)

    OPT = Wsnr * Psnr + WPr * Pr
    return levels_arr, OPT, Wsnr, WPr


def pick_level_by_opt_inflection(levels: np.ndarray, OPT: np.ndarray) -> int:
    """
    Paper chooses the level where OPT change-rate has an inflection.
    Operationalization:
      d[k]  = |OPT[k+1]-OPT[k]|
      dd[k] = d[k+1]-d[k]
    Pick level at index (argmin dd) + 1  (matches: drop from big change to small change).
    """
    if OPT.size < 3:
        return int(levels[len(levels)//2])
    d = np.abs(np.diff(OPT))
    dd = np.diff(d)
    if dd.size == 0:
        return int(levels[np.argmax(d) + 1])
    k = int(np.argmin(dd))  # most negative drop
    return int(levels[k + 1])


def compute_bands(df: pd.DataFrame, row_idx: int, mic_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DC removal -> linear detrend -> optional bandpass at 40/50/60 kHz."""
    x = np.asarray(df.iloc[row_idx][mic_col], float)
    x = x - np.mean(x)
    x = detrend(x)

    if PREPROCESS_MODE == "raw":
        # Keep three-band API shape while bypassing bandpass filtering.
        return x.copy(), x.copy(), x.copy()

    if PREPROCESS_MODE == "bp50k_bw25k":
        wide = bandpass(
            x,
            fs=FS_HZ,
            f0=BANDPASS_CFG_WIDEBAND_50K["f0"],
            bw=BANDPASS_CFG_WIDEBAND_50K["bw"],
            order=BANDPASS_CFG_WIDEBAND_50K["order"],
        )
        # Keep three-band API shape while using the same wideband signal.
        return wide.copy(), wide.copy(), wide.copy()

    cfg40 = BANDPASS_CFG_BY_BAND["40k"]
    cfg50 = BANDPASS_CFG_BY_BAND["50k"]
    cfg60 = BANDPASS_CFG_BY_BAND["60k"]

    s40 = bandpass(x, fs=FS_HZ, f0=cfg40["f0"], bw=cfg40["bw"], order=cfg40["order"])
    s50 = bandpass(x, fs=FS_HZ, f0=cfg50["f0"], bw=cfg50["bw"], order=cfg50["order"])
    s60 = bandpass(x, fs=FS_HZ, f0=cfg60["f0"], bw=cfg60["bw"], order=cfg60["order"])

    return s40, s50, s60


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"[PREPROCESS] mode={PREPROCESS_MODE}")

    ref_dir = Path(REF_DIR)
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir.resolve()}")

    ref_files = sorted(ref_dir.glob("*.pickle"))
    if not ref_files:
        raise FileNotFoundError(f"No reference pickle files found in: {ref_dir.resolve()}")

    # Collect all signal pickle files from the requested class subdirectories
    sig_root = Path(SIG_DATA_DIR)
    sig_files: List[Path] = []
    for cls in SIG_CLASSES:
        cls_dir = sig_root / cls
        if not cls_dir.is_dir():
            print(f"WARNING: class directory not found, skipping: {cls_dir}")
            continue
        sig_files.extend(sorted(cls_dir.glob("*.pickle")))
    if not sig_files:
        raise FileNotFoundError(f"No .pickle files found under {sig_root} for classes {SIG_CLASSES}")

    if DEBUG_SINGLE_FILE_ONLY:
        if DEBUG_FILE_RELATIVE:
            target_rel = DEBUG_FILE_RELATIVE.replace("\\", "/")
            selected = [p for p in sig_files if p.relative_to(sig_root).as_posix() == target_rel]
            if not selected:
                raise FileNotFoundError(
                    f"DEBUG_FILE_RELATIVE not found under {sig_root}: {DEBUG_FILE_RELATIVE}"
                )
            sig_files = selected
        else:
            sig_files = [sig_files[0]]
        print(f"DEBUG: single-file mode enabled -> {sig_files[0].relative_to(sig_root)}")

    print(f"Found {len(sig_files)} signal pickle files across {SIG_CLASSES}")

    print(f"Loading {len(ref_files)} reference pickle files from: {ref_dir}")
    ref_frames = []
    for ref_path in ref_files:
        ref_frames.append(pd.read_pickle(ref_path))
    df_ref = pd.concat(ref_frames, ignore_index=True)
    if MIC_COL not in df_ref.columns:
        raise KeyError(f"MIC_COL='{MIC_COL}' not in reference columns: {list(df_ref.columns)}")

    n_ref = len(df_ref) if MAX_ROWS_REF is None else min(len(df_ref), int(MAX_ROWS_REF))

    # -------------------------------------------------------------------------
    # 1) Build reference band cache + sigma_noise
    #    - sigma_noise per band from reference for thresholding
    #    - cached reference band signals reused in candidate-coupled scoring
    # -------------------------------------------------------------------------
    print(f"Building reference noise stats using {n_ref} rows...")
    noise_sig = {"40k": [], "50k": [], "60k": []}
    ref_band_cache: Dict[str, List[np.ndarray]] = {"40k": [], "50k": [], "60k": []}

    for i in range(n_ref):
        r40, r50, r60 = compute_bands(df_ref, i, MIC_COL)
        for bname, r in zip(["40k", "50k", "60k"], [r40, r50, r60]):
            if not r.size:
                continue
            noise_sig[bname].append(robust_sigma_from_time_domain(r))
            ref_band_cache[bname].append(r)

    sigma_noise = {k: float(np.median(v)) for k, v in noise_sig.items()}

    print("Reference sigma_noise:", sigma_noise)
    print(f"Reference band cache built with {n_ref} rows per band (before any skips).")

    ref_subset_idx_by_band: Dict[str, List[int]] = {}
    for band in ["40k", "50k", "60k"]:
        n_band = len(ref_band_cache[band])
        subset_idx = deterministic_subset_indices(n_band, REF_SUBSET_SIZE)
        ref_subset_idx_by_band[band] = subset_idx
        print(
            f"Reference subset [{band}]: {len(subset_idx)} of {n_band} rows "
            f"(agg={REF_SUBSET_AGG})"
        )

    subset_rows = []
    for band, idxs in ref_subset_idx_by_band.items():
        for idx in idxs:
            subset_rows.append({
                "band": band,
                "ref_row_idx": idx,
                "ref_subset_size_requested": REF_SUBSET_SIZE,
                "ref_subset_agg": REF_SUBSET_AGG,
            })

    pd.DataFrame(subset_rows).to_csv(OUT_DIR / "reference_subset_indices.csv", index=False)
    print("Saved:", OUT_DIR / "reference_subset_indices.csv")

    run_config = {
        "REF_DIR": str(REF_DIR),
        "SIG_DATA_DIR": str(SIG_DATA_DIR),
        "SIG_CLASSES": SIG_CLASSES,
        "MIC_COL": MIC_COL,
        "MAX_ROWS_REF": MAX_ROWS_REF,
        "MAX_ROWS_SIG": MAX_ROWS_SIG,
        "FS_HZ": FS_HZ,
        "PREPROCESS_MODE": PREPROCESS_MODE,
        "BANDPASS_CFG_BY_BAND": BANDPASS_CFG_BY_BAND,
        "BANDPASS_CFG_WIDEBAND_50K": BANDPASS_CFG_WIDEBAND_50K,
        "WAVELETS": WAVELETS,
        "LEVELS_TO_TEST": LEVELS_TO_TEST,
        "THRESH_RULE": THRESH_RULE,
        "THR_MODE": THR_MODE,
        "SNR_SCOPE": SNR_SCOPE,
        "REF_SUBSET_SIZE": REF_SUBSET_SIZE,
        "REF_SUBSET_AGG": REF_SUBSET_AGG,
        "REF_RMS_FLOOR_FRAC": REF_RMS_FLOOR_FRAC,
        "INVERT_SMOOTHNESS_FOR_OPT": INVERT_SMOOTHNESS_FOR_OPT,
    }

    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("Saved:", OUT_DIR / "run_config.json")

    # -------------------------------------------------------------------------
    # 2) Evaluate metrics for each (band, wavelet, level) across all signal files
    # -------------------------------------------------------------------------
    rows = []
    debug_logs_emitted = 0
    ref_feature_cache: Dict[Tuple[str, str, int, int, int], Tuple[float, float, float]] = {}
    for file_idx, sig_path in enumerate(sig_files):
        print(f"[{file_idx+1}/{len(sig_files)}] {sig_path.relative_to(sig_root)}")
        try:
            df_sig = pd.read_pickle(sig_path)
        except Exception as e:
            print(f"  WARNING: could not load {sig_path.name}: {e}")
            continue
        if MIC_COL not in df_sig.columns:
            print(f"  WARNING: MIC_COL='{MIC_COL}' not found in {sig_path.name}, skipping.")
            continue

        n_sig = len(df_sig) if MAX_ROWS_SIG is None else min(len(df_sig), int(MAX_ROWS_SIG))
        if DEBUG_SINGLE_FILE_ONLY:
            n_sig = min(n_sig, int(DEBUG_MAX_ROWS_SIG))
        source_label = f"{sig_path.parent.name}/{sig_path.stem}"

        # Parse distance so we can look up the correct ROI window
        dist_cm = parse_distance_cm(sig_path.name)
        if dist_cm is None:
            print(f"  WARNING: cannot parse distance from '{sig_path.name}', skipping file.")
            continue
        if dist_cm not in ROI_WINDOWS:
            print(f"  WARNING: no ROI window defined for distance {dist_cm}cm (file={sig_path.name}), skipping file.")
            continue
        roi_start, roi_end = ROI_WINDOWS[dist_cm]

        for row_idx in range(n_sig):
            s40, s50, s60 = compute_bands(df_sig, row_idx, MIC_COL)
            bands = {"40k": s40, "50k": s50, "60k": s60}

            for bname, orig in bands.items():
                if DEBUG_BAND_FILTER and bname not in DEBUG_BAND_FILTER:
                    continue
                ref_subset_idx = ref_subset_idx_by_band.get(bname, [])
                if not ref_subset_idx:
                    continue
                for w in WAVELETS:
                    if DEBUG_WAVELET_FILTER and w != DEBUG_WAVELET_FILTER:
                        continue
                    try:
                        wav = pywt.Wavelet(w)
                    except Exception:
                        continue
                    maxL = pywt.dwt_max_level(len(orig), wav.dec_len)
                    levels = [L for L in LEVELS_TO_TEST if 1 <= L <= maxL]
                    if DEBUG_LEVEL_FILTER is not None:
                        levels = [L for L in levels if L == DEBUG_LEVEL_FILTER]
                    if not levels:
                        continue

                    for L in levels:
                        den_sig = wavelet_denoise(orig, wavelet=w, level=L,
                                                  sigma_noise=sigma_noise[bname],
                                                  thr_rule=THRESH_RULE,
                                                  thr_mode=THR_MODE)

                        roi_sig = den_sig[roi_start:roi_end]          # keep ROI for peak diagnostic
                        snr_sig = get_snr_segment(den_sig, roi_start, roi_end)

                        if SNR_SCOPE == "whole_signal":
                            ref_cache_key = (bname, w, int(L), "whole_signal")
                        else:
                            ref_cache_key = (bname, w, int(L), int(roi_start), int(roi_end))
                        cached = ref_feature_cache.get(ref_cache_key)
                        if cached is None:
                            ref_rms_vals = []
                            ref_peak_vals = []
                            ref_rms_raw_vals = []

                            for idx_ref in ref_subset_idx:
                                ref_orig = ref_band_cache[bname][idx_ref]

                                raw_snr_ref = get_snr_segment(ref_orig, roi_start, roi_end)
                                ref_rms_raw_vals.append(segment_rms(raw_snr_ref))

                                den_ref = wavelet_denoise(ref_orig, wavelet=w, level=L,
                                                          sigma_noise=sigma_noise[bname],
                                                          thr_rule=THRESH_RULE,
                                                          thr_mode=THR_MODE)
                                roi_ref = den_ref[roi_start:roi_end]          # keep ROI for peak diagnostic
                                snr_ref = get_snr_segment(den_ref, roi_start, roi_end)

                                ref_rms_vals.append(segment_rms(snr_ref))
                                ref_peak_vals.append(float(np.max(np.abs(roi_ref))) if roi_ref.size else 0.0)

                            if REF_SUBSET_AGG == "median":
                                ref_rms_agg = float(np.median(ref_rms_vals)) if ref_rms_vals else 0.0
                                ref_peak_agg = float(np.median(ref_peak_vals)) if ref_peak_vals else 0.0
                                ref_rms_raw_agg = float(np.median(ref_rms_raw_vals)) if ref_rms_raw_vals else 0.0
                            elif REF_SUBSET_AGG == "mean":
                                ref_rms_agg = float(np.mean(ref_rms_vals)) if ref_rms_vals else 0.0
                                ref_peak_agg = float(np.mean(ref_peak_vals)) if ref_peak_vals else 0.0
                                ref_rms_raw_agg = float(np.mean(ref_rms_raw_vals)) if ref_rms_raw_vals else 0.0
                            else:
                                raise ValueError(f"Unknown REF_SUBSET_AGG: {REF_SUBSET_AGG}")

                            cached = (ref_rms_agg, ref_peak_agg, ref_rms_raw_agg)
                            ref_feature_cache[ref_cache_key] = cached

                        ref_rms_agg, ref_peak_agg, ref_rms_raw_agg = cached
                        ref_rms_eff = max(ref_rms_agg, REF_RMS_FLOOR_FRAC * ref_rms_raw_agg, 1e-12)
                        ref_floor_active = ref_rms_eff > ref_rms_agg + 1e-15

                        sig_peak = float(np.max(np.abs(roi_sig))) if roi_sig.size else 0.0
                        sig_rms = segment_rms(snr_sig)
                        peak_not_above_reference_diag = sig_peak <= ref_peak_agg
                        rms_not_above_reference = sig_rms <= ref_rms_eff

                        snr_peak = snr_db_roi_peak(roi_sig, ref_peak_agg)
                        # still ROI-based diagnostic
                        snr_eval_rms_ratio = snr_db_ratio(sig_rms, ref_rms_eff)
                        # Smoothness stays on the full bandpassed signal by design
                        r = smoothness_r_paper(den_sig, orig)

                        if DEBUG_SINGLE_FILE_ONLY and debug_logs_emitted < DEBUG_LOG_LIMIT:
                            print(
                                "DEBUG_SNR "
                                f"file={source_label} row={row_idx} band={bname} wavelet={w} L={L} "
                                f"dist_cm={dist_cm} roi=[{roi_start}:{roi_end}] "
                                f"roi_len_sig={roi_sig.size} ref_subset_n={len(ref_subset_idx)} "
                                f"ref_peak_agg={ref_peak_agg:.6e} sig_peak={sig_peak:.6e} snr_peak_db={snr_peak:.6f} "
                                f"ref_rms_agg={ref_rms_agg:.6e} ref_rms_raw_agg={ref_rms_raw_agg:.6e} "
                                f"ref_rms_eff={ref_rms_eff:.6e} sig_rms={sig_rms:.6e} "
                                f"rms_not_above_reference={rms_not_above_reference} "
                                f"snr_eval_rms_ratio_db={snr_eval_rms_ratio:.6f} r={r:.6f}"
                            )
                            debug_logs_emitted += 1

                        rows.append({
                            "source_file": source_label,
                            "source_filename": sig_path.name,
                            "row_idx":     row_idx,
                            "band":        bname,
                            "wavelet":     w,
                            "level":       L,
                            "distance_cm": dist_cm,
                            "roi_start":   roi_start,
                            "roi_end":     roi_end,
                            "snr_db_peak": snr_peak,
                            "snr_db_eval_rms_ratio": snr_eval_rms_ratio,
                            "ref_subset_n": len(ref_subset_idx),
                            "ref_rms_agg": ref_rms_agg,
                            "ref_rms_raw_agg": ref_rms_raw_agg,
                            "ref_rms_eff": ref_rms_eff,
                            "ref_peak_agg": ref_peak_agg,
                            "rms_not_above_reference": rms_not_above_reference,
                            "peak_not_above_reference_diag": peak_not_above_reference_diag,
                            "ref_floor_active": ref_floor_active,
                            "r":           r,
                        })

    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(OUT_DIR / "per_row_wavelet_level.csv", index=False)

    problematic_cols = [
        "source_file", "source_filename", "row_idx", "band", "wavelet", "level",
        "distance_cm", "roi_start", "roi_end", "snr_db_peak", "snr_db_eval_rms_ratio",
        "ref_subset_n", "ref_rms_agg", "ref_rms_raw_agg", "ref_rms_eff", "ref_floor_active", "ref_peak_agg", "rms_not_above_reference",
        "peak_not_above_reference_diag", "r",
    ]
    if not df_eval.empty:
        df_problematic = df_eval[df_eval["rms_not_above_reference"] == True].copy()
        df_problematic = df_problematic[problematic_cols]
    else:
        df_problematic = pd.DataFrame(columns=problematic_cols)
    df_problematic.to_csv(OUT_DIR / "problematic_measurements.csv", index=False)
    print("Saved:", OUT_DIR / "problematic_measurements.csv")
    print(f"Problematic measurements (rms_not_above_reference=True): {len(df_problematic)} / {len(df_eval)}")

    # -------------------------------------------------------------------------
    # 3) Aggregate per (band, wavelet, level)
    # -------------------------------------------------------------------------
    # Primary ultrasonic adaptation: use all rows in the main summary.
    summary = (df_eval
               .groupby(["band", "wavelet", "level"], as_index=False)
               .agg(
                    snr_db_eval_rms_ratio_mean=("snr_db_eval_rms_ratio",  "mean"),
                    snr_db_eval_rms_ratio_std=("snr_db_eval_rms_ratio",   "std"),
                    snr_db_eval_rms_ratio_min=("snr_db_eval_rms_ratio",   "min"),
                    snr_db_eval_rms_ratio_max=("snr_db_eval_rms_ratio",   "max"),
                    snr_db_peak_mean=("snr_db_peak", "mean"),
                    snr_db_peak_std=("snr_db_peak",  "std"),
                    snr_db_peak_min=("snr_db_peak",  "min"),
                    snr_db_peak_max=("snr_db_peak",  "max"),
                    r_mean=("r", "mean"),
                    r_std=("r",  "std"),
                    r_min=("r",  "min"),
                    r_max=("r",  "max"),
                    rms_fail_rate=("rms_not_above_reference", "mean"),
                    peak_fail_rate_diag=("peak_not_above_reference_diag", "mean"),
                    ref_floor_active_rate=("ref_floor_active", "mean"),
                    n_samples=("snr_db_eval_rms_ratio", "count"),
                ))
    summary.to_csv(OUT_DIR / "summary_by_band_wavelet_level.csv", index=False)
    print("Saved:", OUT_DIR / "summary_by_band_wavelet_level.csv")

    # Keep filtered summary only as QC / sensitivity analysis.
    df_eval_selection = df_eval[df_eval["rms_not_above_reference"] == False].copy()
    summary_qc_filtered = (df_eval_selection
               .groupby(["band", "wavelet", "level"], as_index=False)
               .agg(
                    snr_db_eval_rms_ratio_mean=("snr_db_eval_rms_ratio",  "mean"),
                    snr_db_eval_rms_ratio_std=("snr_db_eval_rms_ratio",   "std"),
                    snr_db_eval_rms_ratio_min=("snr_db_eval_rms_ratio",   "min"),
                    snr_db_eval_rms_ratio_max=("snr_db_eval_rms_ratio",   "max"),
                    snr_db_peak_mean=("snr_db_peak", "mean"),
                    snr_db_peak_std=("snr_db_peak",  "std"),
                    snr_db_peak_min=("snr_db_peak",  "min"),
                    snr_db_peak_max=("snr_db_peak",  "max"),
                    r_mean=("r", "mean"),
                    r_std=("r",  "std"),
                    r_min=("r",  "min"),
                    r_max=("r",  "max"),
                    n_samples=("snr_db_eval_rms_ratio", "count"),
                ))
    summary_qc_filtered.to_csv(OUT_DIR / "summary_by_band_wavelet_level_qc_filtered.csv", index=False)
    print("Saved:", OUT_DIR / "summary_by_band_wavelet_level_qc_filtered.csv")
    print(f"QC filtered rows (sig_rms > ref_rms_eff): {len(df_eval_selection)} / {len(df_eval)}")

    # -------------------------------------------------------------------------
    # 4-6) Per-band two-step selection:
    #      Step 1) choose one fixed wavelet per band by ranking across levels
    #      Step 2) with that wavelet fixed, choose decomposition level via OPT
    # -------------------------------------------------------------------------
    final_choices = {}
    for band in ["40k", "50k", "60k"]:
        sb = summary[summary["band"] == band].copy()
        if sb.empty:
            print(f"[{band}] No data — skipping.")
            continue

        # Step 1: rank wavelets across levels and pick one wavelet per band.
        # Keep only wavelets present at all tested levels; rank by robust SNR stats.
        wavelet_rank = (
            sb.groupby("wavelet", as_index=False)
              .agg(
                  snr_mean=("snr_db_eval_rms_ratio_mean", "mean"),
                  snr_median=("snr_db_eval_rms_ratio_mean", "median"),
                  snr_std=("snr_db_eval_rms_ratio_mean", "std"),
                  fail_rate_mean=("rms_fail_rate", "mean"),
                  floor_active_rate_mean=("ref_floor_active_rate", "mean"),
                  n_levels=("level", "nunique"),
              )
        )

        wavelet_rank = wavelet_rank[
            wavelet_rank["n_levels"] == len(LEVELS_TO_TEST)
        ].copy()

        if wavelet_rank.empty:
            print(f"[{band}] WARNING: no wavelet covers all tested levels; using best available coverage fallback.")
            wavelet_rank = (
                sb.groupby("wavelet", as_index=False)
                  .agg(
                      snr_mean=("snr_db_eval_rms_ratio_mean", "mean"),
                      snr_median=("snr_db_eval_rms_ratio_mean", "median"),
                      snr_std=("snr_db_eval_rms_ratio_mean", "std"),
                      fail_rate_mean=("rms_fail_rate", "mean"),
                      floor_active_rate_mean=("ref_floor_active_rate", "mean"),
                      n_levels=("level", "nunique"),
                  )
            )

        wavelet_rank = wavelet_rank.sort_values(
            ["n_levels", "snr_median", "fail_rate_mean", "snr_mean", "snr_std"],
            ascending=[False, False, True, False, True],
        ).reset_index(drop=True)

        wavelet_rank.to_csv(OUT_DIR / f"wavelet_rank_{band}.csv", index=False)
        print("Saved:", OUT_DIR / f"wavelet_rank_{band}.csv")

        if wavelet_rank.empty:
            print(f"[{band}] ERROR: no eligible wavelets after ranking; skipping band.")
            continue

        chosen_wavelet = str(wavelet_rank.iloc[0]["wavelet"])
        print(f"[{band}] Step-1 chosen fixed wavelet: {chosen_wavelet}")

        # Step 2: with wavelet fixed, build OPT across levels only.
        level_curve = (
            sb[sb["wavelet"] == chosen_wavelet]
              .sort_values("level")
              [["level", "snr_db_eval_rms_ratio_mean", "snr_db_peak_mean", "r_mean"]]
              .reset_index(drop=True)
        )

        n_levels_used = len(level_curve)
        if n_levels_used < len(LEVELS_TO_TEST):
            print(
                f"[{band}] WARNING: chosen wavelet {chosen_wavelet} only has "
                f"{n_levels_used}/{len(LEVELS_TO_TEST)} tested levels."
            )

        levels_arr, OPT, Wsnr, Wr = opt_curve_from_snr_r(
            level_curve["level"].astype(int).tolist(),
            level_curve["snr_db_eval_rms_ratio_mean"].to_numpy(float),
            level_curve["r_mean"].to_numpy(float),
        )
        best_level_inflection = pick_level_by_opt_inflection(levels_arr, OPT)
        best_level_maxopt = int(levels_arr[int(np.argmax(OPT))])

        print(f"[{band}] OPT weights: Wsnr={Wsnr:.3f}, Wr={Wr:.3f}")
        print(f"[{band}] Chosen level by OPT inflection: {best_level_inflection}")
        print(f"[{band}] Chosen level by max OPT: {best_level_maxopt}")
        if best_level_inflection != best_level_maxopt:
            print(f"[{band}] NOTE: inflection and max-OPT levels disagree.")

        final_choices[band] = {
            "wavelet": chosen_wavelet,
            "level_inflection": best_level_inflection,
            "level_maxopt": best_level_maxopt,
            "n_levels_used": n_levels_used,
        }

        opt_df = pd.DataFrame({
            "level": levels_arr,
            "wavelet": [chosen_wavelet] * len(levels_arr),
            "snr_db_eval_rms_ratio_mean": level_curve["snr_db_eval_rms_ratio_mean"].to_numpy(float),
            "snr_db_peak_mean": level_curve["snr_db_peak_mean"].to_numpy(float),
            "r_mean": level_curve["r_mean"].to_numpy(float),
            "OPT": OPT,
            "OPT_Wsnr": [Wsnr] * len(levels_arr),
            "OPT_Wr": [Wr] * len(levels_arr),
            "smoothness_inverted_for_OPT": [INVERT_SMOOTHNESS_FOR_OPT] * len(levels_arr),
        })
        opt_df.to_csv(OUT_DIR / f"chosen_wavelet_opt_curve_{band}.csv", index=False)
        print("Saved:", OUT_DIR / f"chosen_wavelet_opt_curve_{band}.csv")

        # OPT curve plot
        plt.figure()
        plt.plot(levels_arr, OPT, marker="o")
        plt.xlabel("Decomposition level")
        plt.ylabel("OPT")
        plt.title(f"OPT vs level — {band} (fixed wavelet: {chosen_wavelet})")
        plt.grid(True, alpha=0.3)
        plt.savefig(OUT_DIR / f"chosen_wavelet_opt_curve_{band}.png", dpi=160, bbox_inches="tight")
        plt.close()

        # Change-rate plot
        if len(OPT) >= 2:
            d = np.abs(np.diff(OPT))
            plt.figure()
            plt.plot(levels_arr[1:], d, marker="o")
            plt.axvline(best_level_inflection, linestyle="--")
            plt.xlabel("Level (transition endpoint)")
            plt.ylabel("|ΔOPT|")
            plt.title(f"OPT change rate — {band} (fixed wavelet: {chosen_wavelet})")
            plt.grid(True, alpha=0.3)
            plt.savefig(OUT_DIR / f"chosen_wavelet_opt_change_rate_{band}.png", dpi=160, bbox_inches="tight")
            plt.close()

    # -------------------------------------------------------------------------
    # 7) Save consolidated results table
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame([
        {
            "band": band,
            "wavelet": c["wavelet"],
            "level_inflection": c["level_inflection"],
            "level_maxopt": c["level_maxopt"],
            "n_levels_used": c["n_levels_used"],
        }
        for band, c in final_choices.items()
    ])
    results_df.to_csv(OUT_DIR / "best_wavelet_level_per_band.csv", index=False)
    print("\nBest wavelet and decomposition level per band:")
    print(results_df.to_string(index=False))
    print("\nSaved:", OUT_DIR / "best_wavelet_level_per_band.csv")
    print(f"Outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
