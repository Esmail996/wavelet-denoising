import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import detrend

from denoising.preprocess import bandpass
from roi_windows import ROI_WINDOWS, parse_distance_cm

# ============================================================
# CONFIG
# ============================================================

FS_HZ = 2_000_000
MIC = "Mic2"

REFERENCE_PICKLE = r"Multifrequenz Dataset\\Multifrequenz\\referenz\\referenz.pickle"
ROI_DATASET_PICKLE = r"outputs_roi\\roi_dataset.pkl"
OUTPUT_DIR = r"outputs\\snr_estimation"
DENOISED_ROOT = r"Multifrequenz Dataset\\Multifrequenz_denoised_multiband_mic2"
DETAIL_CSV = r"outputs\\snr_estimation\\snr_per_trial_reference_roi.csv"

# Reference preprocessing settings used to reproduce the delivered results
BANDPASS_ORDER = 2
BANDPASS_BW_HZ = 5000
BP_METHOD = "iir"
FIR_NUMTAPS = 71
FIR_WINDOW = "hamming"
EPS = 1e-12
BANDS_HZ = [40_000, 50_000, 60_000]
PREPROCESSING_MODES = ["bandpassed", "detrended_only"]

# ============================================================
# I/O
# ============================================================

def load_pickle(path):
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def parse_distance_from_filename(filename: str) -> int | None:
    return parse_distance_cm(filename)

# ============================================================
# SIGNAL PROCESSING
# ============================================================

def preprocess_dc_detrend(signal):
    """Apply only DC removal and linear detrending."""
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    x = detrend(x, type="linear")
    return x


def preprocess_reference_trial(signal, fs_hz, center_hz, bw_hz=5000, order=1, method="fir", fir_numtaps=71, fir_window="hamming"):
    """
    Preprocess one reference trial exactly as used for the delivered SNR results:
      1) DC removal
      2) linear detrend
      3) Butterworth bandpass around the requested center frequency

    Notes
    -----
    The roi_dataset.pkl object ROIs are already band-specific arrays. We do not
    reprocess them here. We only process the raw reference trials to obtain
    matched reference ROIs in the same frequency band.
    """
    x = preprocess_dc_detrend(signal)

    # x = bandpass(x, fs_hz, center_hz, bw=bw_hz, order=order, method="iir")
    x = bandpass(x, fs_hz, center_hz, bw=bw_hz, order=order, method=method, numtaps=fir_numtaps, window=fir_window)

    return x

# ============================================================
# CORE SNR METHOD
# ============================================================

def compute_reference_noise_lookup(reference_df, mic="Mic2", fs_hz=2_000_000,
                                   bw_hz=5000, order=1, method="fir", fir_numtaps=71, fir_window="hamming"):
    """
    For each band and each distance ROI, compute the mean reference noise power
    across reference trials.

    Power for one trial is computed with the mean-power equation:
        P = (1/N) * sum(x[n]^2)

    Returns
    -------
    ref_lookup : dict
        ref_lookup[band_hz][distance_cm] = mean reference ROI power
    reference_noise_summary : pd.DataFrame
        One row per (band, distance) with mean/median/std of reference noise power
    """
    if mic not in reference_df.columns:
        raise KeyError(f"Column {mic!r} not found in reference DataFrame.")

    ref_trials_raw = np.stack(
        reference_df[mic].apply(lambda t: np.asarray(t, dtype=float)).to_list()
    )

    bands = BANDS_HZ
    ref_lookup = {"bandpassed": {}, "detrended_only": {}}
    rows = []

    detrended_only = np.apply_along_axis(preprocess_dc_detrend, axis=1, arr=ref_trials_raw)

    for distance_cm, (start, end) in ROI_WINDOWS.items():
        trial_powers = np.mean(detrended_only[:, start:end] ** 2, axis=1)
        mean_power = float(np.mean(trial_powers))
        median_power = float(np.median(trial_powers))
        std_power = float(np.std(trial_powers, ddof=1)) if len(trial_powers) > 1 else 0.0
        ref_lookup["detrended_only"][distance_cm] = mean_power

        rows.append({
            "preprocessing": "detrended_only",
            "band_center_hz": np.nan,
            "distance_cm": distance_cm,
            "roi_start": start,
            "roi_end": end,
            "roi_length": end - start,
            "reference_noise_power_mean": mean_power,
            "reference_noise_power_median": median_power,
            "reference_noise_power_std": std_power,
            "reference_noise_rms_from_mean_power": float(np.sqrt(mean_power)),
        })

    for band_hz in bands:
        processed_bandpassed = np.apply_along_axis(
            lambda x: preprocess_reference_trial(
                x,
                fs_hz=fs_hz,
                center_hz=band_hz,
                bw_hz=bw_hz,
                order=order,
                method=method,
                fir_numtaps=fir_numtaps,
                fir_window=fir_window,
            ),
            axis=1,
            arr=ref_trials_raw
        )

        ref_lookup["bandpassed"][band_hz] = {}

        for distance_cm, (start, end) in ROI_WINDOWS.items():
            trial_powers = np.mean(processed_bandpassed[:, start:end] ** 2, axis=1)
            mean_power = float(np.mean(trial_powers))
            median_power = float(np.median(trial_powers))
            std_power = float(np.std(trial_powers, ddof=1)) if len(trial_powers) > 1 else 0.0

            ref_lookup["bandpassed"][band_hz][distance_cm] = mean_power

            rows.append({
                "preprocessing": "bandpassed",
                "band_center_hz": band_hz,
                "distance_cm": distance_cm,
                "roi_start": start,
                "roi_end": end,
                "roi_length": end - start,
                "reference_noise_power_mean": mean_power,
                "reference_noise_power_median": median_power,
                "reference_noise_power_std": std_power,
                "reference_noise_rms_from_mean_power": float(np.sqrt(mean_power)),
            })

    return ref_lookup, pd.DataFrame(rows)


def compute_snr_detail(roi_dataset_df, ref_lookup):
    """
    Compute SNR for every row in roi_dataset.pkl.

    The object ROI is already stored in roi_dataset.pkl as a band-specific array.

    For one row:
        P_total  = mean(obj_roi^2)
        P_noise  = mean reference power for the same band and distance ROI
        P_signal = max(P_total - P_noise, EPS)
        SNR_dB   = 10 * log10(P_signal / P_noise)

    Returns
    -------
    detail_df : pd.DataFrame
        One row per ROI entry from roi_dataset.pkl
    """
    rows = []
    has_preprocessing = "preprocessing" in roi_dataset_df.columns

    for _, r in roi_dataset_df.iterrows():
        band_hz = float(r["band"]) if pd.notna(r["band"]) else np.nan
        distance_cm = int(r["distance"])
        angle_deg = float(r["angle"])
        preprocessing = str(r["preprocessing"]) if has_preprocessing else "bandpassed"
        obj_roi = np.asarray(r["roi"], dtype=float)

        p_total = float(np.mean(obj_roi ** 2))
        if preprocessing == "bandpassed":
            p_noise = float(ref_lookup["bandpassed"][int(band_hz)][distance_cm])
        else:
            p_noise = float(ref_lookup["detrended_only"][distance_cm])
        p_noise_safe = max(p_noise, EPS)
        p_signal = max(p_total - p_noise, EPS)
        snr_db = float(10.0 * np.log10(p_signal / p_noise_safe))

        rows.append({
            "object": r["object"],
            "distance_cm": distance_cm,
            "angle_deg": angle_deg,
            "mic": r["mic"],
            "trial": int(r["trial"]),
            "band_center_hz": int(band_hz) if np.isfinite(band_hz) else np.nan,
            "preprocessing": preprocessing,
            "filename": r["filename"],
            "roi_start": int(r["ROI start"]),
            "roi_end": int(r["ROI end"]),
            "roi_length": int(r["ROI length"]),
            "object_roi_power_mean": p_total,
            "reference_roi_noise_power_mean": p_noise,
            "estimated_signal_power": p_signal,
            "snr_db": snr_db,
        })

    return pd.DataFrame(rows)


def summarize_snr(detail_df):
    """
    Build several useful summary tables.
    """
    summary_condition = detail_df.groupby(
        ["preprocessing", "object", "distance_cm", "angle_deg", "band_center_hz"],
        as_index=False,
        dropna=False
    ).agg(
        n_trials=("trial", "count"),
        snr_db_mean=("snr_db", "mean"),
        snr_db_median=("snr_db", "median"),
        snr_db_std=("snr_db", "std"),
        snr_db_q25=("snr_db", lambda x: x.quantile(0.25)),
        snr_db_q75=("snr_db", lambda x: x.quantile(0.75)),
        signal_power_mean=("estimated_signal_power", "mean"),
        total_power_mean=("object_roi_power_mean", "mean"),
        reference_noise_power_mean=("reference_roi_noise_power_mean", "mean"),
    )

    summary_object_band = detail_df.groupby(
        ["preprocessing", "object", "band_center_hz"], as_index=False, dropna=False
    ).agg(
        snr_db_mean=("snr_db", "mean"),
        snr_db_median=("snr_db", "median"),
        snr_db_std=("snr_db", "std"),
    )

    summary_distance_band = detail_df.groupby(
        ["preprocessing", "distance_cm", "band_center_hz"], as_index=False, dropna=False
    ).agg(
        snr_db_mean=("snr_db", "mean"),
        snr_db_median=("snr_db", "median"),
        snr_db_std=("snr_db", "std"),
    )

    summary_overall_band = detail_df.groupby(
        ["preprocessing", "band_center_hz"], as_index=False, dropna=False
    ).agg(
        snr_db_mean=("snr_db", "mean"),
        snr_db_median=("snr_db", "median"),
        snr_db_std=("snr_db", "std"),
    )

    return {
        "summary_condition": summary_condition,
        "summary_object_band": summary_object_band,
        "summary_distance_band": summary_distance_band,
        "summary_overall_band": summary_overall_band,
    }


def compute_snr_gain(detail_df):
    """Compare bandpassed and detrended-only SNR on matched ROI entries."""
    key_cols = [
        "object",
        "distance_cm",
        "angle_deg",
        "mic",
        "trial",
        "filename",
    ]

    bandpassed_df = detail_df[detail_df["preprocessing"] == "bandpassed"].copy()
    detrended_df = detail_df[detail_df["preprocessing"] == "detrended_only"].copy()

    if bandpassed_df.empty or detrended_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    detrended_df = detrended_df[key_cols + ["snr_db"]].rename(
        columns={"snr_db": "snr_db_detrended_only"}
    )

    wide = bandpassed_df.merge(detrended_df, on=key_cols, how="inner")
    wide = wide.rename(columns={"snr_db": "snr_db_bandpassed"})
    wide["snr_gain_db"] = wide["snr_db_bandpassed"] - wide["snr_db_detrended_only"]

    summary_condition = wide.groupby(
        ["object", "distance_cm", "angle_deg", "band_center_hz"], as_index=False
    ).agg(
        n_trials=("trial", "count"),
        snr_gain_db_mean=("snr_gain_db", "mean"),
        snr_gain_db_median=("snr_gain_db", "median"),
        snr_gain_db_std=("snr_gain_db", "std"),
        snr_db_bandpassed_mean=("snr_db_bandpassed", "mean"),
        snr_db_detrended_only_mean=("snr_db_detrended_only", "mean"),
    )

    summary_overall_band = wide.groupby(["band_center_hz"], as_index=False).agg(
        snr_gain_db_mean=("snr_gain_db", "mean"),
        snr_gain_db_median=("snr_gain_db", "median"),
        snr_gain_db_std=("snr_gain_db", "std"),
    )

    return wide, summary_condition, summary_overall_band


def compute_denoising_snr_improvement(detail_df: pd.DataFrame, denoised_root: str):
    """
    Compute ROI SNR before/after denoising on matched (file, trial, band, mic) keys.

    "before" is the preprocessed ROI SNR already in detail_df.
    "after" is recomputed from denoised signals over the exact same ROI window,
    using denoised reference noise from denoised_root/referenz/referenz.pickle.
    """
    denoised_path = Path(denoised_root)
    if not denoised_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_cols = {
        "object",
        "distance_cm",
        "mic",
        "trial",
        "band_center_hz",
        "filename",
        "roi_start",
        "roi_end",
        "reference_roi_noise_power_mean",
        "snr_db",
        "preprocessing",
    }
    missing = required_cols.difference(detail_df.columns)
    if missing:
        raise ValueError(f"detail_df missing required columns for denoising SNR gain: {sorted(missing)}")

    bandpassed_rows = detail_df[detail_df["preprocessing"] == "bandpassed"].copy()
    if bandpassed_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build denoised reference-noise lookup by (band, distance) from the already
    # denoised reference file.
    den_ref_path = denoised_path / "referenz" / "referenz.pickle"
    if not den_ref_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    den_ref = load_pickle(str(den_ref_path))
    if not isinstance(den_ref, dict):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    den_ref_lookup: dict[int, dict[int, float]] = {}
    for band_hz in BANDS_HZ:
        if band_hz in den_ref:
            band_key = band_hz
        else:
            possible = [k for k in den_ref.keys() if np.isclose(float(k), float(band_hz))]
            if not possible:
                continue
            band_key = possible[0]

        band_trials = np.asarray(den_ref[band_key], dtype=object)
        den_ref_lookup[int(band_hz)] = {}
        for distance_cm, (start, end) in ROI_WINDOWS.items():
            trial_powers = []
            for t in band_trials:
                sig = np.asarray(t, dtype=float)
                i0 = max(0, int(start))
                i1 = min(int(end), len(sig))
                if i1 <= i0:
                    continue
                roi = sig[i0:i1]
                if roi.size == 0:
                    continue
                trial_powers.append(float(np.mean(roi ** 2)))
            if trial_powers:
                den_ref_lookup[int(band_hz)][int(distance_cm)] = float(np.mean(trial_powers))

    loaded_pickles: dict[tuple[str, str], dict] = {}
    rows = []

    for _, r in bandpassed_rows.iterrows():
        obj = str(r["object"])
        file_name = str(r["filename"])
        mic = str(r["mic"])
        trial = int(r["trial"])
        band_hz = float(r["band_center_hz"])
        i0 = int(r["roi_start"])
        i1 = int(r["roi_end"])
        p_noise_pre = max(float(r["reference_roi_noise_power_mean"]), EPS)
        snr_pre_db = float(r["snr_db"])

        if mic != "Mic2":
            continue

        cache_key = (obj, file_name)
        if cache_key not in loaded_pickles:
            file_path = denoised_path / obj / file_name
            if not file_path.exists():
                continue
            loaded_pickles[cache_key] = load_pickle(str(file_path))

        data = loaded_pickles[cache_key]
        if not isinstance(data, dict):
            continue

        if band_hz not in data:
            possible = [k for k in data.keys() if np.isclose(float(k), band_hz)]
            if not possible:
                continue
            band_key = possible[0]
        else:
            band_key = band_hz

        band_trials = data[band_key]
        if trial < 0 or trial >= len(band_trials):
            continue

        den_sig = np.asarray(band_trials[trial], dtype=float)
        if i0 < 0 or i1 <= i0:
            continue
        if i1 > len(den_sig):
            i1 = len(den_sig)
        if i1 <= i0:
            continue

        den_roi = den_sig[i0:i1]
        if den_roi.size == 0:
            continue

        band_int = int(band_hz)
        dist_int = int(r["distance_cm"])
        if band_int not in den_ref_lookup or dist_int not in den_ref_lookup[band_int]:
            continue
        p_noise_den = max(float(den_ref_lookup[band_int][dist_int]), EPS)

        p_total_den = float(np.mean(den_roi ** 2))
        p_signal_den = max(p_total_den - p_noise_den, EPS)
        snr_den_db = float(10.0 * np.log10(p_signal_den / p_noise_den))
        snr_improve_db = snr_den_db - snr_pre_db

        rows.append(
            {
                "object": obj,
                "distance_cm": int(r["distance_cm"]),
                "angle_deg": float(r["angle_deg"]),
                "mic": mic,
                "trial": trial,
                "filename": file_name,
                "band_center_hz": int(band_hz),
                "roi_start": i0,
                "roi_end": i1,
                "roi_length": i1 - i0,
                "snr_db_preprocessed": snr_pre_db,
                "snr_db_denoised": snr_den_db,
                "snr_improvement_db": snr_improve_db,
                "reference_roi_noise_power_mean_pre": p_noise_pre,
                "reference_roi_noise_power_mean_denoised": p_noise_den,
                "denoised_roi_power_mean": p_total_den,
                "denoised_estimated_signal_power": p_signal_den,
            }
        )

    per_trial_df = pd.DataFrame(rows)
    if per_trial_df.empty:
        return per_trial_df, pd.DataFrame(), pd.DataFrame()

    summary_condition = per_trial_df.groupby(
        ["object", "distance_cm", "angle_deg", "band_center_hz"], as_index=False
    ).agg(
        n_trials=("trial", "count"),
        snr_db_preprocessed_mean=("snr_db_preprocessed", "mean"),
        snr_db_denoised_mean=("snr_db_denoised", "mean"),
        snr_improvement_db_mean=("snr_improvement_db", "mean"),
        snr_improvement_db_median=("snr_improvement_db", "median"),
        snr_improvement_db_std=("snr_improvement_db", "std"),
    )

    summary_overall_band = per_trial_df.groupby(["band_center_hz"], as_index=False).agg(
        snr_db_preprocessed_mean=("snr_db_preprocessed", "mean"),
        snr_db_denoised_mean=("snr_db_denoised", "mean"),
        snr_improvement_db_mean=("snr_improvement_db", "mean"),
        snr_improvement_db_median=("snr_improvement_db", "median"),
        snr_improvement_db_std=("snr_improvement_db", "std"),
    )

    return per_trial_df, summary_condition, summary_overall_band

# ============================================================
# Direct raw-vs-denoised comparison (no roi_dataset.pkl needed)
# ============================================================

def compute_direct_raw_vs_denoised_snr(
    raw_root: str,
    denoised_root: str,
    denoised_ref_path: str | None,
    bands_hz: list,
    bw_hz: float,
    bp_method: str,
    iir_order: int,
    fir_numtaps: int,
    fir_window: str,
    mic: str,
    fs_hz: float,
    raw_ref_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk every raw measurement pickle, extract ROIs, and compute SNR for:
      - raw signal (DC+detrend+bandpass) vs raw reference noise
      - denoised signal (loaded from denoised_root) vs denoised reference noise

    Does NOT rely on roi_dataset.pkl.  Works for any single-band or multi-band
    denoised output.

    Returns
    -------
    per_trial_df : DataFrame
        One row per (file, trial, band) with both raw and denoised SNR values.
    summary_condition : DataFrame
        Mean/median/std grouped by (object, distance_cm, band_center_hz).
    summary_overall : DataFrame
        Mean/median/std grouped by band_center_hz only.
    """
    raw_root_path = Path(raw_root)
    den_root_path = Path(denoised_root)

    # ---- build raw reference noise lookup {band_hz -> {dist_cm -> mean_power}} ----
    raw_ref = load_pickle(raw_ref_path)
    if not isinstance(raw_ref, pd.DataFrame):
        raise TypeError(f"Raw reference pickle at {raw_ref_path!r} must be a DataFrame.")
    if mic not in raw_ref.columns:
        raise KeyError(f"Column {mic!r} not found in raw reference DataFrame.")

    raw_ref_signals = np.stack(
        raw_ref[mic].apply(lambda t: np.asarray(t, dtype=float)).to_list()
    )

    raw_ref_noise: dict[float, dict[int, float]] = {}
    for band_hz in bands_hz:
        band_hz_f = float(band_hz)
        processed = np.apply_along_axis(
            lambda x: preprocess_reference_trial(
                x, fs_hz=fs_hz, center_hz=band_hz_f, bw_hz=bw_hz,
                order=iir_order, method=bp_method,
                fir_numtaps=fir_numtaps, fir_window=fir_window,
            ),
            axis=1,
            arr=raw_ref_signals,
        )
        raw_ref_noise[band_hz_f] = {}
        for dist_cm, (start, end) in ROI_WINDOWS.items():
            roi_slice = processed[:, start:end]
            if roi_slice.shape[1] == 0:
                continue
            powers = np.mean(roi_slice ** 2, axis=1)
            raw_ref_noise[band_hz_f][dist_cm] = float(np.mean(powers))

    # ---- build denoised reference noise lookup {band_hz -> {dist_cm -> mean_power}} ----
    den_ref_noise: dict[float, dict[int, float]] = {}
    if denoised_ref_path:
        den_ref_pkl = Path(denoised_ref_path)
    else:
        den_ref_pkl = den_root_path / "referenz" / "referenz.pickle"

    if den_ref_pkl.exists():
        den_ref = load_pickle(str(den_ref_pkl))
        if isinstance(den_ref, dict):
            for band_hz in bands_hz:
                band_hz_f = float(band_hz)
                band_key = next(
                    (k for k in den_ref if np.isclose(float(k), band_hz_f)), None
                )
                if band_key is None:
                    continue
                den_ref_trials = den_ref[band_key]
                den_ref_noise[band_hz_f] = {}
                for dist_cm, (start, end) in ROI_WINDOWS.items():
                    powers = []
                    for t in den_ref_trials:
                        sig = np.asarray(t, dtype=float)
                        i1 = min(int(end), len(sig))
                        if i1 > int(start):
                            roi = sig[int(start):i1]
                            if roi.size:
                                powers.append(float(np.mean(roi ** 2)))
                    if powers:
                        den_ref_noise[band_hz_f][dist_cm] = float(np.mean(powers))

    # ---- walk measurement files ----
    rows = []
    loaded_den: dict[tuple, dict] = {}

    for raw_pickle in sorted(raw_root_path.rglob("*.pickle")):
        if "referenz" in str(raw_pickle).lower():
            continue

        obj = raw_pickle.parent.name
        file_name = raw_pickle.name
        dist_cm_val = parse_distance_from_filename(file_name)
        if dist_cm_val is None:
            continue
        dist_cm = int(dist_cm_val)
        if dist_cm not in ROI_WINDOWS:
            continue
        roi_start, roi_end = ROI_WINDOWS[dist_cm]

        try:
            raw_data = load_pickle(str(raw_pickle))
        except Exception:
            continue

        # Extract raw Mic2 signal list
        if isinstance(raw_data, pd.DataFrame):
            if mic not in raw_data.columns:
                continue
            raw_signals = [np.asarray(raw_data[mic].iloc[i], dtype=float)
                           for i in range(len(raw_data))]
        elif isinstance(raw_data, np.ndarray) and raw_data.ndim == 3:
            mic_idx = {"Mic1": 0, "Mic2": 1, "Mic3": 2}.get(mic)
            if mic_idx is None:
                continue
            raw_signals = [np.asarray(raw_data[mic_idx, i], dtype=float)
                           for i in range(raw_data.shape[1])]
        elif isinstance(raw_data, dict) and mic in raw_data:
            raw_signals = [np.asarray(raw_data[mic][i], dtype=float)
                           for i in range(len(raw_data[mic]))]
        else:
            continue

        n_trials = len(raw_signals)

        # Load matching denoised pickle (cached)
        den_cache_key = (obj, file_name)
        if den_cache_key not in loaded_den:
            den_file = den_root_path / obj / file_name
            if den_file.exists():
                try:
                    loaded_den[den_cache_key] = load_pickle(str(den_file))
                except Exception:
                    loaded_den[den_cache_key] = None
            else:
                loaded_den[den_cache_key] = None
        denoised_data = loaded_den[den_cache_key]

        for band_hz in bands_hz:
            band_hz_f = float(band_hz)

            raw_ref_pow = raw_ref_noise.get(band_hz_f, {}).get(dist_cm)
            den_ref_pow = den_ref_noise.get(band_hz_f, {}).get(dist_cm)

            # Denoised band key lookup
            den_band_key = None
            if isinstance(denoised_data, dict):
                den_band_key = next(
                    (k for k in denoised_data if np.isclose(float(k), band_hz_f)), None
                )

            for trial_idx in range(n_trials):
                # Raw SNR
                raw_snr_db = np.nan
                raw_roi_power = np.nan
                if raw_ref_pow is not None:
                    try:
                        pp = preprocess_reference_trial(
                            raw_signals[trial_idx], fs_hz=fs_hz,
                            center_hz=band_hz_f, bw_hz=bw_hz,
                            order=iir_order, method=bp_method,
                            fir_numtaps=fir_numtaps, fir_window=fir_window,
                        )
                        i1 = min(roi_end, len(pp))
                        if i1 > roi_start:
                            raw_roi = pp[roi_start:i1]
                            raw_roi_power = float(np.mean(raw_roi ** 2))
                            p_sig = max(raw_roi_power - raw_ref_pow, EPS)
                            raw_snr_db = float(10.0 * np.log10(p_sig / max(raw_ref_pow, EPS)))
                    except Exception:
                        pass

                # Denoised SNR
                den_snr_db = np.nan
                den_roi_power = np.nan
                if den_band_key is not None and den_ref_pow is not None:
                    band_list = denoised_data[den_band_key]
                    if trial_idx < len(band_list):
                        den_sig = np.asarray(band_list[trial_idx], dtype=float)
                        i1 = min(roi_end, len(den_sig))
                        if i1 > roi_start:
                            den_roi = den_sig[roi_start:i1]
                            den_roi_power = float(np.mean(den_roi ** 2))
                            p_sig = max(den_roi_power - den_ref_pow, EPS)
                            den_snr_db = float(10.0 * np.log10(p_sig / max(den_ref_pow, EPS)))

                rows.append({
                    "object": obj,
                    "filename": file_name,
                    "distance_cm": dist_cm,
                    "mic": mic,
                    "trial": trial_idx,
                    "band_center_hz": int(band_hz_f),
                    "roi_start": roi_start,
                    "roi_end": roi_end,
                    "roi_length": roi_end - roi_start,
                    "raw_roi_power": raw_roi_power,
                    "raw_ref_noise_power": raw_ref_pow if raw_ref_pow is not None else np.nan,
                    "snr_db_raw": raw_snr_db,
                    "denoised_roi_power": den_roi_power,
                    "denoised_ref_noise_power": den_ref_pow if den_ref_pow is not None else np.nan,
                    "snr_db_denoised": den_snr_db,
                    "snr_improvement_db": (
                        float(den_snr_db - raw_snr_db)
                        if np.isfinite(den_snr_db) and np.isfinite(raw_snr_db)
                        else np.nan
                    ),
                })

    per_trial_df = pd.DataFrame(rows)
    if per_trial_df.empty:
        return per_trial_df, pd.DataFrame(), pd.DataFrame()

    summary_condition = per_trial_df.groupby(
        ["object", "distance_cm", "band_center_hz"], as_index=False
    ).agg(
        n_trials=("trial", "count"),
        snr_db_raw_mean=("snr_db_raw", "mean"),
        snr_db_raw_median=("snr_db_raw", "median"),
        snr_db_raw_std=("snr_db_raw", "std"),
        snr_db_denoised_mean=("snr_db_denoised", "mean"),
        snr_db_denoised_median=("snr_db_denoised", "median"),
        snr_db_denoised_std=("snr_db_denoised", "std"),
        snr_improvement_db_mean=("snr_improvement_db", "mean"),
        snr_improvement_db_median=("snr_improvement_db", "median"),
        snr_improvement_db_std=("snr_improvement_db", "std"),
    )

    summary_overall = per_trial_df.groupby(["band_center_hz"], as_index=False).agg(
        snr_db_raw_mean=("snr_db_raw", "mean"),
        snr_db_raw_median=("snr_db_raw", "median"),
        snr_db_raw_std=("snr_db_raw", "std"),
        snr_db_denoised_mean=("snr_db_denoised", "mean"),
        snr_db_denoised_median=("snr_db_denoised", "median"),
        snr_db_denoised_std=("snr_db_denoised", "std"),
        snr_improvement_db_mean=("snr_improvement_db", "mean"),
        snr_improvement_db_median=("snr_improvement_db", "median"),
        snr_improvement_db_std=("snr_improvement_db", "std"),
    )

    return per_trial_df, summary_condition, summary_overall

# ============================================================
# MAIN
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Estimate SNR from ROI dataset and reference trials.")
    parser.add_argument("--reference-pickle", type=str, default=REFERENCE_PICKLE, help="Path to reference pickle.")
    parser.add_argument("--roi-dataset-pickle", type=str, default=ROI_DATASET_PICKLE, help="Path to ROI dataset pickle.")
    parser.add_argument("--detail-csv", type=str, default=DETAIL_CSV, help="Optional existing SNR detail CSV fallback.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to write SNR outputs.")
    parser.add_argument("--mic", type=str, default=MIC, help="Mic column to use for reference signal.")
    parser.add_argument("--fs-hz", type=float, default=FS_HZ, help="Sampling frequency in Hz.")
    parser.add_argument("--bw-hz", type=float, default=BANDPASS_BW_HZ, help="Bandpass half-bandwidth in Hz.")
    parser.add_argument("--bp-method", type=str, choices=["fir", "iir"], default=BP_METHOD, help="Bandpass method.")
    parser.add_argument("--iir-order", type=int, default=BANDPASS_ORDER, help="IIR filter order.")
    parser.add_argument("--fir-numtaps", type=int, default=FIR_NUMTAPS, help="FIR taps.")
    parser.add_argument("--fir-window", type=str, default=FIR_WINDOW, help="FIR window.")
    parser.add_argument(
        "--denoised-root",
        type=str,
        default=DENOISED_ROOT,
        help="Optional root directory containing denoised multiband Mic2 pickles to compute ROI SNR improvement.",
    )
    # Direct raw-vs-denoised comparison (no roi_dataset.pkl required)
    parser.add_argument(
        "--compare-raw-root",
        type=str,
        default=r"Multifrequenz Dataset\Multifrequenz",
        help="Root directory of raw measurement pickles for direct comparison.",
    )
    parser.add_argument(
        "--compare-denoised-root",
        type=str,
        default=None,
        help="Root directory of denoised pickles for direct comparison. Set this to activate the direct SNR comparison mode.",
    )
    parser.add_argument(
        "--compare-denoised-ref",
        type=str,
        default=None,
        help="Path to denoised reference pickle. Defaults to <compare-denoised-root>/referenz/referenz.pickle.",
    )
    parser.add_argument(
        "--compare-bands-hz",
        type=float,
        nargs="+",
        default=[50_000.0],
        help="Band center frequencies (Hz) present in the denoised output. E.g. 50000 for the 25-75 kHz wideband run.",
    )
    parser.add_argument(
        "--compare-bw-hz",
        type=float,
        default=25_000.0,
        help="Bandpass half-bandwidth (Hz) used when preprocessing raw signals for comparison. Use 25000 for 25-75 kHz.",
    )
    parser.add_argument(
        "--compare-bp-method",
        type=str,
        choices=["fir", "iir"],
        default="iir",
        help="Bandpass method used when preprocessing raw signals for comparison.",
    )
    parser.add_argument(
        "--compare-iir-order",
        type=int,
        default=6,
        help="IIR order used when preprocessing raw signals for comparison.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[2]

    reference_path = Path(args.reference_pickle)
    if not reference_path.is_absolute():
        reference_path = project_root / reference_path

    roi_dataset_path = Path(args.roi_dataset_pickle)
    if not roi_dataset_path.is_absolute():
        roi_dataset_path = project_root / roi_dataset_path

    detail_csv_path = Path(args.detail_csv)
    if not detail_csv_path.is_absolute():
        detail_csv_path = project_root / detail_csv_path

    denoised_root = Path(args.denoised_root)
    if not denoised_root.is_absolute():
        denoised_root = project_root / denoised_root

    reference_df = load_pickle(str(reference_path))
    roi_dataset_df = None
    detail_df = None
    roi_load_error = None
    try:
        roi_dataset_df = load_pickle(str(roi_dataset_path))
    except Exception as exc:
        roi_load_error = exc
        if detail_csv_path.exists():
            detail_df = pd.read_csv(detail_csv_path)
        else:
            raise

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic validation
    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("referenz.pickle must contain a pandas DataFrame.")
    if detail_df is None and not isinstance(roi_dataset_df, pd.DataFrame):
        raise TypeError("roi_dataset.pkl must contain a pandas DataFrame.")

    # Compute matched reference noise powers
    ref_lookup, reference_noise_summary = compute_reference_noise_lookup(
        reference_df=reference_df,
        mic=args.mic,
        fs_hz=float(args.fs_hz),
        bw_hz=float(args.bw_hz),
        order=int(args.iir_order),
        method=str(args.bp_method),
        fir_numtaps=int(args.fir_numtaps),
        fir_window=str(args.fir_window),
    )

    # Compute per-row SNR (or load existing detail CSV if ROI pickle is incompatible)
    if detail_df is None:
        detail_df = compute_snr_detail(roi_dataset_df, ref_lookup)

    # Summaries
    summaries = summarize_snr(detail_df)
    gain_per_trial, gain_summary_condition, gain_summary_overall_band = compute_snr_gain(detail_df)
    denoise_gain_per_trial, denoise_gain_summary_condition, denoise_gain_summary_overall_band = compute_denoising_snr_improvement(
        detail_df,
        denoised_root=denoised_root,
    )

    # Save CSV files
    detail_path = out_dir / "snr_per_trial_reference_roi.csv"
    cond_path = out_dir / "snr_summary_reference_roi.csv"
    ref_path = out_dir / "reference_noise_summary.csv"
    obj_band_path = out_dir / "snr_summary_object_band.csv"
    dist_band_path = out_dir / "snr_summary_distance_band.csv"
    overall_band_path = out_dir / "snr_summary_overall_band.csv"
    gain_trial_path = out_dir / "snr_gain_per_trial.csv"
    gain_summary_condition_path = out_dir / "snr_gain_summary_reference_roi.csv"
    gain_summary_band_path = out_dir / "snr_gain_summary_overall_band.csv"
    denoise_gain_trial_path = out_dir / "snr_improvement_denoising_per_trial.csv"
    denoise_gain_summary_condition_path = out_dir / "snr_improvement_denoising_summary_reference_roi.csv"
    denoise_gain_summary_band_path = out_dir / "snr_improvement_denoising_summary_overall_band.csv"

    detail_df.to_csv(str(detail_path), index=False)
    summaries["summary_condition"].to_csv(str(cond_path), index=False)
    reference_noise_summary.to_csv(str(ref_path), index=False)
    summaries["summary_object_band"].to_csv(str(obj_band_path), index=False)
    summaries["summary_distance_band"].to_csv(str(dist_band_path), index=False)
    summaries["summary_overall_band"].to_csv(str(overall_band_path), index=False)
    if not gain_per_trial.empty:
        gain_per_trial.to_csv(str(gain_trial_path), index=False)
        gain_summary_condition.to_csv(str(gain_summary_condition_path), index=False)
        gain_summary_overall_band.to_csv(str(gain_summary_band_path), index=False)
    if not denoise_gain_per_trial.empty:
        denoise_gain_per_trial.to_csv(str(denoise_gain_trial_path), index=False)
        denoise_gain_summary_condition.to_csv(str(denoise_gain_summary_condition_path), index=False)
        denoise_gain_summary_overall_band.to_csv(str(denoise_gain_summary_band_path), index=False)

    # Direct raw-vs-denoised comparison
    compare_denoised_root = args.compare_denoised_root
    if compare_denoised_root:
        compare_denoised_root_abs = Path(compare_denoised_root)
        if not compare_denoised_root_abs.is_absolute():
            compare_denoised_root_abs = project_root / compare_denoised_root_abs

        compare_raw_root_abs = Path(args.compare_raw_root)
        if not compare_raw_root_abs.is_absolute():
            compare_raw_root_abs = project_root / compare_raw_root_abs

        compare_denoised_ref_abs = None
        if args.compare_denoised_ref:
            compare_denoised_ref_abs = Path(args.compare_denoised_ref)
            if not compare_denoised_ref_abs.is_absolute():
                compare_denoised_ref_abs = project_root / compare_denoised_ref_abs

        print(f"\nRunning direct raw-vs-denoised SNR comparison...")
        print(f"  Raw root       : {compare_raw_root_abs}")
        print(f"  Denoised root  : {compare_denoised_root_abs}")
        print(f"  Bands (Hz)     : {args.compare_bands_hz}")
        print(f"  BW (Hz)        : {args.compare_bw_hz}")
        print(f"  BP method      : {args.compare_bp_method}")
        cmp_per_trial, cmp_summary_condition, cmp_summary_overall = compute_direct_raw_vs_denoised_snr(
            raw_root=str(compare_raw_root_abs),
            denoised_root=str(compare_denoised_root_abs),
            denoised_ref_path=str(compare_denoised_ref_abs) if compare_denoised_ref_abs else None,
            bands_hz=args.compare_bands_hz,
            bw_hz=float(args.compare_bw_hz),
            bp_method=str(args.compare_bp_method),
            iir_order=int(args.compare_iir_order),
            fir_numtaps=int(args.fir_numtaps),
            fir_window=str(args.fir_window),
            mic=str(args.mic),
            fs_hz=float(args.fs_hz),
            raw_ref_path=str(reference_path),
        )
        cmp_trial_path = out_dir / "direct_snr_comparison_per_trial.csv"
        cmp_cond_path = out_dir / "direct_snr_comparison_summary_condition.csv"
        cmp_overall_path = out_dir / "direct_snr_comparison_summary_overall.csv"
        if not cmp_per_trial.empty:
            cmp_per_trial.to_csv(str(cmp_trial_path), index=False)
            cmp_summary_condition.to_csv(str(cmp_cond_path), index=False)
            cmp_summary_overall.to_csv(str(cmp_overall_path), index=False)
            print(f"  Saved comparison per-trial  : {cmp_trial_path}")
            print(f"  Saved comparison by condition: {cmp_cond_path}")
            print(f"  Saved comparison overall    : {cmp_overall_path}")
            print("\nDirect comparison overall:")
            print(cmp_summary_overall.to_string(index=False))
        else:
            print("  No matched rows produced. Check paths and band settings.")

    print("Saved:")
    print("  ", detail_path)
    print("  ", cond_path)
    print("  ", ref_path)
    print("  ", obj_band_path)
    print("  ", dist_band_path)
    print("  ", overall_band_path)
    if not gain_per_trial.empty:
        print("  ", gain_trial_path)
        print("  ", gain_summary_condition_path)
        print("  ", gain_summary_band_path)
    else:
        print("  SNR gain tables were not produced (requires both preprocessing modes).")
    if not denoise_gain_per_trial.empty:
        print("  ", denoise_gain_trial_path)
        print("  ", denoise_gain_summary_condition_path)
        print("  ", denoise_gain_summary_band_path)
    else:
        print("  Denoising SNR improvement tables were not produced (missing/invalid --denoised-root or no matched rows).")

    if roi_load_error is not None:
        print("\nNote: ROI pickle could not be loaded in this environment; reused existing detail CSV instead.")
        print(f"  ROI load error: {roi_load_error}")
        print(f"  Detail CSV: {str(detail_csv_path)}")

    print("\nOverall band summary:")
    print(summaries["summary_overall_band"].to_string(index=False))

if __name__ == "__main__":
    main()
