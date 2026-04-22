import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import detrend

from bandpass import bandpass


MIC_ORDER = ["Mic1", "Mic2", "Mic3"]
DEFAULT_DATA_ROOT = Path("Multifrequenz Dataset") / "Multifrequenz"
DEFAULT_OUTPUT_ROOT = Path("Multifrequenz Dataset") / "Multifrequenz_denoised_db4_L5_paper"
DEFAULT_SUMMARY_CSV = Path("outputs") / "wavelet_denoising" / "full_signal_denoising_summary_paper.csv"
DEFAULT_SNR_CSV = Path("outputs") / "snr_estimation" / "snr_per_trial_reference_roi.csv"

WAVELET = "db4"
DECOMPOSITION_LEVEL = 5
MODE = "symmetric"

ETA = 10.0
SIGMA = 1.5
GAMMA = 0.09
DEFAULT_SNR_DB = 0.0
EPS = 1e-12
FS_HZ = 2_000_000.0
BP_BW_HZ = 5_000.0
BP_METHOD = "iir"
BP_IIR_ORDER = 2
BP_FIR_NUMTAPS = 71
BP_FIR_WINDOW = "hamming"


def iter_pickle_files(data_root: Path):
    for category_dir in sorted([path for path in data_root.iterdir() if path.is_dir()]):
        for file_path in sorted(category_dir.glob("*.pickle")):
            yield category_dir.name, file_path


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def flatten_coeffs(coeffs: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int, tuple[int, ...]]]]:
    flat_parts = []
    layout = []
    cursor = 0
    for idx, arr in enumerate(coeffs):
        arr = np.asarray(arr, dtype=float)
        n_values = arr.size
        flat_parts.append(arr.ravel())
        layout.append((idx, cursor, cursor + n_values, arr.shape))
        cursor += n_values
    flat = np.concatenate(flat_parts) if flat_parts else np.array([], dtype=float)
    return flat, layout


def unflatten_coeffs(flat: np.ndarray, layout: list[tuple[int, int, int, tuple[int, ...]]]) -> list[np.ndarray]:
    coeffs = []
    for _, start, end, shape in layout:
        coeffs.append(np.asarray(flat[start:end], dtype=float).reshape(shape))
    return coeffs


def risk_vector_from_flat_coeffs(flat_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    abs_sorted = np.sort(np.abs(np.asarray(flat_coeffs, dtype=float)))
    varpi = abs_sorted ** 2
    n_coeffs = len(varpi)
    if n_coeffs == 0:
        raise ValueError("No coefficients to process.")

    risk = np.zeros(n_coeffs, dtype=float)
    cumsum = np.cumsum(varpi)
    for index in range(1, n_coeffs + 1):
        risk[index - 1] = (
            n_coeffs
            - 2 * index
            + (n_coeffs - index) * varpi[index - 1]
            + cumsum[index - 1]
        ) / n_coeffs

    idx_min = int(np.argmin(risk))
    omega_min = float(abs_sorted[idx_min])
    return abs_sorted, varpi, risk, omega_min


def lambda_eq13_literal(omega_min: float, snr_used: float) -> float:
    lam = np.sqrt(omega_min / ((1.0 + np.sqrt(np.exp(snr_used))) ** 2))
    return float(lam)


def lambda_mad_signal(signal: np.ndarray) -> float:
    """Compute lambda as MAD of the signal (robust scale estimate)."""
    x = np.asarray(signal, dtype=float).ravel()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(mad, EPS))


def preprocess_signal_for_denoising(
    signal: np.ndarray,
    preprocessing: str,
    fs_hz: float,
    band_center_hz: float | None,
    bw_hz: float,
    bp_method: str,
    iir_order: int,
    fir_numtaps: int,
    fir_window: str,
) -> np.ndarray:
    x = np.asarray(signal, dtype=float).ravel()
    x = x - np.mean(x)
    x = detrend(x, type="linear")

    if str(preprocessing) == "bandpassed":
        if band_center_hz is None:
            raise ValueError("band_center_hz must be provided when preprocessing='bandpassed'.")
        x = bandpass(
            x,
            fs=fs_hz,
            f0=float(band_center_hz),
            bw=float(bw_hz),
            order=int(iir_order),
            method=str(bp_method),
            numtaps=int(fir_numtaps),
            window=str(fir_window),
        )

    return np.asarray(x, dtype=float)


def improved_threshold_eq10(
    flat_coeffs: np.ndarray,
    lam: float,
    eta: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    w = np.asarray(flat_coeffs, dtype=float)
    absw = np.abs(w)
    sgn = np.sign(w)

    out = np.zeros_like(w)

    mask_hi = absw > lam
    out[mask_hi] = sgn[mask_hi] * (
        absw[mask_hi]
        - (lam / (1.0 + sigma)) * gamma ** np.sqrt(absw[mask_hi] ** 2 - lam ** 2)
    )

    mask_lo = ~mask_hi
    out[mask_lo] = sgn[mask_lo] * (
        (sigma / (1.0 + sigma))
        * np.exp(eta * (absw[mask_lo] - lam))
        * abs(lam)
    )

    return out


def paper_threshold_denoise_signal(
    signal: np.ndarray,
    snr_db: float = DEFAULT_SNR_DB,
    wavelet: str = WAVELET,
    level: int = DECOMPOSITION_LEVEL,
    mode: str = MODE,
    eta: float = ETA,
    sigma: float = SIGMA,
    gamma: float = GAMMA,
    lambda_source: str = "eq13_snr",
    lambda_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    x = np.asarray(signal, dtype=float).ravel()
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=wavelet_obj.dec_len)
    if max_level < 1:
        raise ValueError(f"Signal too short for wavelet '{wavelet}'.")

    level_used = min(level, max_level)
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level_used, mode=mode)
    flat, layout = flatten_coeffs(coeffs)

    _, _, risk, omega_min = risk_vector_from_flat_coeffs(flat)
    if str(lambda_source) == "mad_signal":
        lam = lambda_mad_signal(x)
    else:
        lam = lambda_eq13_literal(omega_min=omega_min, snr_used=snr_db)
    lam = float(max(EPS, lam * float(lambda_scale)))
    flat_thresholded = improved_threshold_eq10(flat, lam=lam, eta=eta, sigma=sigma, gamma=gamma)
    coeffs_thresholded = unflatten_coeffs(flat_thresholded, layout)

    denoised = pywt.waverec(coeffs_thresholded, wavelet=wavelet, mode=mode)
    denoised = np.asarray(denoised[: len(x)], dtype=float)

    info = {
        "wavelet": wavelet,
        "level_used": int(level_used),
        "snr_db_used": float(snr_db),
        "omega_min": float(omega_min),
        "lambda": float(lam),
        "risk_min": float(np.min(risk)),
        "risk_argmin_1based": int(np.argmin(risk) + 1),
        "eta": float(eta),
        "sigma": float(sigma),
        "gamma": float(gamma),
        "lambda_source": str(lambda_source),
        "lambda_scale": float(lambda_scale),
        "coeff_rms_before": rms(flat),
        "coeff_rms_after": rms(flat_thresholded),
        "signal_rms_before": rms(x),
        "signal_rms_after": rms(denoised),
    }
    return denoised, info


def universal_soft_denoise_signal(
    signal: np.ndarray,
    wavelet: str = WAVELET,
    level: int = DECOMPOSITION_LEVEL,
    mode: str = MODE,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Wavelet denoising using universal threshold with soft-thresholding."""
    x = np.asarray(signal, dtype=float).ravel()
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=wavelet_obj.dec_len)
    if max_level < 1:
        raise ValueError(f"Signal too short for wavelet '{wavelet}'.")

    level_used = min(level, max_level)
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level_used, mode=mode)

    # Estimate noise sigma from finest-detail coefficients via MAD.
    detail_finest = np.asarray(coeffs[-1], dtype=float)
    sigma_hat = float(np.median(np.abs(detail_finest)) / 0.6745) if detail_finest.size else 0.0
    sigma_hat = max(sigma_hat, EPS)
    uthresh = float(sigma_hat * np.sqrt(2.0 * np.log(max(len(x), 2))))

    coeffs_thr = [np.asarray(coeffs[0], dtype=float)]
    for c in coeffs[1:]:
        coeffs_thr.append(pywt.threshold(np.asarray(c, dtype=float), value=uthresh, mode="soft"))

    denoised = pywt.waverec(coeffs_thr, wavelet=wavelet, mode=mode)
    denoised = np.asarray(denoised[: len(x)], dtype=float)

    flat_before, _ = flatten_coeffs([np.asarray(c, dtype=float) for c in coeffs])
    flat_after, _ = flatten_coeffs([np.asarray(c, dtype=float) for c in coeffs_thr])

    info = {
        "wavelet": wavelet,
        "level_used": int(level_used),
        "snr_db_used": float(np.nan),
        "omega_min": float(np.nan),
        "lambda": float(uthresh),
        "risk_min": float(np.nan),
        "risk_argmin_1based": int(0),
        "eta": float(np.nan),
        "sigma": float(np.nan),
        "gamma": float(np.nan),
        "lambda_source": "universal_soft",
        "lambda_scale": float(1.0),
        "coeff_rms_before": rms(flat_before),
        "coeff_rms_after": rms(flat_after),
        "signal_rms_before": rms(x),
        "signal_rms_after": rms(denoised),
    }
    return denoised, info


def build_trial_snr_lookup(
    snr_df: pd.DataFrame,
    preprocessing: str,
    band_center_hz: float | None,
    auto_band_from_snr: bool,
) -> tuple[dict[tuple[str, str, int], float], dict[tuple[str, str, int], float | None]]:
    required_columns = {"filename", "mic", "trial", "preprocessing", "snr_db"}
    missing = required_columns.difference(snr_df.columns)
    if missing:
        raise ValueError(f"SNR CSV missing required columns: {sorted(missing)}")

    filtered = snr_df[snr_df["preprocessing"].astype(str) == str(preprocessing)].copy()
    if filtered.empty:
        raise ValueError(f"No SNR rows found for preprocessing={preprocessing!r}.")

    if preprocessing == "bandpassed":
        if band_center_hz is None:
            raise ValueError("band_center_hz must be provided when preprocessing='bandpassed'.")
        if "band_center_hz" not in filtered.columns:
            raise ValueError("SNR CSV does not include 'band_center_hz' column.")

    lookup: dict[tuple[str, str, int], float] = {}
    band_lookup: dict[tuple[str, str, int], float | None] = {}

    if preprocessing == "bandpassed" and auto_band_from_snr:
        idx_best = filtered.groupby(["filename", "mic", "trial"], dropna=False)["snr_db"].idxmax()
        selected = filtered.loc[idx_best]
        for _, row in selected.iterrows():
            key = (str(row["filename"]), str(row["mic"]), int(row["trial"]))
            lookup[key] = float(row["snr_db"])
            band_lookup[key] = float(row["band_center_hz"])
        return lookup, band_lookup

    if preprocessing == "bandpassed":
        band_value = float(band_center_hz)
        filtered = filtered[np.isclose(filtered["band_center_hz"].astype(float), band_value)]
        if filtered.empty:
            raise ValueError(f"No SNR rows found for band_center_hz={band_value}.")

    for _, row in filtered.iterrows():
        key = (str(row["filename"]), str(row["mic"]), int(row["trial"]))
        lookup[key] = float(row["snr_db"])
        band_lookup[key] = float(row["band_center_hz"]) if preprocessing == "bandpassed" else None
    return lookup, band_lookup


def denoise_dataframe_object(
    df: pd.DataFrame,
    file_name: str,
    wavelet: str,
    level: int,
    mode: str,
    snr_lookup: dict[tuple[str, str, int], float],
    snr_band_lookup: dict[tuple[str, str, int], float | None],
    fallback_snr_db: float,
    eta: float,
    sigma: float,
    gamma: float,
    fs_hz: float,
    snr_preprocessing: str,
    snr_band_hz: float | None,
    bp_bw_hz: float,
    bp_method: str,
    bp_iir_order: int,
    bp_fir_numtaps: int,
    bp_fir_window: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    denoised_df = df.copy(deep=True)
    summary_rows: list[dict[str, object]] = []

    col_lookup = {column.lower(): column for column in denoised_df.columns}
    missing = [mic for mic in MIC_ORDER if mic.lower() not in col_lookup]
    if missing:
        raise ValueError(f"Missing microphone columns: {missing}. Available columns: {list(denoised_df.columns)}")

    for row_idx in range(len(denoised_df)):
        for mic in MIC_ORDER:
            column = col_lookup[mic.lower()]
            key = (file_name, mic, int(row_idx))
            snr_db = float(snr_lookup.get(key, fallback_snr_db))
            band_hz_used = snr_band_lookup.get(key, snr_band_hz)
            if str(snr_preprocessing) == "bandpassed" and band_hz_used is None:
                band_hz_used = snr_band_hz
            signal_preprocessed = preprocess_signal_for_denoising(
                signal=denoised_df.at[row_idx, column],
                preprocessing=snr_preprocessing,
                fs_hz=fs_hz,
                band_center_hz=band_hz_used,
                bw_hz=bp_bw_hz,
                bp_method=bp_method,
                iir_order=bp_iir_order,
                fir_numtaps=bp_fir_numtaps,
                fir_window=bp_fir_window,
            )
            denoised_signal, info = paper_threshold_denoise_signal(
                signal=signal_preprocessed,
                snr_db=snr_db,
                wavelet=wavelet,
                level=level,
                mode=mode,
                eta=eta,
                sigma=sigma,
                gamma=gamma,
            )
            denoised_df.at[row_idx, column] = denoised_signal

            summary_rows.append(
                {
                    "trial": int(row_idx),
                    "mic": mic,
                    "snr_band_hz_used": band_hz_used,
                    **info,
                }
            )

    return denoised_df, summary_rows


def denoise_array_object(
    arr: np.ndarray,
    file_name: str,
    wavelet: str,
    level: int,
    mode: str,
    snr_lookup: dict[tuple[str, str, int], float],
    snr_band_lookup: dict[tuple[str, str, int], float | None],
    fallback_snr_db: float,
    eta: float,
    sigma: float,
    gamma: float,
    fs_hz: float,
    snr_preprocessing: str,
    snr_band_hz: float | None,
    bp_bw_hz: float,
    bp_method: str,
    bp_iir_order: int,
    bp_fir_numtaps: int,
    bp_fir_window: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    object_array = np.asarray(arr, dtype=object)
    if object_array.ndim != 2 or object_array.shape[0] < 3:
        raise ValueError(f"Unsupported pickle array shape: {object_array.shape}")

    denoised = object_array.copy()
    summary_rows: list[dict[str, object]] = []

    for mic_idx, mic in enumerate(MIC_ORDER):
        for trial_idx in range(object_array.shape[1]):
            key = (file_name, mic, int(trial_idx))
            snr_db = float(snr_lookup.get(key, fallback_snr_db))
            band_hz_used = snr_band_lookup.get(key, snr_band_hz)
            if str(snr_preprocessing) == "bandpassed" and band_hz_used is None:
                band_hz_used = snr_band_hz
            signal_preprocessed = preprocess_signal_for_denoising(
                signal=object_array[mic_idx, trial_idx],
                preprocessing=snr_preprocessing,
                fs_hz=fs_hz,
                band_center_hz=band_hz_used,
                bw_hz=bp_bw_hz,
                bp_method=bp_method,
                iir_order=bp_iir_order,
                fir_numtaps=bp_fir_numtaps,
                fir_window=bp_fir_window,
            )
            denoised_signal, info = paper_threshold_denoise_signal(
                signal=signal_preprocessed,
                snr_db=snr_db,
                wavelet=wavelet,
                level=level,
                mode=mode,
                eta=eta,
                sigma=sigma,
                gamma=gamma,
            )
            denoised[mic_idx, trial_idx] = denoised_signal

            summary_rows.append(
                {
                    "trial": int(trial_idx),
                    "mic": mic,
                    "snr_band_hz_used": band_hz_used,
                    **info,
                }
            )

    return denoised, summary_rows


def denoise_pickle_object(
    obj,
    file_name: str,
    wavelet: str,
    level: int,
    mode: str,
    snr_lookup: dict[tuple[str, str, int], float],
    snr_band_lookup: dict[tuple[str, str, int], float | None],
    fallback_snr_db: float,
    eta: float,
    sigma: float,
    gamma: float,
    fs_hz: float,
    snr_preprocessing: str,
    snr_band_hz: float | None,
    bp_bw_hz: float,
    bp_method: str,
    bp_iir_order: int,
    bp_fir_numtaps: int,
    bp_fir_window: str,
):
    if isinstance(obj, pd.DataFrame):
        return denoise_dataframe_object(
            obj,
            file_name,
            wavelet,
            level,
            mode,
            snr_lookup,
            snr_band_lookup,
            fallback_snr_db,
            eta,
            sigma,
            gamma,
            fs_hz,
            snr_preprocessing,
            snr_band_hz,
            bp_bw_hz,
            bp_method,
            bp_iir_order,
            bp_fir_numtaps,
            bp_fir_window,
        )
    return denoise_array_object(
        obj,
        file_name,
        wavelet,
        level,
        mode,
        snr_lookup,
        snr_band_lookup,
        fallback_snr_db,
        eta,
        sigma,
        gamma,
        fs_hz,
        snr_preprocessing,
        snr_band_hz,
        bp_bw_hz,
        bp_method,
        bp_iir_order,
        bp_fir_numtaps,
        bp_fir_window,
    )


def process_dataset(
    data_root: Path,
    output_root: Path,
    summary_csv: Path,
    snr_csv: Path,
    snr_preprocessing: str,
    snr_band_hz: float | None,
    auto_band_from_snr: bool,
    wavelet: str,
    level: int,
    mode: str,
    fallback_snr_db: float,
    eta: float,
    sigma: float,
    gamma: float,
    fs_hz: float,
    bp_bw_hz: float,
    bp_method: str,
    bp_iir_order: int,
    bp_fir_numtaps: int,
    bp_fir_window: str,
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    file_count = 0

    snr_df = pd.read_csv(snr_csv)
    snr_lookup, snr_band_lookup = build_trial_snr_lookup(
        snr_df=snr_df,
        preprocessing=snr_preprocessing,
        band_center_hz=snr_band_hz,
        auto_band_from_snr=auto_band_from_snr,
    )

    for category, file_path in iter_pickle_files(data_root):
        file_count += 1
        obj = pd.read_pickle(file_path)

        denoised_obj, file_summary_rows = denoise_pickle_object(
            obj=obj,
            file_name=file_path.name,
            wavelet=wavelet,
            level=level,
            mode=mode,
            snr_lookup=snr_lookup,
            snr_band_lookup=snr_band_lookup,
            fallback_snr_db=fallback_snr_db,
            eta=eta,
            sigma=sigma,
            gamma=gamma,
            fs_hz=fs_hz,
            snr_preprocessing=snr_preprocessing,
            snr_band_hz=snr_band_hz,
            bp_bw_hz=bp_bw_hz,
            bp_method=bp_method,
            bp_iir_order=bp_iir_order,
            bp_fir_numtaps=bp_fir_numtaps,
            bp_fir_window=bp_fir_window,
        )

        output_path = output_root / category / file_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(denoised_obj, output_path)

        for row in file_summary_rows:
            summary_rows.append(
                {
                    "category": category,
                    "file": file_path.name,
                    "snr_preprocessing": snr_preprocessing,
                    "snr_band_hz": snr_band_hz,
                    **row,
                }
            )

        print(f"Processed {file_count}: {file_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    return summary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the paper threshold denoising method to the complete full-signal dataset "
            "and save denoised full-signal pickle files with the same structure as the originals."
        )
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Root folder containing full-signal pickle files.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output folder for denoised full-signal pickle files.")
    parser.add_argument("--summary-csv", type=str, default=str(DEFAULT_SUMMARY_CSV), help="CSV path for denoising summary metrics.")
    parser.add_argument("--snr-csv", type=str, default=str(DEFAULT_SNR_CSV), help="Per-trial SNR CSV (from snr_estimation.py).")
    parser.add_argument("--snr-preprocessing", type=str, choices=["bandpassed", "detrended_only"], default="bandpassed", help="Which SNR mode to use for lambda.")
    parser.add_argument("--snr-band-hz", type=float, default=50_000.0, help="Band center to use when --snr-preprocessing=bandpassed.")
    parser.add_argument("--auto-band-from-snr", action="store_true", help="For bandpassed mode, auto-pick per-signal band from SNR CSV by max snr_db over 40/50/60 kHz.")
    parser.add_argument("--wavelet", type=str, default=WAVELET, help="Wavelet name.")
    parser.add_argument("--level", type=int, default=DECOMPOSITION_LEVEL, help="Requested decomposition level.")
    parser.add_argument("--mode", type=str, default=MODE, help="Wavelet extension mode.")
    parser.add_argument("--snr-db", type=float, default=DEFAULT_SNR_DB, help="Fallback SNR if a (file,mic,trial) key is missing in --snr-csv.")
    parser.add_argument("--eta", type=float, default=ETA, help="Paper Eq. (10) eta parameter.")
    parser.add_argument("--sigma", type=float, default=SIGMA, help="Paper Eq. (10) sigma parameter.")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Paper Eq. (10) gamma parameter.")
    parser.add_argument("--fs-hz", type=float, default=FS_HZ, help="Sampling frequency used for preprocessing.")
    parser.add_argument("--bp-bw-hz", type=float, default=BP_BW_HZ, help="Bandpass half-bandwidth in Hz for preprocessing.")
    parser.add_argument("--bp-method", type=str, choices=["fir", "iir"], default=BP_METHOD, help="Bandpass method for preprocessing.")
    parser.add_argument("--bp-iir-order", type=int, default=BP_IIR_ORDER, help="IIR order for preprocessing bandpass.")
    parser.add_argument("--bp-fir-numtaps", type=int, default=BP_FIR_NUMTAPS, help="FIR taps for preprocessing bandpass.")
    parser.add_argument("--bp-fir-window", type=str, default=BP_FIR_WINDOW, help="FIR window for preprocessing bandpass.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    here = Path(__file__).resolve().parent
    data_root = here / args.data_root
    output_root = here / args.output_root
    summary_csv = here / args.summary_csv
    snr_csv = here / args.snr_csv
    snr_band_hz = float(args.snr_band_hz) if str(args.snr_preprocessing) == "bandpassed" else None

    summary_df = process_dataset(
        data_root=data_root,
        output_root=output_root,
        summary_csv=summary_csv,
        snr_csv=snr_csv,
        snr_preprocessing=str(args.snr_preprocessing),
        snr_band_hz=snr_band_hz,
        auto_band_from_snr=bool(args.auto_band_from_snr),
        wavelet=str(args.wavelet),
        level=int(args.level),
        mode=str(args.mode),
        fallback_snr_db=float(args.snr_db),
        eta=float(args.eta),
        sigma=float(args.sigma),
        gamma=float(args.gamma),
        fs_hz=float(args.fs_hz),
        bp_bw_hz=float(args.bp_bw_hz),
        bp_method=str(args.bp_method),
        bp_iir_order=int(args.bp_iir_order),
        bp_fir_numtaps=int(args.bp_fir_numtaps),
        bp_fir_window=str(args.bp_fir_window),
    )

    print("Saved:")
    print(f"  - {output_root}")
    print(f"  - {summary_csv}")
    print(f"  - Files processed: {summary_df['file'].nunique() if not summary_df.empty else 0}")
    print(f"  - Signals processed: {len(summary_df)}")


if __name__ == "__main__":
    main()