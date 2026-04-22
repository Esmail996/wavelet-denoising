"""
Multiband wavelet denoising for Mic2 only.

For each trial:
  - Preprocess 3 times (once for each band: 40K, 50K, 60K)
  - Denoise each preprocessed signal using its corresponding SNR
  - Output 3 denoised signals per trial (one per band)
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from wavelet_denoising import (
    paper_threshold_denoise_signal,
    universal_soft_denoise_signal,
    preprocess_signal_for_denoising,
    MIC_ORDER,
)

# Constants
DEFAULT_SNR_DB = 0.0
DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 5
DEFAULT_MODE = "symmetric"
DEFAULT_ETA = 10.0
DEFAULT_SIGMA = 1.5
DEFAULT_GAMMA = 0.09
DEFAULT_FS_HZ = 2e6
DEFAULT_BP_BW_HZ = 5000
DEFAULT_BP_METHOD = "iir"
DEFAULT_BP_IIR_ORDER = 5
DEFAULT_BP_FIR_NUMTAPS = 101
DEFAULT_BP_FIR_WINDOW = "hamming"
AVAILABLE_BANDS = [40000.0, 50000.0, 60000.0]
BAND_WAVELET_LEVEL = {
    40000.0: ("bior1.1", 6),
    50000.0: ("bior1.3", 6),
    60000.0: ("db4", 5),
}


def denoise_mic2_multiband(
    signal: np.ndarray,
    file_name: str,
    trial_idx: int,
    snr_values: dict[float, float],  # band_hz -> snr_db
    wavelet: str,
    level: int,
    mode: str,
    eta: float,
    sigma: float,
    gamma: float,
    fs_hz: float,
    bp_bw_hz: float,
    bp_method: str,
    bp_iir_order: int,
    bp_fir_numtaps: int,
    bp_fir_window: str,
    use_mad_lambda: bool = False,
    lambda_scale: float = 1.0,
    threshold_method: str = "universal_soft",
) -> dict[float, dict]:
    """
    Denoise a Mic2 signal for all 3 bands.
    
    Returns:
        dict: {band_hz -> {denoised_signal, info_dict}}
    """
    results = {}
    
    for band_hz in AVAILABLE_BANDS:
        wavelet_used, level_used = BAND_WAVELET_LEVEL.get(float(band_hz), (wavelet, level))

        # Preprocess for this band
        signal_preprocessed = preprocess_signal_for_denoising(
            signal=signal,
            preprocessing="bandpassed",
            fs_hz=fs_hz,
            band_center_hz=band_hz,
            bw_hz=bp_bw_hz,
            bp_method=bp_method,
            iir_order=bp_iir_order,
            fir_numtaps=bp_fir_numtaps,
            fir_window=bp_fir_window,
        )
        
        # Get SNR for this band
        snr_db = snr_values.get(band_hz, DEFAULT_SNR_DB)
        
        # Threshold choice and method:
        # - "universal_soft": universal threshold with soft-thresholding
        # - "paper_eq10": original paper nonlinear thresholding
        if str(threshold_method) == "paper_eq10":
            denoised_signal, info = paper_threshold_denoise_signal(
                signal=signal_preprocessed,
                snr_db=snr_db,
                wavelet=wavelet_used,
                level=level_used,
                mode=mode,
                eta=eta,
                sigma=sigma,
                gamma=gamma,
                lambda_source="mad_signal" if use_mad_lambda else "eq13_snr",
                lambda_scale=lambda_scale,
            )
        else:
            denoised_signal, info = universal_soft_denoise_signal(
                signal=signal_preprocessed,
                wavelet=wavelet_used,
                level=level_used,
                mode=mode,
            )
            info["snr_db_used"] = float(snr_db)
            info["threshold_method"] = "universal_soft"
        
        results[band_hz] = {
            "signal": denoised_signal,
            "info": {**info, "band_hz": band_hz, "snr_db_used": snr_db},
        }
    
    return results


def process_pickle_multiband(
    pickle_path: Path,
    file_name: str,
    snr_df: pd.DataFrame,
    wavelet: str,
    level: int,
    mode: str,
    eta: float,
    sigma: float,
    gamma: float,
    fs_hz: float,
    bp_bw_hz: float,
    bp_method: str,
    bp_iir_order: int,
    bp_fir_numtaps: int,
    bp_fir_window: str,
    lambda_scale: float,
    threshold_method: str,
) -> tuple[dict[float, list], list[dict]]:
    """
    Process a pickle file for Mic2 only, all bands.
    
    Returns:
        (denoised_dict, summary_rows)
        denoised_dict: {band_hz -> [denoised_signal_per_trial]}
        summary_rows: List of dicts with metadata for each (trial, band) combination
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    # Extract Mic2 data
    if isinstance(data, pd.DataFrame):
        # DataFrame format
        if "Mic2" not in data.columns:
            raise ValueError("Mic2 not found in DataFrame columns")
        mic2_data = data["Mic2"].values  # shape (n_trials,) with each element as signal
    elif isinstance(data, np.ndarray):
        # Array format: shape (n_mics, n_trials, signal_length)
        if data.ndim != 3 or data.shape[0] < 3:
            raise ValueError(f"Unexpected pickle shape: {data.shape}")
        mic2_idx = 1  # Mic2 is index 1
        mic2_data = data[mic2_idx]  # shape (n_trials, signal_length)
    elif isinstance(data, dict):
        # Dict format (converted from DataFrame)
        if "Mic2" not in data:
            raise ValueError("Mic2 not found in pickle dict")
        mic2_data = data["Mic2"]  # list or array
    else:
        raise ValueError(f"Unsupported pickle format: {type(data)}")
    
    n_trials = len(mic2_data)
    is_reference_file = str(file_name).lower() == "referenz.pickle"
    denoised_dict = {band_hz: [] for band_hz in AVAILABLE_BANDS}
    summary_rows = []
    
    for trial_idx in range(n_trials):
        signal = np.asarray(mic2_data[trial_idx])
        
        # Get SNR values for this trial from SNR table
        snr_values = {}
        for band_hz in AVAILABLE_BANDS:
            key_rows = snr_df[
                (snr_df["filename"] == file_name)
                & (snr_df["trial"] == trial_idx)
                & (np.isclose(snr_df["band_center_hz"].fillna(-1), band_hz))
            ]
            if not key_rows.empty:
                snr_db = float(key_rows.iloc[0]["snr_db"])
            else:
                snr_db = DEFAULT_SNR_DB
            snr_values[band_hz] = snr_db
        
        # Denoise for all bands
        results = denoise_mic2_multiband(
            signal=signal,
            file_name=file_name,
            trial_idx=trial_idx,
            snr_values=snr_values,
            wavelet=wavelet,
            level=level,
            mode=mode,
            eta=eta,
            sigma=sigma,
            gamma=gamma,
            fs_hz=fs_hz,
            bp_bw_hz=bp_bw_hz,
            bp_method=bp_method,
            bp_iir_order=bp_iir_order,
            bp_fir_numtaps=bp_fir_numtaps,
            bp_fir_window=bp_fir_window,
            use_mad_lambda=is_reference_file,
            lambda_scale=lambda_scale,
            threshold_method=threshold_method,
        )
        
        # Collect results
        for band_hz, result in results.items():
            denoised_dict[band_hz].append(result["signal"])
            summary_rows.append(
                {
                    "file": file_name,
                    "trial": trial_idx,
                    "band_hz": band_hz,
                    "mic": "Mic2",
                    **result["info"],
                }
            )
    
    return denoised_dict, summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Multiband wavelet denoising for Mic2 only"
    )
    parser.add_argument(
        "--snr-csv",
        type=str,
        default="outputs/snr_estimation/snr_per_trial_reference_roi.csv",
        help="Path to SNR CSV file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="Multifrequenz Dataset/Multifrequenz",
        help="Input directory with pickle files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Multifrequenz Dataset/Multifrequenz_denoised_multiband_mic2",
        help="Output directory for denoised pickles",
    )
    parser.add_argument("--wavelet", type=str, default=DEFAULT_WAVELET)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--fs-hz", type=float, default=DEFAULT_FS_HZ)
    parser.add_argument("--bp-bw-hz", type=float, default=DEFAULT_BP_BW_HZ)
    parser.add_argument("--bp-method", type=str, default=DEFAULT_BP_METHOD)
    parser.add_argument("--bp-iir-order", type=int, default=DEFAULT_BP_IIR_ORDER)
    parser.add_argument("--bp-fir-numtaps", type=int, default=DEFAULT_BP_FIR_NUMTAPS)
    parser.add_argument("--bp-fir-window", type=str, default=DEFAULT_BP_FIR_WINDOW)
    parser.add_argument("--lambda-scale", type=float, default=1.0, help="Scale factor applied to lambda (lower values reduce denoising strength).")
    parser.add_argument("--threshold-method", type=str, choices=["universal_soft", "paper_eq10"], default="universal_soft")
    
    args = parser.parse_args()
    
    # Load SNR table
    snr_df = pd.read_csv(args.snr_csv)
    print(f"Loaded SNR CSV: {args.snr_csv}")
    print(f"  Bands: {sorted([b for b in snr_df['band_center_hz'].unique() if pd.notna(b)])}")
    print(f"  Mics: {snr_df['mic'].unique()}")
    print(f"  Rows: {len(snr_df)}")
    
    # Filter to only bandpassed preprocessing
    snr_df_bp = snr_df[snr_df["preprocessing"] == "bandpassed"].copy()
    print(f"Using {len(snr_df_bp)} rows with preprocessing='bandpassed'")
    
    # Discover input files
    input_dir = Path(args.input_dir)
    pickle_files = sorted(input_dir.glob("**/*.pickle"))
    print(f"\nFound {len(pickle_files)} pickle files")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    all_summary_rows = []
    files_processed = 0
    
    for pickle_path in pickle_files:
        rel_path = pickle_path.relative_to(input_dir)
        file_name = pickle_path.stem + ".pickle"  # e.g., "100cm_-10Grad.pickle"
        
        # Create output structure
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {rel_path}")
        
        try:
            denoised_dict, summary_rows = process_pickle_multiband(
                pickle_path=pickle_path,
                file_name=file_name,
                snr_df=snr_df_bp,
                wavelet=args.wavelet,
                level=args.level,
                mode=args.mode,
                eta=args.eta,
                sigma=args.sigma,
                gamma=args.gamma,
                fs_hz=args.fs_hz,
                bp_bw_hz=args.bp_bw_hz,
                bp_method=args.bp_method,
                bp_iir_order=args.bp_iir_order,
                bp_fir_numtaps=args.bp_fir_numtaps,
                bp_fir_window=args.bp_fir_window,
                lambda_scale=float(args.lambda_scale),
                threshold_method=str(args.threshold_method),
            )
            
            all_summary_rows.extend(summary_rows)
            
            # Save denoised data: organize as {band_hz -> [signal_per_trial]}
            output_pickle = output_subdir / pickle_path.name
            with open(output_pickle, "wb") as f:
                pickle.dump(denoised_dict, f)
            
            n_trials = len(denoised_dict[AVAILABLE_BANDS[0]])
            print(f"  OK - Denoised {n_trials} trials x 3 bands = {n_trials * 3} signals")
            files_processed += 1
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save summary CSV
    output_csv = output_dir / "multiband_denoising_summary.csv"
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(output_csv, index=False)
    print(f"\nOK - Saved summary: {output_csv} ({len(summary_df)} rows)")
    
    print(f"\nComplete: {files_processed} files processed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
