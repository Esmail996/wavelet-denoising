from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft, detrend

# =========================================================
# USER SETTINGS
# =========================================================
#DATA_DIR = Path("./Multifrequenz Dataset/Multifrequenz")
DATA_DIR = Path("./Multifrequenz Dataset")  # change if your dataset is in a different location
OUT_DIR = Path("./outputs/stft")
FS_HZ = 2_000_000   # change if you want to use your final confirmed fs
MIC = "Mic2"        # Mic1, Mic2, Mic3
MODE = "mean"       # "mean" or "trial"
TRIAL_INDEX = 0     # only used if MODE == "trial"

# STFT parameters
NPERSEG = 512
NOVERLAP = 384
NFFT = 1024

# Frequency display range in Hz
FMIN = 1
FMAX = 100_000 


# =========================================================
# HELPERS
# =========================================================
def parse_distance_angle(file_path: Path):
    """
    Extract distance and angle from filenames like:
    25cm_-5Grad.pickle
    """
    m = re.search(r'(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad', file_path.stem)
    if m:
        dist_cm = float(m.group("dist"))
        ang_deg = float(m.group("ang"))
        return dist_cm, ang_deg
    return None, None


def load_pickle_signals(file_path: Path, mic="Mic2"):
    """
    Load one pickle file and return all trials for the selected mic
    as a 2D numpy array with shape: (n_trials, n_samples)

    Expected file format:
    - pandas DataFrame
    - columns include Mic1, Mic2, Mic3
    - each row is one trial
    - each cell contains a 1D signal array of length ~18000
    """
    df = pd.read_pickle(file_path)

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{file_path} is not a pandas DataFrame.")

    if mic not in df.columns:
        raise ValueError(f"{file_path} does not contain column {mic}.")

    signals = []
    for i in range(len(df)):
        x = np.asarray(df.iloc[i][mic], dtype=float).squeeze()
        if x.ndim != 1:
            raise ValueError(f"{file_path} row {i} {mic} is not 1D.")
        signals.append(x)

    signals = np.vstack(signals)  # shape: (n_trials, n_samples)
    return df, signals


def load_npy_signals(file_path: Path):
    """
    Load one .npy file and return the signal(s).
    
    Expected file format:
    - 1D numpy array: shape (n_samples,) -> returns as 2D array (1, n_samples)
    - 2D numpy array: shape (n_trials, n_samples) -> returns as-is
    
    Returns:
    - None (no DataFrame for .npy files)
    - signals as 2D numpy array with shape: (n_trials, n_samples)
    """
    data = np.load(file_path)
    
    signals = np.asarray(data, dtype=float)
    
    # If 1D, reshape to (1, n_samples) for consistency
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    elif signals.ndim != 2:
        raise ValueError(f"{file_path} must be 1D or 2D array, got {signals.ndim}D.")
    
    return None, signals


def load_signals(file_path: Path, mic="Mic2"):
    """
    Load signals from either .pickle or .npy file.
    Returns tuple: (df_or_none, signals_2d)
    """
    if file_path.suffix == ".pickle":
        return load_pickle_signals(file_path, mic=mic)
    elif file_path.suffix == ".npy":
        return load_npy_signals(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def choose_signal(signals, mode="mean", trial_index=0):
    """
    Choose how to represent the file.
    mode = "mean"  -> average over all trials
    mode = "trial" -> use one selected trial
    """
    if mode == "mean":
        return np.mean(signals, axis=0)
    elif mode == "trial":
        if not (0 <= trial_index < len(signals)):
            raise IndexError(f"trial_index={trial_index} out of range for {len(signals)} trials.")
        return signals[trial_index]
    else:
        raise ValueError("mode must be 'mean' or 'trial'")


def compute_stft(signal, fs, nperseg=512, noverlap=384, nfft=1024):
    """
    Compute STFT and return frequency, time, and power.
    """
    x = detrend(np.asarray(signal, dtype=float))

    f, t, Zxx = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None
    )

    power = np.abs(Zxx) ** 2
    power_db = 10 * np.log10(power + 1e-12)
    return f, t, power_db


def plot_and_save_stft(f, t, power_db, out_path, title, fmin=20_000, fmax=80_000):
    """
    Plot STFT spectrogram and save to disk.
    """
    freq_mask = (f >= fmin) & (f <= fmax)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        t * 1e3,                 # seconds -> ms
        f[freq_mask] / 1e3,      # Hz -> kHz
        power_db[freq_mask, :],
        shading="gouraud"
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (kHz)")
    plt.title(title)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN DATASET PIPELINE
# =========================================================
def process_dataset_stft(
    data_dir=DATA_DIR,
    out_dir=OUT_DIR,
    fs_hz=FS_HZ,
    mic=MIC,
    mode=MODE,
    trial_index=TRIAL_INDEX,
    nperseg=NPERSEG,
    noverlap=NOVERLAP,
    nfft=NFFT,
    fmin=FMIN,
    fmax=FMAX
):
    """
    Loop through all pickle and .npy files in the dataset and save one STFT plot per file.
    """
    pickle_files = sorted(data_dir.rglob("*.pickle"))
    npy_files = sorted(data_dir.rglob("*.npy"))
    all_files = sorted(pickle_files + npy_files)

    if not all_files:
        print(f"No pickle or .npy files found under: {data_dir}")
        return

    summary_rows = []

    for file_path in all_files:
        try:
            material = file_path.parent.name
            dist_cm, ang_deg = parse_distance_angle(file_path)

            df, signals = load_signals(file_path, mic=mic)
            x = choose_signal(signals, mode=mode, trial_index=trial_index)

            f, t, power_db = compute_stft(
                x,
                fs=fs_hz,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft
            )

            mode_tag = f"{mode}" if mode == "mean" else f"trial{trial_index}"
            out_name = f"{file_path.stem}__{mic}__{mode_tag}.png"
            out_path = out_dir / material / out_name

            title = (
                f"{material} | {file_path.stem} | {mic} | {mode_tag}\n"
                f"dist={dist_cm} cm, angle={ang_deg}°, fs={fs_hz/1e6:.3f} MHz"
            )

            plot_and_save_stft(
                f, t, power_db,
                out_path=out_path,
                title=title,
                fmin=fmin,
                fmax=fmax
            )

            summary_rows.append({
                "file": str(file_path),
                "material": material,
                "distance_cm": dist_cm,
                "angle_deg": ang_deg,
                "mic": mic,
                "mode": mode_tag,
                "n_trials": signals.shape[0],
                "n_samples": signals.shape[1],
                "output_plot": str(out_path)
            })

            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / f"stft_summary__{mic}__{mode}.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSummary saved to: {summary_csv}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    process_dataset_stft()