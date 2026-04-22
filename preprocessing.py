import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, filtfilt, firwin

from bandpass import bandpass


def extract_trial_signal(data, mic: str, trial: int) -> np.ndarray:
    """Extract one trial signal from DataFrame/dict/ndarray pickle payloads."""
    mic_l = str(mic).lower()

    if isinstance(data, pd.DataFrame):
        col_lut = {str(c).lower(): c for c in data.columns}
        if mic_l not in col_lut:
            raise ValueError(f"Mic column '{mic}' not found. Available: {list(data.columns)}")
        if not (0 <= int(trial) < len(data)):
            raise ValueError(f"--trial {trial} out of range. len(df)={len(data)}")
        return np.asarray(data.iloc[int(trial)][col_lut[mic_l]], dtype=float)

    if isinstance(data, dict):
        # Case 1: multiband denoising output {band_hz -> [signals_per_trial]}
        # Use first available band for plotting this utility script.
        if data:
            first_key = next(iter(data.keys()))
            first_val = data[first_key]
            if isinstance(first_val, (list, tuple, np.ndarray)):
                n_trials = len(first_val)
                if not (0 <= int(trial) < n_trials):
                    raise ValueError(f"--trial {trial} out of range. len(trials)={n_trials}")
                return np.asarray(first_val[int(trial)], dtype=float)

        # Case 2: mic-keyed dict {'Mic1': ..., 'Mic2': ...}
        mic_key = None
        for k in data.keys():
            if str(k).lower() == mic_l:
                mic_key = k
                break
        if mic_key is None:
            raise ValueError(f"Mic key '{mic}' not found in dict. Available: {list(data.keys())}")
        mic_trials = data[mic_key]
        if not (0 <= int(trial) < len(mic_trials)):
            raise ValueError(f"--trial {trial} out of range. len(trials)={len(mic_trials)}")
        return np.asarray(mic_trials[int(trial)], dtype=float)

    arr = np.asarray(data, dtype=object)
    if arr.ndim == 2:
        mic_map = {"mic1": 0, "mic2": 1, "mic3": 2}
        if mic_l not in mic_map:
            raise ValueError(f"Unsupported mic '{mic}'.")
        mic_idx = mic_map[mic_l]
        if mic_idx >= arr.shape[0]:
            raise ValueError(f"Array does not contain {mic}. Shape: {arr.shape}")
        if not (0 <= int(trial) < arr.shape[1]):
            raise ValueError(f"--trial {trial} out of range. n_trials={arr.shape[1]}")
        return np.asarray(arr[mic_idx, int(trial)], dtype=float)

    raise ValueError(f"Unsupported pickle payload type/shape: {type(data)} / {arr.shape}")


def visualize_preprocessing_steps(original_data, dc_removed_data, detrended_data, title="Preprocessing Visualization"):
    """Visualize data at different preprocessing stages."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(original_data, color="blue", linewidth=1)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Original Data")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dc_removed_data, color="green", linewidth=1)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("After DC Removal")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(detrended_data, color="red", linewidth=1)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Sample")
    axes[2].set_title("After Detrending")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.axvline(2915, color="black", linestyle="--", linewidth=1.2, label="ROI start (2915)")
        ax.axvline(3545, color="black", linestyle=":",  linewidth=1.2, label="ROI end (3545)")
        ax.axvline(5800, color="gray", linestyle="--", linewidth=1.2, label="ROI2 start (5800)")
        ax.axvline(6430, color="gray", linestyle=":",  linewidth=1.2, label="ROI2 end (6430)")
        ax.axvline(8650, color="darkorange", linestyle="--", linewidth=1.2, label="ROI3 start (8650)")
        ax.axvline(9280, color="darkorange", linestyle=":",  linewidth=1.2, label="ROI3 end (9280)")
        ax.axvline(11550, color="navy", linestyle="--", linewidth=1.2, label="ROI4 start (11550)")
        ax.axvline(12180, color="navy", linestyle=":",  linewidth=1.2, label="ROI4 end (12180)")
        ax.axvline(14400, color="crimson", linestyle="--", linewidth=1.2, label="ROI5 start (14400)")
        ax.axvline(15030, color="crimson", linestyle=":",  linewidth=1.2, label="ROI5 end (15030)")
        ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()


def visualize_bandpass_signals(bp_signals: dict, title="Bandpass Filtered Signals"):
    """Plot one subplot per bandpass-filtered signal."""
    fig, axes = plt.subplots(len(bp_signals), 1, figsize=(12, 3 * len(bp_signals)))
    if len(bp_signals) == 1:
        axes = [axes]
    colors = ["purple", "darkorange", "teal"]
    for ax, (f0, sig), color in zip(axes, bp_signals.items(), colors):
        ax.plot(sig, color=color, linewidth=1)
        ax.axvline(2915, color="black", linestyle="--", linewidth=1.2, label="ROI start (2915)")
        ax.axvline(3545, color="black", linestyle=":",  linewidth=1.2, label="ROI end (3545)")
        ax.axvline(5800, color="gray", linestyle="--", linewidth=1.2, label="ROI2 start (5800)")
        ax.axvline(6430, color="gray", linestyle=":",  linewidth=1.2, label="ROI2 end (6430)")
        ax.axvline(8650, color="darkorange", linestyle="--", linewidth=1.2, label="ROI3 start (8650)")
        ax.axvline(9280, color="darkorange", linestyle=":",  linewidth=1.2, label="ROI3 end (9280)")
        ax.axvline(11550, color="navy", linestyle="--", linewidth=1.2, label="ROI4 start (11550)")
        ax.axvline(12180, color="navy", linestyle=":",  linewidth=1.2, label="ROI4 end (12180)")
        ax.axvline(14400, color="crimson", linestyle="--", linewidth=1.2, label="ROI5 start (14400)")
        ax.axvline(15030, color="crimson", linestyle=":",  linewidth=1.2, label="ROI5 end (15030)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Bandpass @ {int(f0/1000)} kHz")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Sample")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()



def main():
    ap = argparse.ArgumentParser(description="Preprocess a signal and save output files.")
    ap.add_argument(
        "--file",
        default=r"Multifrequenz Dataset\Multifrequenz\Glas\50cm_-10Grad.pickle",
        help="Path to input .pickle file.",
    )
    ap.add_argument("--mic", default="Mic2", help="Mic column to process (Mic1/Mic2/Mic3).")
    ap.add_argument("--trial", type=int, default=50, help="Trial index (row) to process.")
    ap.add_argument("--outdir", default="outputs/preprocessing", help="Output folder.")
    ap.add_argument("--fs_hz", type=float, default=2_000_000.0, help="Sampling rate in Hz.")
    ap.add_argument("--new_bw_hz", type=float, default=5000.0, help="Half-bandwidth for new FIR bandpass.")
    ap.add_argument("--new_numtaps", type=int, default=71, help="Number of taps for new FIR bandpass.")
    ap.add_argument("--new_window", default="hamming", help="Window for new FIR bandpass.")
    args = ap.parse_args()

    input_path = Path(args.file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(input_path)
    original_data = extract_trial_signal(data, mic=args.mic, trial=args.trial)
    dc_removed_data = original_data - np.mean(original_data)
    detrended_data = detrend(dc_removed_data)

    parent_name = input_path.parent.name
    stem = f"{parent_name}__{input_path.stem}__{args.mic}__trial{args.trial}"
    preprocessed_path = outdir / f"{stem}__preprocessed.npy"

    np.save(preprocessed_path, detrended_data)

    title = f"{input_path.name} | {args.mic} | trial={args.trial}"
    visualize_preprocessing_steps(original_data, dc_removed_data, detrended_data, title=title)

    print("Saved preprocessed signal:")
    print(f"- {preprocessed_path}")

    # Bandpass step: apply old and new designs at each carrier frequency
    bands_hz = [40_000.0, 50_000.0, 60_000.0]
    old_bp_signals = {}
    new_bp_signals = {}
    for f0 in bands_hz:
        old_bp_signals[f0] = bandpass(detrended_data, args.fs_hz, f0, bw=7000.0, order=1)

    bp_title = f"{input_path.name} | {args.mic} | trial={args.trial} | BW=5000 Hz, order=1"
    visualize_bandpass_signals(old_bp_signals, title=bp_title)


    print("Saved bandpass signals:")
    for f0 in bands_hz:
        old_bp_path = outdir / f"{stem}__bp_old_{int(f0/1000)}kHz.npy"
        new_bp_path = outdir / f"{stem}__bp_new_{int(f0/1000)}kHz.npy"
        np.save(old_bp_path, old_bp_signals[f0])
        np.save(new_bp_path, new_bp_signals[f0])
        print(f"- {old_bp_path}")
        print(f"- {new_bp_path}")


if __name__ == "__main__":
    main()
