#!/usr/bin/env python3
"""
Inspect and plot multiband denoised Mic2 signals.

Expected denoised pickle format (from wavelet_denoising_multiband_mic2.py):
    {band_hz -> [denoised_signal_per_trial]}

Examples:
    python plot_multiband_denoised_signals.py \
      --denoised-pickle "Multifrequenz Dataset/Multifrequenz_denoised_multiband_mic2/Box/50cm_0Grad.pickle" \
      --trial 10

    python plot_multiband_denoised_signals.py \
      --denoised-pickle "Multifrequenz Dataset/Multifrequenz_denoised_multiband_mic2/Box/50cm_0Grad.pickle" \
      --raw-pickle "Multifrequenz Dataset/Multifrequenz/Box/50cm_0Grad.pickle" \
      --overlay-raw --trial 10 --normalize
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FS_HZ = 2_000_000.0
DEFAULT_BANDS = [40000.0, 50000.0, 60000.0]


def as_1d_float(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {arr.shape}")
    return arr


def _band_key_to_float(k: Any) -> Optional[float]:
    try:
        return float(k)
    except (TypeError, ValueError):
        return None


def load_denoised_multiband(path: Path) -> Dict[float, List[np.ndarray]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in denoised pickle, got {type(obj)}")

    out: Dict[float, List[np.ndarray]] = {}
    for key, value in obj.items():
        band = _band_key_to_float(key)
        if band is None:
            continue

        if isinstance(value, np.ndarray) and value.ndim == 2:
            trials = [as_1d_float(row) for row in value]
        elif isinstance(value, (list, tuple)):
            trials = [as_1d_float(v) for v in value]
        else:
            raise TypeError(
                f"Band {band}: expected list/tuple or 2D ndarray, got {type(value)}"
            )
        out[band] = trials

    if not out:
        raise ValueError("No numeric band keys found in denoised pickle")

    return out


def load_raw_mic_trial(path: Path, mic: str, trial: int) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, pd.DataFrame):
        if mic not in obj.columns:
            raise KeyError(f"Missing mic column '{mic}' in DataFrame")
        if trial < 0 or trial >= len(obj):
            raise IndexError(f"Trial {trial} out of range 0..{len(obj)-1}")
        return as_1d_float(obj.iloc[trial][mic])

    if isinstance(obj, dict):
        if mic not in obj:
            raise KeyError(f"Missing mic key '{mic}' in dict")
        arr = obj[mic]
        if trial < 0 or trial >= len(arr):
            raise IndexError(f"Trial {trial} out of range 0..{len(arr)-1}")
        return as_1d_float(arr[trial])

    if isinstance(obj, np.ndarray):
        if obj.ndim == 3:
            mic_to_idx = {"Mic1": 0, "Mic2": 1, "Mic3": 2}
            if mic not in mic_to_idx:
                raise KeyError(f"Unsupported mic '{mic}'. Use one of {list(mic_to_idx)}")
            mic_idx = mic_to_idx[mic]
            if mic_idx >= obj.shape[0]:
                raise IndexError(f"Mic index {mic_idx} out of bounds for array shape {obj.shape}")
            if trial < 0 or trial >= obj.shape[1]:
                raise IndexError(f"Trial {trial} out of range 0..{obj.shape[1]-1}")
            return as_1d_float(obj[mic_idx, trial, :])
        if obj.ndim == 2:
            if trial < 0 or trial >= obj.shape[0]:
                raise IndexError(f"Trial {trial} out of range 0..{obj.shape[0]-1}")
            return as_1d_float(obj[trial, :])

    raise TypeError(f"Unsupported raw pickle payload type: {type(obj)}")


def maybe_normalize(x: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        return x
    return x / peak


def choose_bands(available: Iterable[float], requested: Optional[List[float]]) -> List[float]:
    available_sorted = sorted(float(b) for b in available)
    if not requested:
        return [b for b in DEFAULT_BANDS if b in available_sorted] or available_sorted

    chosen: List[float] = []
    for req in requested:
        # Accept exact or nearest match within 1 Hz tolerance for float safety.
        matches = [b for b in available_sorted if abs(b - float(req)) <= 1.0]
        if matches:
            chosen.append(matches[0])
    if not chosen:
        raise ValueError(
            f"Requested bands {requested} not found. Available bands: {available_sorted}"
        )
    return chosen


def plot_trial(
    denoised: Dict[float, List[np.ndarray]],
    trial: int,
    bands: List[float],
    fs_hz: float,
    raw_signal: Optional[np.ndarray],
    normalize: bool,
    title: str,
) -> plt.Figure:
    nrows = len(bands)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 2.8 * nrows), sharex=False)
    if nrows == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        trials = denoised[band]
        if trial < 0 or trial >= len(trials):
            raise IndexError(f"Trial {trial} out of range for band {band} (0..{len(trials)-1})")

        den = maybe_normalize(as_1d_float(trials[trial]), normalize)
        t_ms_den = np.arange(den.size, dtype=float) / fs_hz * 1e3
        ax.plot(t_ms_den, den, color="#0f766e", linewidth=1.0, label=f"denoised {int(band)} Hz")

        if raw_signal is not None:
            raw = maybe_normalize(raw_signal, normalize)
            n = min(raw.size, den.size)
            t_ms_raw = np.arange(n, dtype=float) / fs_hz * 1e3
            ax.plot(t_ms_raw, raw[:n], color="#64748b", linewidth=0.9, alpha=0.7, label="raw Mic2")

        ax.set_ylabel("Amplitude")
        ax.set_title(f"Band {int(band)} Hz | trial {trial} | n={den.size}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot denoised multiband signals for one trial.")
    p.add_argument(
        "--denoised-pickle",
        type=str,
        required=True,
        help="Path to denoised multiband pickle (dict[band_hz] -> list of signals).",
    )
    p.add_argument(
        "--raw-pickle",
        type=str,
        default=r"Multifrequenz Dataset\Multifrequenz_denoised_multiband_mic2_universal_soft\Box\25cm_-5Grad.pickle",
        help="Optional path to original raw pickle for overlay.",
    )
    p.add_argument("--mic", type=str, default="Mic2", help="Mic column/key for raw overlay.")
    p.add_argument("--trial", type=int, default=0, help="Trial index to plot.")
    p.add_argument("--fs-hz", type=float, default=DEFAULT_FS_HZ, help="Sampling frequency in Hz.")
    p.add_argument(
        "--bands",
        type=float,
        nargs="*",
        default=None,
        help="Optional subset of bands in Hz, e.g. --bands 40000 60000",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/denoise_plots",
        help="Directory where figure is saved.",
    )
    p.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Optional output filename; default is auto-generated.",
    )
    p.add_argument(
        "--overlay-raw",
        action="store_true",
        help="Overlay raw mic signal (requires --raw-pickle).",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each plotted signal by its own absolute max.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also display the plot window after saving.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    denoised_path = Path(args.denoised_pickle)
    if not denoised_path.exists():
        raise FileNotFoundError(f"Denoised pickle not found: {denoised_path}")

    denoised = load_denoised_multiband(denoised_path)
    bands = choose_bands(denoised.keys(), args.bands)

    raw_signal = None
    if args.overlay_raw:
        if not args.raw_pickle:
            raise ValueError("--overlay-raw requires --raw-pickle")
        raw_path = Path(args.raw_pickle)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw pickle not found: {raw_path}")
        raw_signal = load_raw_mic_trial(raw_path, mic=args.mic, trial=args.trial)

    title = f"Multiband denoised inspection | {denoised_path.parent.name}/{denoised_path.name}"
    fig = plot_trial(
        denoised=denoised,
        trial=int(args.trial),
        bands=bands,
        fs_hz=float(args.fs_hz),
        raw_signal=raw_signal,
        normalize=bool(args.normalize),
        title=title,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_name:
        out_name = args.save_name
    else:
        bands_str = "-".join(str(int(b)) for b in bands)
        out_name = f"multiband_inspect_{denoised_path.stem}_trial{args.trial}_{bands_str}.png"

    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=170)
    print(f"Saved plot: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
