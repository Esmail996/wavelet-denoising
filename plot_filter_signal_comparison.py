#!/usr/bin/env python3
"""
Plot raw vs multiple filtered signals for visual comparison.

Features:
- Loads waveforms from dataset pickle files.
- Uses recommendation CSVs to choose top filter/band combinations.
- Plots each filtered signal next to the same raw signal.

Example:
    python plot_filter_signal_comparison.py \
      --data-root "Multifrequenz Dataset/Multifrequenz" \
      --ranking-csv "outputs/filter_comparison_full_dataset/recommendation_ranking_by_distance_balanced.csv" \
      --designs-json "example_filter_designs.json" \
      --mic Mic2 --trial 50 --top-k 3 --max-files 3
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, firwin, sosfiltfilt


DIST_RE = re.compile(r"(?P<dist>\d+)\s*cm", re.IGNORECASE)


@dataclass(frozen=True)
class FilterDesign:
    name: str
    method: str
    bw_hz: float
    order: Optional[int] = None
    numtaps: Optional[int] = None
    window: str = "hamming"


def as_1d_float(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {arr.shape}")
    return arr


def parse_distance_cm(filename: str) -> Optional[int]:
    m = DIST_RE.search(filename)
    if not m:
        return None
    return int(m.group("dist"))


def load_pickle_df(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(obj)}")
    return obj


def iter_measurement_pickles(data_root: Path) -> Iterable[Path]:
    for object_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if object_dir.name.lower() == "referenz":
            continue
        for pkl in sorted(object_dir.glob("*.pickle")):
            yield pkl


def load_designs(path: Path) -> Dict[str, FilterDesign]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Design JSON must be a list")
    out: Dict[str, FilterDesign] = {}
    for item in raw:
        d = FilterDesign(**item)
        out[d.name] = d
    return out


def butter_bandpass(x: np.ndarray, fs_hz: float, f0_hz: float, bw_hz: float, order: int) -> np.ndarray:
    low = (f0_hz - bw_hz) / (fs_hz / 2.0)
    high = (f0_hz + bw_hz) / (fs_hz / 2.0)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def fir_bandpass(x: np.ndarray, fs_hz: float, f0_hz: float, bw_hz: float, numtaps: int, window: str) -> np.ndarray:
    b = firwin(numtaps, [f0_hz - bw_hz, f0_hz + bw_hz], pass_zero=False, fs=fs_hz, window=window)
    return filtfilt(b, [1.0], x)


def apply_design(x: np.ndarray, fs_hz: float, f0_hz: float, design: FilterDesign) -> np.ndarray:
    if design.method == "butter":
        if design.order is None:
            raise ValueError(f"Design {design.name} has no order")
        return butter_bandpass(x, fs_hz, f0_hz, float(design.bw_hz), int(design.order))
    if design.method == "fir":
        if design.numtaps is None:
            raise ValueError(f"Design {design.name} has no numtaps")
        return fir_bandpass(x, fs_hz, f0_hz, float(design.bw_hz), int(design.numtaps), str(design.window))
    raise ValueError(f"Unsupported method {design.method}")


def get_top_rows_for_file(ranking: pd.DataFrame, distance_cm: Optional[int], top_k: int) -> pd.DataFrame:
    df = ranking.copy()
    if "distance_cm" in df.columns and distance_cm is not None:
        per_dist = df[df["distance_cm"] == distance_cm]
        if not per_dist.empty:
            return per_dist.sort_values("balanced_score", ascending=False).head(top_k)
    return df.sort_values("balanced_score", ascending=False).head(top_k)


def plot_one(
    pkl_path: Path,
    mic: str,
    trial: int,
    fs_hz: float,
    top_rows: pd.DataFrame,
    designs_by_name: Dict[str, FilterDesign],
    output_dir: Path,
) -> Path:
    df = load_pickle_df(pkl_path)
    if mic not in df.columns:
        raise KeyError(f"{pkl_path.name}: missing mic column {mic}")
    if trial < 0 or trial >= len(df):
        raise IndexError(f"{pkl_path.name}: trial {trial} out of range 0..{len(df) - 1}")

    raw = as_1d_float(df.iloc[trial][mic])
    raw_det = detrend(raw - np.mean(raw), type="linear")
    t_ms = np.arange(raw_det.size, dtype=float) / fs_hz * 1e3

    nrows = max(1, len(top_rows))
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 3.3 * nrows), sharex=True)
    if nrows == 1:
        axes = np.asarray([axes])

    for i, (_, row) in enumerate(top_rows.iterrows()):
        design_name = str(row["design_name"])
        band_hz = float(row["band_hz"])
        if design_name not in designs_by_name:
            raise KeyError(f"Design {design_name} not found in designs JSON")
        design = designs_by_name[design_name]

        filtered = apply_design(raw_det, fs_hz, band_hz, design)

        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        ax_l.plot(t_ms, raw_det, color="#1f2937", linewidth=0.9)
        ax_l.set_title("Raw detrended signal")
        ax_l.set_ylabel("Amplitude")
        ax_l.grid(True, alpha=0.25)

        ax_r.plot(t_ms, filtered, color="#1d4ed8", linewidth=0.9)
        ax_r.plot(t_ms, raw_det, color="#64748b", linewidth=0.7, alpha=0.45, label="raw")
        label = f"{design_name} @ {int(band_hz)} Hz"
        ax_r.set_title(label)
        ax_r.grid(True, alpha=0.25)
        ax_r.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")

    fig.suptitle(f"Signal comparison | {pkl_path.parent.name}/{pkl_path.name} | {mic} | trial={trial}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"compare_{pkl_path.parent.name}_{pkl_path.stem}_{mic}_trial{trial}.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot raw vs multiple filtered signals side-by-side.")
    p.add_argument("--data-root", default="Multifrequenz Dataset/Multifrequenz", help="Dataset root folder.")
    p.add_argument("--ranking-csv", default="outputs/filter_comparison_full_dataset/recommendation_ranking_by_distance_balanced.csv", help="CSV with ranked recommendations.")
    p.add_argument("--designs-json", default="example_filter_designs.json", help="Filter design config JSON.")
    p.add_argument("--output-dir", default="outputs/filter_comparison_signal_plots", help="Folder for plot images.")
    p.add_argument("--mic", default="Mic2", help="Mic column name.")
    p.add_argument("--trial", type=int, default=50, help="Trial index in each file.")
    p.add_argument("--fs-hz", type=float, default=2_000_000.0, help="Sampling frequency in Hz.")
    p.add_argument("--top-k", type=int, default=3, help="How many recommended filters to plot per file.")
    p.add_argument("--max-files", type=int, default=3, help="How many measurement files to plot.")
    p.add_argument("--files", nargs="*", default=None, help="Optional explicit list of pickle files to plot.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    ranking_csv = Path(args.ranking_csv)
    designs_json = Path(args.designs_json)
    output_dir = Path(args.output_dir)

    ranking = pd.read_csv(ranking_csv)
    designs_by_name = load_designs(designs_json)

    if args.files:
        files = [Path(p) for p in args.files]
    else:
        files = list(iter_measurement_pickles(data_root))[: int(args.max_files)]

    if not files:
        raise SystemExit("No files selected for plotting.")

    saved: List[Path] = []
    for pkl_path in files:
        dist = parse_distance_cm(pkl_path.name)
        top_rows = get_top_rows_for_file(ranking, dist, int(args.top_k))
        out_path = plot_one(
            pkl_path=pkl_path,
            mic=args.mic,
            trial=int(args.trial),
            fs_hz=float(args.fs_hz),
            top_rows=top_rows,
            designs_by_name=designs_by_name,
            output_dir=output_dir,
        )
        saved.append(out_path)

    print("Saved comparison plots:")
    for p in saved:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
