import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from bandpass import bandpass
from roi_windows import ROI_DISTANCES_CM, ROI_STARTS, ROI_LEN, parse_distance_cm


def parse_meta(filename: str):
    """Parse distance/angle from names like 25cm_-10Grad.pickle."""
    m_angle = re.search(r"(-?\d+)\s*Grad", filename)
    dist_int = parse_distance_cm(filename)
    dist_cm = float(dist_int) if dist_int is not None else np.nan
    angle_deg = float(m_angle.group(1)) if m_angle else np.nan
    return dist_cm, angle_deg


def object_from_path(file_path: Path, root: Path) -> str:
    """Return object class from path (first folder under root, e.g. Box/Glas/Dose)."""
    try:
        rel = file_path.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else file_path.parent.name
    except ValueError:
        return file_path.parent.name


def build_distance_window_map(distances, starts):
    if len(distances) != len(starts):
        raise ValueError("--window_distances and --window_starts must have the same length.")
    return {float(d): int(s) for d, s in zip(distances, starts)}


def get_window_start_for_distance(dist_cm: float, dist_to_start: dict) -> int | None:
    """Map distance to ROI window start using exact or near-integer distance matching."""
    if not np.isfinite(dist_cm):
        return None

    dist_key = float(dist_cm)
    if dist_key in dist_to_start:
        return dist_to_start[dist_key]

    rounded = float(round(dist_cm))
    if rounded in dist_to_start:
        return dist_to_start[rounded]

    return None


def extract_roi(sig: np.ndarray, roi_start: int, roi_len: int):
    """Extract an ROI by sample index window; clip safely to signal bounds."""
    i0 = max(0, int(roi_start))
    i1 = max(i0, int(roi_start) + int(roi_len))
    i1 = min(i1, len(sig))
    return sig[i0:i1], i0, i1


def preprocess_dc_detrend(sig: np.ndarray) -> np.ndarray:
    """Apply only DC removal and linear detrending."""
    x = np.asarray(sig, dtype=float)
    x = x - np.mean(x)
    x = detrend(x, type="linear")
    return x


def main():
    ap = argparse.ArgumentParser(description="Create ROI dataset with fixed index windows per distance.")

    ap.add_argument("--root", default=r"./Multifrequenz Dataset/Multifrequenz", help="Root folder to scan for *.pickle.")
    ap.add_argument("--outdir", default="outputs_roi", help="Output folder.")

    ap.add_argument("--mics", nargs="+", default=["Mic2"], help="Mic columns to process.")
    ap.add_argument("--n_trials", type=int, default=100, help="How many trials (rows) per file.")

    ap.add_argument("--fs_hz", type=float, default=2_000_000.0, help="Sampling rate in Hz.")
    ap.add_argument("--bw_hz", type=float, default=5000.0, help="Bandpass half-bandwidth in Hz.")
    ap.add_argument("--bp_method", default="fir", choices=["fir", "iir"], help="Bandpass method.")
    ap.add_argument("--order", type=int, default=1, help="IIR bandpass filter order.")
    ap.add_argument("--fir_numtaps", type=int, default=71, help="FIR taps for bandpass mode.")
    ap.add_argument("--fir_window", default="hamming", help="FIR window for bandpass mode.")
    ap.add_argument("--bands", nargs="+", type=float, default=[40_000, 50_000, 60_000], help="Carrier frequencies (Hz).")

    ap.add_argument(
        "--window_distances",
        nargs="+",
        type=float,
        default=ROI_DISTANCES_CM,
        help="Distances (cm) mapped to ROI windows in order.",
    )
    ap.add_argument(
        "--window_starts",
        nargs="+",
        type=int,
        default=ROI_STARTS,
        help="ROI start indices mapped to --window_distances.",
    )
    ap.add_argument("--roi_len", type=int, default=ROI_LEN, help="ROI length in samples.")
    ap.add_argument(
        "--pipelines",
        nargs="+",
        default=["bandpassed", "detrended_only"],
        choices=["bandpassed", "detrended_only"],
        help="ROI preprocessing variants to export.",
    )

    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dist_to_start = build_distance_window_map(args.window_distances, args.window_starts)

    files = sorted(root.rglob("*.pickle"))
    if not files:
        raise SystemExit(f"No .pickle files found under: {root}")

    rows = []
    skipped_no_distance_window = 0

    for p in files:
        dist_cm, angle_deg = parse_meta(p.name)
        roi_start = get_window_start_for_distance(dist_cm, dist_to_start)

        if roi_start is None:
            skipped_no_distance_window += 1
            continue

        obj_name = object_from_path(p, root)
        df = pd.read_pickle(p)

        n_use = min(args.n_trials, len(df))
        if n_use <= 0:
            continue

        available_mics = [m for m in args.mics if m in df.columns]
        if not available_mics:
            continue

        for trial_idx in range(n_use):
            for mic in available_mics:
                sig = np.asarray(df.iloc[trial_idx][mic], dtype=float)
                sig_detrended = preprocess_dc_detrend(sig)

                if "detrended_only" in args.pipelines:
                    roi, roi_i0, roi_i1 = extract_roi(
                        sig_detrended, roi_start=roi_start, roi_len=args.roi_len
                    )

                    if roi.size != 0 and np.isfinite(roi).all():
                        rows.append(
                            {
                                "object": obj_name,
                                "distance": float(dist_cm),
                                "angle": float(angle_deg) if np.isfinite(angle_deg) else np.nan,
                                "mic": mic,
                                "trial": int(trial_idx),
                                "band": np.nan,
                                "preprocessing": "detrended_only",
                                "ROI start": int(roi_i0),
                                "ROI end": int(roi_i1),
                                "ROI length": int(roi.size),
                                "roi": roi,
                                "file": str(p),
                                "filename": p.name,
                            }
                        )

                if "bandpassed" in args.pipelines:
                    for f0 in args.bands:
                        # bp = bandpass(sig_detrended, args.fs_hz, f0, bw=args.bw_hz, order=args.order, method="iir")
                        bp = bandpass(
                            sig_detrended,
                            args.fs_hz,
                            f0,
                            bw=args.bw_hz,
                            method=args.bp_method,
                            order=args.order,
                            numtaps=args.fir_numtaps,
                            window=args.fir_window,
                        )
                        roi, roi_i0, roi_i1 = extract_roi(bp, roi_start=roi_start, roi_len=args.roi_len)

                        if roi.size == 0 or not np.isfinite(roi).all():
                            continue

                        rows.append(
                            {
                                "object": obj_name,
                                "distance": float(dist_cm),
                                "angle": float(angle_deg) if np.isfinite(angle_deg) else np.nan,
                                "mic": mic,
                                "trial": int(trial_idx),
                                "band": float(f0),
                                "preprocessing": "bandpassed",
                                "ROI start": int(roi_i0),
                                "ROI end": int(roi_i1),
                                "ROI length": int(roi.size),
                                "roi": roi,
                                "file": str(p),
                                "filename": p.name,
                            }
                        )

    out_df = pd.DataFrame(rows)

    out_pkl = outdir / "roi_dataset.pkl"
    out_df.to_pickle(out_pkl)

    # CSV with requested metadata columns only (without array-valued ROI column)
    metadata_cols = [
        "object",
        "distance",
        "angle",
        "mic",
        "trial",
        "band",
        "preprocessing",
        "ROI start",
        "ROI end",
        "ROI length",
    ]
    out_csv = outdir / "roi_dataset_metadata.csv"
    out_df[metadata_cols].to_csv(out_csv, index=False)

    print(f"Saved ROI dataset rows: {len(out_df)}")
    print(f"ROI dataset (with ROI arrays): {out_pkl}")
    print(f"ROI metadata CSV: {out_csv}")
    print(f"Skipped files (distance not mapped to any window): {skipped_no_distance_window}")
    print("Distance to window mapping:")
    for d, s in dist_to_start.items():
        print(f"- {d:.0f} cm -> start={s}, end={s + args.roi_len}")


if __name__ == "__main__":
    main()
