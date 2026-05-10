"""
extract_features_all_trials.py — build the full feature matrix.

Reads the denoised trial pickles + the calibrated TOF CSV, and produces
a single CSV with one row per (object, distance, angle, trial), columns
being all Family 1 + Family 2 features per (Mic, Tx).

USAGE (your machine):
    python extract_features_all_trials.py \
        --denoised-dir "Multifrequenz Dataset/Multifrequenz_denoised_branchA_sym6" \
        --tof-csv      "outputs/tof_matched/multifrequenz_denoised_branchA_sym6.csv" \
        --output       "outputs/features_all.csv" \
        --use-kymatio                              # enable F2 scattering

Tips:
- Set --max-files for smoke tests
- Set --skip-f2 to skip Kymatio if not installed
- Mic3 included by default for completeness; you can drop columns later
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Local imports
from features_family1 import extract_family1_for_trial
from features_family15 import extract_family15_for_trial
from features_family2 import build_scattering, extract_family2_for_trial


NAME_RE = re.compile(r"(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad", re.IGNORECASE)
MICS = ("Mic1", "Mic2", "Mic3")
TXS = ("Tx1", "Tx5", "Tx8")
CARRIER_PER_TX = {"Tx1": 40_000.0, "Tx5": 50_000.0, "Tx8": 60_000.0}
TX_HALF_BW_HZ = {"Tx1": 4_000.0, "Tx5": 5_000.0, "Tx8": 6_000.0}
FS = 2_000_000.0
WAVELET_NUMERIC_COLS = (
    "top1_mu_sc", "top2_mu_sc", "top3_mu_sc", "top4_mu_sc", "top5_mu_sc",
)


def parse_dist_angle(filename: str) -> tuple[int | None, int | None]:
    m = NAME_RE.search(filename)
    if not m:
        return None, None
    return int(m.group("dist")), int(m.group("ang"))


def load_pickle_trials(p: Path) -> dict[str, list[np.ndarray]]:
    """Load a denoised pickle. Adjust the schema if your pickles differ."""
    with open(p, "rb") as f:
        obj = pickle.load(f)
    # The Multifrequenz pickles in our pipeline are pandas DataFrames with
    # columns Mic1, Mic2, Mic3 and one row per trial.
    if isinstance(obj, pd.DataFrame):
        return {
            mic: [np.asarray(obj[mic].iloc[i]) for i in range(len(obj))]
            for mic in MICS
        }
    elif isinstance(obj, dict):
        return obj
    else:
        raise ValueError(f"Unsupported pickle schema in {p}: type={type(obj)}")


def build_tof_lookup(tof_csv_path: Path) -> dict:
    """Lookup keyed by (category, distance_cm, angle_deg, trial, mic, tx) →
    Path-B TOF in seconds (no tau subtraction)."""
    df = pd.read_csv(tof_csv_path)
    lut = {}
    for r in df.itertuples():
        lut[(r.category, int(r.distance_cm), int(r.angle_deg), int(r.trial),
             r.mic, r.tx)] = float(r.tof_path_b_s)
    return lut


def build_wavelet_sparsity_lookup(csv_paths: list[Path]) -> dict[tuple[str, int, int, int], dict[str, float]]:
    """Build sparse feature lookup keyed by (category, distance, angle, trial).

    Each CSV contributes mic-scoped numeric fields prefixed as:
      ws_<MicX>_top1_mu_sc ... ws_<MicX>_top5_mu_sc
      ws_<MicX>_gap_12 ... ws_<MicX>_gap_15
    """
    required_cols = {
        "category", "distance_cm", "angle_deg", "meas_idx", "mic", *WAVELET_NUMERIC_COLS,
    }
    lut: dict[tuple[str, int, int, int], dict[str, float]] = {}

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Wavelet sparsity CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = required_cols - set(df.columns)
        if missing:
            missing_txt = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {csv_path}: {missing_txt}")

        for r in df.itertuples(index=False):
            if pd.isna(r.distance_cm) or pd.isna(r.angle_deg) or pd.isna(r.meas_idx):
                # Skip rows that cannot be mapped to trial keys (e.g. reference-only files).
                continue
            key = (str(r.category), int(r.distance_cm), int(r.angle_deg), int(r.meas_idx))
            mic = str(r.mic)
            pfx = f"ws_{mic}"

            t1 = float(r.top1_mu_sc)
            t2 = float(r.top2_mu_sc)
            t3 = float(r.top3_mu_sc)
            t4 = float(r.top4_mu_sc)
            t5 = float(r.top5_mu_sc)
            feats = {
                f"{pfx}_top1_mu_sc": t1,
                f"{pfx}_top2_mu_sc": t2,
                f"{pfx}_top3_mu_sc": t3,
                f"{pfx}_top4_mu_sc": t4,
                f"{pfx}_top5_mu_sc": t5,
                f"{pfx}_gap_12": t1 - t2,
                f"{pfx}_gap_23": t2 - t3,
                f"{pfx}_gap_34": t3 - t4,
                f"{pfx}_gap_45": t4 - t5,
                f"{pfx}_gap_15": t1 - t5,
            }
            if key not in lut:
                lut[key] = feats
            else:
                # Merge mic-specific fields from multiple CSVs into one row key.
                lut[key].update(feats)

    return lut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoised-dir", type=Path, required=True)
    parser.add_argument("--tof-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-kymatio", action="store_true")
    parser.add_argument("--skip-f2", action="store_true")
    parser.add_argument("--include-mic3", action="store_true",
                        help="Add Mic3 channels (degraded mic). Off by default.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--roi-n", type=int, default=1024)
    parser.add_argument("--roi-half-us-f1", type=float, default=200.0)
    parser.add_argument("--no-realign", action="store_true",
                        help="Disable per-channel envelope re-alignment within ROI.")
    parser.add_argument("--no-normalise", action="store_true",
                        help="Disable per-trial ROI energy normalisation.")
    parser.add_argument("--skip-f15", action="store_true",
                        help="Skip Family 1.5 (cepstral + cross-channel) extraction.")
    parser.add_argument("--wavelet-csv-mic1", type=Path, default=None,
                        help="Optional wavelet-choice CSV for Mic1 sparsity-profile features.")
    parser.add_argument("--wavelet-csv-mic2", type=Path, default=None,
                        help="Optional wavelet-choice CSV for Mic2 sparsity-profile features.")
    args = parser.parse_args()

    realign = not args.no_realign
    normalise = not args.no_normalise
    print(f"ROI preprocessing — realign: {realign}, normalise energy: {normalise}")

    print(f"Loading TOF lookup from {args.tof_csv}")
    tof_lut = build_tof_lookup(args.tof_csv)
    print(f"  {len(tof_lut)} TOF entries")

    wavelet_csvs = [p for p in (args.wavelet_csv_mic1, args.wavelet_csv_mic2) if p is not None]
    wavelet_lut: dict[tuple[str, int, int, int], dict[str, float]] = {}
    if wavelet_csvs:
        print(f"Loading wavelet sparsity lookup from {len(wavelet_csvs)} CSV(s)")
        wavelet_lut = build_wavelet_sparsity_lookup(wavelet_csvs)
        print(f"  {len(wavelet_lut)} trial keys in wavelet sparsity lookup")

    scattering = None
    if not args.skip_f2 and args.use_kymatio:
        print(f"Building Kymatio Scattering1D N={args.roi_n}, J=8, Q=8 ...")
        scattering = build_scattering(N=args.roi_n, J=8, Q=8)

    pickles = sorted(args.denoised_dir.rglob("*.pickle"))
    if not pickles:
        raise FileNotFoundError(f"No .pickle in {args.denoised_dir}")
    if args.max_files is not None:
        pickles = pickles[: args.max_files]
    print(f"Processing {len(pickles)} pickle files")

    mics_used = MICS if args.include_mic3 else ("Mic1", "Mic2")

    rows = []
    rows_with_wavelet = 0
    for fpath in pickles:
        rel = fpath.relative_to(args.denoised_dir).as_posix()
        category = rel.split("/", 1)[0] if "/" in rel else fpath.parent.name
        dist_cm, ang_deg = parse_dist_angle(fpath.name)
        if dist_cm is None:
            print(f"Skipping {fpath.name} (no dist/angle pattern)")
            continue
        trial_map = load_pickle_trials(fpath)
        n_trials = min(len(trial_map[m]) for m in mics_used)
        print(f"  {rel}: {n_trials} trials")

        for trial_idx in range(n_trials):
            signals = {m: trial_map[m][trial_idx] for m in MICS}
            tof_dict = {(m, t): tof_lut[(category, dist_cm, ang_deg, trial_idx, m, t)]
                        for m in mics_used for t in TXS}
            row: dict[str, Any] = {
                "category": category, "distance_cm": dist_cm,
                "angle_deg": ang_deg, "trial": trial_idx,
            }
            wk = (category, int(dist_cm), int(ang_deg), int(trial_idx))
            if wk in wavelet_lut:
                row.update(wavelet_lut[wk])
                rows_with_wavelet += 1
            # Family 1
            f1 = extract_family1_for_trial(
                signals, tof_dict, FS, CARRIER_PER_TX,
                tx_half_bw_hz=TX_HALF_BW_HZ,
                roi_half_us=args.roi_half_us_f1, mics=mics_used, txs=TXS,
                realign=realign, normalise=normalise,
            )
            row.update(f1)
            # Family 1.5 (cepstral + cross-channel ratios)
            if not args.skip_f15:
                f15 = extract_family15_for_trial(
                    signals, tof_dict, FS, CARRIER_PER_TX,
                    tx_half_bw_hz=TX_HALF_BW_HZ,
                    roi_half_us=args.roi_half_us_f1,
                    realign=realign, normalise=normalise,
                    mics=mics_used, txs=TXS,
                )
                row.update(f15)
            # Family 2
            if not args.skip_f2:
                f2 = extract_family2_for_trial(
                    signals, tof_dict, FS, CARRIER_PER_TX,
                    tx_half_bw_hz=TX_HALF_BW_HZ,
                    scattering=scattering, roi_n=args.roi_n,
                    mics=mics_used, txs=TXS,
                    use_kymatio=args.use_kymatio,
                    realign=realign, normalise=normalise,
                )
                row.update(f2)
            rows.append(row)

    df_out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved {len(df_out)} rows × {df_out.shape[1]} cols to {args.output}")
    if wavelet_csvs:
        print(f"  Rows with wavelet sparsity features: {rows_with_wavelet}/{len(df_out)}")
    f15_cols = sum(1 for c in df_out.columns if "_cep_" in c or
                   "_dlog_energy" in c or "_dlog_peak" in c or c == "signed_amp_ratio")
    f2_cols = sum(1 for c in df_out.columns
                  if c.split("_")[-1].startswith("S") and c.split("_")[-1][1:].isdigit())
    f1_cols = df_out.shape[1] - f15_cols - f2_cols - 4   # minus meta cols
    print(f"  Family 1   columns: {f1_cols}")
    print(f"  Family 1.5 columns: {f15_cols}")
    print(f"  Family 2   columns: {f2_cols}")


if __name__ == "__main__":
    main()
