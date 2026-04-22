from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, hilbert

from bandpass import bandpass
from roi_windows import ROI_WINDOWS, parse_distance_cm


NAME_RE = re.compile(r"(?P<dist>\d+)\s*cm[_-](?P<ang>-?\d+)\s*Grad", re.IGNORECASE)
BANDS_HZ = (40_000.0, 50_000.0, 60_000.0)
EPS = 1e-12


@dataclass(frozen=True)
class FilterConfig:
    cfg_id: str
    method: str
    order: int
    bw_hz: float


def parse_dist_angle(filename: str) -> tuple[int | None, int | None]:
    match = NAME_RE.search(filename)
    if not match:
        return None, None
    return int(match.group("dist")), int(match.group("ang"))


def iter_pickle_files(data_root: Path, include_reference: bool = False):
    for category_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        if not include_reference and category_dir.name.lower() == "referenz":
            continue
        for fp in sorted(category_dir.glob("*.pickle")):
            yield category_dir.name, fp


def extract_mic_trials(obj, mic_col: str):
    mic_col_l = mic_col.lower()

    if isinstance(obj, pd.DataFrame):
        col_lut = {c.lower(): c for c in obj.columns}
        if mic_col_l not in col_lut:
            raise ValueError(
                f"Expected column '{mic_col}' in DataFrame. Got columns: {list(obj.columns)}"
            )
        col = col_lut[mic_col_l]
        return [np.asarray(obj.iloc[i][col], dtype=float) for i in range(len(obj))]

    arr = np.asarray(obj, dtype=object)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Unsupported pickle array shape: {arr.shape}")

    mic_idx = {"mic1": 0, "mic2": 1, "mic3": 2}[mic_col_l]
    if mic_idx >= arr.shape[0]:
        raise ValueError(f"Array does not contain {mic_col}. Shape: {arr.shape}")

    return [np.asarray(arr[mic_idx, j], dtype=float) for j in range(arr.shape[1])]


def mean_groups(trials: list[np.ndarray], group_size: int) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if len(trials) < group_size:
        raise ValueError(f"Not enough trials: {len(trials)} < group_size={group_size}")

    n_groups = len(trials) // group_size
    if n_groups == 0:
        raise ValueError("No groups can be formed.")

    out_means: list[np.ndarray] = []
    out_ranges: list[tuple[int, int]] = []
    for g in range(n_groups):
        i0 = g * group_size
        i1 = i0 + group_size
        chunk = trials[i0:i1]

        lengths = {len(x) for x in chunk}
        if len(lengths) != 1:
            raise ValueError(f"Signals in one group have different lengths: {sorted(lengths)}")

        out_means.append(np.mean(np.stack(chunk, axis=0), axis=0))
        out_ranges.append((i0, i1 - 1))

    return out_means, out_ranges


def preprocess_dc_detrend(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    x = detrend(x, type="linear")
    return x


def roi_indices_for_file(filename: str) -> tuple[int, int] | None:
    distance_cm = parse_distance_cm(filename)
    if distance_cm is None:
        return None
    return ROI_WINDOWS.get(int(distance_cm))


def safe_ratio(numer: float, denom: float) -> float:
    return float(numer / (denom + EPS))


def db10(ratio: float) -> float:
    return float(10.0 * np.log10(max(ratio, EPS)))


def db20(ratio: float) -> float:
    return float(20.0 * np.log10(max(ratio, EPS)))


def spectrum(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 2:
        return np.array([0.0]), np.array([0.0])
    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    power = (np.abs(spec) ** 2).astype(float)
    return freqs, power


def band_energy(freqs: np.ndarray, power: np.ndarray, center_hz: float, halfwidth_hz: float) -> float:
    lo = center_hz - halfwidth_hz
    hi = center_hz + halfwidth_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def peak_in_band(freqs: np.ndarray, power: np.ndarray, center_hz: float, halfwidth_hz: float) -> tuple[float, float]:
    lo = center_hz - halfwidth_hz
    hi = center_hz + halfwidth_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0, center_hz
    local_power = power[mask]
    local_freqs = freqs[mask]
    idx = int(np.argmax(local_power))
    return float(local_power[idx]), float(local_freqs[idx])


def compute_leakage_metrics(roi: np.ndarray, fs_hz: float, target_hz: float, energy_halfwidth_hz: float) -> dict[str, float]:
    freqs, power = spectrum(roi, fs_hz)

    energies = {
        40_000.0: band_energy(freqs, power, 40_000.0, energy_halfwidth_hz),
        50_000.0: band_energy(freqs, power, 50_000.0, energy_halfwidth_hz),
        60_000.0: band_energy(freqs, power, 60_000.0, energy_halfwidth_hz),
    }

    main_energy = energies[target_hz]
    off_energy = sum(v for k, v in energies.items() if k != target_hz)

    main_peak_power, dominant_freq_hz = peak_in_band(freqs, power, target_hz, energy_halfwidth_hz)
    secondary_peak_power = max(
        peak_in_band(freqs, power, k, energy_halfwidth_hz)[0] for k in energies.keys() if k != target_hz
    )

    main_off_ratio = safe_ratio(main_energy, off_energy)
    peak_ratio = safe_ratio(main_peak_power, secondary_peak_power)

    return {
        "target_energy": float(main_energy),
        "off_energy": float(off_energy),
        "main_off_ratio": float(main_off_ratio),
        "main_off_ratio_db": db10(main_off_ratio),
        "main_peak_power": float(main_peak_power),
        "secondary_peak_power": float(secondary_peak_power),
        "main_secondary_peak_ratio": float(peak_ratio),
        "main_secondary_peak_ratio_db": db20(peak_ratio),
        "dominant_freq_hz": float(dominant_freq_hz),
        "dominant_freq_error_hz": float(abs(dominant_freq_hz - target_hz)),
        "energy_40k": float(energies[40_000.0]),
        "energy_50k": float(energies[50_000.0]),
        "energy_60k": float(energies[60_000.0]),
    }


def compute_detectability_metrics(roi: np.ndarray, fs_hz: float) -> dict[str, float]:
    x = np.asarray(roi, dtype=float)
    if len(x) == 0:
        return {
            "roi_rms": 0.0,
            "envelope_peak": 0.0,
            "envelope_median": 0.0,
            "envelope_peak_prominence_ratio": 0.0,
            "envelope_peak_to_second_ratio": 0.0,
            "peak_count_over_35pct": 0.0,
            "false_peak_penalty": 0.0,
            "detectability_score": -1e9,
            "peak_index": 0.0,
            "peak_time_us": 0.0,
        }

    env = np.abs(hilbert(x))

    # Use local maxima for peak ambiguity; avoid neighboring-sample artifacts.
    peak_indices, _ = find_peaks(env)
    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(env))], dtype=int)

    peak_values = env[peak_indices]
    order_desc = np.argsort(peak_values)[::-1]
    ranked_peak_indices = peak_indices[order_desc]
    ranked_peak_values = peak_values[order_desc]

    main_peak = float(ranked_peak_values[0])
    peak_index = int(ranked_peak_indices[0])

    rms = float(np.sqrt(np.mean(x ** 2)))
    env_median = float(np.median(env))
    prominence_ratio = safe_ratio(main_peak, env_median)

    peak_threshold = 0.35 * main_peak
    peak_count = int(np.sum(peak_values >= peak_threshold))

    second_peak = float(ranked_peak_values[1]) if len(ranked_peak_values) >= 2 else 0.0
    main_second_ratio = safe_ratio(main_peak, second_peak)

    false_peak_penalty = max(0.0, float(peak_count - 1))

    score = (
        0.55 * np.log10(max(prominence_ratio, EPS))
        + 0.45 * np.log10(max(main_second_ratio, EPS))
        - 0.10 * false_peak_penalty
    )

    return {
        "roi_rms": float(rms),
        "envelope_peak": float(main_peak),
        "envelope_median": float(env_median),
        "envelope_peak_prominence_ratio": float(prominence_ratio),
        "envelope_peak_to_second_ratio": float(main_second_ratio),
        "peak_count_over_35pct": float(peak_count),
        "false_peak_penalty": float(false_peak_penalty),
        "detectability_score": float(score),
        "peak_index": float(peak_index),
        "peak_time_us": float((peak_index / fs_hz) * 1e6),
    }


def aggregate_trial_metrics(df_trials: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["target_band_hz", "cfg_id", "order", "bw_hz"]
    _NON_METRIC = frozenset([
        "category", "file", "distance_cm", "angle_deg", "mic",
        "run_idx", "run_trial_start", "run_trial_end",
        "roi_start", "roi_end", "roi_len", "method", "energy_halfwidth_hz",
    ])

    numeric_cols = [c for c in df_trials.select_dtypes(include="number").columns
                    if c not in group_keys and c not in _NON_METRIC]

    # mean + std for every metric column
    agg_spec = {col: ["mean", "std"] for col in numeric_cols}
    grouped = df_trials.groupby(group_keys, as_index=False)
    summary = grouped.agg(agg_spec)

    # flatten multi-level column index
    summary.columns = [
        "_".join(filter(None, col)).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # add n_samples
    n_samples = df_trials.groupby(group_keys, as_index=False).size().rename(columns={"size": "n_samples"})
    summary = summary.merge(n_samples, on=group_keys, how="left")

    # fill NaN stds with 0
    std_cols = [c for c in summary.columns if c.endswith("_std")]
    for col in std_cols:
        summary[col] = summary[col].fillna(0.0)

    summary = summary.sort_values(
        by=[
            "target_band_hz",
            "detectability_score_mean",
            "main_off_ratio_db_mean",
            "main_secondary_peak_ratio_db_mean",
            "dominant_freq_error_hz_mean",
            "detectability_score_std",
        ],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)

    return summary


def choose_band_winners(df_summary: pd.DataFrame) -> pd.DataFrame:

    max_cols = [
        "detectability_score_mean",
        "main_off_ratio_db_mean",
        "main_secondary_peak_ratio_db_mean",
    ]
    min_cols = [
        "dominant_freq_error_hz_mean",
        "detectability_score_std",
    ]

    def dominates(a: pd.Series, b: pd.Series) -> bool:
        ge_all = all(float(a[c]) >= float(b[c]) for c in max_cols)
        le_all = all(float(a[c]) <= float(b[c]) for c in min_cols)
        gt_any = any(float(a[c]) > float(b[c]) for c in max_cols)
        lt_any = any(float(a[c]) < float(b[c]) for c in min_cols)
        return ge_all and le_all and (gt_any or lt_any)

    winners = []
    for band_hz in sorted(df_summary["target_band_hz"].unique()):
        sub = df_summary[df_summary["target_band_hz"] == band_hz].copy()
        if sub.empty:
            continue

        n = len(sub)
        dominated = np.zeros(n, dtype=bool)
        sub_rows = [row for _, row in sub.iterrows()]

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if dominates(sub_rows[j], sub_rows[i]):
                    dominated[i] = True
                    break

        pareto_front = sub.loc[~dominated].copy()
        front_size = int(len(pareto_front))

        pareto_front = pareto_front.sort_values(
            by=[
                "detectability_score_mean",
                "main_off_ratio_db_mean",
                "dominant_freq_error_hz_mean",
                "main_secondary_peak_ratio_db_mean",
                "detectability_score_std",
            ],
            ascending=[False, False, True, False, True],
        )

        chosen = pareto_front.iloc[0].copy()
        chosen["pareto_front_size"] = front_size
        chosen["pareto_selected_by_tiebreak"] = bool(front_size > 1)
        winners.append(chosen)

    if not winners:
        return pd.DataFrame(columns=df_summary.columns)

    return pd.DataFrame(winners).reset_index(drop=True)


def parse_csv_numbers(text: str, cast=float) -> list:
    vals = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(cast(tok))
    return vals


def build_filter_grid(orders: list[int], bws_hz: list[float]) -> list[FilterConfig]:
    cfgs: list[FilterConfig] = []
    for order in orders:
        for bw_hz in bws_hz:
            cfgs.append(
                FilterConfig(
                    cfg_id=f"iir_o{order}_bw{int(round(bw_hz))}",
                    method="iir",
                    order=int(order),
                    bw_hz=float(bw_hz),
                )
            )
    return cfgs


def evaluate_grid(
    data_root: Path,
    out_dir: Path,
    mic: str = "Mic2",
    fs_hz: float = 2_000_000.0,
    include_reference: bool = False,
    orders: list[int] | None = None,
    bws_hz: list[float] | None = None,
    energy_halfwidth_hz: float = 1_500.0,
    group_size: int = 25,
    max_files: int | None = None,
    max_trials: int | None = None,
    exclude_distances: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if orders is None:
        orders = [1, 2, 3, 4, 5, 6, 7, 8]
    if bws_hz is None:
        bws_hz = [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]

    grid = build_filter_grid(orders=orders, bws_hz=bws_hz)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_set = set(exclude_distances or [])

    rows = []
    file_count = 0
    run_count = 0

    for category, fp in iter_pickle_files(data_root, include_reference=include_reference):
        if max_files is not None and file_count >= max_files:
            break

        roi_idx = roi_indices_for_file(fp.name)
        if roi_idx is None:
            continue

        start, end = roi_idx
        dist_cm, angle_deg = parse_dist_angle(fp.name)

        if dist_cm in excluded_set:
            continue
        payload = pd.read_pickle(fp)
        mic_trials = extract_mic_trials(payload, mic_col=mic)
        if max_trials is not None:
            mic_trials = mic_trials[:max_trials]

        run_trials = mic_trials
        trial_ranges = [(i, i) for i in range(len(mic_trials))]

        # Mean-group fallback (disabled by default on purpose):
        # run_trials, trial_ranges = mean_groups(mic_trials, group_size=group_size)

        run_count += len(run_trials)

        file_count += 1

        for run_idx, (trial_signal, trial_range) in enumerate(zip(run_trials, trial_ranges)):
            trial_pre = preprocess_dc_detrend(np.asarray(trial_signal, dtype=float))
            n = len(trial_pre)
            if n <= start:
                continue
            end_idx = min(end, n)
            if end_idx - start < 8:
                continue

            for cfg in grid:
                for target_hz in BANDS_HZ:
                    try:
                        filtered = bandpass(
                            trial_pre,
                            fs=fs_hz,
                            f0=target_hz,
                            bw=cfg.bw_hz,
                            method=cfg.method,
                            order=cfg.order,
                        )
                    except ValueError:
                        continue

                    roi = np.asarray(filtered[start:end_idx], dtype=float)
                    if len(roi) < 8:
                        continue

                    detectability = compute_detectability_metrics(roi=roi, fs_hz=fs_hz)
                    leakage = compute_leakage_metrics(
                        roi=roi,
                        fs_hz=fs_hz,
                        target_hz=target_hz,
                        energy_halfwidth_hz=energy_halfwidth_hz,
                    )

                    rows.append(
                        {
                            "category": category,
                            "file": fp.name,
                            "distance_cm": dist_cm,
                            "angle_deg": angle_deg,
                            "mic": mic,
                            "run_idx": int(run_idx),
                            "run_trial_start": int(trial_range[0]),
                            "run_trial_end": int(trial_range[1]),
                            "roi_start": int(start),
                            "roi_end": int(end_idx),
                            "roi_len": int(end_idx - start),
                            "target_band_hz": float(target_hz),
                            "cfg_id": cfg.cfg_id,
                            "method": cfg.method,
                            "order": int(cfg.order),
                            "bw_hz": float(cfg.bw_hz),
                            "energy_halfwidth_hz": float(energy_halfwidth_hz),
                            **detectability,
                            **leakage,
                        }
                    )

    df_trials = pd.DataFrame(rows)
    if df_trials.empty:
        raise RuntimeError("No trial rows were produced. Check dataset path and ROI mapping.")

    df_summary = aggregate_trial_metrics(df_trials)
    df_winners = choose_band_winners(df_summary)

    df_trials.to_csv(out_dir / "pilot_iir_grid_trials.csv", index=False)
    df_summary.to_csv(out_dir / "pilot_iir_grid_summary.csv", index=False)
    df_winners.to_csv(out_dir / "pilot_iir_grid_winners.csv", index=False)

    meta = {
        "fs_hz": fs_hz,
        "mic": mic,
        "include_reference": include_reference,
        "orders": orders,
        "bws_hz": bws_hz,
        "energy_halfwidth_hz": energy_halfwidth_hz,
        "group_size": group_size,
        "averaging_mode": "per_trial_no_grouping",
        "n_configs": len(grid),
        "n_grouped_runs": run_count,
        "n_runs": run_count,
        "n_trial_rows": int(len(df_trials)),
        "exclude_distances": list(exclude_distances) if exclude_distances else [],
    }
    with open(out_dir / "pilot_iir_grid_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df_trials, df_summary, df_winners


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pilot IIR bandpass grid search for 40/50/60 kHz channel separation. "
            "Ranks configs by ROI detectability first and leakage as tie-breaker."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"Multifrequenz Dataset\Multifrequenz",
        help="Root folder containing object folders with pickle files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"outputs\bandpass_grid_search",
        help="Directory for trial/summary/winner outputs.",
    )
    parser.add_argument("--mic", type=str, default="Mic2", choices=["Mic1", "Mic2", "Mic3"])
    parser.add_argument("--fs-hz", type=float, default=2_000_000.0)
    parser.add_argument(
        "--orders",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated IIR orders, e.g. 1,2,3,4,5,6,7,8",
    )
    parser.add_argument(
        "--bws-hz",
        type=str,
        default="2500,3000,3500,4000,4500,5000,5500,6000,6500,7000",
        help="Comma-separated half-bandwidths in Hz.",
    )
    parser.add_argument(
        "--energy-halfwidth-hz",
        type=float,
        default=1500.0,
        help="Half-width around 40/50/60 kHz used for spectral energy accounting.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=25,
        help="Average this many consecutive trials into one run (25 -> 4 runs for 100 trials).",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Include referenz folder in search input (off by default).",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick dry runs.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional per-file trial cap.")
    parser.add_argument(
        "--exclude-distances",
        type=str,
        default="",
        help="Comma-separated distances in cm to exclude (e.g. '100,125'). Empty means include all.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    orders = parse_csv_numbers(args.orders, cast=int)
    bws_hz = parse_csv_numbers(args.bws_hz, cast=float)
    exclude_distances = parse_csv_numbers(args.exclude_distances, cast=int) if args.exclude_distances.strip() else []

    df_trials, df_summary, df_winners = evaluate_grid(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        mic=args.mic,
        fs_hz=float(args.fs_hz),
        include_reference=bool(args.include_reference),
        orders=orders,
        bws_hz=bws_hz,
        energy_halfwidth_hz=float(args.energy_halfwidth_hz),
        group_size=int(args.group_size),
        max_files=args.max_files,
        max_trials=args.max_trials,
        exclude_distances=exclude_distances if exclude_distances else None,
    )

    print(f"Trial rows: {len(df_trials)}")
    print(f"Summary rows: {len(df_summary)}")
    print(f"Winner rows: {len(df_winners)}")
    print(f"Output dir: {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
