from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import paper_opt_pipeline as pop


def _fmt(x: float) -> str:
    return f"{x:.12e}"


def find_negative_case(
    per_row_csv: Path,
    band: Optional[str],
    wavelet: Optional[str],
    level: Optional[int],
    chunksize: int = 200_000,
) -> Dict[str, object]:
    usecols = [
        "source_file",
        "source_filename",
        "row_idx",
        "band",
        "wavelet",
        "level",
        "distance_cm",
        "roi_start",
        "roi_end",
        "snr_db_eval_rms_ratio",
        "snr_db_peak",
        "r",
        "ref_rms_agg",
        "ref_rms_raw_agg",
        "ref_rms_eff",
    ]

    for chunk in pd.read_csv(per_row_csv, usecols=usecols, chunksize=chunksize):
        mask = chunk["snr_db_eval_rms_ratio"] < 0.0
        if band is not None:
            mask &= chunk["band"] == band
        if wavelet is not None:
            mask &= chunk["wavelet"] == wavelet
        if level is not None:
            mask &= chunk["level"] == int(level)

        hit = chunk[mask]
        if not hit.empty:
            return hit.iloc[0].to_dict()

    raise RuntimeError(
        "No negative case found with current filters. "
        "Try removing wavelet/level constraints or changing band."
    )


def build_reference_state() -> tuple[Dict[str, float], Dict[str, List[np.ndarray]], Dict[str, List[int]]]:
    ref_dir = Path(pop.REF_DIR)
    ref_files = sorted(ref_dir.glob("*.pickle"))
    if not ref_files:
        raise FileNotFoundError(f"No reference pickle files in {ref_dir}")

    ref_frames = [pd.read_pickle(p) for p in ref_files]
    df_ref = pd.concat(ref_frames, ignore_index=True)

    if pop.MIC_COL not in df_ref.columns:
        raise KeyError(f"MIC_COL '{pop.MIC_COL}' not in reference dataframe columns")

    n_ref = len(df_ref) if pop.MAX_ROWS_REF is None else min(len(df_ref), int(pop.MAX_ROWS_REF))

    noise_sig: Dict[str, List[float]] = {"40k": [], "50k": [], "60k": []}
    ref_band_cache: Dict[str, List[np.ndarray]] = {"40k": [], "50k": [], "60k": []}

    for i in range(n_ref):
        r40, r50, r60 = pop.compute_bands(df_ref, i, pop.MIC_COL)
        for bname, r in zip(["40k", "50k", "60k"], [r40, r50, r60]):
            if r.size == 0:
                continue
            noise_sig[bname].append(pop.robust_sigma_from_time_domain(r))
            ref_band_cache[bname].append(r)

    sigma_noise = {k: float(np.median(v)) for k, v in noise_sig.items()}

    ref_subset_idx_by_band: Dict[str, List[int]] = {}
    for b in ["40k", "50k", "60k"]:
        ref_subset_idx_by_band[b] = pop.deterministic_subset_indices(
            len(ref_band_cache[b]), pop.REF_SUBSET_SIZE
        )

    return sigma_noise, ref_band_cache, ref_subset_idx_by_band


def locate_signal_path(case: Dict[str, object]) -> Path:
    source_file = str(case["source_file"]).replace("\\", "/")
    source_filename = str(case["source_filename"])

    # source_file format in pipeline is: Class/stem
    cls = source_file.split("/")[0]
    sig_path = Path(pop.SIG_DATA_DIR) / cls / source_filename

    if not sig_path.exists():
        raise FileNotFoundError(f"Signal file not found: {sig_path}")
    return sig_path


def debug_case(case: Dict[str, object], max_ref_print: int) -> None:
    band = str(case["band"])
    wavelet = str(case["wavelet"])
    level = int(case["level"])
    row_idx = int(case["row_idx"])
    roi_start = int(case["roi_start"])
    roi_end = int(case["roi_end"])

    print("=" * 90)
    print("DEBUG CASE")
    print("=" * 90)
    print(f"source_file      : {case['source_file']}")
    print(f"source_filename  : {case['source_filename']}")
    print(f"row_idx          : {row_idx}")
    print(f"band             : {band}")
    print(f"wavelet          : {wavelet}")
    print(f"level            : {level}")
    print(f"distance_cm      : {case['distance_cm']}")
    print(f"roi              : [{roi_start}:{roi_end}] (len={roi_end - roi_start})")
    print(f"SNR_SCOPE        : {pop.SNR_SCOPE}")
    print(f"THRESH_RULE      : {pop.THRESH_RULE}")
    print(f"THR_MODE         : {pop.THR_MODE}")
    print(f"REF_SUBSET_AGG   : {pop.REF_SUBSET_AGG}")
    print(f"REF_RMS_FLOOR_FRAC: {pop.REF_RMS_FLOOR_FRAC}")

    sig_path = locate_signal_path(case)
    df_sig = pd.read_pickle(sig_path)
    if pop.MIC_COL not in df_sig.columns:
        raise KeyError(f"MIC_COL '{pop.MIC_COL}' not in signal dataframe columns")

    s40, s50, s60 = pop.compute_bands(df_sig, row_idx, pop.MIC_COL)
    bands = {"40k": s40, "50k": s50, "60k": s60}
    orig = bands[band]

    sigma_noise, ref_band_cache, ref_subset_idx_by_band = build_reference_state()
    sigma_band = sigma_noise[band]
    ref_subset_idx = ref_subset_idx_by_band[band]

    print("\n" + "-" * 90)
    print("STEP 1: SIGNAL DENOISING")
    print("-" * 90)
    print(f"orig_len         : {len(orig)}")
    print(f"sigma_noise[{band}] : {_fmt(sigma_band)}")

    den_sig = pop.wavelet_denoise(
        orig,
        wavelet=wavelet,
        level=level,
        sigma_noise=sigma_band,
        thr_rule=pop.THRESH_RULE,
        thr_mode=pop.THR_MODE,
    )

    roi_sig = den_sig[roi_start:roi_end]
    snr_sig = pop.get_snr_segment(den_sig, roi_start, roi_end)
    sig_rms = pop.segment_rms(snr_sig)
    sig_peak = float(np.max(np.abs(roi_sig))) if roi_sig.size else 0.0

    print(f"den_sig_len      : {len(den_sig)}")
    print(f"snr_seg_len      : {len(snr_sig)}")
    print(f"sig_rms          : {_fmt(sig_rms)}")
    print(f"sig_peak (ROI)   : {_fmt(sig_peak)}")

    print("\n" + "-" * 90)
    print("STEP 2: REFERENCE DENOMINATOR (candidate-coupled)")
    print("-" * 90)
    print(f"reference subset n: {len(ref_subset_idx)}")

    ref_rms_vals: List[float] = []
    ref_peak_vals: List[float] = []
    ref_rms_raw_vals: List[float] = []

    for j, idx_ref in enumerate(ref_subset_idx):
        ref_orig = ref_band_cache[band][idx_ref]

        raw_snr_ref = pop.get_snr_segment(ref_orig, roi_start, roi_end)
        raw_rms = pop.segment_rms(raw_snr_ref)
        ref_rms_raw_vals.append(raw_rms)

        den_ref = pop.wavelet_denoise(
            ref_orig,
            wavelet=wavelet,
            level=level,
            sigma_noise=sigma_band,
            thr_rule=pop.THRESH_RULE,
            thr_mode=pop.THR_MODE,
        )

        roi_ref = den_ref[roi_start:roi_end]
        snr_ref = pop.get_snr_segment(den_ref, roi_start, roi_end)
        den_rms = pop.segment_rms(snr_ref)
        peak = float(np.max(np.abs(roi_ref))) if roi_ref.size else 0.0

        ref_rms_vals.append(den_rms)
        ref_peak_vals.append(peak)

        if j < max_ref_print:
            print(
                f"ref[{idx_ref:3d}] raw_rms={_fmt(raw_rms)} den_rms={_fmt(den_rms)} "
                f"roi_peak={_fmt(peak)}"
            )

    if pop.REF_SUBSET_AGG == "median":
        ref_rms_agg = float(np.median(ref_rms_vals)) if ref_rms_vals else 0.0
        ref_peak_agg = float(np.median(ref_peak_vals)) if ref_peak_vals else 0.0
        ref_rms_raw_agg = float(np.median(ref_rms_raw_vals)) if ref_rms_raw_vals else 0.0
    elif pop.REF_SUBSET_AGG == "mean":
        ref_rms_agg = float(np.mean(ref_rms_vals)) if ref_rms_vals else 0.0
        ref_peak_agg = float(np.mean(ref_peak_vals)) if ref_peak_vals else 0.0
        ref_rms_raw_agg = float(np.mean(ref_rms_raw_vals)) if ref_rms_raw_vals else 0.0
    else:
        raise ValueError(f"Unknown REF_SUBSET_AGG: {pop.REF_SUBSET_AGG}")

    floor_term = pop.REF_RMS_FLOOR_FRAC * ref_rms_raw_agg
    ref_rms_eff = max(ref_rms_agg, floor_term, 1e-12)
    ref_floor_active = ref_rms_eff > ref_rms_agg + 1e-15

    print("\nAggregates:")
    print(f"ref_rms_agg      : {_fmt(ref_rms_agg)}")
    print(f"ref_rms_raw_agg  : {_fmt(ref_rms_raw_agg)}")
    print(f"floor_term       : {_fmt(floor_term)} = REF_RMS_FLOOR_FRAC * ref_rms_raw_agg")
    print(f"ref_rms_eff      : {_fmt(ref_rms_eff)} = max(ref_rms_agg, floor_term, 1e-12)")
    print(f"ref_floor_active : {ref_floor_active}")
    print(f"ref_peak_agg     : {_fmt(ref_peak_agg)}")

    print("\n" + "-" * 90)
    print("STEP 3: METRICS")
    print("-" * 90)

    snr_eval = pop.snr_db_ratio(sig_rms, ref_rms_eff)
    snr_peak = pop.snr_db_roi_peak(roi_sig, ref_peak_agg)
    r_metric = pop.smoothness_r_paper(den_sig, orig)

    ratio_lin = (sig_rms + 1e-12) / (ref_rms_eff + 1e-12)

    print(f"linear RMS ratio : {_fmt(ratio_lin)} = (sig_rms+eps)/(ref_rms_eff+eps)")
    print(f"snr_db_eval      : {snr_eval:.12f} dB = 20*log10(linear ratio)")
    print(f"snr_db_peak      : {snr_peak:.12f} dB")
    print(f"smoothness r     : {r_metric:.12f}")
    print(f"rms_not_above_ref: {sig_rms <= ref_rms_eff}")

    print("\n" + "-" * 90)
    print("STEP 4: COMPARE WITH STORED per_row VALUE")
    print("-" * 90)

    if "snr_db_eval_rms_ratio" in case:
        stored_snr = float(case["snr_db_eval_rms_ratio"])
        print(f"stored snr_db_eval_rms_ratio : {stored_snr:.12f}")
        print(f"recomputed snr_db_eval       : {snr_eval:.12f}")
        print(f"abs diff                     : {abs(stored_snr - snr_eval):.12e}")

    if "ref_rms_agg" in case:
        print(f"stored ref_rms_agg           : {float(case['ref_rms_agg']):.12e}")
        print(f"recomputed ref_rms_agg       : {ref_rms_agg:.12e}")
    if "ref_rms_raw_agg" in case:
        print(f"stored ref_rms_raw_agg       : {float(case['ref_rms_raw_agg']):.12e}")
        print(f"recomputed ref_rms_raw_agg   : {ref_rms_raw_agg:.12e}")
    if "ref_rms_eff" in case:
        print(f"stored ref_rms_eff           : {float(case['ref_rms_eff']):.12e}")
        print(f"recomputed ref_rms_eff       : {ref_rms_eff:.12e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug one negative OPT/SNR case end-to-end and print all intermediate terms "
            "for the denominator and metric calculation."
        )
    )
    parser.add_argument("--band", default="40k", help="Band filter for auto-selected negative case")
    parser.add_argument("--wavelet", default=None, help="Optional wavelet filter for auto-selection")
    parser.add_argument("--level", type=int, default=None, help="Optional level filter for auto-selection")
    parser.add_argument(
        "--source-file",
        default=None,
        help="Optional source_file exact match (e.g., Box/25cm_-10Grad) from per_row csv",
    )
    parser.add_argument("--row-idx", type=int, default=None, help="Optional exact row_idx from per_row csv")
    parser.add_argument("--max-ref-print", type=int, default=16, help="How many reference rows to print")
    args = parser.parse_args()

    per_row_csv = Path(pop.OUT_DIR) / "per_row_wavelet_level.csv"
    if not per_row_csv.exists():
        raise FileNotFoundError(f"Missing file: {per_row_csv}. Run paper_opt_pipeline.py first.")

    case = find_negative_case(
        per_row_csv=per_row_csv,
        band=args.band,
        wavelet=args.wavelet,
        level=args.level,
    )

    if args.source_file is not None:
        if str(case["source_file"]).replace("\\", "/") != args.source_file.replace("\\", "/"):
            # Try finding exact source_file + row_idx override if provided.
            usecols = [
                "source_file",
                "source_filename",
                "row_idx",
                "band",
                "wavelet",
                "level",
                "distance_cm",
                "roi_start",
                "roi_end",
                "snr_db_eval_rms_ratio",
                "snr_db_peak",
                "r",
                "ref_rms_agg",
                "ref_rms_raw_agg",
                "ref_rms_eff",
            ]
            df = pd.read_csv(per_row_csv, usecols=usecols)
            mask = df["source_file"].astype(str).str.replace("\\", "/") == args.source_file.replace("\\", "/")
            if args.row_idx is not None:
                mask &= df["row_idx"] == int(args.row_idx)
            if args.band is not None:
                mask &= df["band"] == args.band
            if args.wavelet is not None:
                mask &= df["wavelet"] == args.wavelet
            if args.level is not None:
                mask &= df["level"] == int(args.level)
            if not df[mask].empty:
                case = df[mask].iloc[0].to_dict()
            else:
                raise RuntimeError("No matching case found for --source-file/--row-idx filters")

    print("Using case found in per_row_wavelet_level.csv:")
    print(
        f"  source_file={case['source_file']} row_idx={case['row_idx']} "
        f"band={case['band']} wavelet={case['wavelet']} level={case['level']} "
        f"snr_db_eval_rms_ratio={float(case['snr_db_eval_rms_ratio']):.12f}"
    )

    debug_case(case, max_ref_print=max(0, int(args.max_ref_print)))


if __name__ == "__main__":
    main()
