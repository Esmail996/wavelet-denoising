from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pywt import iswt, swt

from data_rw import load_trials
from denoising import _bayes_threshold, _garrote_shrink, mad
from preprocess import preprocess_signal_for_denoising

# ---------------------------------------------------------------------------
# Edit these parameters manually, then run: python debug_branch_a_single_signal.py
# ---------------------------------------------------------------------------
INPUT_PICKLE = r"Multifrequenz Dataset/Multifrequenz/Box/25cm_0Grad.pickle"
MIC = "Mic2"
TRIAL = 0
GATE_START_SAMPLE = 2400

OUTPUT_DIR = r"outputs/branch_a_debug_single"

# Branch A settings (kept aligned with run_wavelet_denoising.py)
FS_HZ = 2_000_000
BP_CENTER_HZ = 50_000
BP_BW_HZ = 25_000
BP_METHOD = "iir"
BP_IIR_ORDER = 4

SWT_WAVELET = "sym6"
SWT_LEVEL = 6
SWT_NOISE_LEVEL = 4
LEVEL_SCALE: dict[int, float] = {
    4: 0.8,
    5: 0.5,
    6: 0.8,
}

# Fixed noise sigma per mic from reference dataset
SIGMA_N = {
    "Mic1": 3.960309300521,
    "Mic2": 2.892926398439,
    "Mic3": 0.951529750934,
}

def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    nonzero = int(np.count_nonzero(x))
    total = int(x.size)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "max_abs": float(np.max(np.abs(x))),
        "energy": float(np.dot(x, x)),
        "nonzero": float(nonzero),
        "nonzero_ratio": float(nonzero / total) if total else 0.0,
    }


def _gate_signal(signal: np.ndarray, gate_start_sample: int) -> np.ndarray:
    gated = np.asarray(signal, dtype=float).copy()
    if gate_start_sample > 0:
        gated[:gate_start_sample] = 0.0
    return gated


def branch_a_debug(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    noise_level: int,
    level_scale: dict[int, float] | None,
    sigma_n_fixed: float | None,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    x = np.asarray(signal, dtype=np.float64)
    n_orig = int(len(x))

    pad = (-n_orig) % (2 ** level)
    if pad:
        x_padded = np.concatenate([x, np.zeros(pad)])
    else:
        x_padded = x

    coeffs = swt(x_padded, wavelet, level=level, trim_approx=True, norm=True)
    cA = np.asarray(coeffs[0], dtype=float)

    noise_idx = level - noise_level + 1
    if sigma_n_fixed is not None:
        sigma_n = float(sigma_n_fixed)
        sigma_source = "fixed"
    else:
        sigma_n = float(mad(coeffs[noise_idx]))
        sigma_source = "mad_noise_level"

    detail_rows: list[dict] = []
    new_coeffs = [cA]

    arrays: dict[str, np.ndarray] = {
        "raw_input": x.copy(),
        "preprocessed_input": x.copy(),
        "padded_input": x_padded.copy(),
        "approximation_cA": cA.copy(),
    }

    for idx in range(1, level + 1):
        physical_level = level - idx + 1
        d_before = np.asarray(coeffs[idx], dtype=float)

        lam_base = float(_bayes_threshold(d_before, sigma_n))
        scale = float(level_scale.get(physical_level, 1.0)) if level_scale is not None else 1.0
        lam_final = lam_base * scale

        d_after = np.asarray(_garrote_shrink(d_before, lam_final), dtype=float)
        new_coeffs.append(d_after)

        arrays[f"detail_before_L{physical_level}"] = d_before.copy()
        arrays[f"detail_after_L{physical_level}"] = d_after.copy()

        detail_rows.append(
            {
                "physical_level": int(physical_level),
                "coeff_index": int(idx),
                "threshold_base": lam_base,
                "threshold_scale": scale,
                "threshold_final": float(lam_final),
                "before_mean": _stats(d_before)["mean"],
                "before_std": _stats(d_before)["std"],
                "before_energy": _stats(d_before)["energy"],
                "before_max_abs": _stats(d_before)["max_abs"],
                "before_nonzero_ratio": _stats(d_before)["nonzero_ratio"],
                "after_mean": _stats(d_after)["mean"],
                "after_std": _stats(d_after)["std"],
                "after_energy": _stats(d_after)["energy"],
                "after_max_abs": _stats(d_after)["max_abs"],
                "after_nonzero_ratio": _stats(d_after)["nonzero_ratio"],
            }
        )

    rec = iswt(new_coeffs, wavelet, norm=True)
    denoised = np.asarray(rec, dtype=float)[:n_orig]

    arrays["denoised_output"] = denoised.copy()

    report = {
        "input": {
            "input_pickle": INPUT_PICKLE,
            "mic": MIC,
            "trial": int(TRIAL),
            "gate_start_sample": int(GATE_START_SAMPLE),
        },
        "preprocessing": {
            "fs_hz": int(FS_HZ),
            "bp_center_hz": int(BP_CENTER_HZ),
            "bp_bw_hz": int(BP_BW_HZ),
            "bp_method": BP_METHOD,
            "bp_iir_order": int(BP_IIR_ORDER),
        },
        "swt": {
            "wavelet": wavelet,
            "level": int(level),
            "noise_level": int(noise_level),
            "noise_idx": int(noise_idx),
            "n_original": n_orig,
            "n_padded": int(len(x_padded)),
            "pad_samples": int(pad),
            "sigma_n": float(sigma_n),
            "sigma_n_source": sigma_source,
            "level_scale": level_scale,
        },
        "signal_stats": {
            "raw": _stats(signal),
            "denoised": _stats(denoised),
        },
        "levels": sorted(detail_rows, key=lambda r: r["physical_level"]),
    }

    return denoised, report, arrays


def _write_text_log(report: dict, txt_path: Path) -> None:
    lines: list[str] = []
    lines.append("Branch A one-signal debug report")
    lines.append("=" * 40)
    lines.append(f"input_pickle: {report['input']['input_pickle']}")
    lines.append(f"mic: {report['input']['mic']} | trial: {report['input']['trial']}")
    lines.append("")

    swt_meta = report["swt"]
    lines.append("SWT meta")
    lines.append(f"wavelet={swt_meta['wavelet']} level={swt_meta['level']} noise_level={swt_meta['noise_level']} noise_idx={swt_meta['noise_idx']}")
    lines.append(f"n_original={swt_meta['n_original']} n_padded={swt_meta['n_padded']} pad_samples={swt_meta['pad_samples']}")
    lines.append(f"sigma_n={swt_meta['sigma_n']:.12g} source={swt_meta['sigma_n_source']}")
    lines.append(f"level_scale={swt_meta['level_scale']}")
    lines.append("")

    lines.append("Per-level thresholds and coeff stats")
    for row in report["levels"]:
        lines.append(
            "L{lvl}: T_base={tb:.6g}, scale={sc:.6g}, T_final={tf:.6g}, "
            "before_std={bs:.6g}, after_std={as_:.6g}, before_nz={bnz:.4f}, after_nz={anz:.4f}".format(
                lvl=row["physical_level"],
                tb=row["threshold_base"],
                sc=row["threshold_scale"],
                tf=row["threshold_final"],
                bs=row["before_std"],
                as_=row["after_std"],
                bnz=row["before_nonzero_ratio"],
                anz=row["after_nonzero_ratio"],
            )
        )

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = load_trials(INPUT_PICKLE)
    if MIC not in trials:
        raise KeyError(f"Mic key '{MIC}' not found. Available keys: {list(trials.keys())}")
    if TRIAL < 0 or TRIAL >= len(trials[MIC]):
        raise IndexError(f"Trial {TRIAL} out of range (0..{len(trials[MIC]) - 1}) for mic {MIC}")

    raw_signal = np.asarray(trials[MIC][TRIAL], dtype=float)
    preprocessed = preprocess_signal_for_denoising(
        signal=raw_signal,
        preprocessing="bandpassed",
        fs_hz=FS_HZ,
        band_center_hz=BP_CENTER_HZ,
        bw_hz=BP_BW_HZ,
        bp_method=BP_METHOD,
        iir_order=BP_IIR_ORDER,
    )
    gated_signal = _gate_signal(preprocessed, GATE_START_SAMPLE)

    sigma_n_fixed = SIGMA_N.get(MIC)
    denoised, report, arrays = branch_a_debug(
        signal=gated_signal,
        wavelet=SWT_WAVELET,
        level=SWT_LEVEL,
        noise_level=SWT_NOISE_LEVEL,
        level_scale=LEVEL_SCALE,
        sigma_n_fixed=sigma_n_fixed,
    )

    arrays["raw_input"] = raw_signal
    arrays["gated_input"] = gated_signal
    arrays["preprocessed_input"] = preprocessed
    arrays["denoised_output"] = denoised
    report["signal_stats"] = {
        "raw": _stats(raw_signal),
        "gated": _stats(gated_signal),
        "preprocessed": _stats(preprocessed),
        "denoised": _stats(denoised),
    }

    stem = f"branchA_debug_{Path(INPUT_PICKLE).stem}_{MIC}_trial{TRIAL}"
    report_json = output_dir / f"{stem}_report.json"
    report_txt = output_dir / f"{stem}_report.txt"
    levels_csv = output_dir / f"{stem}_levels.csv"
    arrays_npz = output_dir / f"{stem}_arrays.npz"

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_text_log(report, report_txt)
    pd.DataFrame(report["levels"]).to_csv(levels_csv, index=False)
    np.savez_compressed(arrays_npz, **arrays)

    print(f"Saved JSON report: {report_json}")
    print(f"Saved text report: {report_txt}")
    print(f"Saved level table: {levels_csv}")
    print(f"Saved arrays dump: {arrays_npz}")


if __name__ == "__main__":
    main()
