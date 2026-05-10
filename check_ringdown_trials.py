from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_trial_signal
from ringdown_handeling import (
    _align_and_scale,
    ringdown_energy_db_reduction,
    ringdown_handle,
    wideband_frontend,
)

FS_HZ = 2_000_000.0
MIC = "Mic2"
TRIAL = 0


def _t_ms(n: int) -> np.ndarray:
    return np.arange(n, dtype=float) / FS_HZ * 1e3


def _plot_ringdown_diagnostic(
    label: str,
    x_raw: np.ndarray,
    x_proc: np.ndarray,
    r_aligned: np.ndarray | None,
    alpha: float | None,
    y: np.ndarray,
    reduction_db: float,
    lag: int | None,
    out_path: Path,
) -> None:
    t = _t_ms(x_raw.size)
    zoom_end_ms = 1.5  # ringdown region to zoom
    zoom_samples = int(zoom_end_ms * FS_HZ / 1e3)

    n_rows = 4 if r_aligned is not None else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 3.0 * n_rows), sharex=False)

    # row 0: full raw signal
    axes[0].plot(t, x_raw, color="#1e3a5f", linewidth=0.7)
    axes[0].set_title("Raw signal (full length)")
    axes[0].set_ylabel("ADC counts")
    axes[0].axvspan(0, zoom_end_ms, color="orange", alpha=0.10, label="ringdown window")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.25)

    # row 1: zoom — raw vs processed
    tz = _t_ms(zoom_samples)
    axes[1].plot(tz, x_proc[:zoom_samples], color="#1e3a5f", linewidth=0.9, label="preprocessed (before subtraction)")
    if r_aligned is not None and alpha is not None:
        axes[1].plot(tz, (alpha * r_aligned)[:zoom_samples], color="#dc2626", linewidth=0.9,
                     alpha=0.75, label=f"alpha·ref (alpha={alpha:.4f}, lag={lag})")
    axes[1].set_title(f"Ringdown window zoom [0 – {zoom_end_ms} ms]")
    axes[1].set_ylabel("ADC counts")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    # row 2: residual (y) zoomed
    axes[2].plot(tz, y[:zoom_samples], color="#065f46", linewidth=0.9, label="after ringdown handling")
    axes[2].plot(tz, x_proc[:zoom_samples], color="#94a3b8", linewidth=0.7, alpha=0.55, label="before")
    axes[2].set_title(f"Residual [0 – {zoom_end_ms} ms]   |   reduction = {reduction_db:.1f} dB")
    axes[2].set_ylabel("ADC counts")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.25)

    # row 3 (template path only): full-length overlay
    if r_aligned is not None:
        axes[3].plot(t, y, color="#065f46", linewidth=0.7, label="after ringdown handling")
        axes[3].plot(t, x_proc, color="#94a3b8", linewidth=0.6, alpha=0.55, label="preprocessed")
        axes[3].axvspan(0, zoom_end_ms, color="orange", alpha=0.08)
        axes[3].set_title("Full length — before vs after")
        axes[3].set_ylabel("ADC counts")
        axes[3].set_xlabel("Time (ms)")
        axes[3].legend(fontsize=8, loc="upper right")
        axes[3].grid(True, alpha=0.25)
    else:
        axes[2].set_xlabel("Time (ms)")

    fig.suptitle(label, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _run_case(
    data_path: str | Path,
    ref_path: str | Path,
    distance_m: float,
    out_dir: Path,
    stem: str,
) -> dict:
    x_raw = load_trial_signal(data_path, MIC, TRIAL)
    r_raw = load_trial_signal(ref_path, MIC, TRIAL)

    y, diag = ringdown_handle(
        x=x_raw, distance_m=distance_m, ref_template=r_raw, fs=FS_HZ,
        return_diagnostics=True,
    )
    reduction_db = ringdown_energy_db_reduction(x_raw, y, fs_hz=FS_HZ)

    np.save(out_dir / f"{stem}_raw.npy", np.asarray(x_raw, dtype=np.float64))
    np.save(out_dir / f"{stem}_processed.npy", np.asarray(y, dtype=np.float64))

    # build aligned reference for plot (only for template subtraction path)
    r_aligned = None
    if diag["used_template_subtraction"]:
        xp = wideband_frontend(x_raw, FS_HZ)
        rp = wideband_frontend(r_raw, FS_HZ)
        r_shifted, _alpha, _lag = _align_and_scale(xp, rp)
        r_aligned = r_shifted
        x_proc = xp
    else:
        x_proc = wideband_frontend(x_raw, FS_HZ)

    _plot_ringdown_diagnostic(
        label=f"Ringdown check | {Path(data_path).name} | {MIC} | trial {TRIAL}",
        x_raw=x_raw,
        x_proc=x_proc,
        r_aligned=r_aligned,
        alpha=diag["alpha"] if diag["used_template_subtraction"] else None,
        y=np.asarray(y, dtype=np.float64),
        reduction_db=reduction_db,
        lag=diag["lag_samples"],
        out_path=out_dir / f"{stem}_diagnostic.png",
    )

    return {
        "diagnostics": diag,
        "ringdown_reduction_db_0_1p2ms": reduction_db,
    }


def main() -> int:
    root = Path.cwd()
    file_25 = root / "Multifrequenz Dataset" / "Multifrequenz" / "Box" / "25cm_0Grad.pickle"
    file_50 = root / "Multifrequenz Dataset" / "Multifrequenz" / "Box" / "50cm_0Grad.pickle"
    ref_file = root / "Multifrequenz Dataset" / "Multifrequenz" / "referenz" / "referenz.pickle"
    out_dir = root / "outputs" / "ringdown_handeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    res25 = _run_case(file_25, ref_file, distance_m=0.25, out_dir=out_dir, stem="ringdown_25cm_trial0")
    res50 = _run_case(file_50, ref_file, distance_m=0.50, out_dir=out_dir, stem="ringdown_50cm_trial0")

    summary = {
        "trial": TRIAL, "mic": MIC, "fs_hz": FS_HZ,
        "files": {"25cm": str(file_25), "50cm": str(file_50), "reference": str(ref_file)},
        "results": {"25cm": res25, "50cm": res50},
    }
    with open(out_dir / "ringdown_check_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs to:", out_dir)
    print("25 cm  reduction (dB):", res25["ringdown_reduction_db_0_1p2ms"])
    print("25 cm  diagnostics   :", res25["diagnostics"])
    print("50 cm  reduction (dB):", res50["ringdown_reduction_db_0_1p2ms"])
    print("50 cm  diagnostics   :", res50["diagnostics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
