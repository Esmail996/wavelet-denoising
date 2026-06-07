"""
Generates the figures and the region-decomposition table for
Section 3.3 (Noise characterisation) of the thesis.

This script mirrors `noise_estimate.py` -- same constants, same
bandpass call, same gating -- and adds three figures plus a
console table summarising the regional noise statistics.

Run from the project root:

    python -m <package>.noise_characterisation

Outputs:
    Pictures/fig_noise_distribution.png
    Pictures/fig_noise_psd.png
    Pictures/fig_noise_per_trial.png
    plus a console table with the numbers cited in Section 3.3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import stats

from helpers.data_rw import load_trials
from .preprocess import preprocess_signal_for_denoising

MAD_TO_STD = 1.0 / 0.6744897501960817  # converts MAD to Gaussian-equivalent sigma

# ---------------------------------------------------------------------------
# Configuration -- mirrors noise_estimate.py. Edit if the canonical values
# in noise_estimate.py change, then re-run.
# ---------------------------------------------------------------------------
REFERENCE_PICKLE = r"Multifrequenz Dataset/Multifrequenz/referenz/referenz.pickle"
MICS             = ("Mic1", "Mic2", "Mic3")
PRIMARY_MIC      = "Mic2"          # mic used for the distribution and PSD figures
# Bandpass preprocessing (must match noise_estimate.py)
FS_HZ            = 2_000_000
BP_CENTER_HZ     = 50_000
BP_BW_HZ         = 25_000
BP_METHOD        = "iir"
BP_IIR_ORDER     = 2
# Gating regions
GATE_START       = 2400
GATE_END         = 17000
MULTIPATH_START  = 2400
MULTIPATH_END    = 10000
THERMAL_START    = 10000
THERMAL_END      = 17000
# Output
OUT_DIR          = Path("Pictures")
DPI              = 180
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mad_std(x: np.ndarray) -> float:
    """Robust Gaussian-equivalent sigma from MAD (matches noise_estimate.py)."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return float(mad * MAD_TO_STD) if mad > 0.0 else float(np.std(arr))


def bandpass_trial(signal: np.ndarray) -> np.ndarray:
    """Apply the same bandpass that noise_estimate.py applies."""
    return preprocess_signal_for_denoising(
        signal=signal,
        preprocessing="bandpassed",
        fs_hz=FS_HZ,
        band_center_hz=BP_CENTER_HZ,
        bw_hz=BP_BW_HZ,
        bp_method=BP_METHOD,
        iir_order=BP_IIR_ORDER,
    )


def stack_trials(trials_by_mic: dict, mic: str) -> np.ndarray:
    """Return a 2-D array of shape (n_trials, n_samples) for one mic."""
    return np.asarray([np.asarray(t, dtype=float) for t in trials_by_mic[mic]])


def pool_region(trials_2d: np.ndarray, start: int, end: int,
                dc_remove: bool = False) -> np.ndarray:
    """Concatenate samples in [start, end) across all trials."""
    seg = trials_2d[:, start:end]
    if dc_remove:
        seg = seg - seg.mean(axis=1, keepdims=True)
    return seg.ravel()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_distribution(
    thermal_pool: np.ndarray,
    operational_pool: np.ndarray,
    out_path: Path,
) -> None:
    """Two-panel histogram: thermal floor (raw) vs operational noise (BP)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    for ax, pool, title, integer_bins in [
        (
            axes[0], thermal_pool,
            f"Thermal floor ({PRIMARY_MIC}, raw, "
            f"samples [{THERMAL_START}, {THERMAL_END}))",
            True,
        ),
        (
            axes[1], operational_pool,
            f"Operational noise ({PRIMARY_MIC}, bandpass "
            f"{BP_CENTER_HZ // 1000 - BP_BW_HZ // 1000}"
            f"-{BP_CENTER_HZ // 1000 + BP_BW_HZ // 1000} kHz, "
            f"samples [{GATE_START}, {GATE_END}))",
            False,
        ),
    ]:
        sigma_std = pool.std()
        sigma_mad = mad_std(pool)
        skew = stats.skew(pool)
        kurt = stats.kurtosis(pool)

        if integer_bins:
            span = max(abs(pool.min()), abs(pool.max()))
            bin_edges = np.arange(-np.ceil(span) - 0.5, np.ceil(span) + 1.5, 1.0)
        else:
            p_lo, p_hi = np.percentile(pool, [0.05, 99.95])
            span = max(abs(p_lo), abs(p_hi))
            bin_edges = np.linspace(-span * 1.05, span * 1.05, 81)

        ax.hist(pool, bins=bin_edges, color="#4C72B0", alpha=0.7, density=True,
                label=f"Empirical (n = {len(pool):,})")
        xline = np.linspace(bin_edges[0], bin_edges[-1], 600)
        ax.plot(xline, stats.norm.pdf(xline, 0.0, sigma_mad),
                color="#C44E52", linewidth=1.8,
                label=fr"Gaussian, $\sigma_{{MAD}}$ = {sigma_mad:.2f}")

        ax.set_xlabel("Amplitude (ADC units)")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

        text = (
            f"$\\sigma_{{std}}$    = {sigma_std:.2f}\n"
            f"$\\sigma_{{MAD}}$  = {sigma_mad:.2f}\n"
            f"skewness   = {skew:+.3f}\n"
            f"excess kurt = {kurt:+.3f}"
        )
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes, verticalalignment="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="grey", alpha=0.85),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_path}")


def figure_psd(thermal_trials: np.ndarray, out_path: Path) -> None:
    """Welch PSD of the thermal-floor region, averaged across trials."""
    f_psd, p_psd = welch(thermal_trials, fs=FS_HZ, nperseg=2048, axis=1)
    p_avg = p_psd.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.semilogy(f_psd / 1000, p_avg, color="#4C72B0", linewidth=1.4)

    lo_khz = (BP_CENTER_HZ - BP_BW_HZ) / 1000
    hi_khz = (BP_CENTER_HZ + BP_BW_HZ) / 1000
    ax.axvspan(lo_khz, hi_khz, alpha=0.15, color="#55A868",
               label=f"wideband front-end passband ({lo_khz:.0f}-{hi_khz:.0f} kHz)")

    for fc, lab in [(40, "40 kHz"), (50, "50 kHz"), (60, "60 kHz")]:
        ax.axvline(fc, color="#C44E52", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD (ADC$^2$/Hz)")
    ax.set_title(
        f"Thermal-floor PSD ({PRIMARY_MIC}, raw, "
        f"samples [{THERMAL_START}, {THERMAL_END}), averaged over "
        f"{thermal_trials.shape[0]} trials)",
        fontsize=10,
    )
    ax.set_xlim([0, FS_HZ / 2 / 1000])
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_path}")


def figure_per_trial(
    trials_by_mic: dict,
    out_path: Path,
) -> None:
    """Per-trial sigma scatter for all configured mics."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = {"Mic1": "#4C72B0", "Mic2": "#55A868", "Mic3": "#C44E52"}

    for mic in MICS:
        sigmas = []
        for signal in trials_by_mic[mic]:
            bp = bandpass_trial(np.asarray(signal, dtype=float))
            sigmas.append(mad_std(bp[GATE_START:GATE_END]))
        sigmas = np.asarray(sigmas)
        median = np.median(sigmas)
        rel_spread = sigmas.std() / median * 100.0
        ax.plot(np.arange(len(sigmas)), sigmas, ".",
                color=colors.get(mic, "#777777"), markersize=4,
                label=f"{mic}: median = {median:.2f}, "
                      f"std/median = {rel_spread:.1f}%")
        ax.axhline(median, color=colors.get(mic, "#777777"),
                   linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel("Trial index")
    ax.set_ylabel(r"Per-trial $\hat{\sigma}_n$ (ADC units)")
    ax.set_title(
        f"Per-trial noise estimate "
        f"(bandpass {(BP_CENTER_HZ - BP_BW_HZ) // 1000}-"
        f"{(BP_CENTER_HZ + BP_BW_HZ) // 1000} kHz, "
        f"gate [{GATE_START}, {GATE_END}), MAD/0.6745)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Table data printed to console (backs the numerical claims in Section 3.3)
# ---------------------------------------------------------------------------

def print_region_table(trials_by_mic: dict) -> None:
    """Print the region-decomposition table cited in Section 3.3.3."""
    print()
    print("=" * 88)
    print("  Region decomposition (median over trials, ADC units)")
    print("=" * 88)
    print(f"  Bandpass: {BP_CENTER_HZ // 1000 - BP_BW_HZ // 1000}-"
          f"{BP_CENTER_HZ // 1000 + BP_BW_HZ // 1000} kHz, "
          f"order {BP_IIR_ORDER} IIR.")
    print()
    header = f'{"Channel":<8} {"Region":<32} {"Pre-proc":<14} {"sigma_MAD":>12}'
    print(header)
    print("-" * len(header))

    regions = [
        ("post-multipath", THERMAL_START, THERMAL_END),
        ("multipath only", MULTIPATH_START, MULTIPATH_END),
        ("operational gate", GATE_START, GATE_END),
    ]

    for mic in MICS:
        trials_2d = stack_trials(trials_by_mic, mic)
        for label, lo, hi in regions:
            raw_sigmas = [mad_std(trials_2d[i, lo:hi] - trials_2d[i, lo:hi].mean())
                          for i in range(trials_2d.shape[0])]
            bp_sigmas = []
            for i in range(trials_2d.shape[0]):
                bp = bandpass_trial(trials_2d[i])
                bp_sigmas.append(mad_std(bp[lo:hi]))
            print(f'{mic:<8} {label + f" [{lo},{hi})":<32} '
                  f'{"raw":<14} {np.median(raw_sigmas):>12.4f}')
            print(f'{mic:<8} {label + f" [{lo},{hi})":<32} '
                  f'{"BP":<14} {np.median(bp_sigmas):>12.4f}')
        print()


def print_distribution_stats(thermal_pool: np.ndarray,
                             operational_pool: np.ndarray) -> None:
    """Print the distribution stats used in the histogram annotations."""
    print()
    print("=" * 88)
    print("  Distribution statistics ({mic})".format(mic=PRIMARY_MIC))
    print("=" * 88)
    for label, pool in [
        (f"Thermal floor (raw, [{THERMAL_START},{THERMAL_END}))", thermal_pool),
        (f"Operational (BP, [{GATE_START},{GATE_END}))", operational_pool),
    ]:
        print(f"\n  {label}")
        print(f"    n           = {pool.size:,}")
        print(f"    mean        = {pool.mean():+.4f}")
        print(f"    std         = {pool.std():.4f}")
        print(f"    MAD-sigma   = {mad_std(pool):.4f}")
        print(f"    skewness    = {stats.skew(pool):+.4f}")
        print(f"    excess kurt = {stats.kurtosis(pool):+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {REFERENCE_PICKLE} ...")
    trials_by_mic = load_trials(REFERENCE_PICKLE)
    for mic in MICS:
        if mic not in trials_by_mic:
            raise KeyError(
                f"Mic '{mic}' not found in pickle. "
                f"Available: {list(trials_by_mic.keys())}"
            )

    # Pools for the primary mic
    primary_trials = stack_trials(trials_by_mic, PRIMARY_MIC)

    thermal_pool = pool_region(
        primary_trials, THERMAL_START, THERMAL_END, dc_remove=True,
    )
    thermal_trials_dc = primary_trials[:, THERMAL_START:THERMAL_END] \
        - primary_trials[:, THERMAL_START:THERMAL_END].mean(axis=1, keepdims=True)

    operational_pool_chunks = []
    for i in range(primary_trials.shape[0]):
        bp = bandpass_trial(primary_trials[i])
        operational_pool_chunks.append(bp[GATE_START:GATE_END])
    operational_pool = np.concatenate(operational_pool_chunks)

    # Figures
    print()
    print(f"Writing figures to {OUT_DIR}/ ...")
    figure_distribution(
        thermal_pool, operational_pool,
        OUT_DIR / "fig_noise_distribution.png",
    )
    figure_psd(
        thermal_trials_dc,
        OUT_DIR / "fig_noise_psd.png",
    )
    figure_per_trial(
        trials_by_mic,
        OUT_DIR / "fig_noise_per_trial.png",
    )

    # Tables (numbers cited in the prose)
    print_distribution_stats(thermal_pool, operational_pool)
    print_region_table(trials_by_mic)


if __name__ == "__main__":
    main()
