from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from helpers.data_rw import load_trials

# ── Edit these values before running ────────────────────────────────────────
PATHS = [
    r"Multifrequenz Dataset\Multifrequenz\Box\50cm_-10Grad.pickle",
    r"Multifrequenz Dataset\Multifrequenz_mean10_denoised_branchA_db6_hard\Box\25cm_-10Grad__mean_trials_000_009.pickle",
]
MIC   = "Mic2"   # "Mic1", "Mic2", or "Mic3"
TRIAL = 0        # trial index
# ────────────────────────────────────────────────────────────────────────────


def plot_signals(paths: list[str | Path], mic: str = "Mic2", trial: int = 0) -> None:
    """Plot one trial from each pickle file, each in its own subplot.

    Args:
        paths: List of pickle file paths to plot.
        mic:   Mic key to read from each pickle (default: "Mic2").
        trial: Trial index to plot from each file (default: 0).
    """
    n = len(paths)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, paths):
        path = Path(path)
        trials = load_trials(path)
        signal = trials[mic][trial]
        ax.plot(signal)
        ax.set_title(path.name)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_signals(PATHS, mic=MIC, trial=TRIAL)
