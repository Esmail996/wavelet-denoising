from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helpers.data_rw import load_trials
from .preprocess import preprocess_signal_for_denoising


# ---------------------------------------------------------------------------
# Edit defaults as needed, then run: python -m denoising.histogram
# ---------------------------------------------------------------------------
REFERENCE_PICKLE = r"Multifrequenz Dataset/Multifrequenz/referenz/referenz.pickle"
MIC = "Mic2"  # "Mic1", "Mic2", or "Mic3"
NOISE_START = 2000  # ignore deterministic early-time crosstalk before this index
NOISE_END = None    # optional tail cut; None means "until end of signal"
BINS = 25

# Optional bandpass preprocessing for each trial before amplitude extraction
USE_BANDPASS = True
FS_HZ = 2_000_000
BP_CENTER_HZ = 50_000
BP_BW_HZ = 25_000
BP_METHOD = "iir"
BP_IIR_ORDER = 4
# ---------------------------------------------------------------------------


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
	"""Return Gaussian PDF values N(mu, sigma^2) evaluated at x."""
	if sigma <= 0.0:
		return np.zeros_like(x, dtype=float)
	z = (x - mu) / sigma
	return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def extract_reference_noise_samples(
	pickle_path: str | Path,
	mic: str,
	noise_start: int,
	noise_end: int | None = None,
	use_bandpass: bool = USE_BANDPASS,
	fs_hz: float = 2_000_000,
	bp_center_hz: float = 50_000,
	bp_bw_hz: float = 25_000,
	bp_method: str = "iir",
	iir_order: int = 4,
) -> np.ndarray:
	"""
	Pool noise-floor samples from all selected reference trials.

	Each trial contributes signal[noise_start:noise_end], optionally after
	bandpass preprocessing. The returned 1D array is the concatenation of all
	pooled noise samples.
	"""
	if noise_start < 0:
		raise ValueError(f"Invalid noise_start={noise_start}. Must be >= 0.")
	if noise_end is not None and noise_end <= noise_start:
		raise ValueError(
			f"Invalid noise window: start={noise_start}, end={noise_end}"
		)

	trials = load_trials(pickle_path)
	if mic not in trials:
		raise KeyError(f"Mic '{mic}' not found. Available: {list(trials.keys())}")

	pooled: list[np.ndarray] = []
	for signal in trials[mic]:
		x = np.asarray(signal, dtype=float).ravel()
		if use_bandpass:
			x = preprocess_signal_for_denoising(
				signal=x,
				preprocessing="bandpassed",
				fs_hz=fs_hz,
				band_center_hz=bp_center_hz,
				bw_hz=bp_bw_hz,
				bp_method=bp_method,
				iir_order=iir_order,
			)

		seg = x[noise_start:noise_end]
		if seg.size == 0:
			continue
		pooled.append(np.asarray(seg, dtype=float))

	if not pooled:
		return np.asarray([], dtype=float)
	return np.concatenate(pooled)


def plot_histogram_with_gaussian(
	samples: np.ndarray,
	bins: int = 25,
	title: str | None = None,
	save_path: str | Path | None = None,
	show: bool = True,
) -> dict:
	"""
	Plot histogram of pooled samples and overlay a Gaussian fit.

	Returns fit statistics: {'mu', 'sigma', 'n'}.
	"""
	x = np.asarray(samples, dtype=float).ravel()
	x = x[np.isfinite(x)]
	if x.size == 0:
		raise ValueError("No valid samples available for histogram/fit.")

	mu = float(np.mean(x))
	sigma = float(np.std(x, ddof=0))

	fig, ax = plt.subplots(figsize=(9, 5))
	_, edges, _ = ax.hist(
		x,
		bins=int(bins),
		alpha=0.65,
		color="#4C78A8",
		edgecolor="black",
		linewidth=0.6,
		label="Reference noise samples",
	)

	x_line = np.linspace(edges[0], edges[-1], 400)
	pdf_line = _gaussian_pdf(x_line, mu=mu, sigma=sigma)
	bin_width = edges[1] - edges[0] if len(edges) > 1 else 1.0
	y_line = pdf_line * x.size * bin_width
	ax.plot(
		x_line,
		y_line,
		color="#E45756",
		linewidth=2.2,
		label=f"Gaussian fit (mu={mu:.3e}, sigma={sigma:.3e})",
	)

	ax.set_xlabel("Noise amplitude")
	ax.set_ylabel("Count")
	ax.set_title(title or "Reference Noise-Floor Histogram")
	ax.grid(alpha=0.25)
	ax.legend()
	fig.tight_layout()

	if save_path is not None:
		out = Path(save_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out, dpi=200)

	if show:
		plt.show()
	else:
		plt.close(fig)

	return {"mu": mu, "sigma": sigma, "n": int(x.size)}


def _build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Histogram of pooled noise-floor samples from reference trials with Gaussian fit."
	)
	p.add_argument("--pickle", default=REFERENCE_PICKLE, help="Path to reference pickle")
	p.add_argument("--mic", default=MIC, help="Microphone key, e.g. Mic1/Mic2/Mic3")
	p.add_argument("--noise-start", type=int, default=NOISE_START)
	p.add_argument(
		"--noise-end",
		type=int,
		default=NOISE_END,
		help="Optional end index of noise window; default is end-of-signal",
	)
	p.add_argument("--bins", type=int, default=BINS)
	p.add_argument("--save", default=None, help="Optional output image path")
	p.add_argument(
		"--no-bandpass",
		action="store_true",
		help="Disable bandpass preprocessing before pooling noise samples",
	)
	p.add_argument("--fs-hz", type=float, default=FS_HZ)
	p.add_argument("--bp-center-hz", type=float, default=BP_CENTER_HZ)
	p.add_argument("--bp-bw-hz", type=float, default=BP_BW_HZ)
	p.add_argument("--bp-method", default=BP_METHOD, choices=["iir", "fir"])
	p.add_argument("--bp-iir-order", type=int, default=BP_IIR_ORDER)
	p.add_argument(
		"--no-show",
		action="store_true",
		help="Create the plot but do not open a UI window",
	)
	return p


def main() -> None:
	args = _build_arg_parser().parse_args()

	samples = extract_reference_noise_samples(
		pickle_path=args.pickle,
		mic=args.mic,
		noise_start=args.noise_start,
		noise_end=args.noise_end,
		use_bandpass=(not args.no_bandpass),
		fs_hz=args.fs_hz,
		bp_center_hz=args.bp_center_hz,
		bp_bw_hz=args.bp_bw_hz,
		bp_method=args.bp_method,
		iir_order=args.bp_iir_order,
	)

	stats = plot_histogram_with_gaussian(
		samples=samples,
		bins=args.bins,
		title=(
			f"Reference noise-floor histogram | {args.mic} | "
			f"window=[{args.noise_start}, {args.noise_end})"
		),
		save_path=args.save,
		show=(not args.no_show),
	)
	print(
		"Gaussian fit: "
		f"mu={stats['mu']:.6e}, sigma={stats['sigma']:.6e}, n={stats['n']}"
	)


if __name__ == "__main__":
	main()


#python.exe -m denoising.histogram --mic Mic3 --noise-start 2000 --bins 40 --no-bandpass --save outputs/reference_noise_hist_mic3_nobp.png