"""
Signal-anatomy investigation for the Multifrequenz dataset.

Generates the figures used in Chapter 3 Section 3.2 (Anatomy of a Multifrequenz
signal trial). All figures are computed from raw pickle data, no preprocessing
beyond DC removal. The default reference trial is Box at 75 cm 0 degrees,
Mic2, averaged across all 100 trials in the file.

Usage:
    python signal_anatomy.py --out-dir figures/

The script produces four figures:
    fig_anatomy_full.png       Full 9 ms waveform with region annotations
    fig_anatomy_rms.png        RMS profile, measurement vs reference
    fig_anatomy_spectrogram.png   Time-frequency content
    fig_anatomy_endpeak.png    End-of-trial peak investigation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import stft

from helpers.data_rw import load_trials

# ---------------------------------------------------------------------------
DATASET_ROOT = "Multifrequenz Dataset/Multifrequenz"
TARGET_OBJECT = "Box"
TARGET_DISTANCE_CM = 75
TARGET_ANGLE = "0Grad"
MIC = "Mic2"

FS_HZ = 2_000_000
TRIAL_LENGTH_SAMPLES = 18000

# Region boundaries determined empirically (described in chapter prose).
# Refined after fitting exponential decay to the reference envelope:
# the real transducer + microphone ringdown (Q ~ 2.5) is essentially
# complete by 1 ms; the slower decay between 1 ms and 5 ms is
# non-monotonic and is consistent with room reflections / stationary
# multipath rather than continued ringdown.
REGION_TX_END = 600        # 0.30 ms - end of transmission burst
REGION_RINGDOWN_END = 2000     # 1.00 ms - end of transducer/mic ringdown
REGION_MULTIPATH_END = 10000   # 5.0 ms - end of room-reflection tail
BLANKING_BOUNDARY = 2000   # 1 ms - current pipeline blanking length

# For end-peak investigation
ALL_DISTANCES = [25, 50, 75, 100, 125]
# ---------------------------------------------------------------------------


def load_trial_array(dataset_root: str | Path, obj: str, distance_cm: int,
                     angle: str, mic: str) -> np.ndarray:
	"""Load all 100 trials for one (object, distance, angle) cell.
	Returns array of shape (100, 18000) with DC-removed waveforms.
	"""
	path = Path(dataset_root) / obj / f"{distance_cm}cm_{angle}.pickle"
	trial_map = load_trials(path)
	if mic not in trial_map:
		raise KeyError(f"Mic {mic!r} not in {list(trial_map.keys())}")
	trials = np.stack([np.asarray(signal, dtype=float) for signal in trial_map[mic]])
	# DC removal: subtract late-trial mean (post-ringdown, near noise floor)
	trials = trials - trials[:, 10000:15000].mean(axis=1, keepdims=True)
	return trials


def load_reference(dataset_root: str | Path, mic: str) -> np.ndarray:
	"""Load all reference (target-absent) trials. Same shape as load_trial_array."""
	path = Path(dataset_root) / "referenz" / "referenz.pickle"
	trial_map = load_trials(path)
	if mic not in trial_map:
		raise KeyError(f"Mic {mic!r} not in {list(trial_map.keys())}")
	trials = np.stack([np.asarray(signal, dtype=float) for signal in trial_map[mic]])
	trials = trials - trials[:, 10000:15000].mean(axis=1, keepdims=True)
	return trials


def windowed_rms(trials: np.ndarray, window: int) -> np.ndarray:
	"""Compute RMS in non-overlapping windows of size `window`. Returns
	shape (n_trials, n_windows)."""
	n_trials, n_samples = trials.shape
	n_win = n_samples // window
	out = np.zeros((n_trials, n_win))
	for i in range(n_win):
		out[:, i] = np.sqrt(
			np.mean(trials[:, i * window:(i + 1) * window] ** 2, axis=1)
		)
	return out


# ---------------------------------------------------------------------------
def fig_anatomy_full(out_dir: Path):
	"""Figure 1: full waveform with region annotations.

	The waveform shown is the 100-trial mean (matching the methodology
	used throughout Section 3.2). The transmission burst is deterministic
	to within < 1% across trials and is preserved by averaging;
	random noise is partially suppressed, making the direct echo more
	visible.
	"""
	measurement = load_trial_array(
		DATASET_ROOT, TARGET_OBJECT, TARGET_DISTANCE_CM, TARGET_ANGLE, MIC
	)
	# 100-trial mean preserves deterministic content (burst, ringdown,
	# multipath, direct echo) and suppresses random noise.
	x = measurement.mean(axis=0)
	t_ms = np.arange(len(x)) / FS_HZ * 1000

	fig, ax = plt.subplots(figsize=(11, 4))
	# Region shading (annotation only, no data)
	ax.axvspan(0, REGION_TX_END / FS_HZ * 1000, alpha=0.18, color="#FE6066",
	           label="A: transmission burst")
	ax.axvspan(REGION_TX_END / FS_HZ * 1000,
	           REGION_RINGDOWN_END / FS_HZ * 1000,
	           alpha=0.18, color="#DD8452",
	           label="B: transducer + mic ringdown")
	ax.axvspan(REGION_RINGDOWN_END / FS_HZ * 1000,
	           REGION_MULTIPATH_END / FS_HZ * 1000,
	           alpha=0.12, color="#8C8C8C",
	           label="C: stationary multipath (room reflections)")
	ax.axvspan(REGION_MULTIPATH_END / FS_HZ * 1000, t_ms[-1],
	           alpha=0.10, color="#55A868", label="D: noise floor")

	ax.plot(t_ms, x, linewidth=0.5, color="#2E2E2E")
	# Mark the blanking boundary
	ax.axvline(BLANKING_BOUNDARY / FS_HZ * 1000, color="#4C72B0",
	           linestyle="--", linewidth=1.2, label="blanking boundary (sample 2400)")

	ax.set_xlim(0, 9.0)
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel("Amplitude (ADC units, DC-removed)")
	ax.set_title(
		f"Multifrequenz signal trial: {TARGET_OBJECT}, {TARGET_DISTANCE_CM} cm, "
		f"{TARGET_ANGLE}, {MIC}, 100-trial mean"
	)
	ax.legend(loc="upper right", fontsize=8)
	ax.grid(alpha=0.25)
	fig.tight_layout()

	out_path = out_dir / "fig_anatomy_full.png"
	fig.savefig(out_path, dpi=180)
	plt.close(fig)
	print(f"Wrote {out_path}")


def fig_anatomy_rms(out_dir: Path):
	"""Figure 2: RMS profile measurement vs reference.
	Demonstrates that the "quiet" pre-echo region is structural ringdown,
	not target multipath."""
	measurement = load_trial_array(
		DATASET_ROOT, TARGET_OBJECT, TARGET_DISTANCE_CM, TARGET_ANGLE, MIC
	)
	reference = load_reference(DATASET_ROOT, MIC)

	window = 200
	rms_meas = windowed_rms(measurement, window).mean(axis=0)
	rms_ref = windowed_rms(reference, window).mean(axis=0)
	t_ms = np.arange(len(rms_meas)) * window / FS_HZ * 1000

	fig, ax = plt.subplots(figsize=(11, 4))
	ax.semilogy(t_ms, rms_meas, color="#C44E52", linewidth=1.5,
	            label=f"Measurement ({TARGET_OBJECT}, {TARGET_DISTANCE_CM} cm, "
	                  f"averaged over 100 trials)")
	ax.semilogy(t_ms, rms_ref, color="#4C72B0", linewidth=1.5,
	            label="Reference (target absent, averaged over 100 trials)")

	# Region shading
	ax.axvspan(REGION_RINGDOWN_END / FS_HZ * 1000,
	           REGION_MULTIPATH_END / FS_HZ * 1000,
	           alpha=0.15, color="#8C8C8C",
	           label="stationary multipath region")
	ax.axvline(BLANKING_BOUNDARY / FS_HZ * 1000, color="#888888",
	           linestyle="--", linewidth=1.0)
	ax.text(BLANKING_BOUNDARY / FS_HZ * 1000 + 0.05, 1.5,
	        "blanking\nboundary", fontsize=8, color="#555555")

	# Annotate where direct echo is
	# Direct echo time for 75cm round trip ≈ 2*0.75/343 s = 4.37 ms
	echo_time = 2 * (TARGET_DISTANCE_CM / 100) / 343 * 1000
	ax.axvline(echo_time, color="#C44E52", linestyle=":", linewidth=1.0)
	ax.text(echo_time + 0.05, 200, "direct\necho", fontsize=8, color="#C44E52")

	ax.set_xlim(0, 9.0)
	ax.set_ylim(1.0, 3000)
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel("RMS (ADC units, log scale)")
	ax.set_title("RMS profile: measurement vs target-absent reference, 200-sample windows")
	ax.legend(loc="upper right", fontsize=9)
	ax.grid(alpha=0.25, which="both")
	fig.tight_layout()

	out_path = out_dir / "fig_anatomy_rms.png"
	fig.savefig(out_path, dpi=180)
	plt.close(fig)
	print(f"Wrote {out_path}")


def fig_anatomy_spectrogram(out_dir: Path):
	"""Figure 3: STFT spectrogram showing time-frequency content."""
	measurement = load_trial_array(
		DATASET_ROOT, TARGET_OBJECT, TARGET_DISTANCE_CM, TARGET_ANGLE, MIC
	)
	x = measurement.mean(axis=0)

	# STFT with moderate resolution
	nperseg = 256
	noverlap = 224
	f, t, Z = stft(x, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap,
	               window="hann", boundary=None)
	mag = np.abs(Z)
	# dB scale, normalised to global maximum
	mag_db = 20 * np.log10(mag / mag.max() + 1e-30)

	# Crop to relevant frequency range
	mask_f = (f >= 20e3) & (f <= 70e3)
	mag_db_crop = mag_db[mask_f]
	f_crop = f[mask_f]

	fig, ax = plt.subplots(figsize=(11, 4))
	im = ax.pcolormesh(t * 1000, f_crop / 1000, mag_db_crop,
	                    cmap="viridis", vmin=-60, vmax=0, shading="auto")
	plt.colorbar(im, ax=ax, label="Magnitude (dB)")

	# Annotate carrier frequencies
	for fc in [40, 50, 60]:
		ax.axhline(fc, color="white", linestyle=":", linewidth=0.8, alpha=0.7)
		ax.text(8.5, fc + 0.5, f"{fc} kHz", color="white", fontsize=8,
		        verticalalignment="bottom")
	# Annotate mic resonance
	ax.axhline(39, color="#FFCC00", linestyle="--", linewidth=1.0, alpha=0.7)
	ax.text(0.1, 39 - 1.5, "mic resonance (39 kHz)", color="#FFCC00",
	        fontsize=8, verticalalignment="top")

	ax.set_xlim(0, 9.0)
	ax.set_ylim(20, 70)
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel("Frequency (kHz)")
	ax.set_title(
		f"STFT spectrogram, {TARGET_OBJECT} {TARGET_DISTANCE_CM} cm "
		f"{TARGET_ANGLE} {MIC} 100-trial mean (window {nperseg} samples, overlap {noverlap})"
	)
	fig.tight_layout()

	out_path = out_dir / "fig_anatomy_spectrogram.png"
	fig.savefig(out_path, dpi=180)
	plt.close(fig)
	print(f"Wrote {out_path}")


def fig_anatomy_decay(out_dir: Path):
	"""Figure: envelope decay on log scale showing real ringdown vs multipath.

	Real transducer ringdown is monotonic exponential decay with a time
	constant determined by Q-factor. Room reflections appear as bumps
	in the envelope. This figure makes the distinction visible.
	"""
	from scipy.signal import hilbert

	reference = load_reference(DATASET_ROOT, MIC)
	# Hilbert envelope, averaged across 100 trials
	env = np.abs(hilbert(reference, axis=1)).mean(axis=0)
	t_ms = np.arange(len(env)) / FS_HZ * 1000

	# Theoretical pure-ringdown decay for Q = 2.5 at f0 = 39 kHz:
	# tau = Q / (pi * f0) = 2.5 / (pi * 39000) ~ 20.4 us.
	Q = 2.5
	f0 = 39e3
	tau = Q / (np.pi * f0)
	# Anchor at end of burst (sample 600, t = 0.30 ms)
	anchor_sample = 600
	anchor_val = env[anchor_sample]
	t_theory = (np.arange(len(env)) - anchor_sample) / FS_HZ
	theory = anchor_val * np.exp(-t_theory / tau)
	theory[t_theory < 0] = anchor_val

	fig, ax = plt.subplots(figsize=(11, 4.5))
	ax.semilogy(t_ms, env, color="#4C72B0", linewidth=1.2,
	            label="Reference envelope, 100-trial mean (Hilbert magnitude)")
	mask_th = (t_ms >= anchor_sample / FS_HZ * 1000) & (theory > 1.0)
	ax.semilogy(t_ms[mask_th], theory[mask_th], color="#C44E52",
	            linewidth=1.2, linestyle="--",
	            label=f"Pure exponential decay: Q = {Q}, f0 = 39 kHz, "
	                  f"tau = {tau*1e6:.0f} us")

	ax.axvline(REGION_RINGDOWN_END / FS_HZ * 1000, color="#888888",
	           linestyle=":", linewidth=1.0)
	ax.text(REGION_RINGDOWN_END / FS_HZ * 1000 + 0.05, 1.5,
	        "end of monotonic\nexponential decay\n(~1 ms)",
	        fontsize=8, color="#555555")

	ax.axhline(2.0, color="#55A868", linestyle=":", linewidth=1.0)
	ax.text(8.0, 1.3, "noise floor (~2 ADC units)",
	        fontsize=8, color="#55A868")

	ax.set_xlim(0, 7.5)
	ax.set_ylim(1.0, 5000)
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel("Envelope amplitude (ADC units, log scale)")
	ax.set_title(
		"Envelope decay: empirical vs pure exponential ringdown"
	)
	ax.legend(loc="upper right", fontsize=9)
	ax.grid(alpha=0.25, which="both")
	fig.tight_layout()

	out_path = out_dir / "fig_anatomy_decay.png"
	fig.savefig(out_path, dpi=180)
	plt.close(fig)
	print(f"Wrote {out_path}")


def fig_anatomy_endpeak(out_dir: Path):
	"""Figure 4: end-of-trial peak investigation across distances."""
	fig, axes = plt.subplots(2, 1, figsize=(11, 7))

	# (a) Zoom on end region for each distance (Box, 0 angle)
	ax = axes[0]
	colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
	for d, color in zip(ALL_DISTANCES, colors):
		try:
			arr = load_trial_array(DATASET_ROOT, "Box", d, "0Grad", MIC)
		except FileNotFoundError:
			continue
		rms = windowed_rms(arr, window=100).mean(axis=0)
		t_ms = np.arange(len(rms)) * 100 / FS_HZ * 1000
		# Zoom into samples 14000-18000 = 7.0-9.0 ms
		mask = (t_ms >= 7.0) & (t_ms <= 9.0)
		ax.plot(t_ms[mask], rms[mask], color=color, linewidth=1.3,
		        label=f"{d} cm")

	# Also show reference noise floor for context
	ref = load_reference(DATASET_ROOT, MIC)
	ref_rms = windowed_rms(ref, window=100).mean(axis=0)
	t_ms_ref = np.arange(len(ref_rms)) * 100 / FS_HZ * 1000
	mask = (t_ms_ref >= 7.0) & (t_ms_ref <= 9.0)
	ax.plot(t_ms_ref[mask], ref_rms[mask], color="black", linewidth=1.2,
	        linestyle="--", label="reference (no target)")

	ax.set_xlim(7.0, 9.0)
	ax.set_xlabel("Time (ms)")
	ax.set_ylabel("RMS (ADC units)")
	ax.set_title("End-of-trial region (samples 14000-18000) for Box at five distances")
	ax.legend(loc="upper left", fontsize=8, ncol=2)
	ax.grid(alpha=0.25)

	# (b) Spectrum of end peak vs direct echo, Box 75 cm
	ax = axes[1]
	arr = load_trial_array(DATASET_ROOT, "Box", 75, "0Grad", MIC)

	def avg_spectrum(arr: np.ndarray, s_start: int, s_end: int):
		n = s_end - s_start
		window = np.hanning(n)
		freqs = rfftfreq(n, 1 / FS_HZ)
		specs = [np.abs(rfft(arr[t, s_start:s_end] * window))
		         for t in range(arr.shape[0])]
		return freqs, np.mean(specs, axis=0)

	f_direct, spec_direct = avg_spectrum(arr, 8800, 9300)
	f_end, spec_end = avg_spectrum(arr, 17400, 17900)

	norm = spec_direct.max()
	mask = (f_direct >= 30e3) & (f_direct <= 70e3)

	ax.plot(f_direct[mask] / 1000, 20 * np.log10(spec_direct[mask] / norm),
	        color="#C44E52", linewidth=1.5, label="direct echo (samples 8800–9300)")
	ax.plot(f_end[mask] / 1000, 20 * np.log10(spec_end[mask] / norm),
	        color="#4C72B0", linewidth=1.5, label="end peak (samples 17400–17900)")

	# Annotate carriers
	for fc in [40, 50, 60]:
		ax.axvline(fc, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

	ax.set_xlim(30, 70)
	ax.set_ylim(-40, 5)
	ax.set_xlabel("Frequency (kHz)")
	ax.set_ylabel("Magnitude (dB, relative to direct-echo peak)")
	ax.set_title("Spectrum of direct echo vs end peak (Box, 75 cm, averaged 100 trials)")
	ax.legend(loc="lower left", fontsize=9)
	ax.grid(alpha=0.25)

	fig.tight_layout()
	out_path = out_dir / "fig_anatomy_endpeak.png"
	fig.savefig(out_path, dpi=180)
	plt.close(fig)
	print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Generate signal-anatomy figures for Chapter 3 Section 3.2."
	)
	p.add_argument("--out-dir", default="figures",
	               help="Output directory for the four PNG figures.")
	p.add_argument("--dataset-root", default=DATASET_ROOT,
	               help=f"Dataset root directory (default {DATASET_ROOT!r}).")
	return p


def main() -> None:
	args = _build_arg_parser().parse_args()
	global DATASET_ROOT
	DATASET_ROOT = args.dataset_root

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	fig_anatomy_full(out_dir)
	fig_anatomy_rms(out_dir)
	fig_anatomy_spectrogram(out_dir)
	fig_anatomy_decay(out_dir)
	fig_anatomy_endpeak(out_dir)
	print("All five figures generated.")


if __name__ == "__main__":
	main()
