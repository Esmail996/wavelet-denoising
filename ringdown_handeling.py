from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.signal import correlate

from preprocess import preprocess_signal_for_denoising


DEFAULT_FS_HZ = 2_000_000.0
DEFAULT_NEAR_DISTANCE_M = 0.50
DEFAULT_BLANK_SAMPLES = 2400


def _as_1d_float64(signal: Any) -> np.ndarray:
	arr = np.asarray(signal, dtype=np.float64).squeeze()
	if arr.ndim != 1:
		raise ValueError(f"Expected 1D waveform, got shape {arr.shape}.")
	return arr


def wideband_frontend(signal: np.ndarray, fs: float = DEFAULT_FS_HZ) -> np.ndarray:
    """
    Wideband frontend for ringdown processing.

    This intentionally applies only baseline cleanup (DC removal + linear detrend)
    to preserve the carrier content needed for reliable phase alignment.
    """
    x = preprocess_signal_for_denoising(
        signal=_as_1d_float64(signal),
        preprocessing="detrended_only",
        fs_hz=float(fs),
    )
    return _as_1d_float64(x)


def _shift_with_zero_fill(signal: np.ndarray, lag: int) -> np.ndarray:
	"""Shift by integer lag and zero-fill wrapped samples."""
	y = np.roll(signal, lag)
	if lag > 0:
		y[:lag] = 0.0
	elif lag < 0:
		y[lag:] = 0.0
	return y


def _clip_mask(
	x: np.ndarray,
	clip_level: float = 0.97,
	x_raw: np.ndarray | None = None,
) -> np.ndarray:
	"""Return boolean mask: True where sample is NOT saturated.

	When ``x_raw`` is provided (the ADC counts before any preprocessing),
	clipping is detected against the raw ADC rails using an absolute margin:
	  positive clip  =>  raw >= rail_max - margin
	  negative clip  =>  raw <= rail_min + margin
	where margin = (1 - clip_level) * (rail_max - rail_min).

	This handles asymmetric rails correctly (e.g. min_raw = 1, max_raw = 4095).

	Without ``x_raw``, per-rail relative thresholds are applied on ``x``.

	A 3-sample guard band is eroded around each clipped sample.
	"""
	ref = _as_1d_float64(x_raw) if x_raw is not None else _as_1d_float64(x)

	r_min = float(ref.min())
	r_max = float(ref.max())
	span = r_max - r_min
	if span == 0.0:
		return np.ones(x.size, dtype=bool)

	margin = (1.0 - clip_level) * span
	clipped = (ref >= r_max - margin) | (ref <= r_min + margin)

	# dilate mask by 3 samples on each side
	for _ in range(3):
		clipped[1:] |= clipped[:-1]
		clipped[:-1] |= clipped[1:]
	return ~clipped


def _align_and_scale(
	x: np.ndarray,
	r: np.ndarray,
	search: int = 40,
	corr_window: tuple[int, int] = (0, 900),
	fit_window: tuple[int, int] = (0, 2400),
	clip_level: float = 0.97,
	x_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
	"""
	Align reference phase to trial and solve scalar LS amplitude in ringdown region.

	Clipped (saturated) samples are automatically excluded from both the
	cross-correlation and the LS amplitude fit so ADC saturation in the
	early ringdown does not corrupt the estimates.

	Returns
	-------
	r_shift : np.ndarray
		Aligned reference template.
	alpha : float
		Least-squares amplitude scale.
	lag : int
		Integer lag in samples (reference shifted by +lag).
	"""
	x = _as_1d_float64(x)
	r = _as_1d_float64(r)
	if x.size != r.size:
		raise ValueError(f"Length mismatch: trial={x.size}, reference={r.size}")

	a, b = corr_window
	if not (0 <= a < b <= x.size):
		raise ValueError(f"Invalid corr_window={corr_window} for signal length {x.size}.")

	# --- phase alignment: skip clipped samples in correlation window ---
	mask_corr = _clip_mask(x[a:b], clip_level, x_raw=x_raw[a:b] if x_raw is not None else None)
	xc = x[a:b].copy()
	rc = r[a:b].copy()
	xc[~mask_corr] = 0.0
	rc[~mask_corr] = 0.0
	cc = correlate(xc, rc, mode="full")
	lag = int(np.argmax(cc) - (b - a - 1))
	lag = int(np.clip(lag, -int(search), int(search)))

	r_shift = _shift_with_zero_fill(r, lag)

	s, e = fit_window
	if not (0 <= s < e <= x.size):
		raise ValueError(f"Invalid fit_window={fit_window} for signal length {x.size}.")

	# --- amplitude fit: skip clipped samples in fit window ---
	mask_fit = _clip_mask(x[s:e], clip_level, x_raw=x_raw[s:e] if x_raw is not None else None)
	xf = x[s:e][mask_fit]
	rf = r_shift[s:e][mask_fit]
	denom = float(np.dot(rf, rf))
	alpha = float(np.dot(xf, rf) / (denom + 1e-12))
	return r_shift, alpha, lag


def ringdown_handle(
	x: np.ndarray,
	distance_m: float | None,
	ref_template: np.ndarray | None = None,
	blank_samples: int = DEFAULT_BLANK_SAMPLES,
	fs: float = DEFAULT_FS_HZ,
	near_distance_m: float = DEFAULT_NEAR_DISTANCE_M,
	frontend: Callable[[np.ndarray, float], np.ndarray] | None = None,
	align_search: int = 40,
	align_corr_window: tuple[int, int] = (0, 900),
	fit_window: tuple[int, int] = (0, 2400),
	return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | bool]]:
	"""
	Handle Tx ringdown using near-field template subtraction or far-field blanking.

	Near field (`distance_m < near_distance_m`):
	1) Apply wideband frontend
	2) Align reference by cross-correlation in first carrier cycle
	3) Solve LS amplitude scale on ringdown window
	4) Subtract alpha * aligned_reference

	Far field: fallback to blanking the first `blank_samples`.

	Output is always float64 and same length as input.
	"""
	x_in = _as_1d_float64(x)
	frontend_fn = wideband_frontend if frontend is None else frontend
	x_proc = _as_1d_float64(frontend_fn(x_in, fs))

	diagnostics: dict[str, float | int | bool] = {
		"used_template_subtraction": False,
		"lag_samples": 0,
		"alpha": 0.0,
	}

	use_template = (distance_m is None) or (float(distance_m) < float(near_distance_m))
	if use_template:
		if ref_template is None:
			raise ValueError("Reference template required for near-field ringdown handling.")
		r_proc = _as_1d_float64(frontend_fn(_as_1d_float64(ref_template), fs))
		r_shift, alpha, lag = _align_and_scale(
			x=x_proc,
			r=r_proc,
			search=align_search,
			corr_window=align_corr_window,
			fit_window=fit_window,
			x_raw=x_in,
		)
		y = x_proc - alpha * r_shift
		# Blank any samples that were hard-clipped in the original trial —
		# template subtraction cannot reconstruct saturated ADC values and
		# leaves a large distorted residual.  Zeroing those samples avoids
		# polluting downstream processing with clipping artefacts.
		s, e = fit_window
		clip_mask_fit = _clip_mask(x_proc[s:e], x_raw=x_in[s:e])
		clipped_indices = np.arange(s, e)[~clip_mask_fit]
		y[clipped_indices] = 0.0
		diagnostics["used_template_subtraction"] = True
		diagnostics["lag_samples"] = int(lag)
		diagnostics["alpha"] = float(alpha)
	else:
		y = x_proc.copy()
		n_blank = int(max(0, blank_samples))
		y[:n_blank] = 0.0
		diagnostics["used_template_subtraction"] = False

	y = np.asarray(y, dtype=np.float64)
	if y.size != x_in.size:
		raise RuntimeError("Internal error: output length changed unexpectedly.")

	if return_diagnostics:
		return y, diagnostics
	return y


def ringdown_energy_db_reduction(
	before: np.ndarray,
	after: np.ndarray,
	fs_hz: float = DEFAULT_FS_HZ,
	window_s: tuple[float, float] = (0.0, 1.2e-3),
) -> float:
	"""Compute ringdown energy reduction in dB inside a time window."""
	b = _as_1d_float64(before)
	a = _as_1d_float64(after)
	if b.size != a.size:
		raise ValueError("before and after must have the same length.")
	s_idx = int(max(0, round(window_s[0] * fs_hz)))
	e_idx = int(min(b.size, round(window_s[1] * fs_hz)))
	if e_idx <= s_idx:
		raise ValueError(f"Invalid window_s={window_s} for fs_hz={fs_hz}.")

	e_before = float(np.sum(b[s_idx:e_idx] ** 2))
	e_after = float(np.sum(a[s_idx:e_idx] ** 2))
	return float(10.0 * np.log10((e_before + 1e-12) / (e_after + 1e-12)))

