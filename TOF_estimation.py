from __future__ import annotations

import numpy as np
import pywt
from scipy.optimize import brentq
from scipy.signal import argrelmax, butter, correlate, hilbert, sosfiltfilt


FS_HZ = 2_000_000.0
SOUND_SPEED_M_S = 343.2
GATE_CM = 15.0
MATCHED_FILTER_CYCLES = 10
MORLET_WAVELET = "cmor1.5-1.0"

# === PCB GEOMETRY — PLACEHOLDER, REPLACE WITH CAD VALUES ============
# Origin: front face of PCB at the geometric mid-line. +x = right,
# +y = forward (towards object).
# Units: metres. Numbers here are visually estimated from slide 3
# of the project PowerPoint and MUST be replaced by IPMS CAD.
PCB = {
	"Mic1": (-0.020, 0.000),
	"Mic2": (0.000, 0.005),
	"Mic3": (+0.020, 0.000),
	"Tx1": (-0.020, 0.000),
	"Tx5": (0.000, 0.005),
	"Tx8": (+0.020, 0.000),
}
# ====================================================================

CARRIERS_HZ = {"Tx1": 40_000.0, "Tx5": 50_000.0, "Tx8": 60_000.0}

# === DELAY CALIBRATION CONSTANTS =====================================
# Fitted per (method, Tx, Mic) from Multifrequenz full dataset using
# Brent root-finding to zero the mean distance error.
# Units: seconds. Keys: ("path_a" | "path_b", tx_name, mic_name)
DELAY_CALIBRATION_S: dict[tuple[str, str, str], float] = {
    # Path A (bandpass matched filter) — median-fitted, nearest-peak picker, per-carrier order
    ("path_a", "Tx1", "Mic1"): 59.016e-6,
    ("path_a", "Tx1", "Mic2"): 80.612e-6,
    ("path_a", "Tx1", "Mic3"): 90.743e-6,
    ("path_a", "Tx5", "Mic1"): 67.741e-6,
    ("path_a", "Tx5", "Mic2"): 101.083e-6,
    ("path_a", "Tx5", "Mic3"): 26.246e-6,
    ("path_a", "Tx8", "Mic1"): 54.162e-6,
    ("path_a", "Tx8", "Mic2"): 64.455e-6,
    ("path_a", "Tx8", "Mic3"): -2.561e-6,
    # Path B (Morlet CWT) — median-fitted, nearest-peak picker, per-carrier order
    ("path_b", "Tx1", "Mic1"): 21.646e-6,
    ("path_b", "Tx1", "Mic2"): 25.047e-6,
    ("path_b", "Tx1", "Mic3"): 48.207e-6,
    ("path_b", "Tx5", "Mic1"): 10.898e-6,
    ("path_b", "Tx5", "Mic2"): 27.699e-6,
    ("path_b", "Tx5", "Mic3"): 93.197e-6,
    ("path_b", "Tx8", "Mic1"):  4.486e-6,
    ("path_b", "Tx8", "Mic2"): 16.565e-6,
    ("path_b", "Tx8", "Mic3"): 67.424e-6,
}
# ====================================================================


def tof_to_distance_m(
	tof_s: float,
	mic_name: str,
	tx_name: str,
	geom: dict[str, tuple[float, float]] = PCB,
	c_m_s: float = SOUND_SPEED_M_S,
	d_min_m: float = 0.001,
	d_max_m: float = 10.0,
) -> float:
	"""Invert a measured ToF to object distance (metres) using the full bistatic
	path model: path = ||Tx - reflector|| + ||reflector - Mic||.

	The reflector is assumed to lie on the sensor axis (x=0) at depth d along y.
	When Tx and Mic are co-located this reduces to the monostatic formula d = tof*c/2
	(corrected for the sensor's y-offset from the origin).

	Solves f(d) = 0 numerically via Brent's method.
	"""
	_validate_keys(mic_name=mic_name, tx_name=tx_name, geom=geom)
	tx_xy = np.asarray(geom[tx_name], dtype=np.float64)
	mic_xy = np.asarray(geom[mic_name], dtype=np.float64)
	total_path_m = float(tof_s) * float(c_m_s)

	def _residual(d: float) -> float:
		reflector = np.array([0.0, d])
		return float(np.linalg.norm(tx_xy - reflector) + np.linalg.norm(reflector - mic_xy)) - total_path_m

	try:
		return float(brentq(_residual, d_min_m, d_max_m, xtol=1e-6))
	except ValueError:
		# ToF outside plausible range — fall back to monostatic approximation
		return total_path_m / 2.0


def tof_to_distance_calibrated(
	tof_s: float,
	mic_name: str,
	tx_name: str,
	method: str,
	geom: dict[str, tuple[float, float]] = PCB,
	c_m_s: float = SOUND_SPEED_M_S,
	calibration: dict[tuple[str, str, str], float] = DELAY_CALIBRATION_S,
) -> float:
	"""Like tof_to_distance_m() but subtracts the fitted per-(method, Tx, Mic)
	delay constant before inverting.  method must be 'path_a' or 'path_b'.
	If the exact key is not found (e.g. path_a constants are commented out),
	falls back to the corresponding path_b constant, then to zero."""
	key = (method, tx_name, mic_name)
	fallback = ("path_b", tx_name, mic_name)
	tau_s = calibration.get(key, calibration.get(fallback, 0.0))
	return tof_to_distance_m(tof_s - tau_s, mic_name, tx_name, geom, c_m_s)


def _as_1d_float64(signal: np.ndarray) -> np.ndarray:
	x = np.asarray(signal, dtype=np.float64).squeeze()
	if x.ndim != 1:
		raise ValueError(f"Expected 1D waveform, got shape {x.shape}.")
	return x


def _validate_keys(mic_name: str, tx_name: str, geom: dict[str, tuple[float, float]]) -> None:
	if tx_name not in CARRIERS_HZ:
		raise ValueError(f"Unsupported tx_name '{tx_name}'. Expected one of {sorted(CARRIERS_HZ)}.")
	if tx_name not in geom:
		raise KeyError(f"Missing transmitter geometry for '{tx_name}'.")
	if mic_name not in geom:
		raise KeyError(f"Missing microphone geometry for '{mic_name}'.")


def _hann_burst(fc_hz: float, n_cycles: int = MATCHED_FILTER_CYCLES, fs_hz: float = FS_HZ) -> np.ndarray:
	duration_s = float(n_cycles) / float(fc_hz)
	n_samples = int(round(duration_s * float(fs_hz)))
	t = np.arange(n_samples, dtype=np.float64) / float(fs_hz)
	return np.hanning(n_samples) * np.sin(2.0 * np.pi * float(fc_hz) * t)


def _gated_peak(env: np.ndarray, idx_lo: int, idx_hi: int) -> int:
	a = max(0, int(idx_lo))
	b = min(env.size, int(idx_hi))
	if b <= a:
		raise ValueError(f"Invalid gate bounds ({idx_lo}, {idx_hi}) for signal length {env.size}.")
	return a + int(np.argmax(env[a:b]))


def _gated_peak_nearest(env: np.ndarray, idx_lo: int, idx_hi: int, n_nom: int, fc_hz: float, fs_hz: float = FS_HZ, top_n: int = 3) -> int:
	"""Among the top-N local maxima in the gate, return the index closest to n_nom.

	This avoids latching onto high-amplitude artifacts (e.g. direct crosstalk or
	early reflections) that happen to be stronger than the true echo.  If no local
	maxima are found (monotone gate) the function falls back to the global argmax.
	
	Applied uniformly across all Mics and Tx channels.
	"""
	a = max(0, int(idx_lo))
	b = min(env.size, int(idx_hi))
	if b <= a:
		raise ValueError(f"Invalid gate bounds ({idx_lo}, {idx_hi}) for signal length {env.size}.")
	gate_env = env[a:b]
	# Minimum separation: half a carrier period in samples — carrier-frequency-aware.
	# At FS=2MHz: Tx1/40kHz→25samp, Tx5/50kHz→20samp, Tx8/60kHz→17samp.
	# Distinct physical echoes are hundreds of samples apart; this only suppresses noise ripple.
	order = max(1, int(round(0.5 * fs_hz / fc_hz)))
	local_idxs = argrelmax(gate_env, order=order)[0]
	if len(local_idxs) == 0:
		return a + int(np.argmax(gate_env))
	abs_idxs = local_idxs + a
	# Keep top-N by amplitude, then pick the one closest to the nominal arrival
	amps = env[abs_idxs]
	top_cut = min(top_n, len(abs_idxs))
	top_abs = abs_idxs[np.argsort(amps)[::-1][:top_cut]]
	return int(top_abs[np.argmin(np.abs(top_abs - n_nom))])


def _parabolic_subsample(r: np.ndarray, k: int) -> float:
	if k <= 0 or k >= (len(r) - 1):
		return 0.0
	denom = float(r[k - 1] - 2.0 * r[k] + r[k + 1])
	if abs(denom) < 1e-18:
		return 0.0
	raw = float((r[k - 1] - r[k + 1]) / (2.0 * denom))
	return max(-0.5, min(0.5, raw))


def _nominal_arrival_index(
	distance_m: float,
	mic_name: str,
	tx_name: str,
	geom: dict[str, tuple[float, float]],
	c_m_s: float,
	fs_hz: float,
) -> int:
	_validate_keys(mic_name=mic_name, tx_name=tx_name, geom=geom)
	tx_xy = np.asarray(geom[tx_name], dtype=np.float64)
	mic_xy = np.asarray(geom[mic_name], dtype=np.float64)
	reflector_xy = np.array([0.0, float(distance_m)], dtype=np.float64)
	d_nom_m = np.linalg.norm(tx_xy - reflector_xy) + np.linalg.norm(reflector_xy - mic_xy)
	return int(round(d_nom_m / float(c_m_s) * float(fs_hz)))


def tof_path_A(
	x_denoised: np.ndarray,
	distance_m: float,
	mic_name: str,
	tx_name: str,
	geom: dict[str, tuple[float, float]] = PCB,
	c_m_s: float = SOUND_SPEED_M_S,
	fs_hz: float = FS_HZ,
	gate_cm: float = GATE_CM,
	use_bandpass: bool = True,
) -> tuple[float, float, int]:
	x = _as_1d_float64(x_denoised)
	fc_hz = CARRIERS_HZ[tx_name]

	if use_bandpass:
		sos = butter(
			2,
			[fc_hz - 3_000.0, fc_hz + 3_000.0],
			btype="band",
			fs=float(fs_hz),
			output="sos",
		)
		x_band = sosfiltfilt(sos, x)
	else:
		x_band = x

	reference = _hann_burst(fc_hz=fc_hz, n_cycles=MATCHED_FILTER_CYCLES, fs_hz=fs_hz)
	matched = correlate(x_band, reference, mode="same")
	envelope = np.abs(hilbert(matched))

	n_nom = _nominal_arrival_index(
		distance_m=distance_m,
		mic_name=mic_name,
		tx_name=tx_name,
		geom=geom,
		c_m_s=c_m_s,
		fs_hz=fs_hz,
	)
	gate_samples = int(round((float(gate_cm) * 1e-2) / float(c_m_s) * float(fs_hz)))
	k = _gated_peak_nearest(envelope, n_nom - gate_samples, n_nom + gate_samples, n_nom, fc_hz=fc_hz, fs_hz=fs_hz)

	delta = _parabolic_subsample(envelope, k)
	tof_s = (k - delta) / float(fs_hz)
	amplitude = float(envelope[k])
	return float(tof_s), amplitude, int(k)


def tof_path_B(
	x_denoised: np.ndarray,
	distance_m: float,
	mic_name: str,
	tx_name: str,
	geom: dict[str, tuple[float, float]] = PCB,
	c_m_s: float = SOUND_SPEED_M_S,
	fs_hz: float = FS_HZ,
	gate_cm: float = GATE_CM,
) -> tuple[float, float, int]:
	x = _as_1d_float64(x_denoised)
	fc_hz = CARRIERS_HZ[tx_name]
	scale = float(fs_hz) / float(fc_hz)

	cwt_coeffs, _ = pywt.cwt(x, [scale], MORLET_WAVELET, sampling_period=1.0 / float(fs_hz))
	envelope = np.abs(cwt_coeffs[0])

	n_nom = _nominal_arrival_index(
		distance_m=distance_m,
		mic_name=mic_name,
		tx_name=tx_name,
		geom=geom,
		c_m_s=c_m_s,
		fs_hz=fs_hz,
	)
	gate_samples = int(round((float(gate_cm) * 1e-2) / float(c_m_s) * float(fs_hz)))
	k = _gated_peak_nearest(envelope, n_nom - gate_samples, n_nom + gate_samples, n_nom, fc_hz=fc_hz, fs_hz=fs_hz)

	delta = _parabolic_subsample(envelope, k)
	tof_s = (k - delta) / float(fs_hz)
	amplitude = float(envelope[k])
	return float(tof_s), amplitude, int(k)
