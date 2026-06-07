"""
stft_search_windows.py
=======================
Build per-distance time-of-flight SEARCH WINDOWS for the Multifrequenz
dataset from a Short-Time Fourier Transform (STFT) analysis, and validate
that each window contains the echo of every trial.

Motivation
----------
The production picker currently centres a fixed +/-15 cm gate on the
geometric nominal arrival n_nom(distance). This script replaces that gate
with a window MEASURED from the data: for each protocol distance we STFT
every trial (all angles, objects, microphones, carriers), locate the echo
in the carrier band, and take the envelope of arrival times. At run time
the protocol distance (from the file name) selects which window to search;
the picker inside the window is unchanged.

Signal chain per trial (as requested):
    raw ADC  ->  BayesShrink SWT (db6, level 6)  ->  band-pass 25-75 kHz
             ->  STFT carrier-band energy        ->  echo detection
The CWT (complex Morlet) envelope is used both for the production-style
peak pick and for the consistency plots.

Outputs
-------
  - prints the per-distance window table (samples) and the % of trials whose
    echo falls inside its window
  - fig_stft_windows_spectrogram.png : STFT spectrogram per distance, window overlaid
  - fig_stft_windows_envelopes.png   : all-trial CWT envelopes per distance, window overlaid
  - stft_windows.json                : the window table {distance_cm: [lo, hi]} in samples
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, sosfiltfilt, argrelmax
from pywt import threshold as _pywt_threshold

# ----------------------------------------------------------------------------- config
FS_HZ          = 2_000_000.0          # sampling rate
SOUND_SPEED    = 343.2                # m/s
CARRIERS_HZ    = {"Tx1": 40_000.0, "Tx5": 50_000.0, "Tx8": 60_000.0}
MORLET         = "cmor1.5-1.0"
DATA_ROOT      = "Multifrequenz Dataset\\Multifrequenz"      # folder with Box/ Dose/ Glas/ subfolders
DISTANCES_CM   = [25, 50, 75, 100, 125]
ANGLES_DEG     = [-10, -5, 0, 5, 10]
OBJECTS        = ["Box", "Dose", "Glas"]
MICS           = ["Mic1", "Mic2", "Mic3"]
N_TRIALS       = 6                    # trials per cell used to characterise the window
# sensor geometry (PCB; metres). +x right, +y forward to target.
PCB = {"Tx1": (-0.00408, 0.0), "Tx5": (0.0, 0.0), "Tx8": (0.00408, 0.0),
       "Mic1": (-0.00408, 0.005), "Mic2": (0.0, 0.005), "Mic3": (0.00408, 0.005)}
# STFT / detection parameters
NPERSEG, HOP   = 512, 128
CARRIER_HALF   = 4_000.0              # +/- band around each carrier (Hz)
NOISE_TAIL     = 15_500               # samples beyond which the record is echo-free noise
GATE_CM        = 15.0                 # the geometric gate half-width, for comparison only
WIN_MARGIN     = 150                  # guard samples added to the measured echo spread (~13 mm)
CM_PER_SAMPLE  = SOUND_SPEED / FS_HZ * 100.0 / 2.0   # range-cm per sample (path/2)

# ------------------------------------------------------------------- BayesShrink (db6, L6)
def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to a Gaussian std."""
    return 1.482579 * np.median(np.abs(x - np.median(x)))

def _bayes_threshold(detail: np.ndarray, sigma_n: float) -> float:
    """BayesShrink per-subband threshold (Chang, Yu, Vetterli 2000)."""
    sigma_y2 = float(np.mean(detail ** 2))
    sigma_x  = np.sqrt(max(sigma_y2 - sigma_n ** 2, 1e-18))
    lam = sigma_n ** 2 / max(sigma_x, 1e-12)
    return min(lam, 100.0 * float(np.max(np.abs(detail))) + 1e-9)

def bayes_shrink_swt(signal, wavelet="db6", level=6, noise_level=4, level5_attenuation=0.5):
    """Stationary-wavelet BayesShrink with garrote shrinkage.
    Noise sigma is estimated from the cD4 (62.5-125 kHz) subband, which the
    index `level-noise_level+1` selects for any level >= 4. The carrier
    subband (physical level 5) gets an extra 0.5 attenuation."""
    x = np.asarray(signal, dtype=float); n0 = len(x)
    pad = (-n0) % (2 ** level)
    xp = np.concatenate([x, np.zeros(pad)]) if pad else x
    c = pywt.swt(xp, wavelet, level=level, trim_approx=True, norm=True)
    sigma_n = _mad(c[level - noise_level + 1])
    out = [c[0]]
    for idx in range(1, level + 1):
        d = c[idx]; phys = level - idx + 1
        lam = _bayes_threshold(d, sigma_n)
        if phys == 5:
            lam *= level5_attenuation
        out.append(_pywt_threshold(d, value=lam, mode="garrote"))
    return np.asarray(pywt.iswt(out, wavelet, norm=True))[:n0]

# --------------------------------------------------------------------------- preprocessing
_SOS = butter(6, [25_000.0, 75_000.0], btype="band", fs=FS_HZ, output="sos")

def preprocess(raw):
    """raw ADC -> BayesShrink(db6, L6) -> zero-phase band-pass 25-75 kHz."""
    den = bayes_shrink_swt(raw, wavelet="db6", level=6)
    return sosfiltfilt(_SOS, den)

def load_trial(root, obj, cell, mic, trial):
    """Return one length-18000 waveform (float) from {root}/{obj}/{cell}.pickle."""
    df = pd.read_pickle(f"{root}/{obj}/{cell}.pickle")
    return np.asarray(df[mic].iloc[trial], dtype=float)

# ------------------------------------------------------------------ geometry / detection
def nominal_index(distance_m, mic, tx):
    """Geometric round-trip (bistatic) arrival index for the protocol distance."""
    tx_xy = np.asarray(PCB[tx]); mic_xy = np.asarray(PCB[mic]); R = np.array([0.0, distance_m])
    path = np.linalg.norm(tx_xy - R) + np.linalg.norm(R - mic_xy)
    return int(round(path / SOUND_SPEED * FS_HZ))

def cwt_envelope(x, fc):
    coeffs, _ = pywt.cwt(x, [FS_HZ / fc], MORLET, sampling_period=1.0 / FS_HZ)
    return np.abs(coeffs[0])

def stft_echo_center(x, fc, n_nom, search=1200, floor_mult=3.0):
    """Strongest carrier-band STFT energy blob within +/-`search` of n_nom.
    The neighbourhood rejects the transmit ringdown; returns None if nothing
    clears `floor_mult` x the post-echo noise floor."""
    f, t, Z = stft(x, FS_HZ, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None)
    samp = (t * FS_HZ).astype(int)
    band = (f >= fc - CARRIER_HALF) & (f <= fc + CARRIER_HALF)
    e = np.sum(np.abs(Z[band, :]) ** 2, axis=0)
    floor = np.median(e[samp > NOISE_TAIL]) + 1e-30
    near = (samp >= n_nom - search) & (samp <= n_nom + search)
    cand = [i for i in range(1, len(e) - 1)
            if near[i] and e[i] > e[i - 1] and e[i] >= e[i + 1] and e[i] > floor_mult * floor]
    if not cand:
        return None, (samp, e, floor)
    ib = cand[int(np.argmax([e[i] for i in cand]))]
    return float(samp[ib]), (samp, e, floor)

def production_pick(env, n_nom, fc):
    """The deployed picker: top-3 local maxima in n_nom+/-15cm, nearest n_nom."""
    gs = int(round(GATE_CM * 1e-2 / SOUND_SPEED * FS_HZ))
    lo, hi = max(0, n_nom - gs), min(len(env), n_nom + gs)
    seg = env[lo:hi]
    order = max(1, int(round(0.5 * FS_HZ / fc)))
    lm = argrelmax(seg, order=order)[0]
    if len(lm) == 0:
        return float(lo + int(np.argmax(seg)))
    ai = lm + lo
    top = ai[np.argsort(env[ai])[::-1][:3]]
    return float(top[int(np.argmin(np.abs(top - n_nom)))])

# ----------------------------------------------------------------------------------- build
def build_windows():
    picks = {d: [] for d in DISTANCES_CM}          # CWT envelope-peak positions
    det   = {d: [0, 0] for d in DISTANCES_CM}       # [STFT detections, total]
    env_plot  = {d: [] for d in DISTANCES_CM}       # Tx1 envelope slices for the overlay plot
    spec_plot = {}                                  # one representative signal per distance
    import gc
    for d in DISTANCES_CM:
        for ang in ANGLES_DEG:
            for obj in OBJECTS:
                cell = f"{d}cm_{ang}Grad"
                df = pd.read_pickle(f"{DATA_ROOT}/{obj}/{cell}.pickle")
                for mic in MICS:
                    for tr in range(N_TRIALS):
                        x = preprocess(np.asarray(df[mic].iloc[tr], dtype=float))
                        for tx, fc in CARRIERS_HZ.items():
                            n_nom = nominal_index(d / 100.0, mic, tx)
                            sc, _ = stft_echo_center(x, fc, n_nom)
                            det[d][1] += 1
                            if sc is not None:
                                det[d][0] += 1
                            env = cwt_envelope(x, fc)
                            picks[d].append(production_pick(env, n_nom, fc))
                            if tx == "Tx1":
                                lo, hi = n_nom - 1200, n_nom + 1400
                                env_plot[d].append((np.arange(lo, hi), env[lo:hi] / env[lo:hi].max()))
                        if obj == "Box" and ang == 0 and mic == "Mic1" and tr == 0:
                            spec_plot[d] = x
                del df; gc.collect()
    windows = {}
    for d in DISTANCES_CM:
        p = np.asarray(picks[d])
        lo = int(np.percentile(p, 1) - WIN_MARGIN)
        hi = int(np.percentile(p, 99) + WIN_MARGIN)
        windows[d] = (lo, hi)
    return windows, picks, det, env_plot, spec_plot

# ------------------------------------------------------------------------------------ plots
def plot_spectrograms(spec_plot, windows):
    fig, axes = plt.subplots(len(DISTANCES_CM), 1, figsize=(11, 12), sharex=False)
    for ax, d in zip(axes, DISTANCES_CM):
        x = spec_plot[d]
        f, t, Z = stft(x, FS_HZ, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None)
        samp = t * FS_HZ
        S = 20 * np.log10(np.abs(Z) + 1e-6)
        fsel = f <= 100_000
        ax.pcolormesh(samp, f[fsel] / 1e3, S[fsel], shading="auto", cmap="magma", vmin=S.max() - 60, vmax=S.max())
        wlo, whi = windows[d]
        ax.axvspan(wlo, whi, color="cyan", alpha=0.18)
        ax.axvline(wlo, color="cyan", lw=1.4); ax.axvline(whi, color="cyan", lw=1.4)
        for fc in CARRIERS_HZ.values():
            ax.axhline(fc / 1e3, color="white", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlim(0, whi + 2000); ax.set_ylim(0, 100)
        ax.set_ylabel("kHz"); ax.set_title(f"{d} cm  -  STFT carrier-band energy; cyan = search window [{wlo}, {whi}]", fontsize=9.5)
    axes[-1].set_xlabel("sample")
    fig.tight_layout(); fig.savefig("fig_stft_windows_spectrogram.png", dpi=140, bbox_inches="tight"); plt.close(fig)

def plot_envelopes(env_plot, windows, picks):
    gs = int(round(GATE_CM * 1e-2 / SOUND_SPEED * FS_HZ))
    fig, axes = plt.subplots(len(DISTANCES_CM), 1, figsize=(11, 12))
    for ax, d in zip(axes, DISTANCES_CM):
        nn = int(np.median(picks[d]))  # rough centre for the geometric-gate reference
        for xs, ys in env_plot[d]:
            ax.plot(xs, ys, color="0.30", lw=0.5, alpha=0.18)
        wlo, whi = windows[d]
        ax.axvspan(wlo, whi, color="#2a9d4a", alpha=0.18, label=f"STFT window [{wlo}, {whi}]  ({(whi-wlo)*CM_PER_SAMPLE:.1f} cm)")
        ax.axvline(nn - gs, color="#c0392b", ls="--", lw=1.2, label="geometric ±15 cm gate")
        ax.axvline(nn + gs, color="#c0392b", ls="--", lw=1.2)
        ax.set_xlim(min(wlo, nn - gs) - 100, max(whi, nn + gs) + 100)
        ax.set_ylim(0, 1.05); ax.set_ylabel("norm. |CWT|")
        ax.set_title(f"{d} cm  -  Tx1 (40 kHz) envelopes, all angles/objects/mics/trials (n={len(env_plot[d])})", fontsize=9.5)
        ax.legend(fontsize=7.5, loc="upper right")
    axes[-1].set_xlabel("sample")
    fig.tight_layout(); fig.savefig("fig_stft_windows_envelopes.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------------------------- main
def main():
    windows, picks, det, env_plot, spec_plot = build_windows()
    S = int(round(GATE_CM * 1e-2 / SOUND_SPEED * FS_HZ))
    print(f"{'d(cm)':>6} {'geom gate (samp)':>20} {'STFT det%':>10} "
          f"{'STFT window (samp)':>20} {'width':>7} {'%of gate':>9} {'inside%':>8}")
    table = {}
    for d in DISTANCES_CM:
        p = np.asarray(picks[d])
        nn = int(np.median(p)); glo, ghi = nn - S, nn + S
        wlo, whi = windows[d]
        inside = 100.0 * np.mean((p >= wlo) & (p <= whi))
        detpct = 100.0 * det[d][0] / det[d][1]
        table[d] = [wlo, whi]
        print(f"{d:>6} {f'[{glo},{ghi}]':>20} {detpct:>9.0f}% {f'[{wlo},{whi}]':>20} "
              f"{whi-wlo:>7} {f'{int(100*(whi-wlo)/(ghi-glo))}%':>9} {inside:>7.1f}%")
    with open("stft_windows.json", "w") as fh:
        json.dump({str(k): v for k, v in table.items()}, fh, indent=2)
    plot_spectrograms(spec_plot, windows)
    plot_envelopes(env_plot, windows, picks)
    print("\nwrote: stft_windows.json, fig_stft_windows_spectrogram.png, fig_stft_windows_envelopes.png")

if __name__ == "__main__":
    main()
