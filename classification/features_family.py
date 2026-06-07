"""features_family.py — carrier + ROI-window feature library (multi-wavelet).

This replaces the merged Family 1 + 1.5 + 3 extractor. It produces ONE feature
vector per (Mic, trial) and obeys two structural controls only — there is no
Cohen's d screen and no per-feature leakage flagging here:

  (1) WINDOW.   Every feature is read from the fixed region of interest about the
      per-trial time-of-flight peak, [peak - WIN_BEFORE, peak + WIN_AFTER].
  (2) CARRIERS. Spectral content is read only at the three emitted carriers
      (40/50/60 kHz). At wavelet-packet level 7 the packet bandwidth is
      fs/2^8 = 7.8 kHz, so the carriers fall in separate freq-ordered packets
      5, 6, 7 for every wavelet (the mapping is frequency-based).

Two transforms carry their own frequency selectivity, so NO Butterworth
band-pass is applied here and the echo is not smeared:

    * Wavelet packet transform (WPT), level 7 — supplies the per-carrier ENERGY
        and SPECTRAL features under the db6 analysis wavelet, and the per-carrier
        SPARSITY under EACH wavelet in `SPARSITY_WAVELETS`. The sparsity is the only
        multi-wavelet read: S = max|c| / sum|c| of a carrier packet is exactly the
        per-level sparsity that feeds mu_sc in wavelet_choice.py (`sparsity()`), and
        computing it under each wavelet exposes how concentrated each one's
        carrier-packet representation is, instead of using mu_sc to pick a single
        wavelet. Everything else in the WPT block uses db6.
  * Single-scale complex Morlet CWT (cmor1.5-1.0) at scale fs/f_c — supplies all
    ENVELOPE / TIMING / TAIL / MODAL features. |W| is the carrier-band envelope
    directly, formed analytically with no IIR ringing. Morlet does not use the
    packet wavelet, so the CWT block is computed ONCE (wavelet-independent).

=====================  FAMILY 1 / 1.5 / 3 -> CARRIER MAPPING  =================
 FF feature (original)                  -> here                         source
 ----------------------------------------------------------------------------
 FF1  SWT subband energy/logenergy      -> wpt_f40/f50/f60              WPT carrier packets, db6
 FF1  SWT subband sparsity              -> {w}_sp40/sp50/sp60           carrier-packet sparsity, EACH wavelet
 FF1  SWT subband entropy               -> wpt_entropy                  carrier-energy distribution, db6
 FF1  SWT subband kurtosis/peakamp/pos  -> cwt_envkurt{c}/peakamp{c}/.. CWT scale fs/f_c
 FF1  Hilbert env peakamp/time/fwhm/..  -> cwt_peakamp/peaktime/fwhm/.. CWT per carrier
 FF1  spectral centroid/spread/dominant -> wpt_centroid/spread/dominant WPT 3 carrier packets, db6
 FF1  spectral skew/kurt                -> band_skew / band_kurt        fine FFT spectrum over 35-65 kHz (once)
 FF1  spectral centroid/spread (fine)   -> band_centroid / band_spread  fine FFT spectrum over 35-65 kHz (once)
 FF1  carrier_band_frac                 -> wpt_f40+wpt_f50+wpt_f60      WPT carrier packets, db6
 FF1  tail late_to_peak/n_peaks/decay   -> cwt_late_to_peak{c}/npeaks{c}/modal_decay{c}/modal_q{c}  CWT per carrier, in window
 FF1  cross-mic peak-time / centroid    -> dpeak_M1M2_{c}_us / dcent_M1M2       CWT, wavelet-independent
 FF1.5 cross-carrier energy tilt        -> wpt_tilt, wpt_dE_50_40       WPT carrier packets, db6
 FF1.5 cross-carrier peak tilt          -> cwt_peak_tilt, dP_50_40      CWT peak amplitudes
 FF1.5 carrier cepstrum                 -> wptcep1/wptcep2              DCT of 3 carrier log-energies, db6
 FF1.5 cross-mic energy/peak ratios     -> dPmic_M1M2_{c}, signed_amp_ratio    CWT peak amps (wavelet-indep)
 FF3  wpt_f / centroid / tilt           -> wpt_f.. / wpt_centroid / wpt_tilt   WPT, db6
 FF3  cwt_width / rise / env_kurt       -> cwt_fwhm{c}/rise{c}/envkurt{c}       CWT per carrier
 FF3  cwt_peak_tilt / centroid          -> cwt_peak_tilt / cwt_centroid_khz     CWT
=============================================================================

Kept on purpose, though they read just off the carriers (a within-band SNR /
leakage reference, computed under db6): wpt_snr_db (carrier energy vs the noise
packet 9) and wpt_neighbour_leak (energy in the +/-1 neighbour packets).

NOT computed: the secondary-echo features. That reflection sits at lag 500-1150
samples, past WIN_AFTER, so it is outside the ROI window.

NOT reproduced from the families (excluded by the carrier/window restriction):
off-carrier SWT subbands, the broadband Hilbert envelope (superseded by the
per-carrier CWT envelopes), and the full out_of_band_frac.

The carrier cepstrum wptcep1/wptcep2 is the WPT carrier-band equivalent of
FF1.5's real cepstrum: a DCT of the three carrier-packet log-energies. FF1.5's
full 13-coefficient cepstrum read the whole-spectrum envelope (mostly off the
carriers) and is not reproduced under the carrier-only rule.

Scope note: every feature is READ from inside the ROI window (the packet slice
[clo:chi], the CWT envelope slice [lo:hi], or the band FFT of x[lo:hi]). The WPT
and CWT are computed on the whole trace and then read in the ROI rather than
being run on a hard-cut ROI, to avoid truncation edge artefacts; the wavelets'
finite support means the ROI-edge values are influenced by samples just outside
the window, which is unavoidable for any time-frequency transform.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Sequence
import numpy as np
import pywt
from scipy.signal import find_peaks
from scipy.stats import kurtosis as _kurt
from scipy.fft import dct as _dct

FS_DEFAULT = 2_000_000.0
CARRIERS = (40_000.0, 50_000.0, 60_000.0)
WPT_WAVELET = "db6"            # analysis wavelet for the carrier ENERGY + SPECTRAL features
# The carrier-packet SPARSITY (only) is computed under each of these wavelets, to
# expose how concentrated each one's carrier-packet representation is. Everything
# else in the WPT block uses WPT_WAVELET (db6).
SPARSITY_WAVELETS = ("db6", "db2", "bior2.2", "sym6", "coif2")
WPT_LEVEL = 7                  # packet BW = fs/2^8 = 7.8 kHz -> carriers in packets 5/6/7
MORLET = "cmor1.5-1.0"
# Fixed ROI about the per-trial TOF peak (same as the production window).
WIN_BEFORE, WIN_AFTER = 256, 768
TAIL_MIN_US, TAIL_MAX_US = 50.0, 300.0
# Fine FFT spectral shape (centroid/spread/skew/kurtosis) is read over this band
# rather than the 3 carrier points, so the higher moments are meaningful.
SPECTRAL_BAND = (35_000.0, 65_000.0)

# wpt_snr_db and wpt_neighbour_leak read just off the carriers (the noise packet
# and the +/-1 neighbour packets). Kept on purpose as a within-band SNR / leakage
# reference. The secondary echo is NOT computed: it sits past the window.
NOISE_PACKET_IDX = 9


# ------------------------------------------------------- WPT freq ordering ----
def _freq_paths(level):
    order = sorted((n ^ (n >> 1), n) for n in range(2 ** level))
    return ["".join("a" if b == "0" else "d" for b in format(nat, f"0{level}b"))
            for _, nat in order]


_PATHS = _freq_paths(WPT_LEVEL)


def _pkt_idx(fc, fs, level):
    return max(0, min(2 ** level - 1, int(round(fc / (fs / 2 ** (level + 1)) - 0.5))))


def _sparsity(c):
    """max|c| / sum|c| — the per-subband sparsity used by wavelet_choice.sparsity()."""
    a = np.abs(np.asarray(c, float))
    s = a.sum()
    return float(a.max() / s) if s > 0 else 0.0


# ---------------------------------------------------- CWT envelope helpers ----
def _cwt_env(x, fc, fs):
    coef, _ = pywt.cwt(x, [fs / fc], MORLET, sampling_period=1.0 / fs)
    return np.abs(coef[0])


def _fwhm(env, pk):
    half = 0.5 * env[pk]
    if half <= 0:
        return 0.0
    l = pk
    while l > 0 and env[l] > half:
        l -= 1
    r = pk
    while r < len(env) - 1 and env[r] > half:
        r += 1
    return float(r - l)


def _rise(env, pk, lo=0.1):
    thr = lo * env[pk]
    i = pk
    while i > 0 and env[i] > thr:
        i -= 1
    return float(pk - i)


def _decay_db_us(env, pk, fs, span_us, eps=1e-12):
    hi = min(len(env), pk + int(span_us * 1e-6 * fs) + 1)
    if hi - pk < 3:
        return 0.0
    y = 20.0 * np.log10(env[pk:hi] + eps)
    t = (np.arange(pk, hi) - pk) / fs * 1e6
    return float(np.polyfit(t, y, 1)[0])


def _band_spectral_moments(x, lo, hi, fs, band=SPECTRAL_BAND, eps=1e-12):
    """Spectral centroid/spread/skew/kurtosis on a fine FFT spectrum restricted to
    the carrier band. The three carrier-packet energies are too coarse for shape
    moments; the band FFT gives ~15 bins across 35-65 kHz at the ROI length, which
    is enough for the higher moments to be meaningful. Wavelet-independent."""
    seg = np.asarray(x[lo:hi], float)
    n = seg.size
    keys = ("band_centroid_khz", "band_spread_khz", "band_skew", "band_kurt")
    if n < 8:
        return OrderedDict((k, 0.0) for k in keys)
    w = np.hanning(n) * seg
    F = np.fft.rfftfreq(n, 1.0 / fs)
    P = np.abs(np.fft.rfft(w)) ** 2
    m = (F >= band[0]) & (F <= band[1])
    Fb, Pb = F[m], P[m]
    s = float(Pb.sum())
    if s < eps or Fb.size < 4:
        return OrderedDict((k, 0.0) for k in keys)
    p = Pb / s
    cen = float((Fb * p).sum())
    spr = float(np.sqrt(((Fb - cen) ** 2 * p).sum()))
    if spr > 1e-9:
        sk = float(((Fb - cen) ** 3 * p).sum() / spr ** 3)
        kt = float(((Fb - cen) ** 4 * p).sum() / spr ** 4) - 3.0
    else:
        sk = kt = 0.0
    return OrderedDict([("band_centroid_khz", cen / 1e3), ("band_spread_khz", spr / 1e3),
                        ("band_skew", sk), ("band_kurt", kt)])


def _tail_modal(envf, kg, fs, win_lo, win_hi, fc, eps=1e-12):
    """Tail energy ratio, secondary-peak count, decay slope and Q for ONE carrier,
    read from its CWT envelope after the echo peak `kg`. ALL reads are clipped to
    the ROI [win_lo, win_hi], so nothing reaches the pre-echo transmit ring-down
    (before the peak) or past the window end. This is the near ECHO ring-down."""
    tag = int(fc / 1000)
    out = OrderedDict()
    a = max(win_lo, kg + int(TAIL_MIN_US * 1e-6 * fs))
    b = min(win_hi, kg + int(TAIL_MAX_US * 1e-6 * fs))
    m_lo = max(win_lo, kg - 25)
    m_hi = min(win_hi, kg + 25)
    main = float(np.sum(envf[m_lo:m_hi] ** 2)) + eps
    tail = float(np.sum(envf[a:b] ** 2)) if b > a else 0.0
    out[f"cwt_late_to_peak{tag}"] = tail / main
    s_lo = max(win_lo, kg - int(300e-6 * fs))
    s_hi = min(win_hi, kg + int(300e-6 * fs))
    seg = envf[s_lo:s_hi]
    if seg.size >= 3 and seg.max() > 0:
        pks, _ = find_peaks(seg, height=0.25 * float(envf[kg]))
        out[f"cwt_npeaks{tag}"] = float(np.sum((pks + s_lo) != kg))
    else:
        out[f"cwt_npeaks{tag}"] = 0.0
    decay_us = min(TAIL_MAX_US, (win_hi - kg) / fs * 1e6)
    slope = _decay_db_us(envf, kg, fs, decay_us)
    out[f"modal_decay{tag}_db_us"] = slope
    if slope < -1e-6:
        tau_s = -1.0 / (slope * 1e6 * np.log(10) / 20.0)        # dB/us -> tau (s)
        out[f"modal_q{tag}"] = float(min(np.pi * fc * tau_s, 1e4))
    else:
        out[f"modal_q{tag}"] = 0.0
    return out


# --------------------------------------------------- WPT carrier reads --------
def _wpt_db6(x, lo, hi, fs, level, carriers, wavelet=WPT_WAVELET, eps=1e-12):
    """Carrier energy fractions + 3-carrier spectral features under ONE analysis
    wavelet (db6), all read from the ROI packet slice [clo:chi]. The sparsity is
    NOT here -- it is the only feature computed under the full wavelet bank."""
    fcs = sorted(carriers)
    n = x.size
    pad = (-n) % (2 ** level)
    xp = np.concatenate([x, np.zeros(pad)]) if pad else x
    wp = pywt.WaveletPacket(xp, wavelet, mode="symmetric", maxlevel=level)
    down = 2 ** level
    paths = _PATHS if level == WPT_LEVEL else _freq_paths(level)
    carr_idx = [_pkt_idx(fc, fs, level) for fc in fcs]

    clo, chi = lo // down, hi // down + 1
    E = np.array([float(np.sum(np.asarray(wp[paths[ci]].data)[clo:chi] ** 2)) for ci in carr_idx])
    c3 = float(E.sum()) + eps
    fc3 = np.array(fcs)

    f = OrderedDict()
    for fc, Ec in zip(fcs, E / c3):
        f[f"wpt_f{int(fc / 1000)}"] = float(Ec)        # carrier energy fraction
    cen = float((fc3 * E).sum() / c3)
    spr = float(np.sqrt(((fc3 - cen) ** 2 * E).sum() / c3))
    f["wpt_centroid_khz"] = cen / 1e3
    f["wpt_spread_khz"] = spr / 1e3
    f["wpt_dominant_khz"] = float(fc3[int(np.argmax(E))] / 1e3)
    f["wpt_tilt"] = float(np.log((E[-1] + eps) / (E[0] + eps)))
    f["wpt_dE_50_40"] = float(np.log((E[1] + eps) / (E[0] + eps)))
    pp = (E / c3)[(E / c3) > eps]
    f["wpt_entropy"] = float(-np.sum(pp * np.log(pp)))
    cep = _dct(np.log(E + eps), norm="ortho")
    f["wptcep1"] = float(cep[1])
    f["wptcep2"] = float(cep[2])
    # within-band off-carrier references (KEPT): SNR against the noise packet and
    # the leak into the immediate +/-1 neighbour packets.
    npkt = 2 ** level
    e_noise = float(np.sum(np.asarray(wp[paths[min(NOISE_PACKET_IDX, npkt - 1)]].data)[clo:chi] ** 2)) + eps
    f["wpt_snr_db"] = 10.0 * float(np.log10(c3 / e_noise))
    nb = [i for i in (carr_idx[0] - 1, carr_idx[-1] + 1) if 0 <= i < npkt]
    e_nb = float(sum(np.sum(np.asarray(wp[paths[i]].data)[clo:chi] ** 2) for i in nb))
    f["wpt_neighbour_leak"] = float(e_nb / (c3 + e_nb + eps))
    return f


def _wpt_sparsity_multi(x, lo, hi, fs, level, wavelets, carriers):
    """Carrier-packet sparsity S = max|c|/sum|c| under EACH analysis wavelet, read
    from the ROI packet slice. This is the ONLY multi-wavelet feature: it exposes
    how concentrated each wavelet's carrier-packet representation is, instead of
    using that to pick a single wavelet. Named {wavelet}_sp{carrier}."""
    fcs = sorted(carriers)
    n = x.size
    pad = (-n) % (2 ** level)
    xp = np.concatenate([x, np.zeros(pad)]) if pad else x
    down = 2 ** level
    paths = _PATHS if level == WPT_LEVEL else _freq_paths(level)
    carr_idx = [_pkt_idx(fc, fs, level) for fc in fcs]
    clo, chi = lo // down, hi // down + 1
    f = OrderedDict()
    for w in wavelets:
        wp = pywt.WaveletPacket(xp, w, mode="symmetric", maxlevel=level)
        tag = w.replace(".", "")                       # bior2.2 -> bior22
        for fc, ci in zip(fcs, carr_idx):
            f[f"{tag}_sp{int(fc / 1000)}"] = _sparsity(np.asarray(wp[paths[ci]].data)[clo:chi])
    return f


# =============================================================== main extractor
def extract_features(
    waveform: np.ndarray,
    fs: float = FS_DEFAULT,
    window: tuple[int, int] | None = None,
    carriers=CARRIERS,
    sparsity_wavelets: Sequence[str] = SPARSITY_WAVELETS,
    wpt_level: int = WPT_LEVEL,
    peak: int | None = None,
    before: int = WIN_BEFORE,
    after: int = WIN_AFTER,
) -> "OrderedDict[str, float]":
    """One (Mic, trial) waveform -> carrier+ROI feature dict.

    Pass `peak` (per-trial TOF sample) so the window is [peak-before, peak+after];
    `window` is the explicit fallback. The carrier energy/spectral block uses db6;
    only the carrier-packet sparsity is computed under each wavelet in
    `sparsity_wavelets`; the Morlet CWT block is wavelet-independent.
    """
    x = np.asarray(waveform, dtype=np.float64).ravel()
    n = x.size
    if peak is not None:
        window = (int(peak) - before, int(peak) + after)
    lo, hi = (0, n) if window is None else (max(0, int(window[0])), min(n, int(window[1])))
    f = OrderedDict()
    eps = 1e-12
    fcs = sorted(carriers)

    # ---------- WPT carrier energy + spectral, db6 only ----------
    f.update(_wpt_db6(x, lo, hi, fs, wpt_level, carriers))
    # ---------- carrier-packet sparsity under each wavelet (the ONLY multi-wavelet read) ----------
    f.update(_wpt_sparsity_multi(x, lo, hi, fs, wpt_level, sparsity_wavelets, carriers))

    # ---------- fine spectral shape over 35-65 kHz (wavelet-independent, once) ----------
    f.update(_band_spectral_moments(x, lo, hi, fs))

    # ---------- CWT (Morlet) block — wavelet-independent, computed once ----------
    peak_amp, peak_t = {}, {}
    envs = {}
    for fc in fcs:
        envf = _cwt_env(x, fc, fs)
        env = envf[lo:hi]
        if env.size == 0:
            env = envf
        k = int(np.argmax(env))
        kg = lo + k
        tag = int(fc / 1000)
        peak_amp[fc] = float(env[k])
        peak_t[fc] = kg / fs * 1e6
        f[f"cwt_peakamp{tag}"] = float(env[k])
        f[f"cwt_fwhm{tag}_us"] = _fwhm(env, k) / fs * 1e6
        f[f"cwt_rise{tag}_us"] = _rise(env, k) / fs * 1e6
        f[f"cwt_decay{tag}_db_us"] = _decay_db_us(envf, kg, fs, min(150.0, (hi - kg) / fs * 1e6))
        f[f"cwt_envkurt{tag}"] = float(_kurt(env)) if env.size > 3 else 0.0
        envs[fc] = (envf, kg)
    for fc in fcs:
        f[f"cwt_peaktime{int(fc / 1000)}_us"] = peak_t[fc]
    f["cwt_peak_tilt"] = float(np.log((peak_amp[fcs[-1]] + eps) / (peak_amp[fcs[0]] + eps)))
    f["cwt_centroid_khz"] = float(sum(fc * peak_amp[fc] for fc in fcs) /
                                  (sum(peak_amp.values()) + eps) / 1e3)
    f["dP_50_40"] = float(np.log((peak_amp[50_000.0] + eps) / (peak_amp[40_000.0] + eps)))
    f["disp_t60_t40_us"] = peak_t[60_000.0] - peak_t[40_000.0]
    f["disp_curvature_us"] = peak_t[50_000.0] - 0.5 * (peak_t[40_000.0] + peak_t[60_000.0])

    # ---------- tail + modal ring-down on EACH carrier's CWT envelope (within window) ----------
    # Read after the echo peak and clipped to the ROI on both ends -> the near ECHO
    # ring-down (the object's decay), not the pre-echo transmit ring-down.
    for fc in fcs:
        envf, kg = envs[fc]
        f.update(_tail_modal(envf, kg, fs, lo, min(len(envf), hi), fc))
    return f


# =============================================================== per-trial driver
def extract_for_trial(
    trial_signals: dict[str, np.ndarray],
    window: tuple[int, int] | None = None,
    fs: float = FS_DEFAULT,
    carriers=CARRIERS,
    sparsity_wavelets: Sequence[str] = SPARSITY_WAVELETS,
    mics: Sequence[str] = ("Mic1", "Mic2", "Mic3"),
    peaks: dict[str, int] | None = None,
    before: int = WIN_BEFORE,
    after: int = WIN_AFTER,
) -> "OrderedDict[str, float]":
    """Run the per-waveform extractor on each mic and add the within-trial cross-mic
    offsets. Cross-mic terms use the Morlet CWT (wavelet-independent)."""
    out = OrderedDict()
    per = {}
    for m in mics:
        if m not in trial_signals:
            continue
        pk = None if peaks is None else peaks.get(m)
        feats = extract_features(trial_signals[m], fs=fs, window=window, carriers=carriers,
                                 sparsity_wavelets=sparsity_wavelets, peak=pk, before=before, after=after)
        per[m] = feats
        for k, v in feats.items():
            out[f"{m}_{k}"] = v
    if "Mic1" in per and "Mic2" in per:
        for fc in sorted(carriers):
            tag = int(fc / 1000)
            out[f"dpeak_M1M2_{tag}_us"] = per["Mic1"][f"cwt_peaktime{tag}_us"] - \
                                          per["Mic2"][f"cwt_peaktime{tag}_us"]
            out[f"dPmic_M1M2_{tag}"] = float(np.log((per["Mic1"][f"cwt_peakamp{tag}"] + 1e-12) /
                                                    (per["Mic2"][f"cwt_peakamp{tag}"] + 1e-12)))
        out["dcent_M1M2"] = per["Mic1"]["cwt_centroid_khz"] - per["Mic2"]["cwt_centroid_khz"]
        out["signed_amp_ratio"] = out["dPmic_M1M2_40"] - out["dPmic_M1M2_60"]
    return out


if __name__ == "__main__":
    import pickle
    with open("data_full/Multifrequenz/Box/75cm_0Grad.pickle", "rb") as fh:
        df = pickle.load(fh)
    x = np.asarray(df["Mic2"].iloc[0], dtype=float)
    feats = extract_features(x, window=(8624, 9648))
    print(f"{len(feats)} features per (Mic, trial); sparsity under {len(SPARSITY_WAVELETS)} wavelets, rest db6:")
    for k, v in feats.items():
        print(f"  {k:24s} {v:+.4f}")
