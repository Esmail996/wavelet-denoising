#!/usr/bin/env python3
"""
make_feature_figures.py
=======================
Regenerates the four figures used in the Feature Extraction chapter:

  1. fig_feature_decomposition  - one echo shown as waveform + WPT packet energy
                                   + CWT carrier envelopes (the two-transform design)
  2. fig_swt_collapse           - SWT detail energy collapses into one octave after
                                   a +/-3 kHz band-pass (justifies dropping the SWT)
  3. fig_object_discriminators  - what the features capture: envelope shape,
                                   wpt_centroid (Box), cwt_echo2_lag (Dose/Glas)
  4. fig_leakage_screen         - echo vs echo-free Cohen's d for every feature

Each figure is written as both .pdf (for the thesis) and .png (for preview).

Requirements
------------
    pip install numpy pandas pywavelets scipy matplotlib
    features_family13.py must be importable (it provides the merged extractor,
    rank_features_cohensd, cohens_d and LEAKAGE_PRONE_FEATURES used by figs 3-4).

Dataset layout (set DATA_DIR below)
-----------------------------------
    DATA_DIR/<Class>/<dist>cm_<angle>Grad.pickle
where <Class> in {Box, Dose, Glas}, <dist> in {25,50,75,100,125},
<angle> in {-10,-5,0,5,10}. Each pickle unpickles to a pandas DataFrame with
columns Mic1, Mic2, Mic3; each row i is one trial, a length-18000 waveform at
fs = 2 MHz.

Run
---
    python make_feature_figures.py            # all four
    python make_feature_figures.py 1 3        # only figs 1 and 3
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pywt
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import butter, sosfiltfilt

import features_family13 as F   # the merged feature module (figs 3-4)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_DIR = "data_full/Multifrequenz"        # <-- point at your dataset root
FS = 2_000_000.0
CARRIERS = [40e3, 50e3, 60e3]
# per-distance search windows (samples) from the STFT analysis (Table 4.1)
WIN = {25: (2610, 3498), 50: (5461, 6184), 75: (8437, 9120),
       100: (11298, 12123), 125: (14063, 15135)}
NULL = (600, 1488)                          # echo-free pre-echo window
ANGLES = [-10, -5, 0, 5, 10]
COL = {"Box": "#c1432e", "Dose": "#2f6fb0", "Glas": "#3f9153"}
CARRIER_BAND = (39e3, 62.5e3)               # packets 5-7 at level 7
WPT_WAVELET = "sym6"
MORLET = "cmor1.5-1.0"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_cell(obj, dist, angle):
    """Return the DataFrame for one (class, distance, angle) recording."""
    return pd.read_pickle(f"{DATA_DIR}/{obj}/{dist}cm_{angle}Grad.pickle")


def coherent_average(obj, dist=75, angle=0, mic="Mic2", n=100):
    """Mean waveform over the trials of one cell (clean for illustration)."""
    df = load_cell(obj, dist, angle)
    return np.mean([np.asarray(df[mic].iloc[i], float) for i in range(n)], axis=0)


def cwt_envelope(x, fc):
    """|W| of a single-scale complex Morlet CWT centred on carrier fc."""
    coef, _ = pywt.cwt(x, [FS / fc], MORLET, sampling_period=1.0 / FS)
    return np.abs(coef[0])


def freq_ordered_paths(level):
    """Frequency-ordered wavelet-packet leaf paths (Gray-code ordering)."""
    order = sorted((n ^ (n >> 1), n) for n in range(2 ** level))
    return ["".join("a" if b == "0" else "d" for b in format(nat, f"0{level}b"))
            for _, nat in order]


def packet_energies(x, lo, hi, level=7):
    """Energy in each level-`level` packet over the sample window [lo, hi)."""
    paths = freq_ordered_paths(level)
    wp = pywt.WaveletPacket(x, WPT_WAVELET, mode="symmetric", maxlevel=level)
    d = 2 ** level
    return np.array([np.sum(np.asarray(wp[p].data)[lo // d: hi // d + 1] ** 2)
                     for p in paths])


def fwhm_bounds(env, k):
    """Left/right indices where the envelope crosses half its peak about k."""
    half = 0.5 * env[k]
    l = k
    while l > 0 and env[l] > half:
        l -= 1
    r = k
    while r < len(env) - 1 and env[r] > half:
        r += 1
    return l, r, half


def save(fig, name):
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------------------------------------
# Figure 1 - two-transform decomposition of one echo
# --------------------------------------------------------------------------
def fig_decomposition(obj="Box", dist=75, angle=0, mic="Mic2"):
    print("Figure 1: feature decomposition")
    x = coherent_average(obj, dist, angle, mic)
    lo, hi = WIN[dist]
    seg = x[lo:hi]
    t = np.arange(len(seg)) / FS * 1e6
    cb = COL[obj]

    # DC-removed packet energies over the window
    npk = 14
    Ep = packet_energies(x - x.mean(), lo, hi, level=7)[:npk]
    frac = Ep / Ep.sum()
    fpk = (np.arange(npk) + 0.5) * (FS / 2 ** 8) / 1e3   # packet centre freqs (kHz)

    fig, ax = plt.subplots(3, 1, figsize=(8.2, 8.6))

    # (a) waveform
    ax[0].plot(t, seg, color="#333", lw=0.7)
    ax[0].set_title(f"(a) Denoised echo in the search window "
                    f"({obj}, {dist} cm, {mic})", loc="left", fontsize=10)
    ax[0].set_xlabel("time in window (\u00b5s)"); ax[0].set_ylabel("amplitude (ADC)")
    ax[0].margins(x=0.01)

    # (b) WPT packet energy (log)
    bars = ax[1].bar(fpk, frac, width=(FS / 2 ** 8) / 1e3 * 0.9,
                     color="#cfcfcf", edgecolor="#999", lw=0.4, log=True)
    for i in (5, 6, 7):
        bars[i].set_color(cb); bars[i].set_edgecolor("k")
    bars[0].set_color("#7a5cad")        # sub-8 kHz drift
    bars[9].set_color("#d6a32f")        # noise reference packet
    ax[1].axvspan(CARRIER_BAND[0] / 1e3, CARRIER_BAND[1] / 1e3, color=cb, alpha=0.08)
    for fc in CARRIERS:
        ax[1].axvline(fc / 1e3, color="k", ls=":", lw=0.6)
    ax[1].set_ylim(max(frac.min() * 0.5, 1e-4), 1.5)
    ax[1].set_title("(b) WPT level-7 packet energy, DC removed "
                    "(\u0394f = 7.8 kHz); carriers isolated in packets 5,6,7",
                    loc="left", fontsize=10)
    ax[1].set_xlabel("packet centre frequency (kHz)")
    ax[1].set_ylabel("energy fraction (log)")
    ax[1].legend(handles=[Patch(fc=cb, label="carriers (read)"),
                          Patch(fc="#7a5cad", label="sub-8 kHz drift (excluded)"),
                          Patch(fc="#d6a32f", label="noise ref, pkt 9"),
                          Patch(fc="#cfcfcf", label="off-carrier / artefact (excluded)")],
                 fontsize=8, loc="upper right", ncol=2)

    # (c) CWT carrier envelopes, 50 kHz annotated
    for fc, c in zip(CARRIERS, ["#9a9a9a", cb, "#555"]):
        ax[2].plot(t, cwt_envelope(x, fc)[lo:hi], color=c, lw=1.3,
                   label=f"{int(fc/1e3)} kHz")
    e = cwt_envelope(x, 50e3)[lo:hi]
    k = int(np.argmax(e)); pk = e[k]
    l, r, half = fwhm_bounds(e, k)
    ax[2].plot(t[k], pk, "o", color="k", ms=5)
    ax[2].annotate("peak", (t[k], pk), textcoords="offset points", xytext=(6, -2), fontsize=8)
    ax[2].hlines(half, t[l], t[r], color="k", lw=1.0, ls="--")
    ax[2].text(t[r] + 4, half, "FWHM", va="center", fontsize=8)
    ax[2].set_title("(c) Single-scale Morlet CWT envelope at each carrier "
                    "(50 kHz: peak and FWHM marked)", loc="left", fontsize=10)
    ax[2].set_xlabel("time in window (\u00b5s)"); ax[2].set_ylabel("|W| envelope")
    ax[2].legend(fontsize=8, loc="upper left"); ax[2].margins(x=0.01)

    fig.tight_layout()
    save(fig, "fig_feature_decomposition")


# --------------------------------------------------------------------------
# Figure 2 - SWT energy collapse after a band-pass
# --------------------------------------------------------------------------
def fig_swt_collapse(dist=75, mic="Mic2"):
    print("Figure 2: SWT energy collapse")
    lo, hi = WIN[dist]
    sos = butter(4, [47e3, 53e3], btype="band", fs=FS, output="sos")
    shares = []
    for obj in ["Box", "Dose", "Glas"]:
        x = coherent_average(obj, dist, 0, mic)
        xb = sosfiltfilt(sos, x - x.mean())[lo:hi]
        nn = len(xb)
        pad = ((nn + 31) // 32) * 32
        xp = np.zeros(pad); xp[:nn] = xb
        coeffs = pywt.swt(xp, WPT_WAVELET, level=5, trim_approx=False, norm=False)
        e = np.array([np.sum(cD[:nn] ** 2) for _, cD in coeffs])   # order L5..L1
        shares.append(e / e.sum())
    sh = np.array(shares).mean(axis=0)

    lvls = [5, 4, 3, 2, 1]
    bands = [f"{FS/2**(l+1)/1e3:.0f}\u2013{FS/2**l/1e3:.0f}" for l in lvls]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(range(5), sh * 100, color=["#c1432e"] + ["#bbb"] * 4, edgecolor="k", lw=0.4)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"L{l}\n{b} kHz" for l, b in zip(lvls, bands)], fontsize=8)
    for i, v in enumerate(sh):
        ax.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=8)
    ax.set_title("SWT detail energy after a \u00b13 kHz band-pass collapses into one octave",
                 fontsize=10)
    ax.set_ylabel("share of detail energy (%)"); ax.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, "fig_swt_collapse")


# --------------------------------------------------------------------------
# Per-trial feature table (used by figs 3 and 4)
# --------------------------------------------------------------------------
def feature_table(distances, mic="Mic2", trial_step=10, with_null=False):
    """Run the merged extractor over the given distances x angles x trials."""
    rows = []
    for obj in ["Box", "Dose", "Glas"]:
        for dist in distances:
            for a in ANGLES:
                df = load_cell(obj, dist, a)
                for tr in range(0, 100, trial_step):
                    x = np.asarray(df[mic].iloc[tr], float)
                    fe = F.extract_family13_features(x, window=WIN[dist])
                    fe.update(object=obj, distance=dist, reg="echo"); rows.append(fe)
                    if with_null:
                        fn = F.extract_family13_features(x, window=NULL)
                        fn.update(object=obj, distance=dist, reg="null"); rows.append(fn)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure 3 - object discriminators
# --------------------------------------------------------------------------
def fig_object_discriminators(dist=75, mic="Mic2"):
    print("Figure 3: object discriminators")
    E = feature_table([dist], mic=mic, trial_step=10)
    lo, hi = WIN[dist]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    # (a) envelope by object
    for obj in ["Box", "Dose", "Glas"]:
        e = cwt_envelope(coherent_average(obj, dist, 0, mic), 50e3)[lo:hi]
        e = e / e.max()
        ax[0].plot(np.arange(len(e)) / FS * 1e6, e, color=COL[obj], lw=1.3, label=obj)
    ax[0].set_title(f"(a) 50 kHz echo envelope by object\n({dist} cm, 0\u00b0, {mic})",
                    fontsize=10)
    ax[0].set_xlabel("time in window (\u00b5s)"); ax[0].set_ylabel("normalised |W|")
    ax[0].legend(fontsize=9)
    # (b),(c) discriminator distributions
    panels = [("wpt_centroid_khz",
               "(b) wpt_centroid: separates Box\nfrom the hard reflectors (spectral tilt)",
               "centroid (kHz)"),
              ("cwt_echo2_lag_us",
               "(c) cwt_echo2_lag: separates Dose/Glas,\nthe hard pair (secondary-echo geometry)",
               "secondary-echo lag (\u00b5s)")]
    for i, (feat, ttl, yl) in enumerate(panels, start=1):
        for j, obj in enumerate(["Box", "Dose", "Glas"]):
            v = E[E.object == obj][feat]
            ax[i].scatter(np.random.normal(j, 0.08, len(v)), v, s=14,
                          color=COL[obj], alpha=0.6, edgecolor="none")
            ax[i].plot([j - 0.25, j + 0.25], [v.median()] * 2, color="k", lw=1.5)
        ax[i].set_xticks([0, 1, 2]); ax[i].set_xticklabels(["Box", "Dose", "Glas"])
        ax[i].set_title(ttl, fontsize=10); ax[i].set_ylabel(yl)
    fig.tight_layout()
    save(fig, "fig_object_discriminators")


# --------------------------------------------------------------------------
# Figure 4 - leakage screen
# --------------------------------------------------------------------------
def fig_leakage_screen(distances=(50, 75, 100), mic="Mic2"):
    print("Figure 4: leakage screen (this one runs the full screen, ~1-2 min)")
    D = feature_table(list(distances), mic=mic, trial_step=20, with_null=True)
    E, N = D[D.reg == "echo"], D[D.reg == "null"]
    fcols = [c for c in D.columns if c not in ("object", "distance", "reg")]
    m = (F.rank_features_cohensd(E, fcols)
         .merge(F.rank_features_cohensd(N, fcols)[["feature", "mean_abs_d"]],
                on="feature", suffixes=("_echo", "_null")))
    m.to_csv("leak_table.csv", index=False)

    leak = set(F.LEAKAGE_PRONE_FEATURES)
    XL, YL = 6.5, 5.5
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.fill_between([0, XL], [0, XL], [YL, YL], color="#3f9153", alpha=0.05)   # genuine
    ax.fill_between([0, XL], [0, 0], [0, XL], color="#c1432e", alpha=0.05)     # leak
    for flagged, col, lab in [(False, "#3a6ea5", "not flagged"),
                              (True, "#c1432e", "flagged leak-prone")]:
        s = m[m.feature.isin(leak) == flagged]
        ax.scatter(s.mean_abs_d_null.clip(upper=XL - 0.05),
                   s.mean_abs_d_echo.clip(upper=YL - 0.05),
                   s=38, color=col, alpha=0.8, edgecolor="k", lw=0.3, label=lab)
    ax.plot([0, min(XL, YL)], [0, min(XL, YL)], "k--", lw=0.9)
    ax.text(4.4, 4.7, "genuine = 0\n(echo = echo-free)", fontsize=8,
            rotation=42, va="bottom", ha="center")
    ax.text(0.15, 5.15, "separates echo\n> background", fontsize=8.5,
            color="#2c6e49", weight="bold")
    ax.text(4.3, 0.25, "separates background > echo  (session leakage)",
            fontsize=8.5, color="#9e3326", weight="bold")
    note = ["cwt_rise40_us", "cwt_peakamp60", "cwt_peakamp40", "disp_t60_t40_us",
            "wpt_dominant_khz", "cwt_late_to_peak50", "wpt_snr_db", "sp_wpt_f50",
            "wpt_carrE60_log", "disp_curvature_us", "wpt_centroid_khz"]
    for _, r in m[m.feature.isin(note)].iterrows():
        ax.annotate(r.feature, (min(r.mean_abs_d_null, XL - 0.05),
                                min(r.mean_abs_d_echo, YL - 0.05)),
                    fontsize=6.6, xytext=(3, 2), textcoords="offset points")
    n_off = int(((m.mean_abs_d_null > XL) | (m.mean_abs_d_echo > YL)).sum())
    ax.set_xlim(0, XL); ax.set_ylim(0, YL)
    ax.set_xlabel("echo-free separability  $|d|$  (recording-session leakage)")
    ax.set_ylabel("echo separability  $|d|$")
    ax.set_title(f"Leakage screen on single trials ({mic}, "
                 f"{'/'.join(str(d) for d in distances)} cm). {n_off} features clipped at the\n"
                 "axis limits; most features sit on or below the line "
                 "because the session signature is in-band", fontsize=9.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_leakage_screen")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    funcs = {1: fig_decomposition, 2: fig_swt_collapse,
             3: fig_object_discriminators, 4: fig_leakage_screen}
    which = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    for k in which:
        funcs[k]()
    print("done.")
