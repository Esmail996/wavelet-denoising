from scipy.signal import butter, filtfilt, firwin, sosfiltfilt


def iir_bandpass(x, fs, f0, bw, order=1):
    """Butterworth IIR bandpass with zero-phase filtering."""
    low = (f0 - bw) / (fs / 2)
    high = (f0 + bw) / (fs / 2)
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid IIR passband: f0={f0}, bw={bw}, fs={fs}")
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def fir_bandpass(x, fs, f0, bw, numtaps=71, window="hamming"):
    """FIR bandpass with linear-phase taps and zero-phase application."""
    if numtaps < 3:
        raise ValueError("numtaps must be >= 3")
    low = f0 - bw
    high = f0 + bw
    if not (0 < low < high < fs / 2):
        raise ValueError(f"Invalid FIR passband: f0={f0}, bw={bw}, fs={fs}")
    b = firwin(numtaps, [low, high], pass_zero=False, fs=fs, window=window)
    return filtfilt(b, [1.0], x)


def bandpass(x, fs, f0, bw, order=1, method="iir", numtaps=71, window="hamming"):
    """Bandpass wrapper supporting both IIR and FIR methods.

    Parameters
    ----------
    bw : float
        Half-bandwidth around center frequency f0.
    method : str
        "iir" (Butterworth) or "fir".
    """
    method_l = str(method).lower()
    if method_l == "iir":
        return iir_bandpass(x, fs=fs, f0=f0, bw=bw, order=order)
    if method_l == "fir":
        return fir_bandpass(x, fs=fs, f0=f0, bw=bw, numtaps=numtaps, window=window)
    raise ValueError(f"Unsupported bandpass method '{method}'. Use 'iir' or 'fir'.")
