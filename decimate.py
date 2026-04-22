import numpy as np
from scipy.signal import resample_poly

def decimate_signal(x, fs, q):
    """
    Anti-aliased integer-factor decimation using polyphase filtering.

    Parameters
    ----------
    x : 1D array Input signal.
    fs : float Original sampling rate [Hz].
    q : int Decimation factor (q >= 1).

    Returns
    -------
    x_ds : 1D array Decimated signal.
    fs_ds : float New sampling rate fs/q.
    """
    x = np.asarray(x, dtype=float)
    q = int(q)
    if q <= 1:
        return x, float(fs)
    
    x_ds = resample_poly(x, up=1, down=q)
    fs_ds = float(fs) / q
    return x_ds, fs_ds