import math

import numpy as np
import scipy.signal
import soxr


def gammatone_filterbank_fir(
    sr,
    cfs,
    fir_dur=0.05,
    order=4,
    bw_mult=None,
):
    """
    Returns finite impulse responses of a Gammatone filter bank.

    Args
    ----
    sr (float): Sampling rate in Hz
    fir_dur (float): Duration of FIR in seconds
    cfs (float or np.ndarray): Center frequencies with shape (n_filters,)
    order (int): Filter order
    bw_mult (float or np.ndarray or None): Bandwidth scaling factor

    Returns
    -------
    fir (np.ndarray): impulse responses with shape (n_filters, int(sr * fir_dur))
    """
    cfs = np.array(cfs).reshape([-1])
    if bw_mult is None:
        bw_mult = np.divide(
            math.factorial(order - 1) ** 2,
            (np.pi * math.factorial(2 * order - 2) * 2 ** (-2 * order + 2)),
        )
    else:
        bw_mult = np.array(bw_mult)
    bw = 2 * np.pi * bw_mult * (24.7 + cfs / 9.265)
    wc = 2 * np.pi * cfs
    t = np.arange(0, fir_dur, 1 / sr)
    a = (
        2
        / math.factorial(order - 1)
        / np.abs(1 / bw**order + 1 / (bw + 2j * wc) ** order)
        / sr
    )
    fir = (
        a[:, None]
        * t ** (order - 1)
        * np.exp(-bw[:, None] * t[None, :])
        * np.cos(wc[:, None] * t[None, :])
    )
    return fir


def ihc_lowpass_filter_fir(sr, fir_dur, cutoff=3e3, order=7):
    """
    Returns finite impulse response of IHC lowpass filter from
    bez2018model/model_IHC_BEZ2018.c
    """
    sr_bez2018 = 100e3

    def resample(x):
        return soxr.resample(
            x,
            in_rate=sr_bez2018,
            out_rate=sr,
            quality="VHQ",
        )

    n_taps = int(sr_bez2018 * fir_dur)
    while len(resample(np.zeros(n_taps))) % 2 == 0:
        n_taps = n_taps + 1
    impulse = np.zeros(n_taps)
    impulse[0] = 1
    fir = np.zeros(n_taps)
    ihc = np.zeros(order + 1)
    ihcl = np.zeros(order + 1)
    c1LP = (sr_bez2018 - 2 * np.pi * cutoff) / (sr_bez2018 + 2 * np.pi * cutoff)
    c2LP = (np.pi * cutoff) / (sr_bez2018 + 2 * np.pi * cutoff)
    for n in range(n_taps):
        ihc[0] = impulse[n]
        for i in range(order):
            ihc[i + 1] = (c1LP * ihcl[i + 1]) + c2LP * (ihc[i] + ihcl[i])
        ihcl = ihc
        fir[n] = ihc[order]
    fir = fir * scipy.signal.windows.hann(n_taps)
    fir = resample(fir)
    fir[fir < 0] = 0
    fir = fir / fir.sum()
    return fir
