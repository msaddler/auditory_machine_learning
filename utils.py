import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.signal


def rms(x):
    """
    Returns root mean square amplitude of x (raises ValueError if NaN).
    """
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError("rms calculation resulted in NaN")
    return out


def get_dbspl(x, mean_subtract=True):
    """
    Returns sound pressure level of x in dB re 20e-6 Pa (dB SPL).
    """
    if mean_subtract:
        x = x - np.mean(x)
    out = 20 * np.log10(rms(x) / 20e-6)
    return out


def set_dbspl(x, dbspl, mean_subtract=True):
    """
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa.
    """
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dbspl / 20)
    return rms_out * x / rms(x)


def logspace(start, stop, num):
    """
    Create an array of numbers uniformly spaced on a log scale.
    """
    return np.exp(np.linspace(np.log(start), np.log(stop), num=num))


def freq2erb(freq):
    """
    Convert frequency in Hz to ERB-number. Same as `freqtoerb.m` in the AMT.
    """
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb2freq(erb):
    """
    Convert ERB-number to frequency in Hz. Same as `erbtofreq.m` in the AMT.
    """
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(start, stop, num):
    """
    Create an array of frequencies in Hz evenly spaced on a ERB-number scale.
    Same as `erbspace.m` in the AMT.

    Args
    ----
    start (float): minimum frequency in Hz
    stop (float): maximum frequency Hz
    num (int): number of frequencies (length of array)

    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    """
    return erb2freq(np.linspace(freq2erb(start), freq2erb(stop), num=num))


def periodogram(x, sr, db=True, p_ref=20e-6, scaling="spectrum", **kwargs):
    """
    Compute power spectrum (default) or power spectral density of signal.

    Args
    ----
    x (np.ndarray): input waveform (Pa)
    sr (int): sampling rate (Hz)
    db (bool): convert output to dB
    p_ref (float): reference pressure for dB conversion (20e-6 Pa for dB SPL)
    scaling (str): "spectrum" (units Pa^2) or "density" (units Pa^2 / Hz)
    kwargs (keyword arguments): passed directly to scipy.signal.periodogram

    Returns
    -------
    fxx (np.ndarray): frequency vector (Hz)
    pxx (np.ndarray): Power spectrum (dB) or power spectral density (dB / Hz)
    """
    fxx, pxx = scipy.signal.periodogram(x=x, fs=sr, scaling=scaling, **kwargs)
    if db:
        p_ref = 1.0 if p_ref is None else p_ref
        pxx = 10.0 * np.log10(pxx / np.square(p_ref))
    return fxx, pxx


def complex_tone(
    sr=50e3,
    dur=50e-3,
    freq=220,
    phase=0,
    amplitude=1,
):
    """
    Generate a tone with the specified parameters.
    """
    freq = np.asarray(freq).reshape([-1])
    phase = np.asarray(phase).reshape([-1])
    amplitude = np.asarray(amplitude).reshape([-1])
    assert freq.max() < sr / 2
    assert freq.shape == phase.shape
    assert freq.shape == amplitude.shape
    t = np.arange(0, dur, 1 / sr)
    x = np.zeros_like(t)
    for f, p, a in zip(freq, phase, amplitude):
        x += a * np.sin(2 * np.pi * f * t + p)
    return x


def harmonic_complex_tone(
    sr=50e3,
    dur=50e-3,
    f0=220,
    harmonics=np.arange(1, 11),
    phase="sine",
    amplitudes=None,
):
    """
    Generate a harmonic complex tone with the specified parameters.
    """
    freq = f0 * np.asarray(harmonics).reshape([-1])
    if isinstance(phase, str):
        if phase.lower() == "sine":
            phase = np.zeros_like(freq)
        elif phase.lower() == "cosine":
            phase = (np.pi / 2) * np.ones_like(freq)
        elif phase.lower() == "rand":
            phase = 2 * np.pi * np.random.randn(*freq.shape)
        elif phase.lower() in ["alt", "alternating"]:
            phase = (np.pi / 2) * np.ones_like(freq)
            phase[::2] = 0
        elif phase.lower() in ["sch", "schroeder"]:
            phase = (np.pi / 2) + (np.pi * np.square(freq) / len(freq))
        else:
            raise ValueError(f"unrecognized {phase=}")
    if amplitudes is None:
        amplitude = np.ones_like(freq)
    else:
        amplitude = np.ones_like(freq) * np.asarray(amplitudes)
    x = complex_tone(
        sr=sr,
        dur=dur,
        freq=freq,
        phase=phase,
        amplitude=amplitude,
    )
    return x


def format_axes(
    ax,
    str_title=None,
    str_xlabel=None,
    str_ylabel=None,
    fontsize_title=12,
    fontsize_labels=12,
    fontsize_ticks=12,
    fontweight_title=None,
    fontweight_labels=None,
    xscale="linear",
    yscale="linear",
    xlimits=None,
    ylimits=None,
    xticks=None,
    yticks=None,
    xticks_minor=None,
    yticks_minor=None,
    xticklabels=None,
    yticklabels=None,
    spines_to_hide=[],
    major_tick_params_kwargs_update={},
    minor_tick_params_kwargs_update={},
):
    """
    Helper function for setting axes-related formatting parameters.
    """
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)

    major_tick_params_kwargs = {
        "axis": "both",
        "which": "major",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 2,
        "direction": "out",
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)

    minor_tick_params_kwargs = {
        "axis": "both",
        "which": "minor",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 4,
        "direction": "out",
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)

    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)

    return ax


def make_periodogram_plot(x, sr, figsize=None, **kwargs):
    """
    Plot power spectrum of input signal(s).
    """
    fig, ax = plt.subplots(figsize=figsize)
    fxx, pxx = periodogram(x, sr=sr)
    if pxx.ndim == 2:
        pxx = pxx.T
    ax.plot(fxx, pxx)
    kwargs_format_axes = {
        "str_xlabel": "Frequency (Hz)",
        "str_ylabel": "Power (dB SPL)",
        "xscale": "log",
        "yscale": "linear",
        "xlimits": [10, sr / 2],
        "ylimits": [-60, None],
    }
    kwargs_format_axes.update(kwargs)
    ax = format_axes(ax, **kwargs_format_axes)
    return fig, ax


def make_spectrogram_plot(
    x,
    sr,
    nfft=1024,
    noverlap=None,
    mode="default",
    scale="default",
    vmin=None,
    vmax=None,
    cmap="magma",
    figsize=None,
    str_title=None,
    **kwargs,
):
    """
    Plot spectrogram of input signal alongside time-aligned waveform.
    """
    fig, ax_arr = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={
            "hspace": 0,
            "height_ratios": [1, 5],
        },
        layout="constrained",
    )
    t = np.arange(0, len(x)) / sr
    ax_arr[0].plot(t, x, color="k", lw=1)
    kwargs_format_axes = {
        "str_title": str_title,
        "xticks": [],
        "yticks": [],
        "ylimits": [
            np.min(x) - (np.max(x) - np.min(x)) / 5,
            np.max(x) + (np.max(x) - np.min(x)) / 5,
        ],
        "major_tick_params_kwargs_update": {"length": 0},
    }
    ax_arr[0] = format_axes(ax_arr[0], **kwargs_format_axes)
    ax_arr[0].set_ylabel("Signal", rotation=0, ha="right")
    spectrum, freqs, t, im = ax_arr[1].specgram(
        x,
        Fs=sr,
        NFFT=nfft,
        mode=mode,
        noverlap=nfft - 1 if noverlap is None else noverlap,
        scale=scale,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    fig.colorbar(im, label="Intensity (dB)", pad=0.025)
    kwargs_format_axes = {
        "str_xlabel": "Time (s)",
        "str_ylabel": "Frequency (Hz)",
        "xlimits": [t[0], t[-1]],
        "ylimits": [freqs[0], freqs[-1]],
    }
    kwargs_format_axes.update(kwargs)
    ax_arr[1] = format_axes(ax_arr[1], **kwargs_format_axes)
    return fig, ax_arr


def plot_nervegram(
    ax,
    nervegram,
    sr=20000,
    cfs=None,
    cmap="gray",
    cbar_on=False,
    fontsize_labels=12,
    fontsize_ticks=12,
    fontweight_labels=None,
    nxticks=6,
    nyticks=5,
    tmin=None,
    tmax=None,
    treset=True,
    vmin=None,
    vmax=None,
    interpolation="none",
    vticks=None,
    str_clabel=None,
    **kwargs_format_axes,
):
    """
    Plot simulated auditory nerve representation on the provided axes.
    """
    # Trim nervegram if tmin and tmax are specified
    nervegram = np.squeeze(nervegram)
    assert len(nervegram.shape) == 2, "nervegram must be freq-by-time array"
    t = np.arange(0, nervegram.shape[1]) / sr
    if (tmin is not None) and (tmax is not None):
        t_IDX = np.logical_and(t >= tmin, t < tmax)
        t = t[t_IDX]
        nervegram = nervegram[:, t_IDX]
    if treset:
        t = t - t[0]
    # Setup time and frequency ticks and labels
    time_idx = np.linspace(0, t.shape[0] - 1, nxticks, dtype=int)
    time_labels = ["{:.0f}".format(1e3 * t[itr0]) for itr0 in time_idx]
    if cfs is None:
        cfs = np.arange(0, nervegram.shape[0])
    else:
        cfs = np.array(cfs)
        msg = f"{cfs.shape[0]=} must match {nervegram.shape[0]=}"
        assert cfs.shape[0] == nervegram.shape[0], msg
    freq_idx = np.linspace(0, cfs.shape[0] - 1, nyticks, dtype=int)
    freq_labels = ["{:.0f}".format(cfs[itr0]) for itr0 in freq_idx]
    # Display nervegram image
    im_nervegram = ax.imshow(
        nervegram,
        origin="lower",
        aspect="auto",
        extent=[0, nervegram.shape[1], 0, nervegram.shape[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    # Add colorbar if `cbar_on == True`
    if cbar_on:
        cbar = plt.colorbar(im_nervegram, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(
            str_clabel, fontsize=fontsize_labels, fontweight=fontweight_labels
        )
        if vticks is not None:
            cbar.set_ticks(vticks)
        else:
            cbar.ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nyticks, integer=True)
            )
        cbar.ax.tick_params(
            direction="out",
            axis="both",
            which="both",
            labelsize=fontsize_ticks,
            length=fontsize_ticks / 2,
        )
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%03d"))
    # Format axes
    ax = format_axes(
        ax,
        xticks=time_idx,
        yticks=freq_idx,
        xticklabels=time_labels,
        yticklabels=freq_labels,
        fontsize_labels=fontsize_labels,
        fontsize_ticks=fontsize_ticks,
        fontweight_labels=fontweight_labels,
        **kwargs_format_axes,
    )
    return ax


def make_nervegram_plot(
    waveform=None,
    nervegram=None,
    sr_waveform=None,
    sr_nervegram=None,
    cfs=None,
    tmin=None,
    tmax=None,
    treset=True,
    vmin=None,
    vmax=None,
    figsize=(9, 5),
    ax_idx_waveform=1,
    ax_idx_spectrum=3,
    ax_idx_nervegram=4,
    ax_idx_excitation=5,
    interpolation="none",
    erb_freq_axis=True,
    nxticks=6,
    nyticks=6,
    kwargs_plot={"color": "k", "lw": 1},
    limits_buffer=0.1,
    **kwargs_format_axes,
):
    """
    Plot simulated auditory nerve representation alongside sound waveform,
    stimulus power spectrum, and time-averaged excitation pattern.
    """
    fig, ax_arr = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=figsize,
        gridspec_kw={
            "wspace": 0.15,
            "hspace": 0.15,
            "width_ratios": [1, 6, 1],
            "height_ratios": [1, 4],
        },
    )
    ax_arr = np.array([ax_arr]).reshape([-1])
    ax_idx_list = []

    # Plot stimulus waveform
    if ax_idx_waveform is not None:
        ax_idx_list.append(ax_idx_waveform)
        y_wav = np.squeeze(waveform)
        assert len(y_wav.shape) == 1, "waveform must be 1D array"
        x_wav = np.arange(0, y_wav.shape[0]) / sr_waveform
        if (tmin is not None) and (tmax is not None):
            IDX = np.logical_and(x_wav >= tmin, x_wav < tmax)
            x_wav = x_wav[IDX]
            y_wav = y_wav[IDX]
        if treset:
            x_wav = x_wav - x_wav[0]
        xlimits_wav = [x_wav[0], x_wav[-1]]
        ylimits_wav = [-np.max(np.abs(y_wav)), np.max(np.abs(y_wav))]
        ylimits_wav = np.array(ylimits_wav) * (1 + limits_buffer)
        ax_arr[ax_idx_waveform].plot(x_wav, y_wav, **kwargs_plot)
        ax_arr[ax_idx_waveform] = format_axes(
            ax_arr[ax_idx_waveform],
            xlimits=xlimits_wav,
            ylimits=ylimits_wav,
            xticks=[],
            yticks=[],
            xticklabels=[],
            yticklabels=[],
            **kwargs_format_axes,
        )

    # Plot stimulus power spectrum
    if ax_idx_spectrum is not None:
        ax_idx_list.append(ax_idx_spectrum)
        fxx, pxx = periodogram(waveform, sr_waveform)
        if cfs is not None:
            msg = "Frequency axes will not align when highest CF exceeds Nyquist"
            assert np.max(cfs) <= np.max(fxx), msg
            IDX = np.logical_and(fxx >= np.min(cfs), fxx <= np.max(cfs))
            pxx = pxx[IDX]
            fxx = fxx[IDX]
        xlimits_pxx = [np.max(pxx) * (1 + limits_buffer), 0]  # Reverses x-axis
        xlimits_pxx = np.ceil(np.array(xlimits_pxx) * 5) / 5
        if erb_freq_axis:
            fxx = freq2erb(fxx)
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ["{:.0f}".format(yt) for yt in erb2freq(yticks)]
        else:
            ylimits_fxx = [np.min(fxx), np.max(fxx)]
            yticks = np.linspace(ylimits_fxx[0], ylimits_fxx[-1], nyticks)
            yticklabels = ["{:.0f}".format(yt) for yt in yticks]
        ax_arr[ax_idx_spectrum].plot(pxx, fxx, **kwargs_plot)
        ax_arr[ax_idx_spectrum] = format_axes(
            ax_arr[ax_idx_spectrum],
            str_xlabel="Power\n(dB SPL)",
            str_ylabel="Frequency (Hz)",
            xlimits=xlimits_pxx,
            ylimits=ylimits_fxx,
            xticks=xlimits_pxx,
            yticks=yticks,
            xticklabels=xlimits_pxx.astype(int),
            yticklabels=yticklabels,
            **kwargs_format_axes,
        )

    # Plot stimulus nervegram
    if ax_idx_nervegram is not None:
        ax_idx_list.append(ax_idx_nervegram)
        if ax_idx_spectrum is not None:
            nervegram_nxticks = nxticks
            nervegram_nyticks = 0
            nervegram_str_xlabel = "Time\n(ms)"
            nervegram_str_ylabel = None
        else:
            nervegram_nxticks = nxticks
            nervegram_nyticks = nyticks
            nervegram_str_xlabel = "Time (ms)"
            nervegram_str_ylabel = "Characteristic frequency (Hz)"
        plot_nervegram(
            ax_arr[ax_idx_nervegram],
            nervegram,
            sr=sr_nervegram,
            cfs=cfs,
            nxticks=nervegram_nxticks,
            nyticks=nervegram_nyticks,
            tmin=tmin,
            tmax=tmax,
            treset=treset,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            str_xlabel=nervegram_str_xlabel,
            str_ylabel=nervegram_str_ylabel,
        )

    # Plot time-averaged excitation pattern
    if ax_idx_excitation is not None:
        ax_idx_list.append(ax_idx_excitation)
        x_exc = np.mean(nervegram, axis=1)
        xlimits_exc = np.array([0, np.max(x_exc) * (1 + limits_buffer)])
        y_exc = np.arange(0, nervegram.shape[0])
        ylimits_exc = [np.min(y_exc), np.max(y_exc)]
        ax_arr[ax_idx_excitation].plot(x_exc, y_exc, **kwargs_plot)
        ax_arr[ax_idx_excitation] = format_axes(
            ax_arr[ax_idx_excitation],
            str_xlabel="Excitation\n(spikes/s)",
            xlimits=xlimits_exc,
            ylimits=ylimits_exc,
            xticks=xlimits_exc,
            yticks=[],
            xticklabels=np.round(xlimits_exc).astype(int),
            yticklabels=[],
            **kwargs_format_axes,
        )

    # Clear unused axes in ax_arr and align x-axis labels
    for ax_idx in range(ax_arr.shape[0]):
        if ax_idx not in ax_idx_list:
            ax_arr[ax_idx].axis("off")
    fig.align_xlabels(ax_arr)
    return fig, ax_arr


def get_example_audiogram(severity="ref"):
    """
    Return a list of frequencies in Hz (`freq`) and hearing
    thresholds in dB HL (`dbhl`) representing an audiogram.
    Example audiograms sourced from the Clarity Challenge
    (https://github.com/claritychallenge/clarity).
    """
    freq = [
        125,
        250,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        4000,
        6000,
        8000,
        10000,
        12000,
        14000,
        16000,
    ]
    dict_dbhl = {
        "ref": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "mild": [5, 10, 15, 18, 19, 22, 25, 28, 31, 35, 38, 40, 40, 45, 50],
        "moderate": [
            15.0,
            20.0,
            20.0,
            22.5,
            25.0,
            30.0,
            35.0,
            40.0,
            45.0,
            50.0,
            55.0,
            55.0,
            60.0,
            65.0,
            65.0,
        ],
        "moderate_severe": [19, 19, 28, 35, 40, 47, 52, 56, 58, 58, 63, 70, 75, 80, 80],
    }
    severity = severity.lower()
    msg = f"{severity=} must be one of {list(dict_dbhl.keys())}"
    assert severity in dict_dbhl.keys(), msg
    dbhl = dict_dbhl[severity]
    return np.asarray(freq), np.asarray(dbhl)


def map_audiogram_to_rate_level_parameters(
    freq=None,
    dbhl=None,
    cfs=None,
    healthy_threshold=0.0,
    healthy_dynamic_range=80.0,
):
    """
    Maps an audiogram (specified by `freq` and `dbhl` to
    sigmoid rate-level function parameters in the simplest
    possible way. Elevated hearing thresholds are added to
    the `healthy_threshold` and subtracted from the
    `healthy_dynamic_range`. If a list of characteristic
    frequencies (`cfs`) is provided, this function returns
    thresholds and dynamic ranges linearly interpolated at
    those `cfs`. If not, the function returns thresholds
    and dynamic ranges at the audiogram frequencies.
    """
    dbhl = np.asarray(dbhl).reshape([-1])
    threshold_at_freq = np.clip(
        a=healthy_threshold + dbhl,
        a_min=healthy_threshold,
        a_max=healthy_threshold + healthy_dynamic_range,
    )
    dynamic_range_at_freq = np.clip(
        a=healthy_dynamic_range - dbhl,
        a_min=0,
        a_max=healthy_dynamic_range,
    )
    if cfs is None:
        return threshold_at_freq, dynamic_range_at_freq
    threshold_at_cfs = np.interp(
        x=cfs,
        xp=freq,
        fp=threshold_at_freq,
    )
    dynamic_range_at_cfs = np.interp(
        x=cfs,
        xp=freq,
        fp=dynamic_range_at_freq,
    )
    return threshold_at_cfs, dynamic_range_at_cfs
