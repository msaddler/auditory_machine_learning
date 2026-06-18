import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils


def visualize_cochlear_model(cochlear_model, example_input, str_title=None):
    """ """
    device = utils.get_device_from_module(cochlear_model)
    waveform = torch.as_tensor(example_input).to(device)
    nervegram = cochlear_model(waveform)
    fig, ax_arr = utils.make_nervegram_plot(
        waveform=waveform.detach().cpu().numpy(),
        nervegram=nervegram.detach().cpu().numpy(),
        sr_waveform=cochlear_model.sr_input,
        sr_nervegram=cochlear_model.sr_output,
        cfs=cochlear_model.cfs,
    )
    if str_title is not None:
        fig.suptitle(str_title, y=0.95)
    plt.tight_layout()
    plt.show()
    return fig, ax_arr


def visualize_cochlear_model_stages(cochlear_model):
    """ """
    fig, ax_arr = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    impulse = torch.zeros(int(cochlear_model.sr_input))
    impulse[0] = 1
    # Plot frequency response of cochlear filterbank (linear Gammatone filterbank)
    cochlear_filterbank_impulse_response = cochlear_model.cochlear_filterbank(impulse)
    fxx, pxx = utils.periodogram(
        cochlear_filterbank_impulse_response.numpy(),
        cochlear_model.sr_input,
    )
    ax_arr[0].plot(fxx, pxx[::4].T)
    ax_arr[0] = utils.format_axes(
        ax_arr[0],
        str_title="Cochlear filterbank",
        str_xlabel="Frequency (Hz)",
        str_ylabel="Power (dB)",
        xscale="log",
        xlimits=[20, cochlear_model.sr_input // 2],
        ylimits=[-40, None],
        fontsize_ticks=10,
    )
    # Plot IHC transduction function (crudely modeled as half-wave rectification)
    x = torch.linspace(-1, 1, 100)
    y = cochlear_model.half_wave_rectification(x)
    ax_arr[1].plot(x.numpy(), y.numpy())
    ax_arr[1].axvline(0, color="k", lw=0.2)
    ax_arr[1].axhline(0, color="k", lw=0.2)
    ax_arr[1].axis("square")
    ax_arr[1] = utils.format_axes(
        ax_arr[1],
        xlimits=[-1, 1],
        ylimits=[-1, 1],
        str_xlabel="BM displacement (a.u.)",
        str_ylabel="IHC potential (a.u.)",
        str_title="Half-wave rectification",
        fontsize_ticks=10,
    )
    # Plot frequency response of IHC lowpass filter (sets upper limit of phase locking)
    ihc_lowpass_filter_impulse_response = cochlear_model.ihc_lowpass_filter(impulse)
    fxx, pxx = utils.periodogram(
        ihc_lowpass_filter_impulse_response.numpy(),
        cochlear_model.sr_output,
    )
    ax_arr[2].plot(fxx, pxx)
    ax_arr[2] = utils.format_axes(
        ax_arr[2],
        str_title="IHC lowpass filter",
        str_xlabel="Frequency (Hz)",
        str_ylabel="Power (dB)",
        xscale="log",
        xlimits=[20, cochlear_model.sr_input // 2],
        ylimits=[-10, None],
        fontsize_ticks=10,
    )
    # Plot auditory nerve rate-level function (determines threshold and dynamic range)
    x_db = torch.arange(-30, 131)
    x_pa = 20e-6 * (10 ** (x_db / 20))
    y = cochlear_model.rate_level_function(x_pa)
    ax_arr[3].plot(x_db.numpy(), y.numpy())
    ax_arr[3] = utils.format_axes(
        ax_arr[3],
        str_xlabel="Sound level (dB SPL)",
        str_ylabel="ANF firing rate (spikes/s)",
        str_title="Rate-level function",
        fontsize_ticks=10,
    )
    plt.tight_layout()
    plt.show()
    return fig, ax_arr


def visualize_cochlear_model_stage_outputs(
    cochlear_model,
    example_input,
    t0=0.025,
    t1=0.075,
):
    """ """
    x = torch.as_tensor(example_input)
    list_model_stage_name = [
        "cochlear_filterbank",
        "half_wave_rectification",
        "ihc_lowpass_filter",
        "rate_level_function",
    ]
    # Iteratively apply cochlear model stages and visualize output after each stage
    fig, ax_arr = plt.subplots(
        nrows=1,
        ncols=len(list_model_stage_name),
        figsize=(12, 4),
    )
    for itr_ax, model_stage_name in enumerate(list_model_stage_name):
        model_stage = getattr(cochlear_model, model_stage_name)
        x = model_stage(x)
        # Plot only a handful of the model frequency channels
        # (excerpted and peak-normalized for visualization)
        sr = cochlear_model.sr_input if itr_ax < 2 else cochlear_model.sr_output
        excerpt = slice(int(sr * t0), int(sr * t1))
        x_to_show = x[5::10, excerpt].numpy()
        x_to_show = x_to_show / np.max(x_to_show)
        ax = ax_arr[itr_ax]
        for itr_channel in range(x_to_show.shape[0]):
            ax.plot(x_to_show[itr_channel] + 1.2 * itr_channel)
        ax = utils.format_axes(
            ax,
            str_title=model_stage_name,
            xticks=[],
            yticks=[],
            str_xlabel=f"Time ({(t1 - t0) * 1e3:.0f} ms)" if itr_ax == 0 else None,
            str_ylabel=f"Outputs from {x_to_show.shape[0]} frequency channels"
            if itr_ax == 0
            else None,
        )
    plt.tight_layout()
    plt.show()
    return fig, ax_arr


def visualize_filterbank(filterbank, str_title=None):
    """"""
    sr = filterbank.sr
    device = utils.get_device_from_module(filterbank)
    impulse = torch.zeros(int(sr)).to(device)
    impulse[0] = 1
    impulse_response = filterbank(impulse).detach().cpu().numpy()

    fig, ax = plt.subplots()
    fxx, pxx = utils.periodogram(impulse_response, sr)
    ax.plot(fxx, pxx.T - pxx.max())
    ax = utils.format_axes(
        ax,
        str_title=str_title,
        str_xlabel="Frequency (Hz)",
        str_ylabel="Power (dB)",
        xscale="log",
        xlimits=[10, sr // 2],
        ylimits=[-40, None],
    )
    plt.tight_layout()
    plt.show()
    return fig, ax


def visualize_hearing_aid_gain(hearing_aid, str_title=None):
    """"""
    sr = getattr(hearing_aid, "sr", 20000)
    device = utils.get_device_from_module(hearing_aid)
    impulse = torch.zeros(int(sr)).to(device)
    impulse[0] = 1
    impulse_response = hearing_aid(impulse).detach().cpu().numpy()

    fig, ax = plt.subplots()
    fxx, pxx = utils.periodogram(impulse_response, sr)
    _, pxx_ref = utils.periodogram(impulse.detach().cpu().numpy(), sr)
    gain = pxx - pxx_ref
    ax.plot(fxx, gain)
    ax = utils.format_axes(
        ax,
        str_title=str_title,
        str_xlabel="Frequency (Hz)",
        str_ylabel="Gain (dB)",
        xscale="log",
        xlimits=[10, sr // 2],
        ylimits=[-20, None],
    )
    ax.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.show()
    return fig, ax


def visualize_hearing_aid_output(hearing_aid, example_input):
    """ """
    sr = getattr(hearing_aid, "sr", 20000)
    device = utils.get_device_from_module(hearing_aid)
    x_unprocessed = torch.as_tensor(example_input).to(device)
    x_processed = hearing_aid(x_unprocessed)
    x_unprocessed = x_unprocessed.detach().cpu().numpy()
    x_processed = x_processed.detach().cpu().numpy()

    fig, ax_arr = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    t = np.arange(0, len(x_unprocessed)) / sr
    ax_arr[0].plot(t, x_unprocessed, color="k", alpha=0.5)
    ax_arr[0].plot(t, x_processed, color="r", alpha=0.5)
    ax_arr[0] = utils.format_axes(
        ax_arr[0],
        str_xlabel="Time (s)",
        str_ylabel="Pressure (Pa)",
        str_title="Time domain",
    )
    fxx, pxx = utils.periodogram(x_unprocessed, sr)
    fyy, pyy = utils.periodogram(x_processed, sr)
    ax_arr[1].plot(fxx, pxx, color="k", alpha=0.5)
    ax_arr[1].plot(fyy, pyy, color="r", alpha=0.5)
    ax_arr[1] = utils.format_axes(
        ax_arr[1],
        xscale="log",
        xlimits=[10, sr / 2],
        ylimits=[-10, None],
        str_xlabel="Frequency (Hz)",
        str_ylabel="Power (dB SPL)",
        str_title="Frequency domain",
    )
    plt.tight_layout()
    plt.show()

    # Play the unprocessed input, followed by the hearing-aid processed output
    data = np.concatenate(
        [
            x_unprocessed,
            np.zeros(int(0.25 * sr)),
            x_processed,
        ]
    )
    print("Unprocessed input followed by processed ouput (no level normalization)")
    ipd.display(ipd.Audio(rate=sr, data=data))

    # Play the unprocessed and processed audio, normalized to the same level
    data = np.concatenate(
        [
            x_unprocessed / np.abs(x_unprocessed).max(),
            np.zeros(int(0.25 * sr)),
            x_processed / np.abs(x_processed).max(),
        ]
    )
    print("Unprocessed input followed by processed ouput (normalized to same level)")
    ipd.display(ipd.Audio(rate=sr, data=data))

    return fig, ax_arr


def visualize_hearing_aid_training_curve(progress_log):
    """ """
    fig, ax = plt.subplots()
    ax.plot(
        range(len(progress_log)),
        [_["loss"] for _ in progress_log],
        color="k",
        marker=".",
    )
    utils.format_axes(
        ax,
        str_title="Hearing aid training curve",
        str_xlabel="Gradient descent steps",
        str_ylabel="Training loss",
        ylimits=[0, None],
    )
    plt.tight_layout()
    plt.show()
    return fig, ax
