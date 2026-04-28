import numpy as np
import torch

import filters
import utils


class CochlearModel(torch.nn.Module):
    def __init__(
        self,
        sr_input=20000,
        sr_output=10000,
        fir_dur=0.05,
        cfs=utils.erbspace(8e1, 8e3, 100),
        bw_mult=1.0,
        ihc_lowpass_cutoff=3000,
        ihc_lowpass_order=7,
        threshold=0.0,
        dynamic_range=80.0,
        dtype=torch.float32,
    ):
        """
        Simple cochlear model in PyTorch.

        Args
        ----
        sr_input (int): sampling rate of the input sound waveform
        sr_output (int): sampling rate of the output representation
        fir_dur (float): duration of finite impulse responses for filtering (s)
        cfs (np.ndarray): characteristic frequencies of the cochlear filters (Hz)
        bw_mult (float): scaling factor applied to cochlear filter bandwidths
        ihc_lowpass_cutoff (float): cutoff frequency of IHC lowpass filter (Hz)
        ihc_lowpass_order (float): order of IHC lowpass filter
        threshold (float): absolute threshold of auditory nerve fibers (dB SPL)
        dynamic_range (float): auditory nerve fiber dynamic range (dB)
        dtype (torch.dtype): datatype for internal intensors and inputs
        """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_output
        self.cfs = cfs
        self.dtype = dtype
        self.cochlear_filterbank = GammatoneFilterbank(
            sr=sr_input,
            fir_dur=fir_dur,
            cfs=self.cfs,
            dtype=dtype,
            bw_mult=bw_mult,
        )
        self.half_wave_rectification = torch.nn.ReLU()
        self.ihc_lowpass_filter = IHCLowpassFilter(
            sr_input=sr_input,
            sr_output=sr_output,
            fir_dur=fir_dur,
            cutoff=ihc_lowpass_cutoff,
            order=ihc_lowpass_order,
        )
        self.rate_level_function = SigmoidRateLevelFunction(
            rate_spont=0.0,
            rate_max=250.0,
            threshold=threshold,
            dynamic_range=dynamic_range,
            dynamic_range_interval=0.95,
            dtype=dtype,
        )

    def forward(self, x):
        """
        Run the cochlear model on input sound waveform(s).

        Args
        ----
        x (torch.Tensor): input sound waveform(s) with shape [batch, time] or [time]

        Returns
        -------
        x (torch.Tensor): output with shape [batch, freq, time] or [freq, time]
        """
        x = torch.as_tensor(x, dtype=self.dtype)
        x = self.cochlear_filterbank(x)
        x = self.half_wave_rectification(x)
        x = self.ihc_lowpass_filter(x)
        x = self.half_wave_rectification(x)
        x = self.rate_level_function(x)
        return x


class AudiogramMatchedCochlearModel(CochlearModel):
    def __init__(
        self,
        audiogram=None,
        sr_input=20000,
        sr_output=10000,
        fir_dur=0.05,
        cfs=utils.erbspace(8e1, 8e3, 100),
        bw_mult=1.0,
        ihc_lowpass_cutoff=3000,
        ihc_lowpass_order=7,
        threshold=0.0,
        dynamic_range=80.0,
        dtype=torch.float32,
    ):
        """
        Same as CochlearModel except the threshold and dynamic range parameters
        are modified in a CF-dependent manner to crudely simulate an audiogram.
        The audiogram should be specified as a dictionary with keys `freqs` (in
        Hz) and `dbhl` (in dB).
        """
        if audiogram is None:
            audiogram = utils.get_example_audiogram(severity="ref")
        if isinstance(audiogram, str):
            audiogram = utils.get_example_audiogram(severity=audiogram)
        msg = f"{audiogram=} must be a dict with keys `freq` and `dbhl`"
        assert isinstance(audiogram, dict), msg
        if isinstance(bw_mult, (list, tuple)):
            min_bw_mult, max_bw_mult = bw_mult
        else:
            min_bw_mult = max_bw_mult = bw_mult
        threshold, dynamic_range, bw_mult = utils.map_audiogram_to_cochlear_model_parameters(
            freq=audiogram["freq"],
            dbhl=audiogram["dbhl"],
            cfs=cfs,
            healthy_threshold=threshold,
            healthy_dynamic_range=dynamic_range,
            min_bw_mult=min_bw_mult,
            max_bw_mult=max_bw_mult,
        )
        super().__init__(
            sr_input=sr_input,
            sr_output=sr_output,
            fir_dur=fir_dur,
            cfs=cfs,
            bw_mult=bw_mult,
            ihc_lowpass_cutoff=ihc_lowpass_cutoff,
            ihc_lowpass_order=ihc_lowpass_order,
            threshold=threshold,
            dynamic_range=dynamic_range,
            dtype=dtype,
        )


class FIRFilterbank(torch.nn.Module):
    def __init__(self, fir, dtype=torch.float32, **kwargs_conv1d):
        """
        Finite impulse response (FIR) filterbank.

        Args
        ----
        fir (array_like): filter impulse response with shape
            [n_taps] or [n_filters, n_taps]
        dtype (torch.dtype): data type to cast `fir` to if `fir`
            is not a `torch.Tensor`
        kwargs_conv1d (kwargs): keyword arguments passed on to
            torch.nn.functional.conv1d (must not include `groups`,
            which is used for batching)
        """
        super().__init__()
        fir = torch.as_tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            msg = f"{fir.shape=} must be [n_taps] or [n_filters, n_taps]"
            raise ValueError(msg)
        self.register_buffer("fir", fir)
        self.kwargs_conv1d = kwargs_conv1d

    def forward(self, x, batching=False):
        """
        Apply filterbank along the time axis (dim=-1) via convolution
        in the time domain (torch.nn.functional.conv1d).

        Args
        ----
        x (torch.Tensor): input signal with shape [..., time]
        batching (bool): if True, the input is assumed to have shape
            [..., n_filters, time] and each channel is filtered with
            its own filter

        Returns
        -------
        x (torch.Tensor): filtered signal
        """
        no_batch_dim = x.ndim == 1
        if no_batch_dim:
            x = x[None, ...]
        if batching:
            assert x.shape[-2] == self.fir.shape[0]
        else:
            x = x.unsqueeze(-2)
        unflatten_shape = x.shape[:-2]
        x = torch.flatten(x, start_dim=0, end_dim=-2 - 1)
        x = torch.nn.functional.conv1d(
            input=torch.nn.functional.pad(x, (self.fir.shape[-1] - 1, 0)),
            weight=self.fir.flip(-1).view(-1, 1, self.fir.shape[-1]),
            **self.kwargs_conv1d,
            groups=x.shape[-2] if batching else 1,
        )
        x = torch.unflatten(x, 0, unflatten_shape)
        if self.fir.ndim == 1:
            x = x.squeeze(-2)
        if no_batch_dim:
            x = x[0]
        return x


class GammatoneFilterbank(FIRFilterbank):
    def __init__(
        self,
        sr=20e3,
        fir_dur=0.05,
        cfs=utils.erbspace(8e1, 8e3, 50),
        dtype=torch.float32,
        **kwargs,
    ):
        """
        Gammatone cochlear filterbank, applied by convolution
        with a set of finite impulse responses.
        """
        fir = filters.gammatone_filterbank_fir(
            sr=sr,
            fir_dur=fir_dur,
            cfs=cfs,
            **kwargs,
        )
        super().__init__(fir, dtype=dtype)


class IHCLowpassFilter(FIRFilterbank):
    def __init__(
        self,
        sr_input=20e3,
        sr_output=10e3,
        fir_dur=0.05,
        cutoff=3e3,
        order=7,
        dtype=torch.float32,
    ):
        """
        Inner hair cell low-pass filter, applied by convolution
        with a finite impulse response.
        """
        fir = filters.ihc_lowpass_filter_fir(
            sr=sr_input,
            fir_dur=fir_dur,
            cutoff=cutoff,
            order=order,
        )
        stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(stride, sr_input / sr_output), msg
        super().__init__(fir, dtype=dtype, stride=stride)


class SigmoidRateLevelFunction(torch.nn.Module):
    def __init__(
        self,
        rate_spont=0.0,
        rate_max=250.0,
        threshold=0.0,
        dynamic_range=80.0,
        dynamic_range_interval=0.95,
        dtype=torch.float32,
    ):
        """
        Sigmoid function to convert sound pressure in Pa to auditory nerve firing
        rates in spikes per second. This function crudely accounts for loss of
        audibility at low sound levels and saturation at high sound levels.

        Args
        ----
        rate_spont (float): spontaneous firing rate in spikes/s
        rate_max (float): maximum firing rate in spikes/s
        threshold (float): auditory nerve fiber threshold for spiking (dB SPL)
        dynamic_range (float): dynamic range over which firing rate changes (dB)
        dynamic_range_interval (float): determines proportion of firing rate change
            within dynamic_range (default is 95%)
        dtype (torch.dtype): data type for inputs and internal tensors
        """
        super().__init__()
        self.rate_spont = torch.as_tensor(rate_spont, dtype=dtype)
        self.rate_max = torch.as_tensor(rate_max, dtype=dtype)
        self.threshold = torch.as_tensor(threshold, dtype=dtype)
        self.dynamic_range = torch.as_tensor(dynamic_range, dtype=dtype)
        y_threshold = torch.as_tensor(
            (1 - dynamic_range_interval) / 2,
            dtype=dtype,
        )
        self.register_buffer(
            "k",
            torch.log((1 / y_threshold) - 1) / (self.dynamic_range / 2),
        )
        self.register_buffer(
            "x0",
            self.threshold - (torch.log((1 / y_threshold) - 1) / (-self.k)),
        )

    def forward(self, x):
        """
        Apply sigmoid auditory nerve rate-level function.

        Args
        ----
        x (torch.Tensor): half-wave rectified and low-pass filtered subbands with
            shape [..., freq, time]

        Returns
        -------
        x (torch.Tensor): instantaneous firing rates with shape [..., freq, time]
        """
        input_ndim = x.ndim
        while x.ndim < 3:
            x = x.unsqueeze(0)
        assert x.ndim == 3, "expected input with shape [batch, freq, time]"
        x = 20 * torch.log10(x / 20e-6)
        x = GradientStableSigmoid.apply(
            x,
            self.k.view(1, -1, 1),
            self.x0.view(1, -1, 1),
        )
        x = self.rate_spont + (self.rate_max - self.rate_spont) * x
        while x.ndim > input_ndim:
            x = x.squeeze(0)
        return x


class GradientStableSigmoid(torch.autograd.Function):
    """
    Custom autograd Function for sigmoid function with stable
    gradient (avoid NaN due to overflow in rate-level function).
    """

    @staticmethod
    def forward(ctx, x, k, x0):
        ctx.save_for_backward(x, k, x0)
        return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

    @staticmethod
    def backward(ctx, grad_output):
        x, k, x0 = ctx.saved_tensors
        grad = k * torch.exp(-k * (x - x0))
        grad = grad / (torch.exp(-k * (x - x0)) + 1.0) ** 2
        grad = torch.nan_to_num(grad, nan=0.0, posinf=None, neginf=None)
        return grad_output * grad, None, None


class AudioConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=None,
        sr=None,
        fir_dur=None,
        **kwargs,
    ):
        """
        Wrapper around torch.nn.Conv1d to support 1-dimensional
        audio convolution with a learnable FIR filter kernel.
        Unlike in standard torch.nn.Conv1d, the kernel is time-
        reversed and "same" padding is applied to the input.

        Args
        ----
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): length of FIR kernel in taps
            (specify `kernel_size` OR `sr` and `fir_dur`)
        sr (int): sampling rate of FIR kernel
        fir_dur (int): duration of FIR kernel in seconds
        """
        msg = f"invalid args: {kernel_size=}, {sr=}, {fir_dur=}"
        if kernel_size is None:
            assert (sr is not None) and (fir_dur is not None), msg
            kernel_size = int(sr * fir_dur)
        elif (sr is not None) and (fir_dur is not None):
            assert kernel_size == int(sr * fir_dur), msg
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
            padding="valid",
            padding_mode="zeros",
            **kwargs,
        )
        self.channelwise = (in_channels == 1) and (out_channels == 1)

    def forward(self, x):
        """
        Forward pass applies filter via 1-dimensional convolution
        with the FIR kernel. Input shape: [batch, channel, time].
        """
        y = torch.nn.functional.pad(
            x,
            pad=(self.kernel_size[0] - 1, 0),
            mode="constant",
            value=0,
        )
        if self.channelwise:
            # Re-shape [batch, channel, time] --> [batch * channel, 1, time]
            y = y.view(y.shape[0] * y.shape[1], 1, y.shape[2])
        y = torch.nn.functional.conv1d(
            input=y,
            weight=self.weight.flip(-1),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if self.channelwise:
            # Re-shape [batch * channel, 1, time] --> [batch, channel, time]
            y = y.view(x.shape[0], x.shape[1], y.shape[2])
        return y


class HalfCosineFilterbank(torch.nn.Module):
    def __init__(
        self,
        sr=20000,
        cf_low=10e0,
        cf_high=10e3,
        cf_num=50,
        scale="log",
        include_highpass=False,
        include_lowpass=False,
    ):
        """
        Half-cosine bandpass filterbank, applied by
        multiplication in the frequency domain.
        """
        super().__init__()
        self.sr = sr
        self.cf_low = cf_low
        self.cf_high = cf_high
        self.cf_num = cf_num
        self.scale = scale
        self.include_lowpass = include_lowpass
        self.include_highpass = include_highpass
        self.cfs = None
        self.filters = None

    def half_cosine_transfer_function(
        self,
        f,
        cf,
        bw,
        lowpass=False,
        highpass=False,
    ):
        """
        Transfer function of a half-cosine filter with center frequency
        `cf` and bandwidth `bw`, evaluated at frequencies `f`.
        """
        out = np.zeros_like(f)
        IDX = np.logical_and(f > cf - bw / 2, f < cf + bw / 2)
        out[IDX] = np.cos(np.pi * (f[IDX] - cf) / bw)
        if lowpass:
            out[f < cf] = 1
        if highpass:
            out[f > cf] = 1
        return out

    def get_frequency_domain_filters(self, f):
        """
        Construct frequency domain half-cosine filterbank with transfer
        functions evaluted at frequencies `f`.
        """
        if self.scale.lower() == "erb":
            f = utils.freq2erb(f)
            cfs = np.linspace(
                start=utils.freq2erb(self.cf_low),
                stop=utils.freq2erb(self.cf_high),
                num=self.cf_num,
            )
            self.cfs = utils.erb2freq(cfs)
        elif self.scale.lower() == "linear":
            cfs = np.linspace(
                start=self.cf_low,
                stop=self.cf_high,
                num=self.cf_num,
            )
            self.cfs = cfs
        elif self.scale.lower() == "log":
            f = np.log(f)
            cfs = np.linspace(
                start=np.log(self.cf_low),
                stop=np.log(self.cf_high),
                num=self.cf_num,
            )
            self.cfs = np.exp(cfs)
        else:
            raise ValueError(f"unrecognized filterbank scale: {self.scale}")
        bw = 2 * (cfs[1] - cfs[0]) if self.cf_num > 1 else np.inf
        filters = np.zeros((self.cf_num, len(f)), dtype=f.dtype)
        for itr, cf in enumerate(cfs):
            filters[itr] = self.half_cosine_transfer_function(
                f=f,
                cf=cf,
                bw=bw,
                lowpass=(itr == 0) and (self.include_lowpass),
                highpass=(itr == self.cf_num - 1) and (self.include_highpass),
            )
        return filters

    def forward(self, x):
        """
        Apply filterbank along time axis (dim=-1) in the frequency domain.
        Construct new filterbank on first call or if input shape changes.
        """
        y = torch.fft.rfft(x, dim=-1)
        y = torch.unsqueeze(y, dim=-2)
        rebuild = self.filters is None
        rebuild = rebuild or (not y.device == self.filters.device)
        rebuild = rebuild or (not y.ndim == self.filters.ndim)
        rebuild = rebuild or (not y.shape[-1] == self.filters.shape[-1])
        if rebuild:
            f = np.fft.rfftfreq(x.shape[-1], d=1 / self.sr)
            self.filters = torch.as_tensor(
                self.get_frequency_domain_filters(f),
                dtype=y.dtype,
                device=x.device,
            )
            while self.filters.ndim < y.ndim:
                self.filters = self.filters[None, ...]
        y = y * self.filters
        y = torch.fft.irfft(y, dim=-1)
        return y
