import numpy as np
import torch

import filters
import utils


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
        if not isinstance(fir, (list, np.ndarray, torch.Tensor)):
            raise TypeError(f"unrecognized {type(fir)=}")
        if isinstance(fir, (list, np.ndarray)):
            fir = torch.tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            raise ValueError(f"invalid {fir.shape=}")
        self.register_buffer("fir", fir)
        self.kwargs_conv1d = kwargs_conv1d

    def forward(self, x, batching=False):
        """
        Apply filterbank along the time axis (dim=-1) via convolution
        in the time domain (torch.nn.functional.conv1d).

        Args
        ----
        x (torch.Tensor): input signal
        batching (bool): if True, the input is assumed to have shape
            [..., n_filters, time] and each channel is filtered with
            its own filter

        Returns
        -------
        y (torch.Tensor): filtered signal
        """
        y = x
        if batching:
            assert y.shape[-2] == self.fir.shape[0]
        else:
            y = y.unsqueeze(-2)
        unflatten_shape = y.shape[:-2]
        y = torch.flatten(y, start_dim=0, end_dim=-2 - 1)
        y = torch.nn.functional.conv1d(
            input=torch.nn.functional.pad(y, (self.fir.shape[-1] - 1, 0)),
            weight=self.fir.flip(-1).view(-1, 1, self.fir.shape[-1]),
            **self.kwargs_conv1d,
            groups=y.shape[-2] if batching else 1,
        )
        y = torch.unflatten(y, 0, unflatten_shape)
        if self.fir.ndim == 1:
            y = y.squeeze(-2)
        return y


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
        scale="erb",
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
        elif self.scale.lower() == "log":
            f = np.log(f)
            cfs = np.linspace(
                start=np.log(self.cf_low),
                stop=np.log(self.cf_high),
                num=self.cf_num,
            )
            self.cfs = np.exp(cfs)
        elif self.scale.lower() == "linear":
            cfs = np.linspace(
                start=self.cf_low,
                stop=self.cf_high,
                num=self.cf_num,
            )
            self.cfs = cfs
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
        Apply filterbank along time axis (dim=1) in the frequency domain.
        Construct new filterbank on first call or if input shape changes.
        """
        assert x.ndim >= 2, "Expected input shape [batch, time, ...]"
        y = torch.fft.rfft(x, dim=1)
        y = torch.unsqueeze(y, dim=1)
        rebuild = self.filters is None
        rebuild = rebuild or (not y.device == self.filters.device)
        rebuild = rebuild or (not y.ndim == self.filters.ndim)
        rebuild = rebuild or (not y.shape[2] == self.filters.shape[2])
        if rebuild:
            f = np.fft.rfftfreq(x.shape[1], d=1 / self.sr)
            self.filters = torch.as_tensor(
                self.get_frequency_domain_filters(f),
                dtype=y.dtype,
                device=x.device,
            )
            self.filters = self.filters[None, ...]
            for _ in range(2, x.ndim):
                self.filters = self.filters[..., None]
        y = y * self.filters
        y = torch.fft.irfft(y, dim=2)
        return y


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
            shape [batch, freq, time]

        Returns
        -------
        x (torch.Tensor): instantaneous firing rates with shape [batch, freq, time]
        """
        assert x.ndim == 3, "expected input with shape [batch, freq, time]"
        x = 20.0 * torch.log10(x / 20e-6)
        x = GradientStableSigmoid.apply(
            x,
            self.k.view(1, -1, 1),
            self.x0.view(1, -1, 1),
        )
        x = self.rate_spont + (self.rate_max - self.rate_spont) * x
        return x
