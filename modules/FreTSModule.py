import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.tsmixer_model import _ConditionalMixerLayer, _FeatureMixing, TimeBatchNorm2d, ACTIVATIONS, raise_log, NORMS, layer_norm_variants
from typing import Union


class FreTSModule(PLMixedCovariatesModule):
    def __init__(self, pred_len: int,
                 enc_in: int,
                 seq_len: int,
                 input_size: int,
                 output_size: int,
                 past_cov_dim: int,
                 future_cov_dim: int,
                 static_cov_dim: int,
                 ff_size: int,
                 num_blocks: int,
                 activation: str,
                 dropout: float,
                 norm_type: Union[str, nn.Module],
                 normalize_before: bool,
                 nr_params: int = 1,
                 channel_independence: int = 1,
                 embed_size: int = 128,
                 hidden_size: int = 256,
                 **kwargs):
        super(FreTSModule, self).__init__(**kwargs)
        self.embed_size = embed_size  # embed_size
        self.hidden_size = hidden_size  # hidden_size
        self.pre_length = pred_len
        self.feature_size = enc_in  # channels
        self.seq_length = seq_len
        self.channel_independence = channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.input_size = input_size
        self.target_size = output_size
        self.nr_params = nr_params
        self.target_length = self.output_chunk_length
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        input_dim = input_size
        self._output_dim = output_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        if activation not in ACTIVATIONS:
            raise_log(
                ValueError(
                    f"Invalid `activation={activation}`. Must be on of {ACTIVATIONS}."
                ),
                logger=logger,
            )
        activation = getattr(nn, activation)()

        if isinstance(norm_type, str):
            if norm_type not in NORMS:
                raise_log(
                    ValueError(
                        f"Invalid `norm_type={norm_type}`. Must be on of {NORMS}."
                    ),
                    logger=logger,
                )
            if norm_type == "TimeBatchNorm2d":
                norm_type = TimeBatchNorm2d
            else:
                norm_type = getattr(layer_norm_variants, norm_type)
        else:
            norm_type = norm_type
        mixer_params = {
            "ff_size": ff_size,
            "activation": activation,
            "dropout": dropout,
            "norm_type": norm_type,
            "normalize_before": normalize_before,
        }

        self.fc_hist = nn.Linear(
            self.input_chunk_length, self.output_chunk_length)
        self.feature_mixing_hist = _FeatureMixing(
            sequence_length=self.output_chunk_length,
            input_dim=input_dim + past_cov_dim + future_cov_dim,
            output_dim=hidden_size,
            **mixer_params,
        )
        if future_cov_dim:
            self.feature_mixing_future = _FeatureMixing(
                sequence_length=self.output_chunk_length,
                input_dim=future_cov_dim,
                output_dim=hidden_size,
                **mixer_params,
            )
        else:
            self.feature_mixing_future = None
        self.conditional_mixer = self._build_mixer(
            prediction_length=self.output_chunk_length,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            **mixer_params,
        )
        self.fc_out = nn.Linear(hidden_size, output_size * nr_params)

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    @staticmethod
    def _build_mixer(
        prediction_length: int,
        num_blocks: int,
        hidden_size: int,
        future_cov_dim: int,
        static_cov_dim: int,
        **kwargs,
    ) -> nn.ModuleList:
        """Build the mixer blocks for the model."""
        # the first block takes `x` consisting of concatenated features with size `hidden_size`:
        # - historic features
        # - optional future features
        input_dim_block = hidden_size * (1 + int(future_cov_dim > 0))

        mixer_layers = nn.ModuleList()
        for _ in range(num_blocks):
            layer = _ConditionalMixerLayer(
                input_dim=input_dim_block,
                output_dim=hidden_size,
                sequence_length=prediction_length,
                static_cov_dim=static_cov_dim,
                **kwargs,
            )
            mixer_layers.append(layer)
            # after the first block, `x` consists of previous block output with size `hidden_size`
            input_dim_block = hidden_size
        return mixer_layers

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) -
            torch.einsum('bijd,dd->bijd', x.imag, i) +
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) +
            torch.einsum('bijd,dd->bijd', x.real, i) +
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x_in):
        # x: [Batch, Input length, Channel]
        x, x_future, x_static = x_in

        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)

        x = self.feature_mixing_hist(x)
        if self.future_cov_dim:
            # feature mixing for future features (B, T, F) -> (B, T, H_S)
            x_future = self.feature_mixing_future(x_future)
            # (B, T, H_S) + (B, T, H_S) -> (B, T, 2*H_S)
            x = torch.cat([x, x_future], dim=-1)

        if self.static_cov_dim:
            # (B, C, S) -> (B, 1, C * S)
            x_static = x_static.reshape(x_static.shape[0], 1, -1)
            # repeat to match horizon (B, 1, C * S) -> (B, T, C * S)
            x_static = x_static.repeat(1, self.output_chunk_length, 1)

        for mixing_layer in self.conditional_mixer:
            # conditional mixer layers with static covariates (B, T, 2 * H_S), (B, T, C * S) -> (B, T, H_S)
            x = mixing_layer(x, x_static=x_static)

        # linear transformation to generate the forecast (B, T, H_S) -> (B, T, C * N_P)
        x = self.fc_out(x)
        # (B, T, C * N_P) -> (B, T, C, N_P)
        x = x.view(-1, self.output_chunk_length,
                   self._output_dim, self.nr_params)
        return x
