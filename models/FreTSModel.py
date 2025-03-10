from typing import Union
from modules.FreTSModule import FreTSModule
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
import torch
import torch.nn as nn
from dataclasses import dataclass
MixedCovariatesTrainTensorType = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class FreTSModel(MixedCovariatesTorchModel):
    def __init__(self,

                 input_chunk_length: int,
                 output_chunk_length: int,
                 output_chunk_shift: int = 0,
                 channel_independence: int = 1,
                 embed_size: int = 128,
                 hidden_size: int = 256,
                 ff_size: int = 64,
                 num_blocks: int = 2,
                 activation: str = "ReLU",
                 dropout: float = 0.1,
                 norm_type: Union[str, nn.Module] = "LayerNorm",
                 normalize_before: bool = False,
                 use_static_covariates: bool = True,

                 **kwargs):
        self._channel_independence = channel_independence
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.activation = activation
        self.normalize_before = normalize_before
        self.norm_type = norm_type
        self._considers_static_covariates = use_static_covariates
        # Initialize the base class
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(
            **self.model_params)

    @property
    def supports_multivariate(self) -> bool:
        return True

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """
        Parameters
        ----------
        train_sample
            contains the following torch.Tensors: `(past_target, past_covariates, historic_future_covariates,
            future_covariates, static_covariates, future_target)`:

            - past/historic torch.Tensors have shape (input_chunk_length, n_variables)
            - future torch.Tensors have shape (output_chunk_length, n_variables)
            - static covariates have shape (component, static variable)
        """
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        input_dim = past_target.shape[1]
        output_dim = future_target.shape[1]

        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None
            else 0
        )
        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        past_cov_dim = past_covariates.shape[1] if past_covariates is not None else 0
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return FreTSModule(
            pred_len=self._output_chunk_length,
            seq_len=self._input_chunk_length,
            enc_in=input_dim,
            hidden_size=self._hidden_size,
            embed_size=self._embed_size,
            channel_independence=self._channel_independence,
            input_size=input_dim,
            output_size=output_dim,
            future_cov_dim=future_cov_dim,
            past_cov_dim=past_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            ff_size=self.ff_size,
            num_blocks=self.num_blocks,
            activation=self.activation,
            dropout=self.dropout,
            norm_type=self.norm_type,
            normalize_before=self.normalize_before,
            **self.pl_module_params,
        )

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_static_covariates(self) -> bool:
        return True

    @property
    def supports_future_covariates(self) -> bool:
        return True

    @property
    def supports_past_covariates(self) -> bool:
        return True
