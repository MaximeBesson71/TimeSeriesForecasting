from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel

import torch

from dataclasses import dataclass
from modules.PatchTSTModule import PatchTSTModule


@dataclass
class Config:
    enc_in: int
    seq_len: int
    pred_len: int
    e_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    fc_dropout: float
    dropout: float
    head_dropout: float
    individual: int
    patch_len: int
    stride: int
    padding_patch: str
    revin: int
    affine: int
    subtract_last: int
    decomposition: int
    kernel_size: int


class PatchTSTModel(PastCovariatesTorchModel):
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 output_chunk_shift: int = 0,
                 enc_in: int = 7,
                 n_layers: int = 8,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 fc_dropout: float = 0.05,
                 dropout: float = 0.05,
                 head_dropout: float = 0,
                 individual: int = 0,
                 patch_len: int = 16,
                 stride: int = 8,
                 padding_patch='end',
                 revin: int = 0,
                 affine: int = 0,
                 subtract_last: int = 0,
                 decomposition: int = 0,
                 kernel_size: int = 25,

                 **kwargs):
        """
        PatchTST implementation with PastCovariatesTorchModel integration.

        Parameters:
        - input_size: Length of the input window (look-back period).
        - patch_len: Length of each patch (default: 16).
        - stride: Stride between patches (default: 8).
        - num_layers: Number of Transformer layers (default: 2).
        - dropout: Dropout probability (default: 0.1).
        - output_chunk_length: Number of time steps to predict.
        - lr: Learning rate.
        - batch_size: Batch size during training.
        - kwargs: Additional arguments for the PastCovariatesTorchModel.
        """
        self._configs = Config(enc_in=enc_in, seq_len=input_chunk_length, pred_len=output_chunk_length, e_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff, fc_dropout=fc_dropout, dropout=dropout, head_dropout=head_dropout,
                               individual=individual, patch_len=patch_len, stride=stride, padding_patch=padding_patch, revin=revin, affine=affine, subtract_last=subtract_last, decomposition=decomposition, kernel_size=kernel_size)

        # Initialize the base class
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(
            **self.model_params)

    @property
    def supports_multivariate(self) -> bool:
        return True

    def _create_model(self, train_sample: tuple[torch.Tensor]) -> torch.nn.Module:
        """
        Creates and returns the PatchTST model.

        Parameters:
        - input_dim: The number of input features (dimensions of the input time series).
        - output_dim: The number of output features (dimensions of the output time series).
        """
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        return PatchTSTModule(
            configs=self._configs, input_size=input_dim,   output_size=output_dim, nr_params=nr_params, **self.pl_module_params
        )
