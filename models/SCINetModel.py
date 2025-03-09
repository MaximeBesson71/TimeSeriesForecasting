from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
import torch
from modules.SCINetModule import SCINetModule
from dataclasses import dataclass


class SCINetModel(PastCovariatesTorchModel):
    def __init__(self,

                 input_chunk_length: int,
                 output_chunk_length: int,
                 output_chunk_shift: int = 0,
                 input_dim: int = 9,
                 hid_size: int = 1,
                 num_stacks: int = 1,
                 num_levels: int = 3,
                 num_decoder_layer: int = 1,
                 concat_len: int = 0,
                 groups: int = 1,
                 kernel: int = 5,
                 dropout: int = 0.5,
                 single_step_output_One: int = 0,
                 input_len_seg: int = 0,
                 positionalE:  bool = False,
                 modified: bool = True,
                 RIN: bool = False,

                 **kwargs):

        self._input_dim = input_dim
        self._hid_size = hid_size
        self._num_stacks = num_stacks
        self._num_levels = num_levels
        self._num_decoder_layer = num_decoder_layer
        self._concat_len = concat_len
        self._groups = groups
        self._kernel = kernel
        self._dropout = dropout
        self._single_step_output_One = single_step_output_One
        self._input_len_seg = input_len_seg
        self._positionalE = positionalE
        self._modified = modified
        self._RIN = RIN
        self._input_chunk_length = input_chunk_length
        self._output_chunk_lenght = output_chunk_length
        # Initialize the base class
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(
            **self.model_params)

    @property
    def supports_multivariate(self) -> bool:
        return True

    def _create_model(self, train_sample: tuple[torch.Tensor]) -> torch.nn.Module:

        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        return SCINetModule(output_len=self._input_chunk_length, input_len=self._input_chunk_length, input_dim=self._input_dim, hid_size=self._hid_size, num_stacks=self._num_stacks, num_levels=self._num_levels, num_decoder_layer=self._num_decoder_layer,
                            concat_len=self._concat_len, groups=self._groups, kernel=self._kernel, dropout=self._dropout, single_step_output_One=self._single_step_output_One,
                            input_len_seg=self._input_len_seg, positionalE=self._positionalE, modified=self._modified, RIN=self._RIN, input_size=input_dim,   output_size=output_dim, nr_params=nr_params, **self.pl_module_params
                            )
