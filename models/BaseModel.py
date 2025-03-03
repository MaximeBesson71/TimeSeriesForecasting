from modules.BaseModule import *
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel


class BaseModel(PastCovariatesTorchModel):
    def __init__(
            self,
            input_chunk_length: int,
            output_chunk_length: int,
            hidden_size: int = 12024,
            output_chunk_shift: int = 0,
            **kwargs):

        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(
            **self.model_params)
        self.hidden_size = hidden_size

    @property
    def supports_multivariate(self) -> bool:
        return True

    def _create_model(self, train_sample: tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _BaseModule(
            input_size=input_dim,
            output_size=output_dim,
            hidden_size=self.hidden_size,
            nr_params=nr_params,
            **self.pl_module_params,
        )
