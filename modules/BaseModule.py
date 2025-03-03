from darts.datasets import *
from darts.models.forecasting.pl_forecasting_module import (
    PLPastCovariatesModule,
    io_processor,)



import torch
import torch.nn as nn



class _BaseModule(PLPastCovariatesModule):
    def __init__(self, input_size, hidden_size, output_size, nr_params :int,  **kwargs):
        super().__init__(**kwargs)
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.target_size = output_size
        self.nr_params = nr_params
        self.target_length = self.output_chunk_length
    @io_processor
    def forward(self, x_in : tuple):
        data, _  = x_in

        data = data.permute(1, 0, 2)
        data = self.hidden(data)
        data = self.activation(data)
        out = self.output(data)

        predictions = out
        predictions = predictions.view(
            -1, self.target_length, self.target_size, self.nr_params
        )
       
        return predictions