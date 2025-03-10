import argparse
import os
import shutil
import sys
import time
import warnings
from models.BaseModel import BaseModel
from models.PatchTSTModel import PatchTSTModel
from models.SCINetModel import SCINetModel
from models.FreTSModel import *
from darts.datasets import *
from sklearn.preprocessing import StandardScaler
from darts.dataprocessing.transformers import Scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BaseModel",
                        help="name of the forecaster model")
    parser.add_argument("--input_chunk_length", type=int, default=24*4)
    parser.add_argument("--input_chunk_length", type=int, default=24*4)
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="number of epochs to train the model")

    args = parser.parse_args()
    input_chunk_length = args.input_chunk_length
    output_chunk_length = args.output_chunk_length
    n_epochs = args.n_epochs
    series = ElectricityConsumptionZurichDataset().load()
    target = series['Value_NE5'][:10000]
    if args.model == "BaseModel":
        model_type = BaseModel
    elif args.model == "PatchTSTModel":
        model_type = PatchTSTModel
    elif args.model == "SCINetModel":
        model_type = SCINetModel
    elif args.model == "FreTS":
        model_type = FreTSModel
    else:
        raise ValueError(f"Model {args.model} not implemented")

    scaler_type = StandardScaler()
    scaler = Scaler(scaler=scaler_type, global_fit=True)
    target_scaled = scaler.fit_transform(target)
    model = model_type(output_chunk_length=output_chunk_length,
                       input_chunk_length=input_chunk_length, n_epochs=n_epochs)
    model.fit(target_scaled)
    pred = model.predict(24*4)
