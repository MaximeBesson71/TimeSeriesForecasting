import pickle
import shutil
import time as time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import polars as pl
import torch
import torch.optim.lr_scheduler as lr_scheduler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import BoxCox, Scaler
from darts.explainability.shap_explainer import ShapExplainer
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.metrics import mae, mape, r2_score, rmse, rmsse, smape,  miw
from darts.models import *
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.losses import MAELoss, MapeLoss, SmapeLoss
from darts.utils.statistics import check_seasonality, plot_acf, plot_pacf
# datetime_attribute normally now OK, still to test it
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.utils import ModelMode, SeasonalityMode
from optuna.exceptions import TrialPruned
from optuna.integration import PyTorchLightningPruningCallback
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import TrialState
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import CometLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from torchmetrics import (MeanAbsoluteError, MeanSquaredError,
                          MetricCollection,
                          SymmetricMeanAbsolutePercentageError)

from tqdm.auto import tqdm

import make_segmentation

from .data import GetIpcFile, meters_dict, meters_exclude
from .display import *
from .profiles import *
from .metrics import CRPS_score, mbe, CRPSS_score, get_error_scale, rmse_s, mae_s
metrics_list = ["R2 score", "MAE", "RMSE", "MBE", "time"]

apart_list = ["A001", "A002", "A101", "A102", "A103", "A104",
              "A105", "A201", "A202", "A203", "A204", "A205", "A301", "A302", "A303", "A401", "B001", "B002",
              "B101", "B102", "B103", "B201", "B202", "B203", "B301", "B302", "B303", "B401", "C001", "C002",
              "C101", "C102", "C103", "C104", "C201", "C202", "C203", "C204", "C301", "C302", "D001"]


def ts_resample(ts, freq="1D", agg="sum"):
    tmp = TimeSeries.from_dataframe(ts.pd_dataframe().resample(freq).agg(agg))
    tmp = tmp.with_static_covariates(ts.static_covariates)
    tmp = tmp.with_hierarchy(ts.hierarchy)
    return tmp


class IdentityScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def concatenate_timeseries(series_list: List[TimeSeries]) -> TimeSeries:
    """
    Concatenates a list of Darts time series into a single one.

    :param series_list: List of time series where the end of each series
                        corresponds to the start of the next one.
    :return: A single concatenated time series.
    """
    if not series_list:
        raise ValueError("the time series list is empty.")
    ts = series_list[0]
    if len(ts) > 1:
        for i in range(len(series_list) - 1):
            ts_2 = series_list[i + 1]
            ts = TimeSeries.concatenate(ts, ts_2)
    return ts


def merge_dict(d1, d2):
    for key in d2:
        if isinstance(d2[key], Iterable):
            d1.setdefault(key, []).extend(d2[key])
        else:
            d1.setdefault(key, []).append(d2[key])
    return d1


def GetCometLogger(key, experiment_name):
    comet_logger = CometLogger(
        api_key=key,
        workspace="nvlaminc",
        # save_dir=".",
        project_name="DTE_AICOMAS",
        # experiment_key = "COMET_EXPERIMENT_KEY",
        experiment_name=experiment_name,
    )
    return comet_logger


def Load_M3(data_folder, process_needed, horizon) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    if process_needed:
        print("building M3 TimeSeries...")
        # Read DataFrame
        df_m3 = pd.read_excel(os.path.join(
            data_folder, "M3", "m3_dataset.xls"), "M3Month")
        # Build TimeSeries
        m3_series = []
        for row in tqdm(df_m3.iterrows()):
            s = row[1]
            start_year = int(s["Starting Year"])
            start_month = int(s["Starting Month"])
            values_series = s[6:].dropna()
            if start_month == 0:
                continue
            start_date = datetime(year=start_year, month=start_month, day=1)
            time_axis = pd.date_range(
                start_date, periods=len(values_series), freq="M")
            series = TimeSeries.from_times_and_values(
                time_axis, values_series.values
            ).astype(np.float32)
            m3_series.append(series)
        print("\nThere are {} monthly series in the M3 dataset".format(len(m3_series)))
        # Split train/test
        print("splitting train/test...")
        m3_train = [s[:-horizon] for s in m3_series]
        m3_test = [s[-horizon:] for s in m3_series]
        # Scale so that the largest value is 1
        print("scaling...")
        scaler_m3 = Scaler(scaler=MaxAbsScaler())
        m3_train_scaled: List[TimeSeries] = scaler_m3.fit_transform(m3_train)
        m3_test_scaled: List[TimeSeries] = scaler_m3.transform(m3_test)
        print(
            "done. There are {} series, with average training length {}".format(
                len(m3_train_scaled), np.mean(
                    [len(s) for s in m3_train_scaled])
            )
        )
        with open(os.path.join(data_folder, "M3", "m3_train_scaled.pkl"), "wb") as file:
            pickle.dump(m3_train_scaled, file)
        with open(os.path.join(data_folder, "M3", "m3_test_scaled.pkl"), "wb") as file:
            pickle.dump(m3_test_scaled, file)
    else:
        with open(os.path.join(data_folder, "M3", "m3_train_scaled.pkl"), "rb") as f:
            m3_train_scaled = pickle.load(f)
        with open(os.path.join(data_folder, "M3", "m3_test_scaled.pkl"), "rb") as f:
            m3_test_scaled = pickle.load(f)
    return m3_train_scaled, m3_test_scaled


def Load_M4(data_folder, process_needed) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    if process_needed:
        # load TimeSeries - the splitting and scaling has already been done
        print("loading M4 TimeSeries...")
        with open(os.path.join(data_folder, "M4", "m4_monthly_scaled.pkl"), "rb") as f:
            m4_series = pickle.load(f)
        # filter and keep only series that contain at least 48 training points
        m4_series = list(filter(lambda t: len(t[0]) >= 48, m4_series))
        m4_train_scaled, m4_test_scaled = zip(*m4_series)
        print(
            "done. There are {} series, with average training length {}".format(
                len(m4_train_scaled), np.mean(
                    [len(s) for s in m4_train_scaled])
            )
        )
        with open(os.path.join(data_folder, "M4", "m4_train_scaled.pkl"), "wb") as file:
            pickle.dump(m4_train_scaled, file)
        with open(os.path.join(data_folder, "M4", "m4_test_scaled.pkl"), "wb") as file:
            pickle.dump(m4_test_scaled, file)
    else:
        with open(os.path.join(data_folder, "M4", "m4_train_scaled.pkl"), "rb") as f:
            m4_train_scaled = pickle.load(f)
        with open(os.path.join(data_folder, "M4", "m4_test_scaled.pkl"), "rb") as f:
            m4_test_scaled = pickle.load(f)
    return m4_train_scaled, m4_test_scaled


def InspectSeasonality(series, meter, outdir):
    max_lag = 4*7*24
    seasonalities = list()
    fig, ax = plt.subplots(figsize=(25, 15))
    plot_acf(series, m=24, alpha=0.05, max_lag=max_lag, axis=ax)
    ax.set_title(f"Autocorrelation with max_lag of one month", fontsize=35)
    ax.set_ylabel(meters_dict[meter], fontsize=30)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=18)
    corr_dir = os.path.join(outdir, "seasonality")
    os.makedirs(corr_dir, exist_ok=True)
    plt.savefig(os.path.join(corr_dir, f"{meter}_seasonality.png"), dpi=200)
    max_lag = 7*24
    fig, ax = plt.subplots(figsize=(25, 15))
    plot_pacf(series, m=24, alpha=0.05, max_lag=max_lag, axis=ax)
    ax.set_title(
        f"Partial autocorrelation with max_lag of one week", fontsize=35)
    ax.set_ylabel(meters_dict[meter], fontsize=30)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=18)
    plt.savefig(os.path.join(
        corr_dir, f"{meter}_partial_seasonality.png"), dpi=200)
    for m in range(2, (7*24) + 1):
        is_seasonal, period = check_seasonality(
            series, m=m, alpha=0.05, max_lag=4*7*24)
        if is_seasonal:
            # print("There is seasonality of order {}.".format(period))
            seasonalities.append(period)
    season_file = open(os.path.join(corr_dir, f"{meter}_seasonality.txt"), "w")
    season_file.write(str(seasonalities))
    season_file.close()


def SplitDataset(series, train_limit, val_limit, display=True):
    train, val_test = series.split_before(pd.Timestamp(train_limit))

    val, test = val_test.split_before(pd.Timestamp(val_limit))

    if display:
        train.plot(new_plot=True, label="training")
        val.plot(label="validation")
        test.plot(label="test")
    train_val, _ = series.split_before(pd.Timestamp(val_limit))
    return train, val, test, train_val, val_test


def get_dic_breakpoints(indir, apart):
    dic_breakpoints = {}
    series = make_segmentation.get_3_signals(indir, apart, "raw")
    breakpoints = make_segmentation.segment_from_cumulative(series)
    dic_breakpoints[apart] = []
    for i in range(len(breakpoints)):
        dic_breakpoints[apart].append(series[0].index[breakpoints[i]])
    return dic_breakpoints


def CustomSplitDataset(indir, series, train_date, val_date, apart, target_list, train_list, val_list, test_list, train_val_list, val_test_list, for_optuna=False, generate_breakpoints=False):

    if generate_breakpoints:
        dic_breakpoints_dates = get_dic_breakpoints(indir, apart)
    else:
        dic_breakpoints_dates = {
            "A001": [],
            "A002": [pd.Timestamp('2021-07-28 18:00:00')],
            "A101": [],
            "A102": [],
            "A103": [pd.Timestamp('2023-03-18 23:00:00')],
            "A104": [],
            "A105": [pd.Timestamp('2023-04-01 15:00:00')],
            "A201": [],
            "A202": [pd.Timestamp('2024-01-01 00:00:00')],
            "A203": [],
            "A204": [pd.Timestamp('2023-08-31 17:00:00')],
            "A205": [],
            "A301": [],
            "A302": [],
            "A303": [pd.Timestamp('2022-10-10 23:00:00'), pd.Timestamp('2023-01-28 23:00:00')],
            "A401": [pd.Timestamp('2021-04-07 11:00:00')],
            "B001": [pd.Timestamp('2022-01-29 23:00:00')],
            "B002": [],
            "B101": [],
            "B102": [],
            "B103": [pd.Timestamp('2022-08-31 23:00:00')],
            "B201": [pd.Timestamp('2022-09-05 23:00:00')],
            "B202": [],
            "B203": [],
            "B301": [],
            "B302": [],
            "B303": [pd.Timestamp('2022-02-06 23:00:00'), pd.Timestamp('2023-03-31 23:00:00')],
            "B401": [pd.Timestamp('2023-07-24 09:00:00')],
            "C001": [],
            "C002": [],
            "C101": [],
            "C102": [],
            "C103": [],
            "C104": [],
            "C201": [pd.Timestamp('2023-08-16 17:00:00')],
            "C202": [pd.Timestamp('2021-09-30 23:00:00')],
            "C203": [],
            "C204": [],
            "C301": [],
            "C302": [pd.Timestamp('2021-08-21 23:00:00'), pd.Timestamp('2023-01-03 23:00:00'), pd.Timestamp('2023-04-07 23:00:00')],
            "D001": [pd.Timestamp('2022-11-06 14:00:00')]
        }

    breakpoints_dates = dic_breakpoints_dates[apart]
    print(breakpoints_dates)

    train_date = pd.Timestamp(train_date)
    val_date = pd.Timestamp(val_date)
    train, val, test, train_val, val_test = SplitDataset(
        series, train_date, val_date)
    val_test_list.append(val_test)
    test_list.append(test)
    target_list.append(series)

    if len(breakpoints_dates) == 0 or breakpoints_dates[0] >= val_date:
        # No breakpoint detected, use of classic SplitDataset function
        train_list.append(train)
        val_list.append(val)
        train_val_list.append(train_val)
        return

    if for_optuna:
        # When optuna is used, the trial perfs are computed on the train_val lists (on unchanged validation set)
        # So we musn't touch to these lists
        train_val_list.append(train_val)

    remain_serie = train_val
    for date in breakpoints_dates:
        if date > val_date:  # can't touch to test set
            continue
        left_serie, right_series = remain_serie.split_before(
            pd.Timestamp(date))

        if for_optuna:
            if pd.Timestamp(left_serie.start_time()) >= train_date:
                # Can't split into train val sets in original validation set, we would train on data that we have to predict
                return

            # Case breakpoint in train
            elif pd.Timestamp(left_serie.end_time()) <= train_date:
                train_size = int(0.8 * len(left_serie))

                new_train = left_serie[:train_size]
                new_val = left_serie[train_size:]

                train_list.append(new_train)
                val_list.append(new_val)

            else:  # Case breakpoint in val

                train_size = int(0.8 * len(left_serie))
                new_train = left_serie[:train_size]
                new_val = left_serie[train_size:]

                if new_train.end_time() > train_date:  # Watch out not to put train section that overlap original validation set
                    new_train = left_serie[:train_date]
                    new_val = left_serie[train_date:]

                train_list.append(new_train)
                val_list.append(new_val)

        else:  # Case where it does not matter to have train in former validation set

            train_size = int(0.8 * len(left_serie))
            new_train = left_serie[:train_size]
            new_val = left_serie[train_size:]

            train_list.append(new_train)
            val_list.append(new_val)
            train_val_list.append(new_train.append(new_val))

        # we take the right hand side of the time serie to check other breakpoints
        remain_serie = right_series

    # right hand side of last breakpoint
    train_size = int(0.8 * len(remain_serie))
    new_train = remain_serie[:train_size]
    new_val = remain_serie[train_size:]

    if for_optuna:
        if new_train.start_time() > train_date:  # check if we are in val set
            return
        # check if end of train overlap val, if yes we change to set it's end before the beginning of val set
        if val_date > new_train.end_time() > train_date:

            new_train = remain_serie[:train_date]
            new_val = remain_serie[train_date:]
    else:  # No restriction if train in original validation set
        train_val_list.append(new_train.append(new_val))

    # WARNING these 2 weeks are hardcoded for shift = context = horizon = 1 week, if changed, change this duration !
    if len(new_val) > 168*2:  # Checks if the right hand side of the time series stuck to the test set is long enough (2 weeks as it has to be > max(self.input_chunk_length, self.shift + self.output_chunk_length)
        train_list.append(new_train)
        val_list.append(new_val)
    return


def from_stochastic_dataframe(pd_pred, meter):
    np_pred = pd_pred.to_numpy()
    np_pred = np.expand_dims(np_pred, axis=1)
    pred = TimeSeries.from_times_and_values(
        pd_pred.index, np_pred)
    pred = pred.with_columns_renamed("0", meters_dict[meter])
    return pred


def remove_negative_values(preds):
    cleaned_ts_list = []
    for pred in preds:
        values = pred.all_values()
        values[values < 0] = 0
        cleaned_ts_list.append(TimeSeries.from_times_and_values(
            pred.time_index, values, columns=pred.components))

    return cleaned_ts_list


def save_preds(preds, preds_dir, filename="preds.pkl"):
    if type(preds) is list:
        preds_array = list()
        for pred in preds:
            np_pred = pred.pd_dataframe()
            preds_array.append(np_pred)
    else:
        preds_array = preds.pd_dataframe()
    os.makedirs(preds_dir, exist_ok=True)
    filename = os.path.join(preds_dir, filename)
    with open(filename, "wb") as file:
        pickle.dump(preds_array, file)


def save_perfs(metrics, outdir, name="metrics.pkl"):
    forecast_dir = os.path.join(outdir, "forecasts")
    os.makedirs(forecast_dir, exist_ok=True)
    filename = os.path.join(forecast_dir, name)
    with open(filename, "wb") as file:
        pickle.dump(metrics, file)


def compute_stats_perfs(times_dict, perfs_dict, metric, median=True):
    times = times_dict.copy()
    perfs = perfs_dict.copy()
    for _, method in enumerate(times_dict.keys()):
        if median:
            times[method] = np.median(times[method])
        else:
            times[method] = np.mean(times[method])
        perf = perfs[method]
        is_inf = np.isinf(perf)
        perf = np.delete(perf, np.where(is_inf), axis=0)
        if metric == "R2 score":
            if len(perf) == 0:
                perf = [0]
        if median:
            perfs[method] = np.nanmedian(perf)
        else:
            perfs[method] = np.nanmean(perf)
    return times, perfs


def compute_individual_perfs(perf_dict, apart, meter, conso_type, out_dir):
    forecast_dir = os.path.join(out_dir, "forecasts")
    os.makedirs(forecast_dir, exist_ok=True)
    perfs = perf_dict[apart][meter][conso_type]
    for metric in metrics_list:
        if metric == "time":
            continue
        time_dict = {x: perfs[x]["time"] for x in perfs.keys()}
        metric_dict = {x: perfs[x][metric] for x in perfs.keys()}
        median_time_dict, median_metric_dict = compute_stats_perfs(
            time_dict, metric_dict, metric, median=True)
        plot_perfs(median_time_dict, median_metric_dict, meter,
                   conso_type, metric, forecast_dir, type="Median")
        mean_time_dict, mean_metric_dict = compute_stats_perfs(
            time_dict, metric_dict, metric, median=False)
        plot_perfs(mean_time_dict, mean_metric_dict, meter,
                   conso_type, metric, forecast_dir, type="Mean")
    plt.close("all")


def compute_global_perfs(perf_dict, out_dir):
    save_perfs(perf_dict, out_dir)
    forecast_dir = os.path.join(out_dir, "forecasts")
    os.makedirs(forecast_dir, exist_ok=True)
    time_backup = {}
    perf_backup = {}
    for meter in meters_dict.keys():
        time_backup[meter] = {}
        perf_backup[meter] = {}
        for conso_type in ["daily", "conso"]:
            time_backup[meter][conso_type] = {}
            perf_backup[meter][conso_type] = {}
            for metric in metrics_list:
                if metric == "time":
                    continue
                global_time_dict = {}
                global_perf_dict = {}
                for _, apart in enumerate(perf_dict.keys()):
                    try:
                        apart_dict = perf_dict[apart][meter][conso_type]
                    except:
                        continue
                    time_dict = {x: apart_dict[x]["time"]
                                 for x in apart_dict.keys()}
                    metric_dict = {x: apart_dict[x][metric]
                                   for x in apart_dict.keys()}
                    merge_dict(global_time_dict, time_dict)
                    merge_dict(global_perf_dict, metric_dict)
                perf_backup[meter][conso_type][metric] = {}
                median_global_time_dict, median_global_perf_dict = compute_stats_perfs(
                    global_time_dict, global_perf_dict, metric, median=True)
                plot_perfs(median_global_time_dict, median_global_perf_dict,
                           meter, conso_type, metric, forecast_dir, type="Median")
                time_backup[meter][conso_type]["median"] = median_global_time_dict
                perf_backup[meter][conso_type][metric]["median"] = median_global_perf_dict
                mean_global_time_dict, mean_global_perf_dict = compute_stats_perfs(
                    global_time_dict, global_perf_dict, metric, median=False)
                plot_perfs(mean_global_time_dict, mean_global_perf_dict,
                           meter, conso_type, metric, forecast_dir, type="Mean")
                time_backup[meter][conso_type]["mean"] = mean_global_time_dict
                perf_backup[meter][conso_type][metric]["mean"] = mean_global_perf_dict

    save_perfs(time_backup, out_dir, name="stats_times.pkl")
    save_perfs(perf_backup, out_dir, name="stats_metrics.pkl")


def get_optimized_params(optuna_folder, apart, meter, conso_type, model):
    # Get the model parameters according optuna tests
    folder = os.path.join(optuna_folder, apart, "models",
                          meter, conso_type, model)
    log_file = f"local_{apart}_{meter}_{conso_type}.log"
    study_name = f"{model}_local_{apart}_{meter}_{conso_type}"
    storage = JournalStorage(JournalFileStorage(
        os.path.join(folder, log_file)))
    study_id = storage.get_study_id_from_name(study_name)
    try:
        best_trial = storage.get_best_trial(study_id)
    except:
        sampler = optuna.samplers.TPESampler(
            seed=None, constant_liar=True, n_startup_trials=50)
        study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize", "maximize"],
                                    study_name=f"{model}_local_{apart}_{meter}_{conso_type}", storage=storage, load_if_exists=True)
        best_trial = min(
            study.best_trials, key=lambda t: t.values[1])
    best_params = best_trial.params
    return best_params


def get_scaled_series(scaler_type, train, val, test, train_val, val_test, series):
    scaler = Scaler(scaler=scaler_type, global_fit=True)
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    train_val_scaled = scaler.transform(train_val)
    val_test_scaled = scaler.transform(val_test)
    series_scaled = scaler.transform(series)
    return scaler, train_scaled, val_scaled, test_scaled, train_val_scaled, val_test_scaled, series_scaled


def get_target_series(apart, apart_dir, freq='1h'):

    target_dict = {}
    for meter in meters_dict:
        if apart in meters_exclude.keys():
            if meter in meters_exclude[apart]:
                continue
        try:
            df = GetIpcFile(apart_dir, meter, "conso_profiles")
        except Exception as error:
            print("An exception occurred:", error)
            continue
        target_dict[meter] = {}
        # Push the dataframe into DARTS framework and get daily cumulative data
        df = df.astype({meters_dict[meter]: "float32"})
        series = TimeSeries.from_dataframe(
            df, value_cols=meters_dict[meter], fill_missing_dates=False)
        pd_series = series.pd_dataframe()
        series_cumsum = RecomputeRawData(pd.DataFrame(), pd_series, meter,
                                         apart_dir, display=False, save=False)
        """
        series_cumsum = series_cumsum.asfreq(freq=freq)
        series_cumsum_diff = series_cumsum.diff()
        series_cumsum_diff = series_cumsum_diff.shift(-1)
        """

        series = TimeSeries.from_dataframe(
            pd_series.resample(freq).agg("sum"),  value_cols=meters_dict[meter], fill_missing_dates=False)
        """
        if not is_daily:

            series_day = GetDailyProfiles(pd_series, series_cumsum, meter,
                                          apart_dir, display=False, save=False, time_attr=False)
            series_daily = TimeSeries.from_dataframe(
                series_day, value_cols=meters_dict[meter], fill_missing_dates=False)
        else:
            series_daily = series
        target_dict[meter]["target"] = series[:-1]
        target_dict[meter]["target_daily"] = series_daily[:-1]
        """
        target_dict[meter] = series
    return target_dict


def get_target_series_bloc(meter, outdir='./TPEE_25062024', freq='1h'):
    bloc_A = None
    bloc_B = None
    bloc_C = None
    res = None
    len_A, len_B, len_C, len_res = 0, 0, 0, 0

    for apart in apart_list:
        if apart not in meters_exclude.keys() or meter not in meters_exclude[apart]:
            apart_dir = os.path.join(outdir, apart)
            series_apart = get_target_series(apart, apart_dir, freq)[meter]
            if not res:
                null_ts = TimeSeries.from_times_and_values(
                    series_apart.time_index, np.zeros(series_apart.values().shape).astype(np.float32))
                bloc_A, bloc_B, bloc_C, res = null_ts, null_ts, null_ts, null_ts
            if apart.startswith("A"):
                bloc_A += series_apart
                len_A += 1
            if apart.startswith("B"):
                bloc_B += series_apart
                len_B += 1
            if apart.startswith("C"):
                bloc_C += series_apart
                len_C += 1
            res += series_apart
            len_res += 1
    return bloc_A/len_A, bloc_B/len_B, bloc_C/len_C, res/len_res


def get_weather_series(out_dir):
    df = GetIpcFile(out_dir, "weather_measures", "weather_profiles")
    series = TimeSeries.from_dataframe(df, fill_missing_dates=False)
    return series


class TimeForecastingModel(object):
    def __init__(self, model_cls, meter, method, outdir, display, transformer=None, load_checkpoint=False, finetune=False, load_path: str = None,  **kwargs):
        self._model_cls = model_cls
        if load_path:
            self._model = model_cls.load(load_path)

        elif load_checkpoint:
            self._model = model_cls.load_from_checkpoint(**kwargs)

        else:
            self._model = model_cls(**kwargs)
        """
        if finetune:
            self._model = self._model.load_weights_from_checkpoint(
                work_dir=self._model.work_dir, best=True)
        """
        self._meter = meter
        self._method = method
        self._outdir = outdir
        self._display = display
        self._transformer = transformer

    def fit(self, train_data, **kwargs):
        start_time = time.time()
        self._model.fit(train_data, **kwargs)
        elapsed_time = time.time() - start_time
        print(
            f"Training time for {self._model_cls.__name__} model is: {elapsed_time:.4f}s")
        return elapsed_time

    def predict(self, test_data, series=None, ind=0,    **kwargs):
        start_time = time.time()
        if series is None:
            pred = self._model.predict(n=len(test_data), **kwargs)
        else:
            pred = self._model.predict(
                n=len(test_data), series=series,  **kwargs)
        if self._transformer != None:
            if series:
                pred = self._transformer.inverse_transform[ind](pred)
            else:
                pred = self._transformer.inverse_transform(pred)
            pred = remove_negative_values(pred, self._meter)
        elapsed_time = time.time() - start_time
        print(
            f"Test time for {self._model_cls.__name__} model is: {elapsed_time:.4f}s")
        test_data.plot(new_plot=True, label="Actual data")
        pred.plot(label=f"{self._model_cls.__name__} forecast")
        return pred, elapsed_time

    def historical_forecasts(self, series, start, horizon, stride, save=True,   **kwargs):
        start_time = time.time()
        preds = self._model.historical_forecasts(series, start=start, forecast_horizon=horizon,
                                                 verbose=True, stride=stride, last_points_only=False, **kwargs)
        elapsed_time = (time.time() - start_time) / len(preds)
        print(
            f"Test time for {self._model_cls.__name__} model is: {elapsed_time}s")
        multi_vars = False
        if preds[0].n_components == 2:
            multi_vars = True
        if multi_vars:
            preds_weather = list()
            for pred_id, pred in enumerate(preds):
                try:
                    pred_weather = pred["Temp NIVRE"]
                except KeyError:
                    pred_weather = pred["Temp GOSSELIES"]
                pred_target = pred[meters_dict[self._meter]]
                preds[pred_id] = pred_target
                preds_weather.append(pred_weather)
        if self._transformer != None:
            for pred_id, pred in enumerate(preds):
                pred = self._transformer.inverse_transform(pred)
                preds[pred_id] = pred
            series = self._transformer.inverse_transform(series)
        if save:
            preds_dir = os.path.join(
                self._outdir, "forecasts", self._meter, self._method, "preds")
            save_preds(preds, preds_dir)
            if multi_vars:
                save_preds(preds_weather, preds_dir,
                           filename="preds_weather.pkl")
        if self._transformer != None:
            preds = remove_negative_values(preds)
        if self._display:
            for pred_id, pred in enumerate(preds):
                test = series.slice_intersect(pred)
                plot_forecast(test, pred, pred_id, self._meter,
                              self._method, preds_dir)
        return preds, elapsed_time

    def global_historical_forecasts(self, apart_meter_list, series, start, horizon, stride, save=True,  **kwargs):
        start_time = time.time()
        all_preds = self._model.global_historical_forecasts(series, start=start, forecast_horizon=horizon,
                                                            verbose=True, stride=stride, last_points_only=False, **kwargs)
        elapsed_time = (time.time() - start_time) / \
            (len(all_preds)+len(all_preds[0]))
        print(
            f"Test time for {self._model_cls.__name__} model is: {elapsed_time}s")
        multi_vars = False
        if all_preds[0][0].n_components == 2:
            multi_vars = True
        for idx, preds in enumerate(all_preds):
            apart = apart_meter_list[idx]
            apart_dir = os.path.join(self._outdir, apart)
            if multi_vars:
                preds_weather = list()
                for pred_id, pred in enumerate(preds):
                    pred_weather = pred["Temp NIVRE"]
                    pred_target = pred[meters_dict[self._meter]]
                    preds[pred_id] = pred_target
                    preds_weather.append(pred_weather)
            if self._transformer != None:
                for pred_id, pred in enumerate(preds):
                    pred = self._transformer[idx].inverse_transform(pred)
                    preds[pred_id] = pred
                series = self._transformer[idx].inverse_transform(series)
            if save:
                preds_dir = os.path.join(
                    apart_dir, "forecasts", self._meter, self._method, "preds")
                save_preds(preds, preds_dir)
                if multi_vars:
                    save_preds(preds_weather, preds_dir,
                               filename="preds_weather.pkl")
            if self._transformer != None:
                preds = remove_negative_values(preds)
            if self._display:
                for pred_id, pred in enumerate(preds):
                    # test = series_daily[idx].slice_intersect(pred)
                    test = series[idx].slice_intersect(pred)
                    plot_forecast(test, pred, pred_id, self._meter,
                                  self._method, preds_dir)
            all_preds[idx] = preds
        return all_preds, elapsed_time

    def compute_metrics(self, series, preds, time, m=1, in_sample=None,  register=True, display=True):
        # For stochastic time series, all metrics are computed for the median prediction
        # (quantile 0.5) by default in Darts
        stochastic = False
        tests = [series] * len(preds)
        if preds[0].n_samples > 1:

            stochastic = True
        if len(preds) > 1:
            mae_list = mae(tests, preds)
            rmse_list = rmse(tests, preds)
            mbe_list = mbe(tests, preds)
            mase_list = mae_s(tests, preds, in_sample=in_sample, m=m)
            rmsse_list = rmse_s(tests, preds, in_sample=in_sample,  m=m)
            if stochastic:
                crps_list = CRPS_score(tests, preds)
                crpss_list = CRPSS_score(
                    tests, preds, in_sample=in_sample, m=m)
                miw_list = miw(tests, preds, q_interval=(0.05, 0.95))
            else:
                r2_list = r2_score(tests, preds)

        else:
            mae_list = [mae(tests, preds)]
            rmse_list = [rmse(tests, preds)]
            mbe_list = [mbe(tests, preds)]
            rmsse_list = [rmse_s(tests, preds, in_sample=in_sample, m=m)]
            mase_list = [mae_s(tests, preds, in_sample=in_sample, m=m)]
            if stochastic:
                crps_list = [CRPS_score(tests, preds)]
                crpss_list = [CRPSS_score(
                    tests, preds, insample=in_sample, m=m)]

                miw_list = [miw(tests, preds)]
            else:
                r2_list = [r2_score(tests, preds)]
        """
        mbe_list = list()
        for test_id, test in enumerate(tests):
            pred = preds[test_id]
            test = test.slice_intersect(pred).values()
            if stochastic:
                pred = pred.quantile_timeseries(
                    quantile=0.5).univariate_values()
            else:
                pred = pred.values()
            mbe = np.mean(pred - test)
            mbe_list.append(mbe)
        """
        if not stochastic:
            metrics = {"R2 score": r2_list, "MAE": mae_list,
                       "RMSE": rmse_list, "MBE": mbe_list, 'RMSSE': rmsse_list, 'MASE': mase_list}

            is_inf = np.isinf(metrics["R2 score"])
            r2_clean = np.delete(
                metrics["R2 score"], np.where(is_inf), axis=0)
            if len(r2_clean) == 0:
                r2_clean = [0]
            if display:
                print(
                    f"Median R2 error for the {self._model_cls.__name__} model (): {np.nanmedian(r2_clean):.4f}")
                print(
                    f"Median MBE error for the {self._model_cls.__name__} model (): {np.nanmedian(mbe_list):.4f}")
                print(
                    f"Median MAE error for the {self._model_cls.__name__} model (): {np.nanmedian(mae_list):.4f}")

                print(
                    f"Median RMSE error for the {self._model_cls.__name__} model (): {np.nanmedian(rmse_list):.4f}")
                print(
                    f"Median RMSSE error for the {self._model_cls.__name__} model (): {np.nanmedian(rmsse_list):.4f}")
                print(
                    f"Median MASE  error for the {self._model_cls.__name__} model (): {np.nanmedian(mase_list):.4f}")
        else:
            metrics = {"MAE": mae_list,
                       "RMSE": rmse_list, "MBE": mbe_list, 'RMSSE': rmsse_list, 'MASE': mase_list, "CRPS": crps_list,
                       "CRPSS": crpss_list,  "MIW": miw_list}

        metrics_dir = os.path.join(
            self._outdir, "forecasts", self._meter, self._method, "metrics")
        if self._display:
            os.makedirs(metrics_dir, exist_ok=True)
            for metric in metrics:
                metrics_histograms(
                    metrics[metric], self._meter, self._method, metrics_dir, metric)
        metrics["time"] = time
        if register:
            os.makedirs(metrics_dir, exist_ok=True)
            filename = os.path.join(metrics_dir, "metrics.pkl")
            with open(filename, "wb") as file:
                pickle.dump(metrics, file)
        return metrics
