import inspect
from collections.abc import Sequence
from functools import wraps
from inspect import signature
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing import dtw
from darts.logging import get_logger, raise_log
from darts.utils.ts_utils import SeriesType, get_series_seq_type, series2seq
from darts.utils.utils import (
    _build_tqdm_iterator,
    _parallel_apply,
    likelihood_component_names,
    n_steps_between,
    quantile_names,
)

from darts.metrics.metrics import _get_values_or_raise, multi_ts_support, multivariate_support, METRIC_OUTPUT_TYPE
from properscoring import crps_ensemble
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from darts.metrics import mae, mape, r2_score, rmse, rmsse, smape,  miw
from scipy.stats import iqr


def get_error_scale(insample: TimeSeries,
                    metric=root_mean_squared_error,
                    m: int = 1):
    values = insample.values().flatten()
    return metric(values[m:], values[: -m])


@multi_ts_support
@multivariate_support
def CRPS_score(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
               pred_series: Union[TimeSeries, Sequence[TimeSeries]],
               intersect: bool = True,
               *,
               q: Optional[Union[float, list[float],
                                 tuple[np.ndarray, pd.Index]]] = None,
               component_reduction: Optional[Callable[[
                   np.ndarray], float]] = np.nanmean,
               series_reduction: Optional[Callable[[
                   np.ndarray], Union[float, np.ndarray]]] = None,
               n_jobs: int = 1,
               verbose: bool = False,
               ) -> METRIC_OUTPUT_TYPE:
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    crps_scores = crps_ensemble(
        y_true.squeeze(axis=2), pred_series.all_values())
    return crps_scores.mean().reshape(1, -1)


@multi_ts_support
@multivariate_support
def mbe(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        intersect: bool = True,
        *,
        q: Optional[Union[float, list[float],
                          tuple[np.ndarray, pd.Index]]] = None,
        component_reduction: Optional[Callable[[
            np.ndarray], float]] = np.nanmean,
        series_reduction: Optional[Callable[[
            np.ndarray], Union[float, np.ndarray]]] = None,
        n_jobs: int = 1,
        verbose: bool = False,
        ) -> METRIC_OUTPUT_TYPE:
    if pred_series.n_samples > 1:
        pred_series = pred_series.quantile_timeseries(quantile=0.5)
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    return np.mean(y_pred - y_true).reshape(1, -1)


@multi_ts_support
@multivariate_support
def rmse_s(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
           pred_series: Union[TimeSeries, Sequence[TimeSeries]],
           intersect: bool = True,
           *,
           q: Optional[Union[float, list[float],
                             tuple[np.ndarray, pd.Index]]] = None,
           component_reduction: Optional[Callable[[
               np.ndarray], float]] = np.nanmean,
           series_reduction: Optional[Callable[[
               np.ndarray], Union[float, np.ndarray]]] = None,
           n_jobs: int = 1,
           verbose: bool = False,
           in_sample: Optional[TimeSeries] = None,
           m: int = 1
           ) -> METRIC_OUTPUT_TYPE:

    scale = get_error_scale(
        insample=in_sample, metric=mean_absolute_error, m=m) if in_sample else np.array(1.0)
    return rmse(actual_series, pred_series).reshape(1, -1)/scale


@multi_ts_support
@multivariate_support
def mae_s(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
          pred_series: Union[TimeSeries, Sequence[TimeSeries]],
          intersect: bool = True,
          *,
          q: Optional[Union[float, list[float],
                            tuple[np.ndarray, pd.Index]]] = None,
          component_reduction: Optional[Callable[[
              np.ndarray], float]] = np.nanmean,
          series_reduction: Optional[Callable[[
              np.ndarray], Union[float, np.ndarray]]] = None,
          n_jobs: int = 1,
          verbose: bool = False,
          in_sample: Optional[TimeSeries] = None,
          m: int = 1
          ) -> METRIC_OUTPUT_TYPE:

    scale = get_error_scale(
        insample=in_sample, metric=mean_absolute_error, m=m) if in_sample else np.array(1.0)
    return mae(actual_series, pred_series).reshape(1, -1)/scale


@multi_ts_support
@multivariate_support
def CRPSS_score(actual_series: Union[TimeSeries, Sequence[TimeSeries]],
                pred_series: Union[TimeSeries, Sequence[TimeSeries]],
                intersect: bool = True,
                *,
                q: Optional[Union[float, list[float],
                                  tuple[np.ndarray, pd.Index]]] = None,
                component_reduction: Optional[Callable[[
                    np.ndarray], float]] = np.nanmean,
                series_reduction: Optional[Callable[[
                    np.ndarray], Union[float, np.ndarray]]] = None,
                n_jobs: int = 1,
                verbose: bool = False,
                in_sample: Optional[TimeSeries] = None,
                m: int = 1,
                scale_method: str = "iqr",
                ) -> METRIC_OUTPUT_TYPE:
    """
    CRPS score scaled by the error  of the mean square error of a naive seasonal in the training set  
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    crps_scores = crps_ensemble(
        y_true.squeeze(axis=2), pred_series.all_values())
    if scale_method == "in_sample_mae":
        scale = get_error_scale(
            insample=in_sample, metric=mean_absolute_error, m=m) if in_sample else np.array(1.0)
    elif scale_method == "iqr":
        iqr_value = iqr(y_true, rng=(2, 98))
        scale = iqr_value + 10

    return crps_scores.mean().reshape(1, -1)/scale
