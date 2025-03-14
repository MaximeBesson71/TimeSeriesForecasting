import os
from pathlib import Path

import optuna
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.visualization import plot_param_importances, plot_slice
from st_aggrid import (AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder,
                       GridUpdateMode)


def get_filtered_study(filtered_trials):
    if not filtered_trials:
        return None  # No data for study
    filtered_study = optuna.create_study(
        direction="minimize", study_name="filtered_study")
    filtered_study.add_trials(filtered_trials)
    return filtered_study


def get_filtered_trials(study, filter_params=None, max_intermediate_value=None, include_pruned=False, index=None):
    if include_pruned:
        filtered_trials = [t for t in study.trials]
    else:
        filtered_trials = [t for t in study.trials if t.state ==
                           optuna.trial.TrialState.COMPLETE]
    # Apply hyperparameter filtering
    if filter_params:
        filtered_trials = [
            t for t in filtered_trials
            if all(t.params.get(key) == value for key, value in filter_params.items())
        ]

    # Filter out trials with any intermediate value exceeding the threshold
    if max_intermediate_value is not None:
        filtered_trials = [
            t for t in filtered_trials
            if all(value <= max_intermediate_value for value in t.intermediate_values.values())
        ]
    return filtered_trials


def render_filtered_intermediate_plot(filtered_trials):
    """
    Renders an intermediate value plot with filtering on trial hyperparameters and intermediate values.

    Parameters:
        study (optuna.study.Study): The Optuna study object.
        filter_params (dict, optional): A dictionary of hyperparameter filters (e.g., {'param_name': value}).
        max_intermediate_value (float, optional): Exclude trials where any intermediate value exceeds this threshold.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """

    if not filtered_trials:
        return None  # No data to display

    # Prepare data for Plotly
    fig = go.Figure()
    for trial in filtered_trials:
        # (step, value) pairs
        trial_data = sorted(trial.intermediate_values.items())
        fig.add_trace(go.Scatter(
            x=[step for step, _ in trial_data],
            y=[value for _, value in trial_data],
            mode='lines+markers',
            name=f"Trial {trial.number}"
        ))

    # Update layout
    fig.update_layout(
        title="Intermediate Value Plot",
        xaxis_title="Step",
        yaxis_title="Intermediate Value",
        legend_title="Trials",
        template="plotly_white",
    )

    return fig


def make_ag_grid(df, subset=None, checkbox=None):
    if subset != None:
        gb = GridOptionsBuilder.from_dataframe(
            df.loc[:, subset])
    else:
        gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gb.configure_column("DateTime", type=[
                        "dateColumnFilter", "customDateTimeFormat"], custom_format_string="yyyy-MM-dd HH:mm:ss", pivot=True)
    gb.configure_pagination(paginationAutoPageSize=False,
                            paginationPageSize=20)  # Add pagination
    gb.configure_side_bar()  # Add a sidebar
    if checkbox:
        gb.configure_selection("single", use_checkbox=True)
    gridOptions = gb.build()

    return gridOptions


@st.cache_data
def get_importance_fig(_study, filter_params, max_intermediate_value, meter, optuna_folder, index=None):
    importance_fig = plot_param_importances(_study)
    return importance_fig


@st.cache_data
def get_trials_dataframe(_study, meter, optuna_folder):
    """
    Converts study trials into a Pandas DataFrame with relevant details.

    Parameters:
        study (optuna.study.Study): The Optuna study object.

    Returns:
        pd.DataFrame: DataFrame containing trial information.
    """
    trials_data = []
    for trial in _study.trials:
        trial_info = {
            "Trial": trial.number,
            "State": trial.state.name,
            "Value": trial.value,
            **{f"Param: {k}": v for k, v in trial.params.items()},
            **{f"User Attr: {k}": v for k, v in trial.user_attrs.items()},
            **{f"System Attr: {k}": v for k, v in trial.system_attrs.items()},
        }
        trials_data.append(trial_info)

    return pd.DataFrame(trials_data)
