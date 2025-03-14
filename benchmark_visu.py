
import streamlit as st
from st_aggrid import (AgGrid, ColumnsAutoSizeMode, GridOptionsBuilder,
                       GridUpdateMode)
from optuna.visualization import plot_param_importances, plot_slice
import io
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import date
import os
import pickle
from darts import TimeSeries, concatenate
from utils.forecasting import *
from utils.streamlit_utils import *

st.set_page_config(
    layout="wide"
)
st.title("Visualisation result")

data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Catégorie obligatoire': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
    'Catégorie opt. 1': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X'],
    'Catégorie opt. 2': ['M', 'M', 'N', 'N', 'M', 'N', 'N', 'M'],
    'Catégorie opt. 3': ['P', 'Q', 'P', 'Q', 'P', 'Q', 'P', 'Q'],
    'Attribute1': [10, 20, 15, 25, 12, 18, 14, 22],
    'Attribute2': [30, 40, 35, 45, 32, 38, 34, 42],
    'Attribute3': [50, 60, 55, 65, 52, 58, 54, 62]
}
apart_list_tot = ["A001", "A002", "A101", "A102", "A103", "A104",
                  "A105", "A201", "A202", "A203", "A204", "A205", "A301", "A302", "A303", "A401", "B001", "B002",
                  "B101", "B102", "B103", "B201", "B202", "B203", "B301", "B302", "B303", "B401", "C001", "C002",
                  "C101", "C102", "C103", "C104", "C201", "C202", "C203", "C204", "C301", "C302", "D001"]

meters_exclude = {"A201": ["Compteur_electrique_double-flux", "Compteur_ECS_tot"], "A203": ["Compteur_electrique_double-flux"],
                  "C101": ["Compteur_ECS_tot"], "C103": ["Compteur_ECS_tot", "Compteur_electrique_coffret", "Compteur_EF", "Compteur_electrique_double-flux"],
                  "C203": ["Compteur_electrique_coffret", "Compteur_electrique_double-flux", "Compteur_EF"]}
models = ["TiDE"]
optuna_folder = './optuna'
default_optuna_folder = '/scratch/Ariac/energy/results/tests_optuna_TiDE/January2025'
df = pd.DataFrame(data)
path = os.path.join(optuna_folder, 'benchmark.pkl')
df2 = pd.read_pickle(path)
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Performance analysis ",  "📉 forecast visualisation", "📈 Optuna Analysis", "📊 Hourly/Daily performance comparaison "])


with tab1:
    # ---------------------------
    # 1. Filtrage obligatoire
    # ---------------------------
    # On récupère les valeurs uniques de la catégorie obligatoire

    st.write("### Complete benchmark")
    st.dataframe(df2[['model_name', 'meter',  'global', 'probabilistic',
                      'use_calendar', 'use_weather', 'frequence', 'horizon']], use_container_width=True)
    mand_options = sorted(df2['frequence'].unique())
    selected_mand = st.selectbox(
        "Choose time granularity :", mand_options)

    mand_options2 = sorted(df2['meter'].unique())
    selected_mand2 = st.selectbox(
        "Choose meter to compare :", mand_options2)

    selected_probabilist = st.selectbox(
        "select only probabilitic model :", df2['probabilistic'].unique())
    metric_det = ['RMSSE', 'R2 score', "RMSE", "MASE", "MAE", "MBE"]
    metric_prob = ['RMSSE', "RMSE", "MASE",
                   "MAE", "MBE", "CRPS", "CRPSS", "MIW"]
    if selected_probabilist == True:
        metric = ['RMSSE', "RMSE", "MASE",
                  "MAE", "MBE", "CRPS", "CRPSS", "MIW"]
    else:
        metric = ['RMSSE', 'R2 score', "RMSE", "MASE", "MAE", "MBE"]
    selected_metric = st.selectbox(
        "Choose the metric :", metric)

    apart_list = []
    for apart in apart_list_tot:
        if apart not in meters_exclude.keys():
            apart_list.append(apart)
        else:
            if selected_mand2 not in meters_exclude[apart]:
                apart_list.append(apart)

    opt1_options = ["All"] + sorted(df2['global'].unique())
    selected_opt1 = st.selectbox(
        "choose global or local forecast (optional) :", opt1_options)

    opt2_options = ["All"] + sorted(df2['use_calendar'].unique())
    selected_opt2 = st.selectbox(
        "use only model with calendar covariates:", opt2_options)

    opt3_options = ["All"] + sorted(df2['use_weather'].unique())
    selected_opt3 = st.selectbox(
        "use only model with weather covariates:", opt3_options)

    # ---------------------------
    # 3. Application des filtres sur le DataFrame
    # ---------------------------
    filtered_df = df2[(df2['frequence'] == selected_mand)
                      & (df2['meter'] == selected_mand2)]
    if selected_probabilist == True:
        filtered_df = filtered_df[filtered_df['probabilistic'] == True]
    if selected_opt1 != "All":
        filtered_df = filtered_df[filtered_df['global'] == selected_opt1]
    if selected_opt2 != "All":
        filtered_df = filtered_df[filtered_df['use_calendar'] == selected_opt2]
    if selected_opt3 != "All":
        filtered_df = filtered_df[filtered_df['use_weather'] == selected_opt3]

    st.write("### Filtered benchmark")
    st.dataframe(filtered_df[['model_name', 'meter',  'global', 'probabilistic',
                              'use_calendar', 'use_weather', 'frequence', 'horizon']], use_container_width=True)

    # ---------------------------
    # 4. Sélection de plusieurs lignes parmi le DataFrame filtré
    # ---------------------------
    st.write("## Select which model you want to compare")
    selected_indexes = st.multiselect(
        "Select the index of the model:", list(filtered_df.index))

    if selected_indexes:
        selected_rows = filtered_df.loc[selected_indexes]

        st.write("### Selected index")
        st.dataframe(selected_rows, use_container_width=True)
        chart_type = st.selectbox(
            "type of plot :", ["global performance", "local performance"])
        # ---------------------------
        # 5. Affichage d'un graphique pour les lignes sélectionnées
        # ---------------------------

        st.write("## Performance visualisation")
        # On affiche ici les attributs numériques sous forme de graphique en barres groupées.
        attributes = ['Attribute1', 'Attribute2', 'Attribute3']
        fig, ax = plt.subplots(figsize=(25, 15))
        x = np.arange(len(attributes))
        markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p']
        # largeur calculée en fonction du nombre de lignes
        width = 0.8 / len(selected_rows)
        shapes = ["o", "s", "D", "*"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                  "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
        styles = list(product(shapes, colors))
        if chart_type == "global performance":
            for idx, index in enumerate(selected_indexes):
                s = styles[idx]
                list_perf = []
                list_time = []
                for apart in apart_list:
                    metric = df2.loc[index,
                                     apart]["perf"][f"{selected_metric}"]
                    is_inf = np.isinf(metric)
                    metric = np.delete(metric, np.where(is_inf), axis=0)
                    list_time.append(df2.loc[index, apart]["training_time"])
                    list_perf.append(np.nanmean(metric))
                avg_perf = np.nanmean(list_perf)
                model_name = df2.loc[index, "model_name"]
                if df2.loc[index, 'global']:
                    training_time = list_time[0]
                    model_name = 'global ' + model_name
                else:
                    training_time = sum(list_time)
                    # model_name = 'local ' + model_name
                    model_name = model_name
                if df2.loc[index, 'probabilistic']:
                    model_name = 'probabilist ' + model_name
                if df2.loc[index, 'use_calendar']:
                    model_name = model_name + '_cal'
                if df2.loc[index, 'use_weather']:
                    model_name = model_name + '_temp'
                plt.semilogx(training_time, avg_perf,
                             s[0], color=s[1], label=f"{model_name}", markersize=13)

            # Création de barres pour chaque ligne sélectionnée
            plt.xlabel("Training time [s]", fontsize=30)
            plt.ylabel(f"{selected_metric}", fontsize=30)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.title(f"Benchmark for {selected_mand2}", fontsize=35)
            plt.legend(fontsize=18, loc="upper right")
            # ax.legend()

            st.pyplot(fig)
        if chart_type == "local performance":
            x_numeric = np.arange(1, len(apart_list) + 1)
            for idx, index in enumerate(selected_indexes):
                # Avoid index out of range
                marker_shape, color = styles[idx % len(styles)]
                list_perf = []
                for apart in apart_list:
                    metric = df2.loc[index,
                                     apart]["perf"][f"{selected_metric}"]
                    list_perf.append(np.nanmean(metric))
                model_name = df2.loc[index, "model_name"]
                if df2.loc[index, 'global']:
                    model_name = 'global ' + model_name
                else:
                    # model_name = 'local ' + model_name
                    model_name = model_name
                ax.scatter(apart_list, list_perf,
                           label=f"{model_name}", color=color, marker=marker_shape, s=70)
            plt.xlabel("Apartment", fontsize=30)
            plt.ylabel(f"{selected_metric}", fontsize=30)
            plt.xticks(x_numeric, fontsize=22)
            ax.set_xticklabels(apart_list, rotation=90)
            plt.yticks(fontsize=22)
            plt.title(f"Benchmark for {selected_mand2}", fontsize=35)
            plt.legend(fontsize=18)

            st.pyplot(fig)

        svg_buffer = io.StringIO()
        fig.savefig(svg_buffer, format='svg')
        svg_data = svg_buffer.getvalue()
        st.download_button(
            label="📥 Download Plot as SVG",
            data=svg_data,
            file_name=f'{selected_mand2}.svg',
            mime='image/svg+xml',
            key="download1"
        )

    else:
        st.write(
            "Veuillez sélectionner au moins une ligne pour afficher le graphique.")
with tab2:
    st.write("### Complete benchmark")
    st.dataframe(df2[['model_name', 'meter',  'global', 'probabilistic',
                      'use_calendar', 'use_weather', 'frequence', 'horizon']], use_container_width=True)
    freq_options = sorted(df2['frequence'].unique())
    selected_freq = st.selectbox(
        "Choose time granularity :", freq_options, key=2)
    proba_options = sorted(df2['probabilistic'].unique())
    selected_proba = st.selectbox(
        "probabilitistic or deterministic model :", proba_options, key=3)

    meter_options = sorted(df2['meter'].unique())
    selected_meter = st.selectbox(
        "Choose meter to compare :", meter_options, key=17)
    apart_list2 = []
    for apart in apart_list_tot:
        if apart not in meters_exclude.keys():
            apart_list2.append(apart)
        else:
            if selected_meter not in meters_exclude[apart]:
                apart_list2.append(apart)

    selected_apart = st.selectbox("choose the apartment", apart_list2)

    # ---------------------------
    # 2. Filtres optionnels
    # ---------------------------
    # Pour chaque filtre optionnel, on propose un choix avec une option "Tous"

    mode_options = ["All"] + sorted(df2['global'].unique())
    selected_mode = st.selectbox(
        "choose global or local forecast (optional) :", mode_options, key=4)

    calendar_options = ["All"] + sorted(df2['use_calendar'].unique())
    selected_calendar = st.selectbox(
        "use only model with calendar covariates:", calendar_options, key=5)

    weather_options = ["All"] + sorted(df2['use_weather'].unique())
    selected_weather = st.selectbox(
        "use only model with weather covariates:", weather_options, key=6)

    date_range = st.date_input("📅 Sélectionnez une plage de dates :", value=[
                               date(2023, 6, 25), date(2024, 6, 24)])

    if len(date_range) == 2:
        start_date, end_date = date_range

        if start_date > end_date:
            st.error(
                "⚠️ Error: The end date must be later than the start date!")
        elif end_date > date(2024, 6, 24):
            st.error("⚠️ data no longer available after 24/06/2024")
        elif start_date < date(2021, 1, 1):
            st.error("⚠️  data only available from 01/01/2021")
        else:
            st.success(
                f"✅ Selected period : **from {start_date} to {end_date}**")
    pd_start_date = pd.to_datetime(start_date)
    pd_end_date = pd.to_datetime(end_date)
    filtered_df_pred = df2[(df2['frequence'] == selected_freq)
                           & (df2['meter'] == selected_meter) & (df2['probabilistic'] == selected_proba)]

    if selected_mode != "All":
        filtered_df_pred = filtered_df_pred[filtered_df_pred['global']
                                            == selected_mode]
    if selected_calendar != "All":
        filtered_df_pred = filtered_df_pred[filtered_df_pred['use_calendar']
                                            == selected_calendar]
    if selected_weather != "All":
        filtered_df_pred = filtered_df_pred[filtered_df_pred['use_weather']
                                            == selected_weather]
    st.write("### Filtered benchmark")
    st.dataframe(filtered_df_pred[['model_name', 'meter',  'global', 'probabilistic',
                                   'use_calendar', 'use_weather', 'frequence', 'horizon']], use_container_width=True)

    # ---------------------------
    # 4. Sélection de plusieurs lignes parmi le DataFrame filtré
    # ---------------------------
    st.write("## Select which model you want to compare")
    selected_indexes = st.multiselect(
        "Select the index of the model:", list(filtered_df_pred.index), key=7)

    if selected_indexes and date_range:
        selected_rows = filtered_df_pred.loc[selected_indexes]

        st.write("### Selected index")
        st.dataframe(selected_rows, use_container_width=True)

        st.write("## Performance visualisation")
        fig = go.Figure()
        results_folder = './TPEE_25062024'
        apart_dir = os.path.join(results_folder, selected_apart)
        meter = selected_meter
        target_series = get_target_series(
            selected_apart, apart_dir,  selected_freq)
        target = target_series[meter]
        target_daily = target_series[meter]
        train, val, test, train_val, val_test = SplitDataset(
            target, "20230101", "20230625", display=False)
        pd_target = target.slice(pd_start_date, pd_end_date)
        pd_target = pd_target.pd_dataframe()
        pd_target = pd_target.reset_index()
        fig.add_trace(go.Scatter(
            x=pd_target['date'],
            y=pd_target[meters_dict[meter]],
            mode='lines+markers',
            name='ground truth',
            line=dict(color='black'),
            marker=dict(symbol='square', size=6)
        ))
        colors = ["blue", "orange", "green", "red",
                  "purple", "brown", "pink", "gray"]
        for idx, index in enumerate(selected_indexes):
            model_name = df2.loc[index, "model_name"]
            global_ = df2.loc[index, "global"]
            filename = df2.loc[index, "meter"] + "_" + df2.loc[index, "model_name"] + \
                '_' + df2.loc[index, "frequence"] + \
                "_" + df2.loc[index, "horizon"]
            if df2.loc[index, "use_calendar"]:
                filename = filename + "_cal"
                model_name = model_name + '_cal'
            if df2.loc[index, "use_weather"]:
                filename = filename + "_wth"
                model_name = model_name + '_temp'
            proba_suffixe = "proba_" if df2.loc[index, "probabilistic"] else ""
            if global_:

                filename = "pred_" + proba_suffixe + "global_" + filename + ".pkl"
                model_name = proba_suffixe + 'global_' + model_name
                path = os.path.join(optuna_folder, selected_meter, filename)
                with open(path, "rb") as file:
                    pred_dict = pickle.load(file)
                pred_historical_forecast = pred_dict["pred"][apart_list.index(
                    selected_apart)]
            if not global_:
                model_name = proba_suffixe + 'local_' + model_name
                filename = "pred_" + proba_suffixe + "local_" + filename + ".pkl"
                path = os.path.join("./optuna", selected_meter,
                                    selected_apart, filename)
                with open(path, "rb") as file:
                    pred_dict = pickle.load(file)
                pred_historical_forecast = pred_dict["pred"]
            ts = pred_historical_forecast[0]
            for i in range(len(pred_historical_forecast) - 1):
                ts_2 = pred_historical_forecast[i + 1]
                ts = TimeSeries.concatenate(ts, ts_2)

            ts = ts.slice(pd_start_date, pd_end_date)
            if df2.loc[index, "probabilistic"]:
                low_quantile = ts.quantile(0.05)
                high_quantile = ts.quantile(0.95)
                fig.add_trace(go.Scatter(
                    x=ts.time_index,
                    y=ts.quantile(0.5).values().flatten(),
                    mode='lines+markers',
                    name='Prévision (Médiane)',
                    line=dict(color='blue'),
                    marker=dict(symbol='circle', size=6)
                ))

                # Tracer l'intervalle de confiance (quantiles 5%-95%)
                fig.add_trace(go.Scatter(
                    x=ts.time_index.tolist() + ts.time_index[::-1].tolist(),
                    y=high_quantile.values().flatten().tolist(
                    ) + low_quantile.values().flatten()[::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(0, 100, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence interval (5%-95%)'
                ))
            else:
                pd_ts = ts.pd_dataframe()
                pd_ts = pd_ts.reset_index()
                fig.add_trace(go.Scatter(
                    x=pd_ts['date'],
                    y=pd_ts[meters_dict[meter]],
                    mode='lines',
                    name=f"{model_name}",
                    line=dict(color=colors[idx]),
                    marker=dict(symbol='circle', size=6)
                ))

        # Scale our data
        # scaler = IdentityScaler() if selected_freq == "1D" else StandardScaler()
        # unit_scaler, train_scaled, val_scaled, test_scaled, train_val_scaled, val_test_scaled, target_scaled = get_scaled_series(
        #    scaler, train, val, test, train_val, val_test, target)
        # Add first time series
        fig.update_layout(
            title=dict(
                text=f"Energy Consumption Over Time for {meter}",
                font=dict(size=24, family='Arial',
                          color='black'),  # Title font size
                x=0.5,
                xanchor="center"
            ),
            xaxis=dict(
                title='Date',
                titlefont=dict(size=18),  # X-axis title size
                tickfont=dict(size=14)    # X-axis tick labels size
            ),
            yaxis=dict(
                title=f'{meters_dict[meter]}',
                titlefont=dict(size=18),  # Y-axis title size
                tickfont=dict(size=14)    # Y-axis tick labels size
            ),
            legend=dict(
                font=dict(size=16),       # Legend font size
                orientation="h",          # Horizontal legend (optional)
                yanchor="bottom", y=1.02,  # Move legend above the plot
                xanchor="center", x=0.5
            ),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
        svg_buffer = io.BytesIO()
        fig.write_image(svg_buffer, format='svg')
        svg_data = svg_buffer.getvalue()
        st.download_button(
            label="📥 Download Plot as SVG",
            data=svg_data,
            file_name=f'{selected_mand2}.svg',
            mime='image/svg+xml',
            key="download2"
        )
with tab3:
    st.title("Optuna Analysis")
    st.dataframe(df2[['model_name', 'meter',  'global', 'probabilistic',
                      'use_calendar', 'use_weather', 'frequence', 'horizon']])
    # Let the user specify the working directory

    st.write("## Select wich model you want to analyse")
    freq_options_optuna = ["All"] + sorted(df2['frequence'].unique())
    selected_freq_optuna = st.selectbox(
        "Choose time granularity :", freq_options_optuna, key=9)

    meter_options_optuna = ["All"] + sorted(df2['meter'].unique())
    selected_meter_optuna = st.selectbox(
        "Choose meter to compare :", meter_options_optuna, key=10)
    mode_options = ["All"] + sorted(df2['global'].unique())
    selected_mode_optuna = st.selectbox(
        "choose global or local forecast (optional) :", mode_options, key=11)

    calendar_options = ["All"] + sorted(df2['use_calendar'].unique())
    selected_calendar_optuna = st.selectbox(
        "use only model with calendar covariates:", calendar_options, key=12)

    weather_options = ["All"] + sorted(df2['use_weather'].unique())
    selected_weather_optuna = st.selectbox(
        "use only model with weather covariates:", weather_options, key=13)
    filtered_df_pred_optuna = df2.copy()
    if selected_freq_optuna != "All":
        filtered_df_pred_optuna = filtered_df_pred_optuna[filtered_df_pred_optuna['frequence']
                                                          == selected_freq_optuna]
    if selected_meter_optuna != "All":
        filtered_df_pred_optuna = filtered_df_pred_optuna[filtered_df_pred_optuna['meter']
                                                          == selected_meter_optuna]

    if selected_mode_optuna != "All":
        filtered_df_pred_optuna = filtered_df_pred_optuna[filtered_df_pred_optuna['global']
                                                          == selected_mode_optuna]
    if selected_calendar != "All":
        filtered_df_pred_optuna = filtered_df_pred_optuna[filtered_df_pred_optuna['use_calendar']
                                                          == selected_calendar_optuna]
    if selected_weather != "All":
        filtered_df_pred_optuna = filtered_df_pred_optuna[filtered_df_pred_optuna['use_weather']
                                                          == selected_weather_optuna]
    st.dataframe(filtered_df_pred_optuna[['model_name', 'meter',  'global', 'probabilistic',
                                          'use_calendar', 'use_weather', 'frequence', 'horizon']], use_container_width=True)
    index = st.selectbox(
        "Select the index of the model:", list(filtered_df_pred_optuna.index), key=8)
    if index and df2.loc[index, "global"] == False:
        selected_apart_optuna = st.selectbox(
            "choose the apartment", apart_list2, key=14)
    model_name = df2.loc[index, "model_name"]
    if index:
        global_ = df2.loc[index, "global"]
        filename = df2.loc[index, "meter"] + "_" + df2.loc[index, "model_name"] + \
            '_' + df2.loc[index, "frequence"] + "_" + df2.loc[index, "horizon"]
        if df2.loc[index, "use_calendar"]:
            filename = filename + "_cal"
            model_name = model_name + '_cal'
        if df2.loc[index, "use_weather"]:
            filename = filename + "_wth"
            model_name = model_name + '_wth'
        if global_:
            filename = "global_" + filename + ".log"
            model_name = 'global_' + model_name
            folder = os.path.join('./optuna', df2.loc[index, "meter"])
        if not global_:
            model_name = 'local_' + model_name
            filename = "local_" + filename + ".log"
            folder = os.path.join(
                optuna_folder, df2.loc[index, "meter"], selected_apart_optuna)
        meter = df2.loc[index, "meter"]
        storage = JournalStorage(JournalFileStorage(
            os.path.join(folder, filename)))
        study_name = storage.get_all_studies()[0].study_name
        # Load Optuna study (replace this with your study loading logic)
        if study_name != None:
            study = optuna.load_study(study_name=study_name, storage=storage)
        # Sidebar for filtering parameters
        st.sidebar.header("Filters for intermediate values plots")
        include_pruned = st.sidebar.checkbox(
            "Include pruned trials", value=False)

        # Collect filterable hyperparameters
        if include_pruned:
            all_trials = [t for t in study.trials]
        else:
            all_trials = [t for t in study.trials if t.state ==
                          optuna.trial.TrialState.COMPLETE]
        hyperparameters = {key for t in all_trials for key in t.params.keys()}

        filter_params = {}
        for param in hyperparameters:
            unique_values = sorted({t.params[param]
                                    for t in all_trials if param in t.params})
            selected_value = st.sidebar.selectbox(
                f"Filter by {param}", options=[None] + unique_values, format_func=lambda x: "All" if x is None else x
            )
            if selected_value is not None:
                filter_params[param] = selected_value

        # Filter for maximum intermediate value
        max_intermediate_value = st.sidebar.number_input(
            "Max Intermediate Value",
            value=10.0,
            step=0.5,
            format="%.2f"
        )

        filtered_trials = get_filtered_trials(
            study, filter_params=None, max_intermediate_value=max_intermediate_value if max_intermediate_value != float(
                "inf") else None, include_pruned=include_pruned, index=index)
        full_filtered_trials = get_filtered_trials(
            study, filter_params, max_intermediate_value=max_intermediate_value if max_intermediate_value != float(
                "inf") else None, include_pruned=include_pruned)
        filtered_study = get_filtered_study(full_filtered_trials)

        # Hyperparameter importance
        st.subheader("Hyperparameter Importance")
        try:
            importance_fig = get_importance_fig(
                filtered_study, filter_params, max_intermediate_value, meter, optuna_folder, index=index)
            st.plotly_chart(importance_fig, use_container_width=True,
                            key="param_importance_plot")
        except ValueError as e:
            st.warning(
                f"Hyperparameter importance could not be calculated: {e}")

        # Slice plots
        st.subheader("Slice plots")
        params_list = list(hyperparameters)
        params_options = st.multiselect(
            "Choose variables to plot",
            params_list,
            [params_list[0]])
        slice_fig = plot_slice(filtered_study, params=params_options)
        slice_fig.update_layout(yaxis=dict(autorange=True))
        st.plotly_chart(slice_fig, use_container_width=True, key="slice_plot")
        # Generate plots
        filtered_fig = render_filtered_intermediate_plot(
            filtered_trials,
        )
        full_filtered_fig = render_filtered_intermediate_plot(
            full_filtered_trials,
        )

        # Display plots side by side
        st.subheader("Unfiltered Trials (only max intermediate value applied)")
        if filtered_fig:
            st.plotly_chart(filtered_fig, use_container_width=True,
                            key="unfiltered_plot")
        else:
            st.write("No data to display for unfiltered trials.")

        st.subheader("Filtered Trials (hyperparameter filter also applied)")
        if full_filtered_fig:
            st.plotly_chart(full_filtered_fig, use_container_width=True,
                            key="filtered_plot")
        else:
            st.write("No data to display for filtered trials.")

        # Generate the trials table
        st.subheader("Trials Table")
        trials_df = get_trials_dataframe(study, meter, optuna_folder)

        if trials_df.empty:
            st.write("No completed trials to display.")
        else:
            gridOptions = make_ag_grid(trials_df)
            AgGrid(trials_df, gridOptions=gridOptions,
                   fit_columns_on_grid_load=False, key="trials_df")

        # Display best trial information
        best_trial = study.best_trial
        st.subheader("Best trial summary")
        st.write(f"Trial number: {best_trial.number}")
        st.write(f"Value: {best_trial.value}")
        st.write("Best hyperparameters:")
        for key, value in best_trial.params.items():
            st.write(f"{key}: {value}")
with tab4:
    st.title("Hourly/Daily performances comparaison")
    st.write("### Complete benchmark")
    tab4_filtered_df = df2[df2["probabilistic"] == False]
    st.dataframe(tab4_filtered_df[['model_name', 'meter',  'global',
                                   'use_calendar', 'use_weather', 'frequence', 'horizon']], use_container_width=True)
    tab4_meter = sorted(tab4_filtered_df['meter'].unique())
    selected_tab4_meter = st.selectbox(
        "Choose meter to compare :", tab4_meter, key=18)
    tab4_training_mode_options = ["All"] + \
        sorted(tab4_filtered_df['global'].unique())
    selected_tab4_training_mode = st.selectbox(
        "choose global or local forecast (optional) :", tab4_training_mode_options, key=19)

    tab4_calendar_options = ["All"] + \
        sorted(tab4_filtered_df['use_calendar'].unique())
    selected_tab4_calendar = st.selectbox(
        "use only model with calendar covariates (optional):", tab4_calendar_options, key=20)

    tab4_weather_options = ["All"] + \
        sorted(tab4_filtered_df['use_weather'].unique())
    selected_tab4_weather = st.selectbox(
        "use only model with weather covariates (optional):", tab4_weather_options, key=21)
    tab4_filtered_df = tab4_filtered_df[(
        tab4_filtered_df['meter'] == selected_tab4_meter)]
    if selected_tab4_training_mode != "All":
        tab4_filtered_df = tab4_filtered_df[tab4_filtered_df["global"]
                                            == selected_tab4_training_mode]
    if selected_tab4_calendar != "All":
        tab4_filtered_df = tab4_filtered_df[tab4_filtered_df['use_calendar']
                                            == selected_tab4_calendar]
    if selected_tab4_weather != "All":
        tab4_filtered_df = tab4_filtered_df[tab4_filtered_df['use_weather']
                                            == selected_tab4_weather]
    tab4_apart_list = []
    for apart in apart_list_tot:
        if apart not in meters_exclude.keys():
            tab4_apart_list.append(apart)
        else:
            if selected_tab4_meter not in meters_exclude[apart]:
                tab4_apart_list.append(apart)
    st.write("### Filtered benchmark")
    st.dataframe(tab4_filtered_df[['model_name', 'meter',  'global',
                                   'use_calendar', 'use_weather', 'frequence', 'horizon']], use_container_width=True)

    st.write("## Select which model you want to compare")
    tab4_selected_indexes = st.multiselect(
        "Select the index of the model:", list(tab4_filtered_df.index), key=24)
    if tab4_selected_indexes:
        tab4_selected_rows = tab4_filtered_df.loc[tab4_selected_indexes]

        st.write("### Selected index")
        st.dataframe(tab4_selected_rows, use_container_width=True)
        analysis_type = st.selectbox(
            "type of plot :", ["metric comparaison", "visualisation"], key=22)
        if analysis_type == "metric comparaison":
            fig, ax = plt.subplots(figsize=(25, 15))
            markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p']
            # largeur calculée en fonction du nombre de lignes
            shapes = ["o", "s", "D", "*"]
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                      "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
            styles = list(product(shapes, colors))

            metric = ['RMSSE', 'R2 score', "RMSE", "MASE", "MAE", "MBE"]
            tab4_selected_metrics = st.selectbox(
                "Choose with which metric to compare  model:", metric, key=23)
            for idx, index in enumerate(tab4_selected_indexes):
                s = styles[idx]
                list_perf = []
                list_time = []
                for apart in tab4_apart_list:
                    metric = df2.loc[index,
                                     apart]["benchmark_perf"][f"{tab4_selected_metrics}"]
                    is_inf = np.isinf(metric)
                    metric = np.delete(metric, np.where(is_inf), axis=0)
                    list_time.append(df2.loc[index, apart]["training_time"])
                    list_perf.append(np.nanmean(metric))
                avg_perf = np.nanmean(list_perf)
                model_name = df2.loc[index, "model_name"]
                if df2.loc[index, 'global']:
                    training_time = list_time[0]
                    model_name = 'global ' + model_name
                else:
                    training_time = sum(list_time)
                    # model_name = 'local ' + model_name
                    model_name = model_name
                if df2.loc[index, 'probabilistic']:
                    model_name = 'probabilist ' + model_name
                if df2.loc[index, 'use_calendar']:
                    model_name = model_name + '_cal'
                if df2.loc[index, 'use_weather']:
                    model_name = model_name + '_temp'
                if df2.loc[index, "frequence"] == "1D":
                    model_name = "daily " + model_name
                if df2.loc[index, "frequence"] == "1h":
                    model_name = "hourly " + model_name
                plt.semilogx(training_time, avg_perf,
                             s[0], color=s[1], label=f"{model_name}", markersize=13)

            # Création de barres pour chaque ligne sélectionnée
            plt.xlabel("Training time [s]", fontsize=30)
            plt.ylabel(f"{tab4_selected_metrics}", fontsize=30)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.title(f"Benchmark for {selected_mand2}", fontsize=35)
            plt.legend(fontsize=18, loc="upper right")
            # ax.legend()
            st.pyplot(fig)
        if analysis_type == "visualisation":
            selected_tab4_apart = st.selectbox(
                "Choose appartment:", tab4_apart_list, key=25)
            fig = go.Figure()
            results_folder = './TPEE_25062024'
            apart_dir = os.path.join(results_folder, selected_tab4_apart)
            meter = selected_tab4_meter
            target_series = get_target_series(
                selected_apart, apart_dir,  "1D")
            target = target_series[meter]
            target_daily = target_series[meter]
            train, val, test, train_val, val_test = SplitDataset(
                target, "20230101", "20230625", display=False)
            pd_target = test
            pd_target = pd_target.pd_dataframe()
            pd_target = pd_target.reset_index()
            fig.add_trace(go.Scatter(
                x=pd_target['date'],
                y=pd_target[meters_dict[meter]],
                mode='lines+markers',
                name='ground truth',
                line=dict(color='black'),
                marker=dict(symbol='square', size=6)
            ))
            colors = ["blue", "orange", "green", "red",
                      "purple", "brown", "pink", "gray"]
            for idx, index in enumerate(tab4_selected_indexes):
                model_name = df2.loc[index, "model_name"]
                global_ = df2.loc[index, "global"]
                filename = df2.loc[index, "meter"] + "_" + df2.loc[index, "model_name"] + \
                    '_' + df2.loc[index, "frequence"] + \
                    "_" + df2.loc[index, "horizon"]
                if df2.loc[index, "use_calendar"]:
                    filename = filename + "_cal"
                    model_name = model_name + '_cal'
                if df2.loc[index, "use_weather"]:
                    filename = filename + "_wth"
                    model_name = model_name + '_temp'
                if df2.loc[index, "frequence"] == "1D":
                    model_name = "daily " + model_name
                if df2.loc[index, "frequence"] == "1h":
                    model_name = "hourly " + model_name
                if global_:

                    filename = "pred_" + "global_" + filename + ".pkl"
                    model_name = 'global ' + model_name
                    path = os.path.join(
                        optuna_folder, selected_tab4_meter, filename)
                    with open(path, "rb") as file:
                        pred_dict = pickle.load(file)
                    pred_historical_forecast = pred_dict["benchmark_pred"][tab4_apart_list.index(
                        selected_tab4_apart)]
                if not global_:
                    model_name = 'local ' + model_name
                    filename = "pred_" + "local_" + filename + ".pkl"
                    path = os.path.join("./optuna", selected_meter,
                                        selected_apart, filename)
                    with open(path, "rb") as file:
                        pred_dict = pickle.load(file)
                    pred_historical_forecast = pred_dict["benchmark_pred"]
                ts = concatenate_timeseries(pred_historical_forecast)
                pd_ts = ts.pd_dataframe()
                pd_ts = pd_ts.reset_index()
                fig.add_trace(go.Scatter(
                    x=pd_ts['date'],
                    y=pd_ts[meters_dict[meter]],
                    mode='lines',
                    name=f"{model_name}",
                    line=dict(color=colors[idx]),
                    marker=dict(symbol='circle', size=6)
                ))

            # Scale our data
            # scaler = IdentityScaler() if selected_freq == "1D" else StandardScaler()
            # unit_scaler, train_scaled, val_scaled, test_scaled, train_val_scaled, val_test_scaled, target_scaled = get_scaled_series(
            #    scaler, train, val, test, train_val, val_test, target)
            # Add first time series
            fig.update_layout(
                title=dict(
                    text=f"Energy Consumption Over Time for {meter}",
                    font=dict(size=24, family='Arial',
                              color='black'),  # Title font size
                    x=0.5,
                    xanchor="center"
                ),
                xaxis=dict(
                    title='Date',
                    titlefont=dict(size=18),  # X-axis title size
                    tickfont=dict(size=14)    # X-axis tick labels size
                ),
                yaxis=dict(
                    title=f'{meters_dict[meter]}',
                    titlefont=dict(size=18),  # Y-axis title size
                    tickfont=dict(size=14)    # Y-axis tick labels size
                ),
                legend=dict(
                    font=dict(size=16),       # Legend font size
                    orientation="h",          # Horizontal legend (optional)
                    yanchor="bottom", y=1.02,  # Move legend above the plot
                    xanchor="center", x=0.5
                ),
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)
            svg_buffer = io.BytesIO()
            fig.write_image(svg_buffer, format='svg')
            svg_data = svg_buffer.getvalue()
            st.download_button(
                label="📥 Download Plot as SVG",
                data=svg_data,
                file_name=f'{selected_mand2}.svg',
                mime='image/svg+xml',
                key="download2"
            )
