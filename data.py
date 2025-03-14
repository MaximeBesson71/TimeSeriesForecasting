import os
from collections import OrderedDict

import polars as pl

from pytpee import config, sqlutils
from pytpee.hpcutils import RemoteClient

item_list = ["Bloc_A", "Bloc_B", "Bloc_C", "building", "A001", "A002", "A101", "A102", "A103", "A104",
             "A105", "A201", "A202", "A203", "A204", "A205", "A301", "A302", "A303", "A401", "B001", "B002",
             "B101", "B102", "B103", "B201", "B202", "B203", "B301", "B302", "B303", "B401", "C001", "C002",
             "C101", "C102", "C103", "C104", "C201", "C202", "C203", "C204", "C301", "C302", "D001", "weather"]

exclude_items = ["Bloc_A", "Bloc_B", "Bloc_C", "building", "weather"]


meters_dict = {"Compteur_chauffage": "energy kWh", "Compteur_electrique_coffret": "energy kWh",
               "Compteur_ECS_tot": "volume m3", "Compteur_EF": "volume m3",
               "Compteur_electrique_double-flux": "corr energy kWh"}


meters_precision = {"Compteur_electrique_coffret": 0.1, "Compteur_electrique_double-flux": 0.1,
                    "Compteur_chauffage": 1.0, "Compteur_ECS_tot": 0.01, "Compteur_EF": 0.01}

meters_exclude = {"A201": ["Compteur_electrique_double-flux", "Compteur_ECS_tot"], "A203": ["Compteur_electrique_double-flux"],
                  "C101": ["Compteur_ECS_tot"], "C103": ["Compteur_ECS_tot", "Compteur_electrique_coffret", "Compteur_EF", "Compteur_electrique_double-flux"],
                  "C203": ["Compteur_electrique_coffret", "Compteur_electrique_double-flux", "Compteur_EF"]}


def GetIpcFile(apart_dir, meter, profiles_type):
    # Import data with Polars which is faster than Pandas
    df = (
        pl.scan_ipc(os.path.join(
            apart_dir, profiles_type, f"{meter}.feat"), memory_map=False)
        .with_columns(pl.col("date").dt.cast_time_unit(time_unit="ns"))
    ).collect()
    # Go back to pandas
    df = df.to_pandas()
    df = df.set_index("date")
    return df


def ConfigureDatabase(credentials='TPEE_PSQL'):
    # Configure Database access
    psql_host = config.pytpee_config[credentials]['host']
    psql_port = config.pytpee_config[credentials]['port']
    psql_user = config.pytpee_config[credentials]['user']
    psql_passwd = config.pytpee_config[credentials]['passwd']
    psql_dbname = config.pytpee_config[credentials]['dbname']
    psqlinstance = sqlutils.psqlhelper(
        dbname=psql_dbname, host=psql_host, port=psql_port, user=psql_user, passwd=psql_passwd)
    psqlurl_string = str(psqlinstance.db_string).replace("+psycopg2", "")
    psqlinstance.db_string = psqlurl_string
    lucia_config = config.pytpee_config["CENAERO_LUCIA"]
    hpc_host = lucia_config["host"]
    hpc_user = lucia_config["user"]
    hpc_passwd = lucia_config["passwd"]
    lucia_session = RemoteClient(
        host=hpc_host,
        user=hpc_user,
        passwd=hpc_passwd
    )
    return psqlinstance, lucia_session


def FixGasMeterIssue(df):
    # Fix issue #37 on gittest cenaero
    df = df.with_columns(pl.when((pl.col("date") < pl.datetime(2020, 10, 12, 11)) & (pl.col(
        "gas m3") < 10000.0)).then(pl.col("gas m3")*10.0).otherwise(pl.col("gas m3")).alias("gas m3"))
    df = df.with_columns(pl.when((pl.col("date") < pl.datetime(2020, 10, 12, 11)) & (pl.col(
        "gas m3") > 13000.0)).then(pl.col("gas m3")/10.0).otherwise(pl.col("gas m3")).alias("gas m3"))
    # Fix new issue
    df = df.with_columns(pl.when((pl.col("date") >= pl.datetime(2020, 3, 29, 4)) & (pl.col(
        "date") < pl.datetime(2020, 9, 11, 6))).then(None).otherwise(pl.col("gas m3")).alias("gas m3"))
    # explain how to get 10047 m3
    df = df.with_columns(pl.when(pl.col("date") >= pl.datetime(2020, 9, 11, 6)).then(
        pl.col("gas m3") + 10047.0).otherwise(pl.col("gas m3")).alias("gas m3"))
    # issue0132 5676 - 560 = 5116
    df = df.with_columns(pl.when((pl.col("date") <= pl.datetime(2020, 4, 1)) & (pl.col("date") > pl.datetime(
        2019, 11, 15))).then(pl.col("gas m3") + 5116.0).otherwise(pl.col("gas m3")).alias("gas m3"))
    return df


def FixCirculatorMeterIssue(df):
    df = df.with_columns((pl.col("date").diff().dt.seconds(
    )/3600.0).alias("sample_times"))  # compute time deltas
    df = df.with_columns(pl.when(pl.col("sample_times") == None).then(
        0).otherwise(pl.col("sample_times")).keep_name())
    df = df.with_columns((pl.max([pl.col("diffK"), 0.0])*4200.0*pl.max(
        [pl.col("volumeF m3/h"), -pl.col("volumeF m3/h")])/3.6).alias("Power W"))
    df = df.with_columns((((pl.col("Power W")+pl.col("Power W").shift())/2.0)
                          * pl.col("sample_times")).cumsum().alias("energy kWh"))
    df = df.with_columns(pl.when(pl.col("date") == pl.datetime(2019, 11, 15)).then(
        0.0).otherwise(pl.col("energy kWh")).alias("energy kWh"))
    df = df.with_columns(pl.when(pl.col("date") >= pl.datetime(2020, 3, 13, 9)).then(
        pl.col("energy kWh") + 1.050028e+07).otherwise(pl.col("energy kWh")).alias("energy kWh"))
    return df


def GetData(psqlinstance, lucia_session, start, end, out_dir):
    # Compteur ECS 1 (Always exist): Compteur Eau Chaude Sanitaire 1
    # Compteur ECS 2 (Only for several apartment): Compteur Eau Chaude Sanitaire 2
    # Compteur EF: Compteur Eau Froide
    # Compteur electrique coffret: Consommation totale electricite
    # Compteur electrique double-flux: Consommation electrique pour le groupe Ventilation Mecanique
    # Get table of instances in the database
    items = item_list
    table_name_list = psqlinstance._get_table_name_list()
    df_table_name = pl.DataFrame(table_name_list)
    df_table_name.write_csv(os.path.join(
        out_dir, "SQL_TableName_list.csv"), separator=";")
    # Put data from aparts_info, clusters_info and meters_info table into pandas dataframe
    for item in ["aparts_info", "clusters_info", "meters_info"]:
        df_data = pl.read_database(
            f"select * from {item}", psqlinstance.db_string)
        df_data.write_csv(os.path.join(out_dir, f"{item}.csv"), separator=";")
    # Get consumption data
    for item in items:
        if item not in ["Bloc_A", "Bloc_B", "Bloc_C", "building", "weather"]:
            apart_meters = OrderedDict(psqlinstance.engine.execute(
                """select type, meter_id from meters_info where location='{}';""".format(item)).fetchall())
            for key, val in apart_meters.items():
                key_name = key.replace(" ", "_")
                os.makedirs(os.path.join(out_dir, item), exist_ok=True)
                if f"meter_{val}_modif" in table_name_list:
                    ############### HOT AND COLD WATER METERS ##################
                    if any(el in key.lower() for el in ["ecs", "ef"]):
                        if "ef" in key.lower() and item in ["C103", "C203"]:
                            pass
                        else:
                            # verify if column exist
                            if "volume,m3,inst-value,0,0,0" in psqlinstance._get_column_name_list(f"meter_{val}_modif"):
                                df_data = pl.read_database(
                                    "select distinct on (created) created as date,\"volume,m3,inst-value,0,0,0\" as \"volume m3\" "
                                    "from "f"meter_{val}_modif "
                                    "where " f"created >='{start}'::date and created <='{end}'::date "
                                    "order by created",
                                    psqlinstance.db_string
                                )
                            else:  # 100 % no column to parse from files
                                df_data = pl.read_database(
                                    "select distinct on (created) created as date "
                                    "from "f"meter_{val}_modif "
                                    "where " f"created >='{start}'::date and created <='{end}'::date "
                                    "order by created",
                                    psqlinstance.db_string
                                )
                                df_data = df_data.with_columns(
                                    pl.lit(None).alias("volume m3"))
                            df_data = df_data.with_columns(pl.when(pl.col("volume m3").min() < 0)
                                                           .then(pl.col("volume m3") - pl.col("volume m3").min())
                                                           .otherwise(pl.col("volume m3")).keep_name())
                    ############# ENERGY (HEAT AND ELEC) METERS #############
                    else:
                        if "electrique" in key.lower() and item in ["C103", "C203"]:
                            pass
                        else:
                            # verify if column exist
                            if "energy,Wh,inst-value,0,0,0" in psqlinstance._get_column_name_list(f"meter_{val}_modif"):
                                df_data = pl.read_database(
                                    # "select distinct on (created), created as date,\"energy,Wh,inst-value,0,0,0\"/1000 as \"energy kWh\" "
                                    "select distinct on (created) * "
                                    "from "f"meter_{val}_modif "
                                    "where " f"created >='{start}'::date and created <='{end}'::date "
                                    "order by created",
                                    psqlinstance.db_string
                                )
                            else:  # 100 % no column to parse from files
                                df_data = pl.read_database(
                                    # "select distinct on (created) created as date "
                                    "select distinct on (created) * "
                                    "from "f"meter_{val}_modif "
                                    "where " f"created >='{start}'::date and created <='{end}'::date "
                                    "order by created",
                                    psqlinstance.db_string
                                )
                                df_data = df_data.with_columns(
                                    pl.lit(None).alias("energy,Wh,inst-value,0,0,0"))
                            df_data = df_data.rename(
                                {"created": "date", "energy,Wh,inst-value,0,0,0": "energy kWh", "power,W,inst-value,0,0,0": "Power W"})
                            df_data = df_data.with_columns((pl.col("date").diff().dt.seconds(
                            )/3600.0).alias("sample_times"))  # compute time deltas
                            df_data = df_data.with_columns(pl.when(pl.col("sample_times") == None).then(
                                0).otherwise(pl.col("sample_times")).keep_name())
                            orig_energy = df_data.select(pl.min("energy kWh"))[
                                0, "energy kWh"]
                            if "electrique double-flux" in key.lower():
                                df_data = df_data.with_columns(
                                    (pl.col("Power W")*pl.col("sample_times")).cumsum().alias("corr energy kWh"))
                                orig_corr_energy = df_data.select(pl.min("corr energy kWh"))[
                                    0, "corr energy kWh"]
                                df_data = df_data.with_columns(
                                    (pl.col("corr energy kWh") - orig_corr_energy + orig_energy)/1000)
                            """
                            if "chauffage" in key.lower():
                                df_data = df_data.rename({"volume-flow,m3/h,inst-value,0,0,0": "volumeF m3/h", "volume,m3,inst-value,0,0,0": "volume m3",
                                                          "diff-temp,K,inst-value,0,0,0": "diffK"})
                                df_data = df_data.with_columns(
                                    (pl.col("volume m3").diff()/pl.col("sample_times")).alias("corr volumeF m3/h"))
                                df_data = df_data.with_columns((pl.max([pl.col("diffK"), 0.0])*4185.0*pl.max(
                                    [pl.col("corr volumeF m3/h"), -pl.col("corr volumeF m3/h")])/3.6).alias("corr Power W"))
                                df_data = df_data.with_columns(
                                    (pl.col("corr Power W")*pl.col("sample_times")).cumsum().alias("corr energy kWh"))
                                df_data = df_data.with_columns((((pl.col("corr Power W")+pl.col("corr Power W").shift(
                                ))/2.0)*pl.col("sample_times")).cumsum().alias("corr energy kWh bis"))
                                orig_corr_energy = df_data.select(pl.min("corr energy kWh"))[
                                    0, "corr energy kWh"]
                                df_data = df_data.with_columns(
                                    (pl.col("corr energy kWh") - orig_corr_energy + orig_energy)/1000)
                                orig_corr_energy_bis = df_data.select(pl.min("corr energy kWh bis"))[
                                    0, "corr energy kWh bis"]
                                df_data = df_data.with_columns(
                                    (pl.col("corr energy kWh bis") - orig_corr_energy_bis + orig_energy)/1000)
                            """
                            df_data = df_data.with_columns(
                                pl.col("energy kWh")/1000)
                            """
                            if "chauffage" in key.lower():
                                last_energy = df_data[-1, "energy kWh"]
                                last_corr_energy = df_data[-1,
                                                           "corr energy kWh"]
                                last_corr_energy_bis = df_data[-1,
                                                               "corr energy kWh bis"]
                                corr_error = (
                                    (last_corr_energy - last_energy)/last_energy)*100
                                corr_error_bis = (
                                    (last_corr_energy_bis - last_energy)/last_energy)*100
                                error_file = open(os.path.join(
                                    out_dir, item, f"{key_name}_corr_error.txt"), "w")
                                error_file.write(str(corr_error) + " \n")
                                error_file.write(str(corr_error_bis))
                                error_file.close()
                            """
                            # df_data = df_data.upsample(time_column = "date", every = "15m")
                    df_data = df_data.fill_nan(None)
                    df_data = df_data.with_columns(
                        pl.col("date").dt.cast_time_unit(time_unit="ms"))
                    df_data.write_csv(os.path.join(
                        out_dir, item, f"{key_name}.csv"), separator=";")
            meters = list(apart_meters.keys())
            if all(meter in " ".join(meters) for meter in ["ECS 1", "ECS 2"]):
                df_ecs_1 = pl.read_csv(os.path.join(
                    out_dir, item, "Compteur_ECS_1.csv"), separator=";")
                df_ecs_1 = df_ecs_1.drop_nulls()
                df_ecs_2 = pl.read_csv(os.path.join(
                    out_dir, item, "Compteur_ECS_2.csv"), separator=";")
                df_ecs_2 = df_ecs_2.drop_nulls()
                df_ecs_tot = df_ecs_1.join(df_ecs_2, on="date")
                df_ecs_tot = df_ecs_tot.with_columns(
                    (pl.col("volume m3")+pl.col("volume m3_right")).alias("volume m3"))
                df_ecs_tot = df_ecs_tot.drop("volume m3_right")
                df_ecs_tot.write_csv(os.path.join(
                    out_dir, item, "Compteur_ECS_tot.csv"), separator=";")
            else:
                os.rename(os.path.join(out_dir, item, "Compteur_ECS_1.csv"),
                          os.path.join(out_dir, item, "Compteur_ECS_tot.csv"))
        else:
            if item in ["Bloc_A", "Bloc_B", "Bloc_C"]:
                # Get circulateur data for building
                bloc_id = item.replace("_", " ")
                circulateur_meterID = psqlinstance.engine.execute(
                    f"select meter_id from meters_info where location = 'Circulateur {bloc_id}'").fetchone()[0]
                df_data = pl.read_database(
                    "select distinct on (created) * "
                    "from "f"meter_{circulateur_meterID}_modif "
                    "where " f"created >='{start}'::date and created <='{end}'::date "
                    "order by created",
                    psqlinstance.db_string
                )
                df_data = df_data.rename({"created": "date", "energy,Wh,inst-value,0,0,0": "energy kWh",
                                          "power,W,inst-value,0,0,0": "Power W", "volume-flow,m3/h,inst-value,0,0,0": "volumeF m3/h",
                                          "diff-temp,K,inst-value,0,0,0": "diffK"})
                df_data = df_data.fill_nan(None)
                if item == "Bloc_C":
                    df_data = FixCirculatorMeterIssue(df_data)
                df_data = df_data.with_columns(pl.col("energy kWh")/1000)
                df_data = df_data.with_columns(
                    pl.col("date").dt.cast_time_unit(time_unit="ms"))
                os.makedirs(os.path.join(out_dir, item), exist_ok=True)
                df_data.write_csv(os.path.join(
                    out_dir, item, f"Circulateur_{item}.csv"), separator=";")
            elif item == "building":
                # Get gas data for building
                building_gas_meterID = psqlinstance.engine.execute(
                    "select meter_id from meters_info where lower(location) like '%%gaz%%'").fetchone()[0]
                df_data = pl.read_database(
                    "select distinct on (created) created as date, \"volume,m3,inst-value,0,2,0\" as \"gas m3\",\"volume,m3,inst-value,0,1,0\" as \"dhw m3\" "
                    "from "f"meter_{building_gas_meterID}_modif "
                    "where " f"created >='{start}'::date and created <='{end}'::date "
                    "order by created",
                    psqlinstance.db_string
                )
                df_data = df_data.fill_nan(None)
                df_data = FixGasMeterIssue(df_data)
                df_data = df_data.with_columns(
                    pl.col("date").dt.cast_time_unit(time_unit="ms"))
                os.makedirs(os.path.join(out_dir, item), exist_ok=True)
                df_data.write_csv(os.path.join(
                    out_dir, item, "building_gas_dhw.csv"), separator=";")
            elif item == "weather":
                weather_dir = "/gpfs/projects/cenaero/p_walecities_enr/dataGede/backupData/ftp/nivre/weather"
                # Download all weather data (bug in the tpee code, has to be in the develop branch to work)
                lucia_session.sftp_get_all(
                    remote_path=weather_dir, local_path=out_dir)
                df_weather = (
                    pl.scan_csv(os.path.join(out_dir, "weather",
                                "*_gosselies.csv"), separator=";", dtypes={"DIRECT HRZ RADIATION (W/m2)": pl.Float32, "SNOW HEIGHT (cm)": pl.Int64, "RELATIVE HUMIDITY (%)": pl.Float32})
                    .select(pl.exclude("PRECIPITATION (mm)", "WEATHER TYPE CODE"))
                    # dtypes={"PRECIPITATION (mm)": pl.Float64}
                    # .select(["YEAR", "MONTH", "DAY", "HOUR", "TEMPERATURE AVERAGE (°C)"])
                    .with_columns(pl.datetime(pl.col("YEAR"), pl.col("MONTH"), pl.col("DAY"), pl.col("HOUR")).alias("date"))
                    .drop(["YEAR", "MONTH", "DAY", "HOUR", "STATION", "CODE", "WEATHER TYPE TEXT"])
                    .unique(maintain_order=False, subset="date", keep="last")
                    .sort("date")
                    # does not work, not the end date included!
                    .filter(pl.col("date").is_between(start, end, closed="both"))
                    .with_columns(pl.col("RELATIVE HUMIDITY (%)"). round(0))
                    .with_columns(pl.col("SNOW HEIGHT (cm)").fill_null(strategy="zero"))
                    # .select([pl.col("date"), pl.col("TEMPERATURE AVERAGE (°C)")])
                    .rename({"TEMPERATURE AVERAGE (°C)": "Temp GOSSELIES"})
                    .fill_nan(None)
                    .with_columns(pl.col("date").dt.cast_time_unit(time_unit="ms"))
                ).collect()
                # df_weather = df_weather.with_columns(
                #    pl.all().is_null().suffix("_isnull"))
                df_weather.write_csv(os.path.join(
                    out_dir, "weather", "weatherGOSSELIES.csv"), separator=";")
                # Configure Synco Database access
                df_synco_info = pl.read_database(
                    "select * from synco_scrap", psqlinstance.db_string)
                df_synco_info = df_synco_info.sort("date")
                df_synco_info = df_synco_info.filter(
                    pl.col("date").is_between(start, end, closed="both"))
                df_synco_info = df_synco_info.with_columns(
                    pl.col("date").dt.cast_time_unit(time_unit="ms"))
                df_synco_info = df_synco_info.rename(
                    {"temp_exterior": "Temp NIVRE"})
                df_synco_info.write_csv(os.path.join(
                    out_dir, "weather", "synco_meter_info.csv"), separator=";")
                # pd_synco = df_synco_info.to_pandas()
                # pd_irm = df_weather.to_pandas()
                df_join = df_synco_info[["date", "Temp NIVRE"]].join(
                    df_weather, on="date", how="outer")
                # df_join = df_join.with_columns(
                #    (pl.col("temp_exterior")-pl.col("Temp C")).alias("delta T"))
                df_join = df_join.sort("date")
                # pd_join = df_join.to_pandas()
                # mean = df_join["delta T"].mean() #2.18
                # std = df_join["delta T"].std() #2.20
                df_join.write_csv(os.path.join(
                    out_dir, "weather", "SYNCO_GOSSELIES.csv"), separator=";")
                df_scrap = pl.read_database(
                    f"select * from weather_scrap", psqlinstance.db_string)
                df_scrap.write_csv(os.path.join(
                    out_dir, "weather", f"weather_scrap.csv"), separator=";")
