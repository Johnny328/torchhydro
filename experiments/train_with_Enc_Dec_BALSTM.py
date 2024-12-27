import os
import pandas as pd
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

gage_id_file = os.path.join(
    SETTING["local_data_path"]["datasets-interim"],
    "attributes",
    "grdc_basin_id_178.csv",
)

basins = pd.read_csv(gage_id_file)['basin_id'].tolist()
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "attributes",
        "grdc_attr_178.csv",
    )
)
df.drop(columns=["basin_id"], inplace=True)
var_c = df.columns.tolist()
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "timeseries",
        "1MS",
        "GRDC_1112200.csv",
    )
)
var_t = df.columns.tolist()
var_t.remove("streamflow")
var_t.remove("time")


def create_config_LongTerm():
    project_name = os.path.join("train_with_LongTerm_Seq2Seq", "178")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "longtermdataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
        },
        ctx=[2],
        model_name="SimpleBALSTM_EncDec",
        model_hyperparam={
            "output_size": 1,
            "hidden_size": 128,
            "forecast_length": 12,
            # "dropout": 0.4,
            "input_size_dyn": len(var_t),
            "input_size_glo": 106,
            "de_input_size": len(var_c)+1+1,
            "output_size": 1,
            "input_size_sta": len(var_c),  # len(var_c) max 195
            "prec_window": 12,
        },
        model_loader={"load_way": "latest"},
        gage_id=basins,
        batch_size=512,
        forecast_history=12,
        forecast_length=12,
        min_time_unit="ME",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        dataset="EncDecBALSTMDataset",
        sampler=None,
        scaler="DapengScaler",
        train_epoch=20,
        save_epoch=1,
        train_period=["1952-01-01", "2020-12-31"],
        test_period=["1982-01-01", "1992-12-31"],
        valid_period=["1982-01-01", "1992-12-31"],
        loss_func="NSELoss",
        opt="Adam",
        lr_scheduler={"lr": 0.0001},
        which_first_tensor="batch",
        rolling=True,
        calc_metrics=False,
        # metrics=["NSE", "RMSE", "R2"],
        early_stopping=False,
        patience=2,
        model_type="Normal",
    )
    update_cfg(config_data, args)
    return config_data


config = create_config_LongTerm()
train_and_evaluate(config)
