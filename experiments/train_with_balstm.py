import os
import pandas as pd
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

basins = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"], "attributes", "basin_list.csv"
    )
).values
basins = [item for sublist in basins for item in sublist]
basins.sort()
df = pd.read_csv(
    os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "attributes",
        "grdc_attributes1.csv",
    )
)
df.drop(columns=["basin_id"], inplace=True)
var_c = df.columns.tolist()
var_t = [
    "tp",
    "d2m",
    "pev",
    "ro",
    "slhf",
    "sp",
    "sro",
    "swvl",
    "u10",
    "t2m",
    "v10",
    "sd",
    "sshf",
]


def create_config_LongTerm():
    project_name = os.path.join("train_with_LongTerm", "bs256,ep20,device1")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "longtermdataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
        },
        ctx=[2],
        model_name="BALSTM",
        model_hyperparam={
            "output_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.4,
            "input_size_dyn": 13,
            "input_size_glo": 106,
            "output_size": 1,
            "input_size_sta": len(var_c),  # len(var_c) max 195
        },
        model_loader={"load_way": "best"},
        gage_id=basins,
        batch_size=512,
        forecast_history=10,
        forecast_length=12,
        min_time_unit="ME",
        min_time_interval=1,
        var_t=var_t,
        var_c=var_c,
        var_out=["streamflow"],
        dataset="BALSTMDataset",
        sampler=None,
        scaler="DapengScaler",
        train_epoch=20,
        save_epoch=1,
        train_period=["1951-01-01", "2000-12-31"],
        test_period=["2001-01-01", "2002-10-31"],
        valid_period=["2001-01-01", "2002-10-31"],
        loss_func="NSELoss",
        opt="Adam",
        lr_scheduler={"lr": 0.001, "lr_factor": 0.1, "lr_patience": 1},
        which_first_tensor="batch",
        rolling=False,
        calc_metrics=True,
        metrics=["NSE", "RMSE", "R2"],
        early_stopping=False,
        patience=10,
        model_type="Normal",
    )
    update_cfg(config_data, args)
    return config_data


config = create_config_LongTerm()
train_and_evaluate(config)
