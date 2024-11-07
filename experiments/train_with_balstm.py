import os
import pandas as pd
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

basins = pd.read_excel("/home/yichengsun/data/basin_list.xlsx").values
basins = [item for sublist in basins for item in sublist]

var_t =[
        "d2m",
        "pev",
        "ro",
        "slhf",
        "sp",
        "sro",
        "swvl",
        "tp",
        "u10",
        "t2m",
        "v10",
        "sd",
        "sshf",
    ]
var_c = [
        "sgr_dk_sav",
        "glc_pc_s06",
        "glc_pc_s07",
        "nli_ix_sav",
        "glc_pc_s04",
        "glc_pc_s05",
        "glc_pc_s02",
        "glc_pc_s03",
        "glc_pc_s01",
        "pet_mm_syr",
    ]

def create_config_LongTerm():
    project_name = os.path.join("train_with_LongTerm", "balstm")
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "longtermdataset",
            "source_path": SETTING["local_data_path"]["basins-longterm"],
        },
        ctx=[1],
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
        batch_size=32,
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
        train_epoch=1,
        save_epoch=1,
        train_period=["1951-01-01", "1970-12-31"],
        test_period=["1971-01-01", "1972-10-31"],
        # test_period=None,
        # valid_period=["1961-01-01", "1963-12-31"],
        valid_period=None,
        loss_func="NSELoss",
        opt="Adam",
        lr_scheduler={"lr": 0.0001},
        which_first_tensor="batch",
        rolling=False,
        calc_metrics=True,
        early_stopping=True,
        patience=1,
        model_type="Normal",
    )
    update_cfg(config_data, args)
    return config_data


config = create_config_LongTerm()
train_and_evaluate(config)
