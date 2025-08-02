"""
Author: Yang Wang
Date: 2025-01-08 15:00:00
LastEditTime: 2025-07-13 18:12:30
LastEditors: Wenyu Ouyang
Description: GNN训练脚本，使用torchhydro框架
FilePath: \torchhydro\experiments\train_with_songliaorrevent_gnn.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import logging
import warnings

from torchhydro.configs.config import default_config_file, cmd, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro import SETTING

# 配置日志和警告 - 最大限度减少详细输出
logging.basicConfig(level=logging.WARNING)  # 只显示WARNING及以上级别的日志
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)  # 减少详细输出

# 抑制所有警告，包括RuntimeWarning
warnings.filterwarnings("ignore")

def create_simple_gnn_config():
    """创建简单的GNN配置"""
    # 获取默认配置
    config_data = default_config_file()

    # 设置实验目录
    project_name = os.path.join("gnn_experiment", "songliao_3h_test_")

    # 使用有有效数据的流域ID（基于3h数据）
    example_gage_ids = [
        "songliao_20800900",
        "songliao_20810200",
        "songliao_21100150",
        "songliao_21110150",
        #"songliao_21113800",
        "songliao_21401050",
        "songliao_21401550"
    ]

    # ==== 参数配置 ====
    warmup_length, hindcast_length, forecast_length = 0, 0, 160
    var_t = ["rain"]
    var_c = ["None"]  # 设置为 "None" 表示不使用静态特征
    station_cols = ["DRP"]  # 站点变量
    use_basin_features = False  # 设置为 False 只使用站点特征，True 则使用站点+流域特征
    
    # ==== 智能计算 in_channels ====
    total_features = len([col for col in station_cols if col != "None"])
    if use_basin_features:
        total_features += len([v for v in var_t if v != "None"])
        total_features += len([v for v in var_c if v != "None"])
    
    in_channels = (warmup_length + hindcast_length + forecast_length) * total_features
    output_size = hindcast_length + forecast_length

    args = cmd(
        sub=project_name,
        # 数据源配置 - 使用正确的松辽流域数据路径
        source_cfgs={
            "source_name": "stationhydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "time_unit": ["3h"],
        },
        # 站点配置
        gage_id=example_gage_ids,  # 指定训练站点
        # 模型配置
        model_name="GCN",  # 使用GCN模型
        model_hyperparam={
            "in_channels": in_channels,
            "hidden_channels": 128,
            "num_hidden": 3,
            "param_sharing": True,
            "edge_orientation": "downstream",
            "output_size": output_size,
            "root_gauge_idx": 1,
        },
        dataset="GNNDataset", scaler="DapengScaler", batch_size=8,
        warmup_length=warmup_length, hindcast_length=hindcast_length, forecast_length=forecast_length,
        min_time_unit="h", min_time_interval=3,
        var_t=var_t, var_c=var_c, var_out=["inflow", "flood_event"],
        train_epoch=60, save_epoch=10,
        train_period=["1980-01-01-02", "2019-12-31-23"],
        valid_period=["2013-01-01-02", "2019-12-31-23"],
        test_period=["2020-01-01-02", "2024-12-30-23"],
        rolling=-1,  # 不使用滚动窗口
        model_loader={"load_way": "best"},
        evaluator={"eval_way": "floodevent"}, which_first_tensor="batch",
        # 优化器配置
        opt="Adam",
        lr_scheduler={
            "lr": 0.001,
            "lr_factor": 0.9,
        },
        # 损失函数
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2.0,
            "flood_strategy": "weight",
            "device": [0],
        },
        # 其他配置
        # num_workers=4,
        # pin_memory=True,
        early_stopping=True,
        patience=5,
        calc_metrics=True,
        continue_train=False,
    )

    # 更新配置
    update_cfg(config_data, args)
    
    # 安全地添加 station_cfgs 参数，不覆盖现有配置
    if "station_cfgs" not in config_data["data_cfgs"]:
        config_data["data_cfgs"]["station_cfgs"] = {}
    config_data["data_cfgs"]["station_cfgs"].update({
        "station_cols": station_cols,
        "use_basin_features": use_basin_features
    })
    
    return config_data


def train_gnn_model():
    """训练GNN模型"""
    try:
        # 创建配置
        config_data = create_simple_gnn_config()

        # 开始训练
        print("开始训练GNN模型...")
        train_and_evaluate(config_data)

        print("训练完成！")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    """
    使用说明：
    1. 根据你的数据路径修改 source_cfgs 中的 source_path
    2. 根据你的站点修改 example_gage_ids 列表
    3. 根据你的数据特征修改 var_t, var_c, var_out 变量
    4. 根据需要调整模型超参数 model_hyperparam
    5. 根据需要调整训练参数（epoch数、学习率等）
    """
    train_gnn_model()
