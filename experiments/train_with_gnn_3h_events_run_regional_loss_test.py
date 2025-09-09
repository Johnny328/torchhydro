"""
Author: Zilin li
Date: 2025-01-08 15:00:00
LastEditTime: 2025-07-13 18:12:30
Description: GNN训练脚本，支持空间敏感性测试（上中下游降水影响分析）
FilePath: \torchhydro\experiments\train_with_gnn_3h_events_run_regional_loss_test.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import logging
import warnings
import sys

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

def create_simple_gnn_config(test_mode="normal"):
    """创建简单的GNN配置
    
    Args:
        test_mode: 测试模式
            - "normal": 正常测试
            - "sim1": 上游降水敏感性测试（上游降水x2）
            - "sim2": 中游降水敏感性测试（中游降水x2）
            - "sim3": 下游降水敏感性测试（下游降水x2）
    """
    # 获取默认配置
    config_data = default_config_file()

    # 设置实验目录
    if test_mode == "normal":
        project_name = os.path.join("gnn_experiment_GCN_auto_1h", "songliao_3h_test_normal")
    else:
        project_name = os.path.join("gnn_experiment_GCN_auto_1h", f"songliao_3h_test_{test_mode}")

    # 使用有有效数据的流域ID（基于3h数据）
    # example_gage_ids = [
    #     "songliao_21110150"
    # ]    
    example_gage_ids = [
        "songliao_20800900",
        "songliao_20810200",
        "songliao_21100150",
        "songliao_21110150",
        #"songliao_21113800",
        "songliao_21401050",
        "songliao_21401550"
    ]

    # ==== 超简化的参数配置 - 解决预测平滑问题 ====
    # 1. 大幅缩短时间窗口
    warmup_length, hindcast_length, forecast_length = 90, 0, 1  # 90*3h历史, 24*3h预测

    # 2. 简化特征
    var_t = ["rain"]  # 气象变量
    var_c = ["None"]  # 暂时不用流域特征
    station_cols = ["DRP"]  # 站点变量
    use_basin_features = False  # 先不用流域特征，简化模型
    
    # ==== 智能计算 in_channels ====
    total_features = len([col for col in station_cols if col != "None"])
    if use_basin_features:
        total_features += len([v for v in var_t if v != "None"])
        total_features += len([v for v in var_c if v != "None"])
    
    # 计算时序相关参数
    seq_len = warmup_length + hindcast_length + forecast_length  # 总时间步长
    model_output_length = hindcast_length + forecast_length  # 模型输出长度
    num_features = total_features  # 每个时间步的特征数
    in_channels = seq_len * num_features  # 展平后的总特征数
    output_size = 1  # 只输出径流量

    args = cmd(
        sub=project_name,
        # 数据源配置 - 使用正确的松辽流域数据路径
        source_cfgs={
            "source_name": "stationhydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "time_unit": ["3h"],
        },
        station_cfgs={
            # 站点数据配置 - 使用3h数据中实际存在的变量
            "station_cols": ["DRP"],  # TM=温度, 从站点数据中选择变量
            "station_rm_nan": True,
            "station_time_units": ["3h"],
            "station_scaler_type": "DapengScaler",
            "use_basin_features": use_basin_features,   # （决定是否使用流域平均的一些属性）或者不设置（默认为True），如果是false，则只使用站点特征
            # 邻接矩阵配置
            "edge_orientation": "bidirectional",  # 邻接矩阵的方向{upstream, downstream,bidirectional}默认是upstream
            "use_adjacency": True,
            "adjacency_src_col": "ID",
            "adjacency_dst_col": "NEXTDOWNID", 
            "adjacency_edge_attr_cols": ["dist_hdn", "elev_diff", "strm_slope"],
            "adjacency_weight_col": None,  # 不使用权重，返回边属性
            "return_edge_weight": False,
            },  
        # 站点配置
        gage_id=example_gage_ids,  # 指定训练站点
        # 模型配置 - 改进的模型，借鉴SimpleLSTM的成功经验
        model_name="GCN",  # 使用改进的GCN模型
        model_hyperparam={
            "in_channels": in_channels,
            "hidden_channels": 128,   # 增加隐藏层维度，与SimpleLSTM对比
            "num_hidden": 4,          # 使用4层GNN
            "param_sharing": False,   # 不共享参数，保持模型容量
            "output_size": output_size,
            "aggregate_to_graph": True,  # True=图级输出(流域级径流), False=节点级输出
            # 强化时序建模参数
            "seq_len": seq_len,
            "num_features": num_features,
            "output_time": model_output_length,  # 输出hindcast+forecast长度
            "dropout": 0.1,  # 适中的dropout，与SimpleLSTM类似
            "use_temporal_modeling": True,  # 确保开启LSTM时序建模
        },
        dataset="GNNDataset", scaler="DapengScaler", batch_size=4,  # 小batch适合时序专注模型
        warmup_length=warmup_length, hindcast_length=hindcast_length, forecast_length=forecast_length,
        min_time_unit="h", min_time_interval=3,
        var_t=var_t, var_c=var_c, var_out=["inflow", "flood_event"],
        train_mode=True,  # 测试模式
        
        train_epoch=60, save_epoch=10,   # 减少epoch，更频繁保存便于调试
        train_period=["1980-01-01-02", "2012-12-31-23"],
        valid_period=["2013-01-01-02", "2019-12-31-23"],
        #test_period=["1980-01-01-02", "2024-12-31-23"],
        test_period=["2020-01-01-02", "2024-12-31-23"],
        rolling=-1,  # 不使用滚动窗口
        model_loader={"load_way": "latest"},  # 对应训练epoch
        evaluator={"eval_way": "floodevent"}, which_first_tensor="batch",
        # 优化器配置 - 更保守的设置
        opt="Adam",
        lr_scheduler={
            "lr": 0.0005,     # 降低学习率，防止训练不稳定
            "lr_factor": 0.7, # 更强的学习率衰减
            "lr_patience": 5,  # 增加patience
        },
        # 损失函数
        loss_func="FloodLoss",
        loss_param={
            "loss_func": "MSELoss",
            "flood_weight": 2,
            "flood_strategy": "weight",
            "device": [0],
        },
        # 其他配置
        # num_workers=4,
        # pin_memory=True,
        # early_stopping=True,
        # patience=5,
        # calc_metrics=True,
        # continue_train=False,
    )

    # 更新配置
    update_cfg(config_data, args)
    
    # 手动添加空间测试参数到training_cfgs（确保参数正确传递）
    if "training_cfgs" not in config_data:
        config_data["training_cfgs"] = {}
    config_data["training_cfgs"]["spatial_test_mode"] = test_mode
    
    return config_data


def train_gnn_model(test_mode="normal"):
    """训练GNN模型
    
    Args:
        test_mode: 测试模式，可选"normal", "sim1", "sim2", "sim3"
    """
    try:
        # 创建配置
        config_data = create_simple_gnn_config(test_mode=test_mode)

        # 开始训练/测试
        print(f"开始{test_mode}模式的GNN模型测试...")
        train_and_evaluate(config_data)

        print(f"{test_mode}模式完成！")

    except Exception as e:
        print(f"{test_mode}模式过程中出现错误: {e}")
        raise


def main():
    """主函数，支持多种运行模式"""
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode in ["normal", "sim1", "sim2", "sim3"]:
            # 运行单个测试模式
            train_gnn_model(test_mode=mode)
        else:
            print(f"未知模式: {mode}")
            print("可用模式: normal, sim1, sim2, sim3")
    else:
        # 默认运行正常模式
        train_gnn_model(test_mode="normal")


if __name__ == "__main__":
    """
    使用说明：
    1. 正常测试：
       python train_with_gnn_3h_events_run_regional_loss_test.py normal
       
    2. 空间敏感性测试：
       python train_with_gnn_3h_events_run_regional_loss_test.py sim1  # 上游降水x2
       python train_with_gnn_3h_events_run_regional_loss_test.py sim2  # 中游降水x2
       python train_with_gnn_3h_events_run_regional_loss_test.py sim3  # 下游降水x2
    
    配置说明：
    - 根据你的数据路径修改 source_cfgs 中的 source_path
    - 根据你的站点修改 example_gage_ids 列表  
    - 根据你的数据特征修改 var_t, var_c, var_out 变量
    - 根据需要调整模型超参数 model_hyperparam
    - 根据需要调整训练参数（epoch数、学习率等）
    
    空间敏感性测试说明：
    - sim1: 测试上游降水的重要性（将上游站点降水x2）
    - sim2: 测试中游降水的重要性（将中游站点降水x2）
    - sim3: 测试下游降水的重要性（将下游站点降水x2）
    - 通过对比不同情况下的预测性能，可以分析不同区域降水对径流的影响
    """
    main()