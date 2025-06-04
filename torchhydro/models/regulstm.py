import torch
import torch.nn as nn
import torch.nn.functional as F
import torchhydro.models.cudnnlstm
import matplotlib
matplotlib.use('Agg')  # 设置为Agg后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


class REGULstmModel(nn.Module):
    """
    创建水库影响的径流预报模型，耦合两个LSTM模型
    第一个模型的输入是15天累积降雨、前20天累积降雨、前10天累积降雨序列，输出是径流衰减系数，后一个模型输入是降雨和年份序列，输出是径流序列
    """

    def __init__(self,
                 regulation_input_features,
                 regulation_output_features,
                 regulation_hidden_states,
                 res_inflow_input_feature,
                 res_inflow_output_feature,
                 res_inflow__hidden_states):
        super(REGULstmModel, self).__init__()
        self.res_regulation_model = torchhydro.models.cudnnlstm.CudnnLstmModel(n_input_features=regulation_input_features,
                                                                               n_output_features=regulation_output_features,
                                                                               n_hidden_states=regulation_hidden_states)

        self.res_inflow_model = torchhydro.models.cudnnlstm.CudnnLstmModel(n_input_features=res_inflow_input_feature,
                                                                           n_output_features=res_inflow_output_feature,
                                                                           n_hidden_states=res_inflow__hidden_states)

    def forward(self, x_normalized, x_origin):
        precip_sum = x_normalized[:, :, 2:3]  # 前15天累积降雨
        is_all_nan = torch.isnan(precip_sum).all().item()
        print(f"precip_sum 是否全为 NaN: {is_all_nan}")
        # lstm_res_regulation_output = torch.nn.functional.softplus(self.res_regulation_model(precip_sum)) + 1
        # limit the range of lstm_res_regulation_output to (0, 2)
        lstm_res_regulation_output = torch.tanh(self.res_regulation_model(precip_sum)) + 1
        lstm_res_regulation_output_numpy = lstm_res_regulation_output.detach().cpu().numpy()
        lstm_res_regulation_output_data = lstm_res_regulation_output_numpy[:, :, :].squeeze()
        start_date = '2010-01-01'
        end_date = '2019-12-31'
        time = pd.date_range(start=start_date, end=end_date, freq='D')

        assert len(time) == len(lstm_res_regulation_output_data), f"时间列长度 ({len(time)}) 与数据列长度 ({len(lstm_res_regulation_output_data)}) 不一致"
        lstm_res_regulation_output_df = pd.DataFrame({'time': time, 'lstm_res_regulation_output': lstm_res_regulation_output_data})

        lstm_res_regulation_output_df_filename = "./lstm_res_regulation_output.csv"
        lstm_res_regulation_output_df.to_csv(lstm_res_regulation_output_df_filename, index=False)

        print(f"数据已保存为 {lstm_res_regulation_output_df_filename}")

        plt.figure(figsize=(10, 6))
        plt.plot(time, lstm_res_regulation_output_data, label='LSTM Regulation Output', color='blue', marker='o', linestyle='-')
        plt.xlabel('Time step')
        plt.ylabel('Regulation Output')
        plt.title('Visualization of lstm_res_regulation_output')

        # 设置时间格式为 '年-月-日'
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[7], bymonthday=1))  # 每年显示7月1日
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 格式为 '年-月-日'

        plt.xticks(rotation=45)  # 使时间轴标签更清晰
        plt.tight_layout()  # 自动调整布局，使标签不重叠
        plt.legend()
        lstm_res_regulation_output_image_filename = "./lstm_res_regulation_output.png"
        plt.savefig(lstm_res_regulation_output_image_filename)
        plt.close()

        print(f"图像已保存：{lstm_res_regulation_output_image_filename}")

        lstm_res_inflow_output = self.res_inflow_model(x_normalized[:, :, :2])  # 降雨和年份序列

        flow_prediction_output = lstm_res_regulation_output * lstm_res_inflow_output
        # lstm_flow_prediction_input = torch.cat((lstm_precipitation_fusion_output, x[:, :, 2:]), dim=2)
        # flow_prediction_output = self.flow_prediction_model(lstm_flow_prediction_input)
        return flow_prediction_output
