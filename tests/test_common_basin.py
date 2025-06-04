import os

from torchhydro import CACHE_DIR
import xarray as xr
from torchhydro.configs.config import update_cfg
import pandas as pd
import torch
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import HydroErr as he
import seaborn as sns
import geopandas as gpd
from torchhydro import CACHE_DIR, SETTING
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi



def find_common_files_in_subdirs(main_dir, compare_dir):
    # 创建一个集合来存储找到的公共文件基本名称
    common_files = set()

    # 获取比较目录中的文件基本名称
    compare_files = {os.path.splitext(f)[0] for f in os.listdir(compare_dir)}

    # 遍历主目录及其所有子目录
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            # 获取文件的基本名称
            base_name = os.path.splitext(file)[0][:8]
            # 检查这个基本名称是否在比较目录的文件集合中
            if base_name in compare_files:
                # 如果是，添加到公共文件集合中
                common_files.add(base_name)

    return common_files

def test_file_list():

    folder_path = r"C:\Users\jgchu\AppData\Local\hydro\Cache\reservoirs\D8"
    output_txt = r"C:\Users\jgchu\AppData\Local\hydro\Cache\reservoirs\D8\input_list.txt"

    with open(output_txt, "w") as f:
        for file in os.listdir(folder_path):
            if file.endswith(".tif"):
                full_path = os.path.join(folder_path, file)
                f.write(full_path + "\n")


# # 定义主目录和比较目录的路径
# main_dir = r"C:\Users\jgchu\AppData\Local\hydro\Cache\gages\basin_mean_forcing\basin_mean_forcing\daymet"
# compare_dir = r"C:\Users\jgchu\AppData\Local\hydro\Cache\mopex"
#
# # 调用函数并打印结果
# common_file_names = find_common_files_in_subdirs(main_dir, compare_dir)
# # 将集合转换为 pandas DataFrame
# df = pd.DataFrame(list(common_file_names), columns=['Basin ID'])
#
# # 指定 Excel 文件路径
# excel_path =  r"C:\Users\jgchu\AppData\Local\hydro\Cache\mopex\commonBasin.xlsx"
#
# # 将 DataFrame 写入 Excel 文件
# df.to_excel(excel_path, index=False, engine='openpyxl')

# 读取训练数据
def test_train_data(fusion_lstm_args, config_data):
    update_cfg(config_data, fusion_lstm_args)
    target_cols = config_data["data_cfgs"]["target_cols"]
    sites_id = config_data["data_cfgs"]["object_ids"]
    time = config_data["data_cfgs"][f"t_range_train"]
    relevant_cols = config_data["data_cfgs"]["relevant_cols"]
    constant_cols = config_data["data_cfgs"]["constant_cols"]
    ts_gages = xr.open_dataset(CACHE_DIR.joinpath("gages_timeseries.nc"))
    ts_mopex = xr.open_dataset(CACHE_DIR.joinpath("mopex_timeseries.nc"))
    attr = xr.open_dataset(CACHE_DIR.joinpath("gages_attributes.nc"))
    target = ts_gages[target_cols].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_gages = ts_gages[relevant_cols[0]].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_mopex = ts_mopex[relevant_cols[1]].sel(basin=sites_id, time=slice(time[0], time[1]))
    forcing = ts_gages[relevant_cols[2:]].sel(basin=sites_id, time=slice(time[0], time[1]))
    constant = attr[constant_cols].sel(basin=sites_id)
    # null_data = target.isnull().sum(dim='time')
    # print(null_data)
    average_target_over_basin = target.mean(dim='time')
    average_p_gages_over_basin = p_gages.mean(dim='time')
    average_p_mopex_over_basin = p_mopex.mean(dim='time')
    average_forcing_over_basin = forcing.mean(dim='time')
    constant_over_basin = constant
    # average_target_over_basin_df = average_target_over_basin.to_dataframe()

    # average_forcing_over_basin_df = average_forcing_over_basin.to_dataframe()
    # average_constant_over_basin_df = average_constant_over_basin.to_dataframe()
    with pd.ExcelWriter('output_data.xlsx') as writer:
        df = average_target_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Target Average')
        # for var_name in average_target_over_basin.data_vars:
        #     df = average_target_over_basin[var_name].to_dataframe()
        #     df.to_excel(writer, sheet_name=var_name)
        df_gages = average_p_gages_over_basin.to_dataframe(name='Average P Gages')
        df_mopex = average_p_mopex_over_basin.to_dataframe(name='Average P Mopex')
        df_p_combined = pd.concat([df_gages, df_mopex], axis=1)
        df_p_combined.to_excel(writer, sheet_name='Combined P Data')
        df = average_forcing_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Forcing Average')
        df = constant_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Constant Data')
        # for var_name in average_forcing_over_basin.data_vars:
        #     df = average_forcing_over_basin[var_name].to_dataframe()
        #     df.to_excel(writer, sheet_name=var_name)

        # average_target_over_basin.to_excel(writer, sheet_name='Target Average')
        # average_p_gages_over_basin.to_excel(writer, sheet_name='Gages Precipitation Average')
        # average_p_mopex_over_basin.to_excel(writer, sheet_name='Mopex Precipitation Average')
        # average_forcing_over_basin.to_excel(writer, sheet_name='Forcing Average')
        # average_constant_over_basin.to_excel(writer, sheet_name='Constant Average')


# 读取nc径流数据
def test_nc_streamflow(fusion_lstm_args, config_data):
    update_cfg(config_data, fusion_lstm_args)
    target_cols = config_data["data_cfgs"]["target_cols"]
    sites_id = config_data["data_cfgs"]["object_ids"]
    time = config_data["data_cfgs"][f"t_range_train"]
    relevant_cols = config_data["data_cfgs"]["relevant_cols"]
    constant_cols = config_data["data_cfgs"]["constant_cols"]
    ts_gages = xr.open_dataset(CACHE_DIR.joinpath("gages_timeseries.nc"))
    ts_mopex = xr.open_dataset(CACHE_DIR.joinpath("mopex_timeseries.nc"))
    attr = xr.open_dataset(CACHE_DIR.joinpath("gages_attributes.nc"))
    target = ts_gages[target_cols].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_gages = ts_gages[relevant_cols[0]].sel(basin=sites_id, time=slice(time[0], time[1]))
    p_mopex = ts_mopex[relevant_cols[1]].sel(basin=sites_id, time=slice(time[0], time[1]))
    forcing = ts_gages[relevant_cols[2:]].sel(basin=sites_id, time=slice(time[0], time[1]))
    constant = attr[constant_cols].sel(basin=sites_id)

    # dates = pd.date_range(start=time[0], end=time[1], periods=366)
    basins = target.basin.size
    times = target.time.size
    results = []

    average_target_over_basin = target.mean(dim='time')
    average_p_gages_over_basin = p_gages.mean(dim='time')
    average_p_mopex_over_basin = p_mopex.mean(dim='time')
    average_forcing_over_basin = forcing.mean(dim='time')
    constant_over_basin = constant

    with pd.ExcelWriter('output_data.xlsx') as writer:
        df = average_target_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Target Average')
        # for var_name in average_target_over_basin.data_vars:
        #     df = average_target_over_basin[var_name].to_dataframe()
        #     df.to_excel(writer, sheet_name=var_name)
        df_gages = average_p_gages_over_basin.to_dataframe(name='Average P Gages')
        df_mopex = average_p_mopex_over_basin.to_dataframe(name='Average P Mopex')
        df_p_combined = pd.concat([df_gages, df_mopex], axis=1)
        df_p_combined.to_excel(writer, sheet_name='Combined P Data')
        df = average_forcing_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Forcing Average')
        df = constant_over_basin.to_dataframe()
        df.to_excel(writer, sheet_name='Constant Data')
        for basin in range(basins):
            basin_id = target.basin.values[basin]
            basin_data = target.streamflow.isel(basin=basin)

            # 计算有效日期范围，忽略 NaN
            valid_dates = basin_data.dropna('time').time
            if valid_dates.size > 0:
                time_range = (pd.to_datetime(valid_dates.min().values).strftime('%Y-%m-%d'),
                              pd.to_datetime(valid_dates.max().values).strftime('%Y-%m-%d'))
            else:
                time_range = (None, None)  # 如果全是NaN

            # 计算缺失率
            missing_rate = basin_data.isnull().mean().item()

            # 计算平均径流，忽略缺失值
            average_streamflow = basin_data.mean().item() if valid_dates.size > 0 else float('nan')
            results.append((basin_id, time_range, missing_rate, average_streamflow))

        df = pd.DataFrame(results, columns=['Basin', 'Time Range', 'Missing Rate', 'Average Streamflow'])

        # 按照流域编号排序
        df = df.sort_values(by='Basin').reset_index(drop=True)
        df.to_excel(writer, sheet_name='Streamflow Data')

def test_reservoir_streamflow_result_analysis():
    stats_per_model = {
        'NSE': [],
        'RMSE': []
    }
    q = []
    model_name = ["reservoir", "reservoir_inflow", "reservoir_regulation"]

    data = xr.open_dataset(os.path.join(os.getcwd(), "results", "rainflow_21100150.nc"))
    time_start = '2010-07-01'
    time_end = '2010-08-31'

    station_id = '21100150'

    prcipitation = data['precip'].sel(time=slice(time_start, time_end))
    flow = data['flow'].sel(time=slice(time_start, time_end))
    time = data['time'].sel(time=slice(time_start, time_end))
    q.append(flow.sel(station=station_id))

    for name in model_name:
        project_name = os.path.join("test_" + name, "exp001")
        obs_streamflow = xr.open_dataset(
            os.path.join(os.getcwd(), "results", "inflow16-reservoir16-regulation16", project_name, "epochmodel.pthflow_obs.nc"))
        pred_streamflow = xr.open_dataset(
            os.path.join(os.getcwd(), "results", "inflow16-reservoir16-regulation16", project_name, "epochmodel.pthflow_pred.nc"))

        obs_2015_2019 = obs_streamflow['flow'].sel(time=slice(time_start, time_end))
        pred_2015_2019 = pred_streamflow['flow'].sel(time=slice(time_start, time_end))

        # 计算NSE和RMSE
        nse = he.nse(obs_2015_2019.sel(station=station_id), pred_2015_2019.sel(station=station_id))
        rmse = he.rmse(obs_2015_2019.sel(station=station_id), pred_2015_2019.sel(station=station_id))

        q.append(pred_2015_2019.sel(station=station_id))

        stats_per_model['NSE'].append(nse)
        stats_per_model['RMSE'].append(rmse)

    print (stats_per_model)
    ps = prcipitation.sel(station=station_id)
    # q_combined = np.concatenate([q_element.squeeze().values for q_element in q], axis=0)
    fig, ax = plot_rainfall_runoff(time, ps, q, prcp_interval=20)
    fig.savefig("rainfall_runoff_plot_20100701_20100831.png", dpi=300)

def plot_rainfall_runoff(
    t,
    p,
    qs,
    fig_size=(8, 6),
    c_lst="rbkgcmy",
    leg_lst=["observation", "Value-lstm", "Value-lstm-regulation", "Value-lstms"],
    dash_lines=None,
    title=None,
    xlabel=None,
    ylabel=None,
    prcp_ylabel="prcp(mm/day)",
    linewidth=1,
    prcp_interval=1,
):
    """Plot rainfall and runoff in one figure

    Parameters
    ----------
    t : a np.array or a list of some np.array and the length of the list is same as the length of qs
        time series, better to be a list with the same length of qs
    p : np.array
        precipitation, a time series
    qs : a np.array or a list of some np.array
        streamflow, a list with multiple time series
    fig_size : tuple, optional
        figure size, by default (8, 6)
    c_lst : str, optional
        colors, by default "rbkgcmy"
    leg_lst : list, optional
        legends, by default None
    dash_lines : list, optional
        if a line is dash line, by default None
    title : str, optional
        the title of the figure, by default None
    xlabel : str, optional
        label of x axis, by default None
    ylabel : str, optional
        label of y axis, by default None
    linewidth : int, optional
        the width of lines, by default 1
    prcp_interval : int, optional
        the interval of precipitation, by default 20
    """
    fig, ax = plt.subplots(figsize=fig_size)
    if dash_lines is not None:
        assert isinstance(dash_lines, list)
    else:
        dash_lines = np.full(len(qs), False).tolist()
    for k in range(len(qs)):
        tt = t[k] if type(t) is list else t
        q = qs[k]
        leg_str = None
        if leg_lst is not None:
            leg_str = leg_lst[k]
        (line_i,) = ax.plot(tt, q, color=c_lst[k], label=leg_str, linewidth=linewidth)
        if dash_lines[k]:
            line_i.set_dashes([2, 2, 10, 2])

    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
    # Create second axes, in order to get the bars from the top you can multiply by -1
    ax2 = ax.twinx()
    # ax2.bar(tt, -p, color="b")
    ax2.fill_between(tt, 0, -p, step="mid", color="b", alpha=0.5)
    # ax2.plot(tt, -p, color="b", alpha=0.7, linewidth=1.5)

    # Now need to fix the axis labels
    # max_pre = max(p)
    max_pre = p.max().item()
    ax2.set_ylim(-max_pre * 5, 0)
    y2_ticks = np.arange(0, max_pre, prcp_interval)
    y2_ticklabels = [str(i) for i in y2_ticks]
    ax2.set_yticks(-1 * y2_ticks)
    ax2.set_yticklabels(y2_ticklabels, fontsize=16)
    # ax2.set_yticklabels([lab.get_text()[1:] for lab in ax2.get_yticklabels()])
    if title is not None:
        ax.set_title(title, loc="center", fontdict={"fontsize": 17})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel(prcp_ylabel, fontsize=8, loc="top")
    # ax2.set_ylabel("precipitation (mm/day)", fontsize=12, loc='top')
    # https://github.com/matplotlib/matplotlib/issues/12318
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(bbox_to_anchor=(0.01, 0.85), loc="upper left", fontsize=16)
    ax.grid()
    return fig, ax

# 统计径流结果
def test_streamflow_result_analysis():
    stats_per_basin = {
        'NSE': [],
        'RMSE': [],
        'R2': [],
        'KGE': []
    }

    stats_1983_1993 = {
        'NSE': [],
        'RMSE': [],
        'R2': [],
        'KGE': []
    }

    stats_1993_2003 = {
        'NSE': [],
        'RMSE': [],
        'R2': [],
        'KGE': []
    }

    dataset_name = ["fusion", "gages", "mopex"]
    for name in dataset_name:
        project_name = os.path.join("test_" + name, "exp001")
        obs_streamflow = xr.open_dataset(
            os.path.join(os.getcwd(), "results", project_name, "epochmodel_Ep30.pthflow_obs.nc")),
        pred_streamflow = xr.open_dataset(
            os.path.join(os.getcwd(), "results", project_name, "epochmodel_Ep30.pthflow_pred.nc")),

        obs_1983_1993 = obs_streamflow[0].sel(time=slice('1983-10-01', '1993-09-30'))
        pred_1983_1993 = pred_streamflow[0].sel(time=slice('1983-10-01', '1993-09-30'))
        obs_1993_2003 = obs_streamflow[0].sel(time=slice('1993-10-01', '2003-09-30'))
        pred_1993_2003 = pred_streamflow[0].sel(time=slice('1993-10-01', '2003-09-30'))

        for basin in range(obs_streamflow[0]['basin'].size):
            obs_flow = obs_streamflow[0]["streamflow"].isel(basin=basin)
            pred_flow = pred_streamflow[0]["streamflow"].isel(basin=basin)

            obs_flow_1983_1993 = obs_1983_1993["streamflow"].isel(basin=basin)
            pred_flow_1983_1993 = pred_1983_1993["streamflow"].isel(basin=basin)

            obs_flow_1993_2003 = obs_1993_2003["streamflow"].isel(basin=basin)
            pred_flow_1993_2003 = pred_1993_2003["streamflow"].isel(basin=basin)

            # 计算 NSE (Nash-Sutcliffe Efficiency)
            nse = he.nse(pred_flow, obs_flow)
            nse_1983_1993 = he.nse(pred_flow_1983_1993, obs_flow_1983_1993)
            nse_1993_2003 = he.nse(pred_flow_1993_2003, obs_flow_1993_2003)

            # 计算 RMSE (Root Mean Square Error)
            rmse = he.rmse(pred_flow, obs_flow)
            rmse_1983_1993 = he.rmse(pred_flow_1983_1993, obs_flow_1983_1993)
            rmse_1993_2003 = he.rmse(pred_flow_1993_2003, obs_flow_1993_2003)

            # 计算 R2 (R-squared)
            r2 = he.r_squared(pred_flow, obs_flow)
            r2_1983_1993 = he.r_squared(pred_flow_1983_1993, obs_flow_1983_1993)
            r2_1993_2003 = he.r_squared(pred_flow_1993_2003, obs_flow_1993_2003)

            # 计算 KGE (Kling-Gupta Efficiency)
            kge = he.kge_2009(pred_flow, obs_flow)
            kge_1983_1993 = he.kge_2009(pred_flow_1983_1993, obs_flow_1983_1993)
            kge_1993_2003 = he.kge_2009(pred_flow_1993_2003, obs_flow_1993_2003)

            # 将每个流域的指标保存到字典中
            stats_per_basin['NSE'].append(nse)
            stats_per_basin['RMSE'].append(rmse)
            stats_per_basin['R2'].append(r2)
            stats_per_basin['KGE'].append(kge)

            stats_1983_1993['NSE'].append(nse_1983_1993)
            stats_1983_1993['RMSE'].append(rmse_1983_1993)
            stats_1983_1993['R2'].append(r2_1983_1993)
            stats_1983_1993['KGE'].append(kge_1983_1993)

            stats_1993_2003['NSE'].append(nse_1993_2003)
            stats_1993_2003['RMSE'].append(rmse_1993_2003)
            stats_1993_2003['R2'].append(r2_1993_2003)
            stats_1993_2003['KGE'].append(kge_1993_2003)

        if name == "fusion":
            df_stats_fusion = pd.DataFrame(stats_per_basin)
            df_stats_1983_1993_fusion = pd.DataFrame(stats_1983_1993)
            df_stats_1993_2003_fusion = pd.DataFrame(stats_1993_2003)

            df_stats_fusion['Time Period'] = '1983-2003'
            df_stats_1983_1993_fusion['Time Period'] = '1983-1993'
            df_stats_1993_2003_fusion['Time Period'] = '1993-2003'
        elif name == "gages":
            df_stats_gages = pd.DataFrame(stats_per_basin)
            df_stats_1983_1993_gages = pd.DataFrame(stats_1983_1993)
            df_stats_1993_2003_gages = pd.DataFrame(stats_1993_2003)

            df_stats_gages['Time Period'] = '1983-2003'
            df_stats_1983_1993_gages['Time Period'] = '1983-1993'
            df_stats_1993_2003_gages['Time Period'] = '1993-2003'
        elif name == "mopex":
            df_stats_mopex = pd.DataFrame(stats_per_basin)
            df_stats_1983_1993_mopex = pd.DataFrame(stats_1983_1993)
            df_stats_1993_2003_mopex = pd.DataFrame(stats_1993_2003)

            df_stats_mopex['Time Period'] = '1983-2003'
            df_stats_1983_1993_mopex['Time Period'] = '1983-1993'
            df_stats_1993_2003_mopex['Time Period'] = '1993-2003'

    # df_combined = pd.concat([df_stats, df_stats_1983_1993, df_stats_1993_2003], ignore_index=True)
    df_combined_1983_2003 = pd.concat([df_stats_fusion, df_stats_gages, df_stats_mopex], ignore_index=True)
    df_combined_1983_1993 = pd.concat([df_stats_1983_1993_fusion, df_stats_1983_1993_gages, df_stats_1983_1993_mopex],
                                      ignore_index=True)
    df_combined_1993_2003 = pd.concat([df_stats_1993_2003_fusion, df_stats_1993_2003_gages, df_stats_1993_2003_mopex],
                                      ignore_index=True)

    df_combined_1983_2003['Dataset'] = ['Fusion'] * len(df_stats_fusion) + ['Gages'] * len(df_stats_gages) + [
        'Mopex'] * len(df_stats_mopex)
    df_combined_1983_1993['Dataset'] = ['Fusion'] * len(df_stats_1983_1993_fusion) + ['Gages'] * len(
        df_stats_1983_1993_gages) + ['Mopex'] * len(df_stats_1983_1993_mopex)
    df_combined_1993_2003['Dataset'] = ['Fusion'] * len(df_stats_1993_2003_fusion) + ['Gages'] * len(
        df_stats_1993_2003_gages) + ['Mopex'] * len(df_stats_1993_2003_mopex)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 绘制箱型图
    # plt.figure(figsize=(12, 8))

    #

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    # 第一行绘制 NSE
    sns.boxplot(x='Dataset', y='NSE', data=df_combined_1983_2003, ax=axes[0, 0], showfliers=False)
    axes[0, 0].set_title('NSE (1983-2003)')
    sns.boxplot(x='Dataset', y='NSE', data=df_combined_1983_1993, ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_title('NSE (1983-1993)')
    sns.boxplot(x='Dataset', y='NSE', data=df_combined_1993_2003, ax=axes[0, 2], showfliers=False)
    axes[0, 2].set_title('NSE (1993-2003)')

    # 第二行绘制 RMSE
    sns.boxplot(x='Dataset', y='RMSE', data=df_combined_1983_2003, ax=axes[1, 0], showfliers=False)
    axes[1, 0].set_title('RMSE (1983-2003)')
    sns.boxplot(x='Dataset', y='RMSE', data=df_combined_1983_1993, ax=axes[1, 1], showfliers=False)
    axes[1, 1].set_title('RMSE (1983-1993)')
    sns.boxplot(x='Dataset', y='RMSE', data=df_combined_1993_2003, ax=axes[1, 2], showfliers=False)
    axes[1, 2].set_title('RMSE (1993-2003)')

    # 第三行绘制 R2
    sns.boxplot(x='Dataset', y='R2', data=df_combined_1983_2003, ax=axes[2, 0], showfliers=False)
    axes[2, 0].set_title('R2 (1983-2003)')
    sns.boxplot(x='Dataset', y='R2', data=df_combined_1983_1993, ax=axes[2, 1], showfliers=False)
    axes[2, 1].set_title('R2 (1983-1993)')
    sns.boxplot(x='Dataset', y='R2', data=df_combined_1993_2003, ax=axes[2, 2], showfliers=False)
    axes[2, 2].set_title('R2 (1993-2003)')

    # 第四行绘制 KGE
    sns.boxplot(x='Dataset', y='KGE', data=df_combined_1983_2003, ax=axes[3, 0], showfliers=False)
    axes[3, 0].set_title('KGE (1983-2003)')
    sns.boxplot(x='Dataset', y='KGE', data=df_combined_1983_1993, ax=axes[3, 1], showfliers=False)
    axes[3, 1].set_title('KGE (1983-1993)')
    sns.boxplot(x='Dataset', y='KGE', data=df_combined_1993_2003, ax=axes[3, 2], showfliers=False)
    axes[3, 2].set_title('KGE (1993-2003)')

    # 调整布局
    plt.tight_layout()

    # 保存图像而不是显示
    plt.savefig('streamflow_statistics_comparison_boxplot.png')  # 保存为 PNG 文件

    # # 调整布局以避免重叠
    # plt.tight_layout()
    #
    # # 显示图表
    # plt.show()

    # print(df_stats)
    # print(df_stats_1983_1993)
    # print(df_stats_1993_2003)


# 站点统计
def test_basin_stations():
    # 加载流域 (面) shapefile 文件
    camelsUS_ChinaBasins = gpd.read_file(CACHE_DIR.joinpath("STATION", "美国+松辽+山东", "basins_shp.shp"))

    # 加载雨量站 (点) shapefile 文件
    camelsUS_rain_stations = gpd.read_file(CACHE_DIR.joinpath("STATION", "iowa_all_locs", "iowa_pp_stations_day.shp"))

    # 将雨量站投影到流域的坐标系
    camelsUS_rain_stations = camelsUS_rain_stations.to_crs(camelsUS_ChinaBasins.crs)

    # 使用空间连接，将雨量站与流域进行空间匹配
    stations_joined_basins = gpd.sjoin(camelsUS_rain_stations, camelsUS_ChinaBasins, how="inner", op="within")

    # 计算每个流域内的雨量站个数
    stations_per_basin = stations_joined_basins.groupby('BASIN_ID').size().reset_index(name='stations_count')

    # 检查流域数据的索引类型
    print(camelsUS_ChinaBasins.index.dtype)

    # 检查 stations_per_watershed 中 BASIN_ID 列的数据类型
    print(stations_per_basin['BASIN_ID'].dtype)

    # 确保 BASIN_ID 列是整数类型
    camelsUS_ChinaBasins['BASIN_ID'] = camelsUS_ChinaBasins['BASIN_ID'].astype(str)
    stations_per_basin['BASIN_ID'] = stations_per_basin['BASIN_ID'].astype(str)

    # 将雨量站个数与流域面积合并
    basin_stations_info = camelsUS_ChinaBasins.merge(stations_per_basin, left_on='BASIN_ID', right_on='BASIN_ID',
                                                     how='left')

    # 如果某个流域没有雨量站，填充 NaN 为 0
    basin_stations_info['stations_count'].fillna(0, inplace=True)

    # 计算雨量站密度（单位为平方公里/站）
    basin_stations_info['stations_density'] = basin_stations_info['AREA'] / basin_stations_info['stations_count']

    # 现在，将流域的属性与雨量站信息合并，生成详细表格
    basin_stations_result = stations_joined_basins[['BASIN_ID', 'ID', 'NAME', 'BEGINTS']].copy()
    basin_stations_result = basin_stations_result.merge(basin_stations_info[['BASIN_ID', 'AREA', 'country',
                                                                             'stations_count', 'stations_density']],
                                                        on='BASIN_ID')
    basin_stations_result.rename(columns={
        'BASIN_ID': '流域编号',
        'AREA': '流域面积',
        'country': '国家',
        'stations_count': '流域内雨量站个数',
        'stations_density': '流域雨量站密度',
        'ID': '雨量站编号',
        'NAME': '雨量站名称',
        'BEGINTS': '雨量站开始时间'
    }, inplace=True)

    # 将结果保存为 Excel 文件
    basin_stations_result.to_excel('basin_stations_info.xlsx', index=False, engine='openpyxl')

    # # 保存为 shapefile 文件
    # basin_stations_info.to_file('basin_with_stations_info.shp')

    # # 或者保存为 CSV 文件
    # basin_stations_info[['BASIN_ID', 'AREA', 'country', 'tz', 'stations_count', 'stations_density']].to_csv(
    #                     'basin_stations_info.csv', index=False)


# 读取nc站点降雨数据
def test_nc_rainfall():
    # 读取nc站点降雨数据
    # 读取nc文件
    nc_file = CACHE_DIR.joinpath("STATION", "iowa_prcp_data_day.nc")

    # 读取nc文件
    with xr.open_dataset(nc_file) as ds:
        # 打印数据集的基本信息
        print(ds)
        # 打印数据集的变量信息
        print(ds.variables)
        # 打印数据集的坐标信息
        print(ds.coords)
        # 打印数据集的维度信息
        print(ds.dims)
        # 打印数据集的属性信息
        print(ds.attrs)
        # 打印数据集的维度坐标信息
        print(ds.dims)
        # 打印数据集的维度坐标信息
        print(ds.coords)
        # 提取降水数据，站点ID，和时间

        prcp_data = ds['prcp_inch'] * 25.4  # 将英寸转换为毫米
        station_ids = ds['station_id'].values
        time_values = ds['utc_valid'].values

        # 创建一个 DataFrame 来存储每个站点的时间范围
        stations_time_ranges = []

        # 创建一个新的时间序列，将时间转换为日期，去掉时间部分
        date_values = pd.to_datetime(time_values).normalize()

        # 创建一个 DataArray，包含站点和日期
        prcp_data.coords['date'] = ('utc_valid', date_values)

        # 按日期聚合降雨量数据（将小时或分钟数据转换为日数据）
        # daily_prcp_data = prcp_data.groupby('date').sum(dim='utc_valid', skipna=False)
        daily_prcp_data = prcp_data.groupby('date').sum(dim='utc_valid')

        # 遍历所有站点
        for i, station in enumerate(station_ids):
            # 提取该站点的所有降水数据
            station_data = prcp_data.sel(station_id=station)

            # 计算缺失值数量和总数据点数量
            total_data_points = station_data.size
            missing_data_points = station_data.isnull().sum().item()  # 计算 NaN 的数量
            missing_rate = (missing_data_points / total_data_points) * 100

            # 找出该站点有数据的时间点（非 NaN 值的时间）
            valid_times = time_values[~station_data.isnull()]

            # 找到起始时间和结束时间
            if len(valid_times) > 0:
                start_time = pd.to_datetime(valid_times[0])
                end_time = pd.to_datetime(valid_times[-1])

                # 过滤出该站点在起始时间和结束时间之间的数据
                time_mask = (time_values >= start_time) & (time_values <= end_time)
                station_data_in_range = station_data[time_mask]

                # 计算有效时间范围内的数据点数和缺失值数
                total_data_points_in_range = station_data_in_range.size
                missing_data_points_in_range = station_data_in_range.isnull().sum().item()  # 计算 NaN 的数量

                # 计算缺失率
                missing_rate_in_range = (missing_data_points_in_range / total_data_points_in_range) * 100
            else:
                start_time, end_time, missing_rate_in_range = None, None, None  # 如果该站点没有数据

            stations_time_ranges.append([station, start_time, end_time, missing_rate, missing_rate_in_range])

        # 将结果转换为 DataFrame
        stations_time_range_df = pd.DataFrame(stations_time_ranges,
                                              columns=['station_id', 'start_time', 'end_time', 'missing_rate',
                                                       'missing_rate_in_range'])

        print(stations_time_range_df)

        # 保存结果为 CSV 文件
        stations_time_range_df.to_csv('station_time_ranges_with_missing_rate.csv', index=False)

        basin_stations_info_file = 'basin_stations_info.xlsx'

        basin_stations_info_df = pd.read_excel(basin_stations_info_file)

        basin_stations_time_range_info = pd.merge(basin_stations_info_df, stations_time_range_df, left_on='雨量站编号',
                                                  right_on='station_id', how='left')

        filtered_basin_stations_time_range_info_df = basin_stations_time_range_info.dropna(subset=['start_time'])

        filtered_basin_stations_time_range_info_df.to_csv('filtered_basin_stations_time_range_info_df.csv', index=False,
                                                          encoding='utf-8-sig')

        time_values = pd.to_datetime(time_values)

        # 设置固定的 X 轴范围
        start_time = pd.to_datetime("2015-01-01 00:00")
        end_time = pd.to_datetime("2024-05-26 00:00")

        station_rainfall_distributions_file = 'station_rainfall_distributions.pdf'

        with PdfPages(station_rainfall_distributions_file) as pdf:
            # 为每个雨量站绘制降雨数据的时间分布图
            for index, row in filtered_basin_stations_time_range_info_df.iterrows():
                station_id = row['雨量站编号']

                # 提取该雨量站的降雨数据
                station_prcp_data = prcp_data.sel(station_id=station_id).values

                # 绘制该雨量站的降雨时间分布图
                plt.figure(figsize=(10, 4))
                plt.plot(time_values, station_prcp_data, label=f'Station {station_id}')
                plt.title(f"Rainfall Time Distribution for Station {station_id} (in mm)")
                plt.xlabel("Date")
                plt.ylabel("Precipitation (mm)")
                plt.xlim(start_time, end_time)  # 设置 X 轴范围
                plt.legend()
                plt.grid(True)

                # 保存当前图表到 PDF 文件
                pdf.savefig()  # 保存当前图表到 PDF
                plt.close()  # 关闭当前图表以节省内存
        print(f"All station rainfall time distribution charts are saved in {station_rainfall_distributions_file}")

        # 创建一个结果列表，存储每个站点的统计结果
        zero_sequence_result = []

        # 遍历每个雨量站
        for index, row in filtered_basin_stations_time_range_info_df.iterrows():
            station_id = row['雨量站编号']

            # 提取该站点的日降雨数据
            station_prcp_data = daily_prcp_data.sel(station_id=station_id).values

            zero_lengths = []
            current_zero_length = 0

            # # 找出日雨量连续为 0 的天数
            # zero_sequences = np.split(station_prcp_data, np.where(station_prcp_data != 0)[0])
            # zero_lengths = [len(seq) for seq in zero_sequences if len(seq) > 0 and np.all(seq == 0)]
            # total_zero_sequences = len(zero_lengths)

            # 遍历降雨数据，统计连续为0的日数
            for value in station_prcp_data:
                if value == 0:
                    current_zero_length += 1  # 如果降雨量为0，增加计数
                else:
                    if current_zero_length > 0:
                        zero_lengths.append(current_zero_length)  # 如果当前计数器不为0，记录当前长度
                    current_zero_length = 0  # 重置计数器

            # 如果最后一个序列是连续为0的，确保它被记录
            if current_zero_length > 0:
                zero_lengths.append(current_zero_length)

            # 统计总的连续为0的序列个数
            total_zero_sequences = len(zero_lengths)

            # 将统计结果保存到结果列表中
            zero_sequence_result.append({
                'station_id': station_id,
                'total_zero_sequences': total_zero_sequences,
                'zero_lengths': zero_lengths
            })

        # 将结果转换为 DataFrame
        zero_sequence_result_df = pd.DataFrame(zero_sequence_result)

        # 查看结果
        print(zero_sequence_result_df)

        # 保存结果为 CSV 文件
        zero_sequence_result_df.to_csv('station_zero_rainfall_sequences.csv', index=False)

        daily_station_rainfall_distributions_file = 'daily_station_rainfall_distributions.pdf'

        # 设置固定的 X 轴范围
        start_time = pd.to_datetime("2015-01-01")
        end_time = pd.to_datetime("2024-05-26")

        time_values = daily_prcp_data['date'].values
        time_values = pd.to_datetime(time_values)

        with PdfPages(daily_station_rainfall_distributions_file) as pdf:
            # 为每个雨量站绘制降雨数据的时间分布图
            for index, row in filtered_basin_stations_time_range_info_df.iterrows():
                station_id = row['雨量站编号']

                # 提取该雨量站的降雨数据
                station_prcp_data = daily_prcp_data.sel(station_id=station_id).values

                # 绘制该雨量站的降雨时间分布图
                plt.figure(figsize=(10, 4))
                plt.plot(time_values, station_prcp_data, label=f'Station {station_id}')
                plt.title(f"Daily Rainfall Time Distribution for Station {station_id} (in mm)")
                plt.xlabel("Date")
                plt.ylabel("Precipitation (mm)")
                plt.xlim(start_time, end_time)  # 设置 X 轴范围
                plt.legend()
                plt.grid(True)

                # 保存当前图表到 PDF 文件
                pdf.savefig()  # 保存当前图表到 PDF
                plt.close()  # 关闭当前图表以节省内存
        print(f"All station rainfall time distribution charts are saved in {daily_station_rainfall_distributions_file}")

        # 将日期转换为年份
        daily_prcp_data.coords['year'] = daily_prcp_data['date.year']

        # 按年分组并求和以计算每个站点的年总降雨量
        # annual_prcp_data = daily_prcp_data.groupby('year').sum(dim='date', skipna=False)
        annual_prcp_data = daily_prcp_data.groupby('year').sum(dim='date')

        # 将年降雨量数据转换为 pandas DataFrame
        annual_prcp_df = annual_prcp_data.to_dataframe().reset_index()

        # 按 'station_id' 和 'year' 进行排序
        annual_prcp_sorted_df = annual_prcp_df.sort_values(by=['station_id', 'year'])

        # 使用 rename 方法将列名 'prcp_inch' 改为 'prcp_mm'
        annual_prcp_sorted_df = annual_prcp_sorted_df.rename(columns={'prcp_inch': 'prcp_mm'})

        # 保存排序后的年降雨量数据为 CSV 文件
        annual_prcp_sorted_df.to_csv('sorted_annual_prcp_per_station.csv', index=False)

        # 去除空值
        annual_prcp_sorted_df = annual_prcp_sorted_df.dropna(subset=['prcp_mm'])
        # 定义合理的年降雨量范围
        MIN_PRCP = 100  # 最小合理年降雨量（单位：毫米）
        MAX_PRCP = 3000  # 最大合理年降雨量（单位：毫米）

        # 筛选合理范围内的年降雨量数据
        filtered_annual_prcp_sorted_df = annual_prcp_sorted_df[(annual_prcp_sorted_df['prcp_mm'] >= MIN_PRCP) &
                                                               (annual_prcp_sorted_df['prcp_mm'] <= MAX_PRCP)]
        print("筛选后的数据：")
        print(filtered_annual_prcp_sorted_df)

        filtered_annual_prcp_sorted_df.to_csv('filtered_sorted_annual_prcp_per_station.csv', index=False)

        # 从 filtered_annual_prcp_sorted_df 获取筛选出的站点列表
        filtered_annual_prcp_sorted_station_ids = filtered_annual_prcp_sorted_df['station_id'].unique()

        # 将 prcp_data 转换为 DataFrame，以便后续分析
        prcp_df = prcp_data.to_dataframe().reset_index()
        prcp_df['timestamp'] = pd.to_datetime(prcp_df['utc_valid'])  # 确保时间为 datetime 格式

        # 从 prcp_data 中提取这些站点的原始数据
        filtered_annual_prcp_sorted_stations_data_df = prcp_df[prcp_df['station_id'].isin(filtered_annual_prcp_sorted_station_ids)]

        # 对每个站点应用时间尺度识别函数
        station_time_scales = filtered_annual_prcp_sorted_stations_data_df.groupby('station_id').apply(identify_time_scale_for_station)
        station_time_scales.columns = ['max_time_scale', 'max_time_interval']

        # 确定统一的最大时间尺度
        # 按时间尺度优先级排序
        time_scale_priority = ['Sub-minute', 'Minute', 'Hourly', '3-Hourly', '6-Hourly', 'Daily']
        station_time_scales['priority'] = station_time_scales['max_time_scale'].apply(lambda x: time_scale_priority.index(x))

        # 选择优先级最高（时间跨度最长）的时间尺度
        uniform_max_time_scale = station_time_scales.loc[station_time_scales['priority'].idxmax(), 'max_time_scale']

        # 输出每个站点的最大时间尺度、最大时间间隔以及统一时间尺度
        print("每个站点的最大时间尺度和最大时间间隔:")
        print(station_time_scales[['max_time_scale', 'max_time_interval']])
        print("\n统一的最大时间尺度:", uniform_max_time_scale)

        # 确保 station_id 是 DataFrame 的一列而不是索引
        station_time_scales = station_time_scales.reset_index()

        # 保存 station_time_scales DataFrame 到 CSV 文件
        station_time_scales[['station_id', 'max_time_scale', 'max_time_interval', 'priority']].to_csv("station_time_scales.csv", index=False)

        print("数据已保存到 station_time_scales.csv 文件中")

        basin_stations_result_df = pd.read_excel('basin_stations_info.xlsx')

        # 筛选流域对应的年降雨合理的雨量站
        filtered_basin_stations = basin_stations_result_df.merge(
            filtered_annual_prcp_sorted_df[['station_id']].drop_duplicates(),
            left_on='雨量站编号',  # `basin_stations_result_df` 中雨量站编号的列名
            right_on='station_id',
            how='inner'
        )

        # 提取每个流域对应的且年降雨量合理的雨量站编号和降雨数据
        extracted_data = []
        # 生成 1小时时间索引（2015年到2024年）
        start_date = "2015-01-01 00:00"
        end_date = "2024-12-31 23:00"
        hourly_time_index = pd.date_range(start=start_date, end=end_date, freq="1H")

        for _, row in filtered_basin_stations.iterrows():
            basin_id = row['流域编号']
            station_id = row['雨量站编号']
            # 获取该站点的合理年份列表
            valid_years = filtered_annual_prcp_sorted_df[filtered_annual_prcp_sorted_df['station_id'] == station_id][
                                                        'year'].unique()
            # 提取该站点的数据
            station_data = prcp_data.sel(station_id=station_id)
            # 提取合理年份数据
            station_data_filtered = station_data.sel(utc_valid=station_data['utc_valid'].dt.year.isin(valid_years))

            station_data_resampled = (
                    station_data_filtered
                    .resample(utc_valid="1H")  # 重采样到1小时尺度
                    .asfreq()  # 保持数据点的稀疏性
                    .reindex(utc_valid=hourly_time_index)  # 对齐到统一的时间索引，填充空值
            )

            # 确保所有新增时间点填充 NaN
            station_data_resampled = station_data_resampled.fillna(np.nan)

            # # 筛选出年份为 2015 至 2024 年的数据
            # station_data = station_data.sel(
            #     utc_valid=slice("2015-01-01", "2024-12-31")
            # )
            #
            # # 将数据重采样为 1 小时尺度并填充缺失值
            # station_data_hourly = station_data.resample(utc_valid='1H').asfreq().fillna(np.nan)

            # 转换为 DataFrame 并添加流域和站点信息
            df = station_data_resampled.to_dataframe().reset_index()
            df['basin_id'] = basin_id
            df['station_id'] = station_id
            # 将 'prcp_inch' 列名改为 'prcp_mm'
            df.rename(columns={'prcp_inch': 'prcp_mm'}, inplace=True)

            extracted_data.append(df)

        # 合并所有流域和站点的数据
        final_df = pd.concat(extracted_data, ignore_index=True)

        # 保存为 CSV 文件
        final_df.to_csv("extracted_hourly_rainfall_2015_2024.csv", index=False)
        print("数据已保存到 extracted_hourly_rainfall_2015_2024.csv 文件中")

        # 确保时间列是 datetime 格式
        final_df['utc_valid'] = pd.to_datetime(final_df['utc_valid'])

        # 将 DataFrame 转换为 xarray 数据集
        ds = xr.Dataset.from_dataframe(
            final_df.set_index(['basin_id', 'station_id', 'utc_valid'])
        )

        # 保存为 NetCDF 文件
        output_nc_file = "extracted_hourly_rainfall_2015_2024.nc"
        ds.to_netcdf(output_nc_file)

        print(f"数据已保存为 NetCDF 文件: {output_nc_file}")

        # 定义要保存的 PDF 文件路径
        pdf_file = 'station_annual_rainfall_distributions.pdf'

        # 使用 PdfPages 将多个图表保存到同一个 PDF 文件
        with PdfPages(pdf_file) as pdf:
            # 获取所有雨量站的列表
            stations = annual_prcp_sorted_df['station_id'].unique()

            # 遍历每个雨量站，绘制年雨量变化图
            for station in stations:
                # 获取该站点的年雨量数据
                station_data = annual_prcp_sorted_df[annual_prcp_sorted_df['station_id'] == station]

                # 创建图形
                plt.figure(figsize=(10, 6))
                plt.plot(station_data['year'], station_data['prcp_mm'], marker='o', label=f'Station {station}')
                plt.title(f"Annual Rainfall Time Distribution for Station {station} (in mm)")
                plt.xlabel("Year")
                plt.ylabel("Annual Precipitation (mm)")
                plt.xticks(range(2015, 2025))  # 设置 x 轴的刻度为2015到2024
                plt.xlim(2015, 2024)  # 确保 X 轴范围是 2015 到 2024
                # plt.xticks(station_data['year'])  # 设置 x 轴为年份
                plt.grid(True)
                plt.legend()

                # 保存当前图表到 PDF 文件
                pdf.savefig()  # 保存当前图表到 PDF
                plt.close()  # 关闭当前图表以节省内存

        print(f"All station annual rainfall time distribution charts are saved in {pdf_file}")

        station_id = 'ELEW1'
        station_prcp_data = prcp_data.sel(station_id=station_id)
        station_prcp_df = station_prcp_data.to_dataframe().reset_index()
        station_prcp_df.to_csv(f'{station_id}_original_time_series.csv', index=False)

        daily_station_prcp_data = daily_prcp_data.sel(station_id=station_id)
        daily_station_prcp_df = daily_station_prcp_data.to_dataframe().reset_index()
        daily_station_prcp_df.to_csv(f'{station_id}_daily_rainfall.csv', index=False)

        annual_station_prcp_data = annual_prcp_data.sel(station_id=station_id)
        annual_station_prcp_df = annual_station_prcp_data.to_dataframe().reset_index()
        annual_station_prcp_df.to_csv(f'{station_id}_annual_rainfall.csv', index=False)


# 识别每个站点的时间尺度
def identify_time_scale_for_station(station_data):
    # 去除降雨数据为空的记录
    non_null_data = station_data.dropna(subset=['prcp_inch']).copy()

    # 计算相邻时间戳之间的时间间隔
    time_diffs = non_null_data['timestamp'].diff().dropna().dt.total_seconds()

    # 获取最大时间间隔
    max_time_diff = time_diffs.max()

    # 判断时间尺度
    if max_time_diff >= 86400:  # >= 1天
        time_scale = 'Daily'
    elif max_time_diff >= 21600:  # >= 6小时
        time_scale = '6-Hourly'
    elif max_time_diff >= 10800:  # >= 3小时
        time_scale = '3-Hourly'
    elif max_time_diff >= 3600:  # >= 1小时
        time_scale = 'Hourly'
    elif max_time_diff >= 60:  # >= 1分钟
        time_scale = 'Minute'
    else:
        time_scale = 'Sub-minute'

    # 返回时间尺度和最大时间间隔
    return pd.Series([time_scale, max_time_diff])

# 面雨量计算
def test_weighted_rainfall():
    # 打开 NetCDF 文件
    nc_file = "extracted_hourly_rainfall_2015_2024.nc"
    ds = xr.open_dataset(nc_file)

    # 提取流域编号和站点编号
    selected_basins = np.unique(ds['basin_id'].values)  # 流域编号
    selected_stations = np.unique(ds['station_id'].values)  # 站点编号

    # 读取站点 Shape 文件
    stations_gdf = gpd.read_file(CACHE_DIR.joinpath("STATION", "iowa_all_locs", "iowa_pp_stations_day.shp"))
    selected_stations_gdf = stations_gdf[stations_gdf['station_id'].isin(selected_stations)]

    # 读取流域 shape 文件
    basins_gdf = gpd.read_file(CACHE_DIR.joinpath("STATION", "美国+松辽+山东", "basins_shp.shp"))
    selected_basins_gdf = basins_gdf[basins_gdf['basin_id'].isin(selected_basins)]
    selected_basins_gdf.to_file("selected_basins.shp")
    print("选中流域的 shape 文件已保存为: selected_basins.shp")

    # 空间关联，将站点与流域关联
    stations_with_basins = gpd.sjoin(selected_stations_gdf, selected_basins_gdf, how="inner", predicate="within")
    # 保存为新的站点 shape 文件
    stations_with_basins.to_file("stations_with_basins.shp")
    print("包含流域信息的站点分布 shape 文件已保存为: stations_with_basins.shp")

    # 初始化结果存储
    all_weighted_rainfall = []

    # 定义时间范围（2015-01-01 00:00 至 2024-12-31 23:00）
    start_time = "2015-01-01 00:00"
    end_time = "2024-12-31 23:00"
    time_range = pd.date_range(start=start_time, end=end_time, freq="1H")  # 1小时间隔

    # 遍历每个流域
    for basin_id, basin_row in selected_basins_gdf.iterrows():
        basin = gpd.GeoDataFrame([basin_row], crs=selected_basins_gdf.crs)  # 当前流域的 GeoDataFrame
        basin_stations = stations_with_basins[selected_basins_gdf['basin_id'] == basin_id]  # 选出属于该流域的站点

        # 遍历每个时间点
        for time in time_range:
            # 提取该时刻的雨量数据
            rainfall_df = ds.sel(utc_valid=time).to_dataframe().reset_index()
            rainfall_df = rainfall_df[rainfall_df['station_id'].isin(basin_stations['station_id'])]
            # 筛选降雨量不为空的站点
            rainfall_df = rainfall_df[~rainfall_df['prcp_mm'].isna()]

            if rainfall_df.empty:
                continue  # 如果没有有效站点，跳过

            # 仅使用有降雨量数据的站点生成泰森多边形
            active_stations = basin_stations[basin_stations['station_id'].isin(rainfall_df['station_id'])]

            # 调用 calculate_voronoi_polygons 生成泰森多边形
            thiesen_polygons = calculate_voronoi_polygons(active_stations, basin)

            # 调用 calculate_weighted_rainfall 计算面雨量
            weighted_rainfall = calculate_weighted_rainfall(thiesen_polygons, rainfall_df)

            # 添加流域和时间信息
            weighted_rainfall['basin_id'] = basin_id
            weighted_rainfall['utc_valid'] = time
            all_weighted_rainfall.append(weighted_rainfall)

    # 将所有流域数据合并为一个 DataFrame
    final_rainfall_df = pd.concat(all_weighted_rainfall, ignore_index=True)

    # 确保时间列是 datetime 格式
    final_rainfall_df['utc_valid'] = pd.to_datetime(final_rainfall_df['utc_valid'])

    # 转换 DataFrame 为 xarray.Dataset
    rainfall_ds = xr.Dataset.from_dataframe(final_rainfall_df.set_index(['basin_id', 'utc_valid']))

    # 保存为 NetCDF 文件
    output_nc_file = "basin_weighted_rainfall.nc"
    rainfall_ds.to_netcdf(output_nc_file)

    print(f"面雨量数据已保存为 NetCDF 文件: {output_nc_file}")

def calculate_voronoi_polygons(self, stations, basin):
    """
    计算泰森多边形并裁剪至流域边界。
    参数：
    stations - 位于流域内部的站点GeoDataFrame。
    basin - 流域shapefile的GeoDataFrame。

    返回：
    clipped_polygons - 裁剪后的泰森多边形GeoDataFrame。
    """
    if len(stations) < 2:
        stations["original_area"] = np.nan
        stations["clipped_area"] = np.nan
        stations["area_ratio"] = 1.0
        return stations

        # 获取流域边界的最小和最大坐标，构建边界框
        x_min, y_min, x_max, y_max = basin.total_bounds

        # 扩展边界框
        x_min -= 1.0 * (x_max - x_min)
        x_max += 1.0 * (x_max - x_min)
        y_min -= 1.0 * (y_max - y_min)
        y_max += 1.0 * (y_max - y_min)

        bounding_box = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        )

        # 提取站点坐标
        points = np.array([point.coords[0] for point in stations.geometry])

        # 将站点坐标与边界框点结合，确保Voronoi多边形覆盖整个流域
        points_extended = np.concatenate((points, bounding_box), axis=0)

        # 计算Voronoi图
        vor = Voronoi(points_extended)

        # 提取每个点对应的Voronoi区域
        regions = [vor.regions[vor.point_region[i]] for i in range(len(points))]

        # 生成多边形
        polygons = [
            Polygon([vor.vertices[i] for i in region if i != -1])
            for region in regions
            if -1 not in region
        ]

        # 创建GeoDataFrame
        gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=stations.crs)
        gdf_polygons["STCD"] = stations["STCD"].values
        gdf_polygons["original_area"] = gdf_polygons.geometry.area

        # 计算流域的总面积
        basin_area = basin.geometry.area.sum()
        print(f"Basin area: {basin_area}")

        # 计算原始泰森多边形的总面积
        total_original_area = gdf_polygons["original_area"].sum()
        print(f"Total original Voronoi polygons area: {total_original_area}")

        # 将多边形裁剪到流域边界
        clipped_polygons = gpd.clip(gdf_polygons, basin)
        clipped_polygons["clipped_area"] = clipped_polygons.geometry.area
        clipped_polygons["area_ratio"] = (
            clipped_polygons["clipped_area"] / clipped_polygons["clipped_area"].sum()
        )

        # 计算裁剪后泰森多边形的总面积
        total_clipped_area = clipped_polygons["clipped_area"].sum()
        print(f"Total clipped Voronoi polygons area: {total_clipped_area}")

        # 打印年度数据汇总并将其追加到日志文件中
        log_file = self.output_log
        with open(log_file, "a") as f:
            log_entries = [
                f"Basin area: {basin_area}",
                f"Total original Voronoi polygons area: {total_original_area}",
                f"Total clipped Voronoi polygons area: {total_clipped_area}",
            ]
            for entry in log_entries:
                print(entry)
                f.write(entry + "\n")

        return clipped_polygons


def calculate_weighted_rainfall(self, thiesen_polygons, rainfall_df):
    """
    计算加权平均降雨量。

    参数：
    thiesen_polygons - 泰森多边形GeoDataFrame。
    rainfall_df - 降雨数据DataFrame。

    返回：
    weighted_average_rainfall - 加权平均降雨量DataFrame。
    """
    thiesen_polygons["STCD"] = thiesen_polygons["STCD"].astype(str)
    rainfall_df["STCD"] = rainfall_df["STCD"].astype(str)

    # 合并泰森多边形和降雨数据
    merged_data = pd.merge(thiesen_polygons, rainfall_df, on="STCD")

    # 计算加权降雨量
    merged_data["weighted_rainfall"] = (
        merged_data["DRP"] * merged_data["area_ratio"]
    )

    # 按时间分组并计算加权平均降雨量
    weighted_average_rainfall = (
        merged_data.groupby("TM")["weighted_rainfall"].sum().reset_index()
    )

    return weighted_average_rainfall

def test_nc_reservoir_timeseries():
    # 读取nc水库时序数据
    # 读取nc文件
    nc_file = CACHE_DIR.joinpath("reservoirs", "reservoirs_timeseries.nc")

    # 读取nc文件
    with xr.open_dataset(nc_file) as ds:
        release_data = ds['Release']
        reservoir_ids = ds['Reservoir_ID'].values
        time_values = ds['Time'].values

        # 创建一个 DataFrame 来存储每个水库的时间范围
        reservoirs_time_ranges = []

        # 创建一个新的时间序列，将时间转换为日期，去掉时间部分
        date_values = pd.to_datetime(time_values).normalize()

        # 创建一个 DataArray，包含站点和日期
        release_data.coords['date'] = ('Time', date_values)

        # 按日期聚合降雨量数据（将小时或分钟数据转换为日数据）
        # daily_prcp_data = prcp_data.groupby('date').sum(dim='utc_valid', skipna=False)
        daily_release_data = release_data.groupby('date').sum(dim='Time')

        # 遍历所有站点
        for i, reservoir in enumerate(reservoir_ids):
            # 提取该站点的所有降水数据
            reservoir_data = release_data.sel(Reservoir_ID=reservoir)

            # 计算缺失值数量和总数据点数量
            total_data_points = reservoir_data.size
            missing_data_points = reservoir_data.isnull().sum().item()  # 计算 NaN 的数量
            missing_rate = (missing_data_points / total_data_points) * 100

            # 找出该站点有数据的时间点（非 NaN 值的时间）
            valid_times = time_values[~reservoir_data.isnull()]

            # 找到起始时间和结束时间
            if len(valid_times) > 0:
                start_time = pd.to_datetime(valid_times[0])
                end_time = pd.to_datetime(valid_times[-1])

                # 过滤出该站点在起始时间和结束时间之间的数据
                time_mask = (time_values >= start_time) & (time_values <= end_time)
                reservoir_data_in_range = reservoir_data[time_mask]

                # 计算有效时间范围内的数据点数和缺失值数
                total_data_points_in_range = reservoir_data_in_range.size
                missing_data_points_in_range = reservoir_data_in_range.isnull().sum().item()  # 计算 NaN 的数量

                # 计算缺失率
                missing_rate_in_range = (missing_data_points_in_range / total_data_points_in_range) * 100
            else:
                start_time, end_time, missing_rate_in_range = None, None, None  # 如果该站点没有数据

            reservoirs_time_ranges.append([reservoir, start_time, end_time, missing_rate, missing_rate_in_range])

        # 将结果转换为 DataFrame
        reservoirs_time_range_df = pd.DataFrame(reservoirs_time_ranges,
                                              columns=['reservoir_id', 'start_time', 'end_time', 'missing_rate',
                                                       'missing_rate_in_range'])

        print(reservoirs_time_range_df)

        # 保存结果为 CSV 文件
        reservoirs_time_range_df.to_csv('reservoir_time_ranges_with_missing_rate.csv', index=False)


