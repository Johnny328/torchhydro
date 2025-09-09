#!/usr/bin/env python3
"""
Normal mode comparison plotting script
Compare observed values with normal mode predictions

Paths:
- Observations: /home/lizilin/code/torchhydro/results/.../songliao_3h_test_normal/epoch60flow_obs.nc
- Predictions: /home/lizilin/code/torchhydro/results/.../songliao_3h_test_normal/epoch60flow_pred.nc
- Output: /home/lizilin/code/torchhydro/results/.../songliao_3h_test_normal/normal_comparison_plots/

Author: Assistant
Date: 2025-09-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 禁用matplotlib的字体警告
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Set font and figure styles - 使用默认字体避免警告
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_netcdf_data(file_path):
    """Load NetCDF file data"""
    try:
        ds = xr.open_dataset(file_path)
        
        # Extract data
        data_dict = {}
        for var in ds.data_vars:
            data_dict[var] = ds[var].values
        
        # Extract coordinate information
        if 'time' in ds.coords:
            data_dict['time'] = ds.coords['time'].values
        if 'basin' in ds.coords:
            data_dict['basin'] = ds.coords['basin'].values
            
        return data_dict
        
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return None

def load_normal_comparison_data(base_dir):
    """Load normal mode comparison data"""
    base_dir = Path(base_dir)
    
    # Define file paths
    files = {
        'obs': base_dir / "songliao_3h_test_normal" / "epoch100flow_obs.nc",
        'normal': base_dir / "songliao_3h_test_normal" / "epoch100flow_pred.nc"
    }
    
    # Check if files exist
    missing_files = []
    existing_files = []
    for name, path in files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
        else:
            existing_files.append(f"{name}: {path}")
    
    print("File status:")
    for file in existing_files:
        print(f"  ✓ {file}")
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  ✗ {file}")
    
    # Load data
    data = {}
    for name, path in files.items():
        if path.exists():
            print(f"Loading {name} data: {path}")
            data[name] = load_netcdf_data(path)
            if data[name] is None:
                print(f"Failed to load {name} data")
            else:
                # Print data structure for debugging
                print(f"  {name} variables: {list(data[name].keys())}")
                for var_name, var_data in data[name].items():
                    if hasattr(var_data, 'shape'):
                        print(f"    {var_name}: shape {var_data.shape}")
        else:
            print(f"Skipping missing file: {name}")
            data[name] = None
    
    return data

def find_flood_events(flow_data, time_array=None, min_duration=6, percentile_threshold=30):
    """Find flood events - very lenient version similar to original find_positive_flow_periods
    
    Parameters:
    -----------
    flow_data : np.ndarray
        Flow data with shape (n_basins, n_time)
    time_array : np.ndarray, optional
        Time array for naming events by year
    min_duration : int, default=6
        Minimum duration (timesteps) for a flood event
    percentile_threshold : float, default=30
        Very low percentile threshold for identifying flow events
    
    Returns:
    --------
    list : List of flood event dictionaries
    """
    
    if flow_data.ndim == 1:
        flow_data = flow_data.reshape(1, -1)
    
    n_basins, n_time = flow_data.shape
    all_periods = []
    
    for basin_idx in range(n_basins):
        basin_flow = flow_data[basin_idx, :]
        
        # Very simple threshold: just above 30th percentile or positive flow
        valid_flow = basin_flow[~np.isnan(basin_flow)]
        if len(valid_flow) < 5:  # Very lenient check
            continue
            
        # Use very low threshold - basically any significant positive flow
        flow_threshold = max(np.percentile(valid_flow, percentile_threshold), 0.01)
        
        print(f"Basin {basin_idx}: threshold={flow_threshold:.2f}")
        
        # Find periods above threshold (similar to original find_positive_flow_periods)
        above_threshold = basin_flow > flow_threshold
        periods = []
        
        start_idx = None
        for i, is_above in enumerate(above_threshold):
            if is_above and start_idx is None:
                start_idx = i
            elif not is_above and start_idx is not None:
                # End a period
                duration = i - start_idx
                if duration >= min_duration:
                    # Find peak within this period
                    period_flow = basin_flow[start_idx:i]
                    peak_idx = start_idx + np.argmax(period_flow)
                    peak_value = basin_flow[peak_idx]
                    
                    periods.append({
                        'basin': basin_idx,
                        'start': start_idx,
                        'end': i,
                        'duration': duration,
                        'peak_idx': peak_idx,
                        'peak_value': peak_value,
                        'threshold': flow_threshold
                    })
                start_idx = None
        
        # Handle case where period continues to the end
        if start_idx is not None:
            duration = n_time - start_idx
            if duration >= min_duration:
                period_flow = basin_flow[start_idx:]
                peak_idx = start_idx + np.argmax(period_flow)
                peak_value = basin_flow[peak_idx]
                
                periods.append({
                    'basin': basin_idx,
                    'start': start_idx,
                    'end': n_time,
                    'duration': duration,
                    'peak_idx': peak_idx,
                    'peak_value': peak_value,
                    'threshold': flow_threshold
                })
        
        # Sort periods by start time (chronological order)
        periods.sort(key=lambda x: x['start'])
        
        # Add year information if time_array is available
        if time_array is not None and len(time_array) == n_time:
            periods_with_year = []
            year_event_counts = {}
            
            for period in periods:
                try:
                    # Try to get actual year from time array
                    if hasattr(time_array[period['peak_idx']], 'year'):
                        year = time_array[period['peak_idx']].year
                    elif hasattr(time_array[0], 'decode'):
                        # Handle string time data
                        time_str = str(time_array[period['peak_idx']])
                        year = int(time_str[:4]) if len(time_str) >= 4 else 2020
                    else:
                        # Try pandas datetime conversion
                        import pandas as pd
                        try:
                            time_pd = pd.to_datetime(time_array[period['peak_idx']])
                            year = time_pd.year
                        except:
                            # Fallback: assume data starts from 2020 (based on user feedback)
                            year = 2020 + (period['peak_idx'] // (365*8))  # 8 timesteps per day for 3-hour data
                except:
                    # Final fallback
                    year = 2020 + (period['peak_idx'] // 2920)  # fallback assuming ~2920 timesteps per year
                
                # Count events per year
                if year not in year_event_counts:
                    year_event_counts[year] = 0
                year_event_counts[year] += 1
                
                period['year'] = year
                period['event_number'] = year_event_counts[year]
                period['event_name'] = f"{year}_{year_event_counts[year]}"
                periods_with_year.append(period)
            
            periods = periods_with_year
        else:
            # No time information, just number events sequentially
            for i, period in enumerate(periods):
                period['event_name'] = f"Event_{i+1}"
        
        print(f"  Found {len(periods)} flood events for basin {basin_idx}")
        all_periods.extend(periods)
    
    return all_periods

def calculate_metrics(obs, pred):
    """Calculate evaluation metrics"""
    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    if valid_mask.sum() < 2:
        return {'error': 'insufficient_data'}
    
    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]
    
    # NSE
    numerator = np.sum((obs_valid - pred_valid) ** 2)
    denominator = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
    nse = 1 - (numerator / denominator) if denominator > 0 else -np.inf
    
    # R²
    r2 = np.corrcoef(obs_valid, pred_valid)[0, 1] ** 2 if len(obs_valid) > 1 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean((obs_valid - pred_valid) ** 2))
    
    # MAE
    mae = np.mean(np.abs(obs_valid - pred_valid))
    
    # Peak comparison
    obs_peak = np.max(obs_valid)
    pred_peak = np.max(pred_valid)
    peak_error = abs(pred_peak - obs_peak) / obs_peak * 100 if obs_peak > 0 else 0
    
    return {
        'NSE': nse,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'obs_peak': obs_peak,
        'pred_peak': pred_peak,
        'peak_error_pct': peak_error
    }

def plot_normal_comparison_period(obs_flow, pred_flow, period, basin_names, time_array, save_dir):
    """Plot normal mode comparison for a single period"""
    
    basin_idx = period['basin']
    start_idx = period['start'] 
    end_idx = period['end']
    
    # Extract period data
    if time_array is not None:
        time_subset = time_array[start_idx:end_idx]
        # Try to convert to proper datetime for better x-axis display
        try:
            import pandas as pd
            time_subset_dt = pd.to_datetime(time_subset)
            use_datetime = True
        except:
            time_subset_dt = time_subset
            use_datetime = False
    else:
        time_subset_dt = range(start_idx, end_idx)
        use_datetime = False
    
    obs_subset = obs_flow[basin_idx, start_idx:end_idx]
    pred_subset = pred_flow[basin_idx, start_idx:end_idx]
    
    # Debug: Print data information
    print(f"  Debug - Basin {basin_idx}, Period {start_idx}-{end_idx}")
    print(f"  Obs shape: {obs_flow.shape}, Obs subset shape: {obs_subset.shape}")
    print(f"  Pred shape: {pred_flow.shape}, Pred subset shape: {pred_subset.shape}")
    print(f"  Obs subset range: {np.min(obs_subset):.3f} - {np.max(obs_subset):.3f}")
    print(f"  Pred subset range: {np.min(pred_subset):.3f} - {np.max(pred_subset):.3f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot observations and predictions
    ax.plot(time_subset_dt, obs_subset, 'k-', linewidth=2.5, label='Observed', alpha=0.8)
    ax.plot(time_subset_dt, pred_subset, '#1f77b4', linewidth=2, label='Predicted (Normal Mode)', alpha=0.9)
    
    # Calculate metrics
    metrics = calculate_metrics(obs_subset, pred_subset)
    
    # Set figure properties
    basin_name = basin_names[basin_idx] if basin_names is not None and len(basin_names) > basin_idx else f"Basin_{basin_idx}"
    event_name = period.get('event_name', f"Event_{start_idx}_{end_idx}")
    peak_value = period.get('peak_value', np.max(obs_subset))
    
    ax.set_title(f'{basin_name} - Normal Mode Comparison\n'
                f'Flood Event: {event_name} (Peak: {peak_value:.2f}, Duration: {period["duration"]} timesteps)', 
                fontsize=14, fontweight='bold')
    
    # Set x-label based on whether we have datetime data
    if use_datetime:
        ax.set_xlabel('Date', fontsize=12)
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax.set_xlabel('Time Step', fontsize=12)
    
    ax.set_ylabel('Flow Rate', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box
    if 'error' not in metrics:
        metrics_text = f"Evaluation Metrics:\n"
        metrics_text += f"NSE: {metrics['NSE']:.3f}\n"
        metrics_text += f"R²: {metrics['R2']:.3f}\n"
        metrics_text += f"RMSE: {metrics['RMSE']:.2f}\n"
        metrics_text += f"MAE: {metrics['MAE']:.2f}\n"
        metrics_text += f"Peak Error: {metrics['peak_error_pct']:.1f}%"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    event_name = period.get('event_name', f"period_{start_idx}_{end_idx}")
    filename = f"{basin_name}_{event_name}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def plot_normal_comparison(data, save_dir):
    """Plot normal mode comparison analysis"""
    
    save_dir = Path(save_dir)
    plots_dir = save_dir / "normal_comparison_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data
    obs_data = data['obs']
    pred_data = data['normal']
    
    if obs_data is None or pred_data is None:
        print("ERROR: Missing observation or prediction data")
        return
    
    # 尝试找到流量数据的变量名
    flow_var_names = ['inflow', 'flow', 'streamflow', 'discharge', 'qobs', 'qsim']
    
    # 提取观测数据
    obs_flow = None
    for var_name in flow_var_names:
        if var_name in obs_data:
            obs_flow = obs_data[var_name]
            print(f"Found observation flow data with variable name: {var_name}")
            break
    
    if obs_flow is None:
        print(f"Available variables in obs data: {list(obs_data.keys())}")
        # 使用第一个数据变量
        first_var = list(obs_data.keys())[0]
        obs_flow = obs_data[first_var]
        print(f"Using first available variable for observations: {first_var}")
    
    # 提取预测数据
    pred_flow = None
    for var_name in flow_var_names:
        if var_name in pred_data:
            pred_flow = pred_data[var_name]
            print(f"Found prediction flow data with variable name: {var_name}")
            break
    
    if pred_flow is None:
        print(f"Available variables in pred data: {list(pred_data.keys())}")
        # 使用第一个数据变量
        first_var = list(pred_data.keys())[0]
        pred_flow = pred_data[first_var]
        print(f"Using first available variable for predictions: {first_var}")
    
    # 检查数据形状
    print(f"Obs flow shape: {obs_flow.shape}")
    print(f"Pred flow shape: {pred_flow.shape}")
    print(f"Obs data range: {np.min(obs_flow):.3f} - {np.max(obs_flow):.3f}")
    print(f"Pred data range: {np.min(pred_flow):.3f} - {np.max(pred_flow):.3f}")
    
    if obs_flow.shape != pred_flow.shape:
        print(f"WARNING: Shape mismatch between observations {obs_flow.shape} and predictions {pred_flow.shape}")
        return
    
    time_array = obs_data.get('time')
    basin_names = obs_data.get('basin')
    
    # Process basin names
    if basin_names is not None:
        try:
            if hasattr(basin_names[0], 'decode'):
                basin_names = [name.decode('utf-8') if hasattr(name, 'decode') else str(name) 
                              for name in basin_names]
        except:
            basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    else:
        basin_names = [f"Basin_{i}" for i in range(obs_flow.shape[0])]
    
    print(f"Processing normal mode comparison for {len(basin_names)} basins:")
    for i, name in enumerate(basin_names):
        print(f"  {i}: {name}")
    
    # Find flood events with very lenient parameters
    periods = find_flood_events(obs_flow, time_array, min_duration=6, percentile_threshold=30)
    print(f"\nFound {len(periods)} flood events")
    
    if len(periods) == 0:
        print("No sufficiently significant flood events found")
        return
    
    # Plot each period
    all_metrics = []
    
    for period in periods:
        print(f"Plotting basin {basin_names[period['basin']]} period {period['start']}-{period['end']}")
        
        # Create basin subfolder
        basin_name = basin_names[period['basin']].replace(' ', '_')
        basin_dir = plots_dir / basin_name
        basin_dir.mkdir(exist_ok=True)
        
        metrics = plot_normal_comparison_period(
            obs_flow, pred_flow, period, basin_names, time_array, basin_dir
        )
        
        # Add period information to metrics
        if 'error' not in metrics:
            event_name = period.get('event_name', f"Event_{period['start']}_{period['end']}")
            metrics.update({
                'basin': basin_names[period['basin']],
                'event_name': event_name,
                'period_start': period['start'],
                'period_end': period['end'],
                'duration': period['duration'],
                'peak_value': period.get('peak_value', 'Unknown')
            })
            all_metrics.append(metrics)
    
    # Save overall metrics statistics
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = plots_dir / "normal_comparison_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Overall statistics
        print(f"\n=== Normal Mode Comparison Results ===")
        print(f"Total periods analyzed: {len(all_metrics)}")
        
        avg_nse = df['NSE'].mean()
        avg_r2 = df['R2'].mean()
        avg_rmse = df['RMSE'].mean()
        avg_mae = df['MAE'].mean()
        avg_peak_error = df['peak_error_pct'].mean()
        
        print(f"\nOverall average metrics:")
        print(f"NSE: {avg_nse:.3f} ± {df['NSE'].std():.3f}")
        print(f"R²: {avg_r2:.3f} ± {df['R2'].std():.3f}")
        print(f"RMSE: {avg_rmse:.3f} ± {df['RMSE'].std():.3f}")
        print(f"MAE: {avg_mae:.3f} ± {df['MAE'].std():.3f}")
        print(f"Peak Error: {avg_peak_error:.1f}% ± {df['peak_error_pct'].std():.1f}%")
        
        # Statistics by basin
        basin_stats = df.groupby('basin').agg({
            'NSE': ['count', 'mean', 'std'],
            'R2': ['mean', 'std'],
            'RMSE': ['mean', 'std'],
            'peak_error_pct': ['mean', 'std']
        }).round(3)
        
        print(f"\nPerformance by basin:")
        print(basin_stats)
        
        # Generate summary report
        summary_text = f"""Normal Mode Comparison Summary Report
        
Number of periods analyzed: {len(all_metrics)}
Number of basins processed: {len(set(df['basin']))}

Overall average metrics:
NSE: {avg_nse:.3f} ± {df['NSE'].std():.3f}
R²: {avg_r2:.3f} ± {df['R2'].std():.3f}
RMSE: {avg_rmse:.3f} ± {df['RMSE'].std():.3f}
MAE: {avg_mae:.3f} ± {df['MAE'].std():.3f}
Peak Error: {avg_peak_error:.1f}% ± {df['peak_error_pct'].std():.1f}%

Performance by basin:
"""
        for basin in set(df['basin']):
            basin_data = df[df['basin'] == basin]
            basin_avg_nse = basin_data['NSE'].mean()
            basin_avg_r2 = basin_data['R2'].mean()
            basin_avg_rmse = basin_data['RMSE'].mean()
            basin_avg_peak_error = basin_data['peak_error_pct'].mean()
            basin_n_events = len(basin_data)
            
            summary_text += f"""
{basin} ({basin_n_events} events):
  NSE: {basin_avg_nse:.3f}
  R²: {basin_avg_r2:.3f}
  RMSE: {basin_avg_rmse:.3f}
  Peak Error: {basin_avg_peak_error:.1f}%
"""
        
        # Save summary
        summary_path = plots_dir / "normal_comparison_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
    
    print(f"\nAll plots and analysis results saved to: {plots_dir}")

def main():
    """Main function"""
    # Set base directory - you can modify this path as needed
    #base_dir = Path("/home/lizilin/code/torchhydro/results/3d_gnn_experiment_flood_gcn_3d")
    base_dir = Path("/home/lizilin/code/torchhydro/results/3d_gnn_experiment_lstm_gnn")
    #base_dir = Path("/home/lizilin/code/torchhydro/results/3d_gnn_experiment_lstm_only")
    
    print("=== Normal Mode Comparison Analysis ===")
    print(f"Base directory: {base_dir}")
    print("Data paths:")
    print(f"  Observations: {base_dir}/songliao_3h_test_normal/epoch30flow_obs.nc")
    print(f"  Predictions: {base_dir}/songliao_3h_test_normal/epoch30flow_pred.nc")
    
    if not base_dir.exists():
        print(f"Base directory does not exist: {base_dir}")
        return
    
    # Load data
    print("\n1. Loading data...")
    data = load_normal_comparison_data(base_dir)
    
    if data is None:
        print("Data loading failed")
        return
    
    print("Data loaded successfully!")
    
    # Plot comparison analysis
    print("\n2. Plotting normal mode comparison...")
    save_dir = base_dir / "songliao_3h_test_normal"
    plot_normal_comparison(data, save_dir)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()
