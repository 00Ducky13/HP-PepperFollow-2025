import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from math import sqrt
import shutil
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

def create_tracking_plots(file_name):
    '''
    Loads tracking data from a CSV file and generates two plots:
    1. Absolute X-Y error vs. time.
    2. A map of the X-Y paths for the person and Pepper.
    
    Plots are saved as 'tracking_error_vs_time.png' and 'tracking_path_xy.png'.
    '''
    
    # Load the dataset
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check for required columns
    required_cols = ['timestamp', 'person_x', 'person_y', 'pepper_x', 'pepper_y']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain the following columns: {', '.join(required_cols)}")
        return

    # 1. Calculate Absolute Error (Euclidean distance in X-Y plane)
    # The error is the straight-line distance between (person_x, person_y) 
    # and (pepper_x, pepper_y)
    df['error_xy'] = np.sqrt((df['person_x'] - df['pepper_x'])**2 + (df['person_y'] - df['pepper_y'])**2)

    # 2. Plot 1: Absolute Error vs. Time
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['error_xy'], label='X-Y Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (m)')
        plt.title('Pepper Tracking Error (X-Y Plane) vs. Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('tracking_error_vs_time.png')
        plt.close() # Close the plot to free up memory
        print("Successfully saved 'tracking_error_vs_time.png'")
    except Exception as e:
        print(f"Error creating error vs. time plot: {e}")

    # 3. Plot 2: X-Y Path Map
    try:
        plt.figure(figsize=(10, 10))
        # Plot paths
        plt.plot(df['person_x'], df['person_y'], label='Person Path', color='blue', linestyle='-')
        plt.plot(df['pepper_x'], df['pepper_y'], label='Pepper Path', color='red', linestyle='--')

        # Plot start points
        plt.plot(df['person_x'].iloc[0], df['person_y'].iloc[0], 'bo', markersize=10, label='Person Start')
        plt.plot(df['pepper_x'].iloc[0], df['pepper_y'].iloc[0], 'ro', markersize=10, label='Pepper Start')

        # Plot end points
        plt.plot(df['person_x'].iloc[-1], df['person_y'].iloc[-1], 'bx', markersize=10, label='Person End')
        plt.plot(df['pepper_x'].iloc[-1], df['pepper_y'].iloc[-1], 'rx', markersize=10, label='Pepper End')

        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('Person vs. Pepper Path (X-Y Plane)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Ensure aspect ratio is equal so paths are not distorted
        plt.tight_layout()
        plt.savefig('tracking_path_xy.png')
        plt.close() # Close the plot
        print("Successfully saved 'tracking_path_xy.png'")
    except Exception as e:
        print(f"Error creating path map plot: {e}")

def create_tracking_plots_multiple(file_names):
    '''
    Loads tracking data from a CSV file and generates two plots:
    1. Absolute X-Y error vs. time.
    2. A map of the X-Y paths for the person and Pepper.
    
    Plots are saved as 'tracking_error_vs_time.png' and 'tracking_path_xy.png'.
    '''
    for i in range(len(file_names)):
        file_name = file_names[i]
        # Load the dataset
        try:
            df = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f"Error: The file '{file_name}' was not found.")
            return
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        # Check for required columns
        required_cols = ['timestamp', 'person_x', 'person_y', 'pepper_x', 'pepper_y']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: The CSV must contain the following columns: {', '.join(required_cols)}")
            return

        # 1. Calculate Absolute Error (Euclidean distance in X-Y plane)
        # The error is the straight-line distance between (person_x, person_y) 
        # and (pepper_x, pepper_y)
        df['error_xy'] = np.sqrt((df['person_x'] - df['pepper_x'])**2 + (df['person_y'] - df['pepper_y'])**2)

        # 2. Plot 1: Absolute Error vs. Time
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['error_xy'], label='X-Y Error')
            plt.xlabel('Time (s)')
            plt.ylabel('Absolute Error (m)')
            plt.title('Pepper Tracking Error (X-Y Plane) vs. Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('tracking_error_vs_time.png')
            plt.close() # Close the plot to free up memory
            print("Successfully saved 'tracking_error_vs_time.png'")
        except Exception as e:
            print(f"Error creating error vs. time plot: {e}")

        # 3. Plot 2: X-Y Path Map
        try:
            plt.figure(figsize=(10, 10))
            # Plot paths
            plt.plot(df['person_x'], df['person_y'], label='Person Path', color='blue', linestyle='-')
            plt.plot(df['pepper_x'], df['pepper_y'], label='Pepper Path', color='red', linestyle='--')

            # Plot start points
            plt.plot(df['person_x'].iloc[0], df['person_y'].iloc[0], 'bo', markersize=10, label='Person Start')
            plt.plot(df['pepper_x'].iloc[0], df['pepper_y'].iloc[0], 'ro', markersize=10, label='Pepper Start')

            # Plot end points
            plt.plot(df['person_x'].iloc[-1], df['person_y'].iloc[-1], 'bx', markersize=10, label='Person End')
            plt.plot(df['pepper_x'].iloc[-1], df['pepper_y'].iloc[-1], 'rx', markersize=10, label='Pepper End')

            plt.xlabel('X Coordinate (m)')
            plt.ylabel('Y Coordinate (m)')
            plt.title('Person vs. Pepper Path (X-Y Plane)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal') # Ensure aspect ratio is equal so paths are not distorted
            plt.tight_layout()
            plt.savefig('tracking_path_xy.png')
            
            # --- SYNTAX ERROR FIX ---
            # These two lines were moved inside the 'try' block
            plt.close() # Close the plot
            print("Successfully saved 'tracking_path_xy.png'")
        except Exception as e:
            print(f"Error creating path map plot: {e}")

def create_tracking_plots_compare(file_names):
    '''
    Loads tracking data from a CSV file and generates two plots:
    1. Absolute X-Y error vs. time.
    2. A map of the X-Y paths for the person and Pepper.
    
    Plots are saved as 'tracking_error_vs_time.png' and 'tracking_path_xy.png'.
    '''
    
    # Load the dataset
    df = []
    for i in range(len(file_names)):
        try:
            dftemp = pd.read_csv(file_names[i])
            df.append(dftemp)
        except FileNotFoundError:
            print(f"Error: The file '{file_names[i]}' was not found.")
            return
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        # Check for required columns
        required_cols = ['timestamp', 'person_x', 'person_y', 'pepper_x', 'pepper_y']
        if not all(col in dftemp.columns for col in required_cols):
            print(f"Error: The CSV must contain the following columns: {', '.join(required_cols)}")
            return
        

    # 1. Calculate Absolute Error (Euclidean distance in X-Y plane)
    # The error is the straight-line distance between (person_x, person_y) 
    # and (pepper_x, pepper_y)
    for j in range(len(df)):
        df[j]['error_xy'] = np.sqrt((df[j]['person_x'] - df[j]['pepper_x'])**2 + (df[j]['person_y'] - df[j]['pepper_y'])**2)

    # 2. Plot 1: Absolute Error vs. Time
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df[0]['timestamp'], df[0]['error_xy'], label='NaiveP',color='red')
        plt.plot(df[1]['timestamp'], df[1]['error_xy'], label='Waypoint',color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (m)')
        plt.title('Pepper Tracking Error (X-Y Plane) vs. Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('tracking_error_vs_time.png')
        plt.close() # Close the plot to free up memory
        print("Successfully saved 'tracking_error_vs_time.png'")
    except Exception as e:
        print(f"Error creating error vs. time plot: {e}")

    # 3. Plot 2: X-Y Path Map
    try:
        plt.figure(figsize=(10, 10))
        # Plot paths
        plt.plot(df[1]['person_x'], df[1]['person_y'], label='Person Path', color='gray', linestyle='--')
        plt.plot(df[0]['pepper_x'], df[0]['pepper_y'], label='NaiveP', color='red', linestyle='-')
        plt.plot(df[1]['pepper_x'], df[1]['pepper_y'], label='Waypoint', color='orange', linestyle='-')

        # Plot start points
        plt.plot(df[1]['person_x'].iloc[0], df[1]['person_y'].iloc[0], marker='o', color='gray', markersize=10, label='Person Start')
        plt.plot(df[0]['pepper_x'].iloc[0], df[0]['pepper_y'].iloc[0], marker='o', color='red', markersize=10, label='NaiveP Start')
        plt.plot(df[1]['pepper_x'].iloc[0], df[1]['person_y'].iloc[0], marker='o', color='orange', markersize=10, label='Waypoint Start')

        # Plot end points
        plt.plot(df[1]['person_x'].iloc[-1], df[1]['person_y'].iloc[-1], marker='x', color='gray', markersize=10, label='Person End')
        plt.plot(df[0]['pepper_x'].iloc[-1], df[0]['pepper_y'].iloc[-1], marker='x', color='red', markersize=10, label='NaiveP End')
        plt.plot(df[1]['pepper_x'].iloc[-1], df[1]['pepper_y'].iloc[-1], marker='x', color='orange', markersize=10, label='Waypoint End')

        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('Person vs. Pepper Path (X-Y Plane)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Ensure aspect ratio is equal so paths are not distorted
        plt.tight_layout()
        plt.savefig('tracking_path_xy.png')
        plt.close() # Close the plot
        print("Successfully saved 'tracking_path_xy.png'")
    except Exception as e:
        print(f"Error creating path map plot: {e}")

def create_tracking_plots_avg(file_names, algo_name, outputFolder):
    '''
    Loads tracking data from a CSV file and generates two plots:
    1. Absolute X-Y error vs. time.
    2. A map of the X-Y paths for the person and Pepper.
    
    Plots are saved as 'tracking_error_vs_time.png' and 'tracking_path_xy.png'.
    '''
    algo_colors = {"NaiveP":"red","Waypoint":"orange","Breadcrumbs":"green","BreadcrumbsTimedPop":"blue","HeadSwivel":"purple"}
    # Load the dataset
    df = []
    for i in range(len(file_names)):
        try:
            dftemp = pd.read_csv(file_names[i])
            # --- Check for 'target_acquired' column (NEW) ---
            if 'target_acquired' not in dftemp.columns:
                print(f"Warning: '{file_names[i]}' is missing 'target_acquired' column. Skipping for stats.")
                # Still add the file for plotting, but it won't be used for loss stats
                # If you want to skip it entirely, use 'continue'
                # continue
            df.append(dftemp)
        except FileNotFoundError:
            print(f"Error: The file '{file_names[i]}' was not found.")
            return
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        # Check for required columns
        required_cols = ['timestamp', 'person_x', 'person_y', 'pepper_x', 'pepper_y']
        if not all(col in dftemp.columns for col in required_cols):
            print(f"Error: The CSV must contain the following columns: {', '.join(required_cols)}")
            return
            
    if not df:
        print(f"No valid data files found for algorithm '{algo_name}'.")
        return

    dfAvg = df[0].copy(deep=True)
    dfAvg['person_x'] = 0.0
    dfAvg['person_y'] = 0.0
    dfAvg['pepper_x'] = 0.0
    dfAvg['pepper_y'] = 0.0
    dfAvg['error_xy'] = 0.0
    dfAvg['person_x_std'] = 0.0
    dfAvg['person_y_std'] = 0.0
    dfAvg['pepper_x_std'] = 0.0
    dfAvg['pepper_y_std'] = 0.0
    dfAvg['error_xy_std'] = 0.0

    # --- NEW: Lists to store per-run statistics ---
    all_mean_xy_errors = []
    all_loss_counts = []
    all_reacquisition_times = [] # Stores *all* individual loss durations


    # 1. Calculate Absolute Error and Stats
    for j in range(len(df)):
        dftemp = df[j] # More readable
        dftemp['error_xy'] = np.sqrt((dftemp['person_x'] - dftemp['pepper_x'])**2 + (dftemp['person_y'] - dftemp['pepper_y'])**2)
        
        # --- START: New Statistics Calculation (per run) ---
        
        # 1. XY Error
        all_mean_xy_errors.append(dftemp['error_xy'].mean())

        # 2. Target Loss & Reacquisition Time
        # Check if 'target_acquired' column exists and is valid
        if 'target_acquired' in dftemp.columns:
            dftemp = dftemp.sort_values('timestamp')
            dftemp['time_diff'] = dftemp['timestamp'].diff().fillna(0)
            dftemp['is_lost'] = ~dftemp['target_acquired'] # 'True' if lost

            # Create unique groups for consecutive 'is_lost' == True blocks
            dftemp['lost_group'] = (dftemp['is_lost'].diff() != 0).cumsum()
            
            # Filter for only the rows where the target is lost
            lost_periods = dftemp[dftemp['is_lost'] == True]
            
            if not lost_periods.empty:
                # Calculate the duration of each unique 'lost' event
                # We filter out durations of 0 which can happen at the start
                lost_durations = lost_periods.groupby('lost_group')['time_diff'].sum()
                lost_durations = lost_durations[lost_durations > 0] 
                
                # Store the count of lost events for this run
                all_loss_counts.append(len(lost_durations))
                
                # Add all durations from this run to the global list
                all_reacquisition_times.extend(lost_durations.tolist())
            else:
                # Target was never lost in this run
                all_loss_counts.append(0)
        else:
            # If column is missing, append 0
            all_loss_counts.append(0)
            
        # --- END: New Statistics Calculation (per run) ---

        # Original averaging logic for plots
        dfAvg['person_x'] = dfAvg['person_x'] + dftemp['person_x']
        dfAvg['person_y'] = dfAvg['person_y'] + dftemp['person_y']
        dfAvg['pepper_x'] = dfAvg['pepper_x'] + dftemp['pepper_x']
        dfAvg['pepper_y'] = dfAvg['pepper_y'] + dftemp['pepper_y']
        dfAvg['error_xy'] = dfAvg['error_xy'] + dftemp['error_xy']
        
    

    dfAvg['person_x'] = dfAvg['person_x'] / len(df)
    dfAvg['person_y'] = dfAvg['person_y'] / len(df)
    dfAvg['pepper_x'] = dfAvg['pepper_x'] / len(df)
    dfAvg['pepper_y'] = dfAvg['pepper_y'] / len(df)
    dfAvg['error_xy'] = dfAvg['error_xy'] / len(df)
    
    for j in range(len(df)):
    	dfAvg['person_x_std'] = dfAvg['person_x_std'] + (df[j]['person_x'] - dfAvg['person_x'])**2
    	dfAvg['person_y_std'] = dfAvg['person_y_std'] + (df[j]['person_y'] - dfAvg['person_y'])**2
    	dfAvg['pepper_x_std'] = dfAvg['pepper_x_std'] + (df[j]['pepper_x'] - dfAvg['pepper_x'])**2
    	dfAvg['pepper_y_std'] = dfAvg['pepper_y_std'] + (df[j]['pepper_y'] - dfAvg['pepper_y'])**2
    	dfAvg['error_xy_std'] = dfAvg['error_xy_std'] + (df[j]['error_xy'] - dfAvg['error_xy'])**2
    
    # Check for division by zero if only one file
    n_runs = len(df)
    if n_runs > 1:
        dfAvg['person_x_std'] = (dfAvg['person_x_std'] / (n_runs - 1))**0.5
        dfAvg['person_y_std'] = (dfAvg['person_y_std'] / (n_runs - 1))**0.5
        dfAvg['pepper_x_std'] = (dfAvg['pepper_x_std'] / (n_runs - 1))**0.5
        dfAvg['pepper_y_std'] = (dfAvg['pepper_y_std'] / (n_runs - 1))**0.5
        dfAvg['error_xy_std'] = (dfAvg['error_xy_std'] / (n_runs - 1))**0.5
    else:
        # If only one run, std dev is 0
        dfAvg['person_x_std'].values[:] = 0
        dfAvg['person_y_std'].values[:] = 0
        dfAvg['pepper_x_std'].values[:] = 0
        dfAvg['pepper_y_std'].values[:] = 0
        dfAvg['error_xy_std'].values[:] = 0

    # --- START: New Statistics Printing ---
    print(f"\n--- Statistics for Algorithm: {algo_name} ({n_runs} runs) ---")
    
    # 1. Overall XY Error (Mean of each run's mean error)
    if all_mean_xy_errors:
        mean_xy_error = np.mean(all_mean_xy_errors)
        std_xy_error = np.std(all_mean_xy_errors)
        print(f"Overall Avg XY Error: {mean_xy_error:.3f} m (Std: {std_xy_error:.3f} m)")
    else:
        print("Overall Avg XY Error: N/A")

    # 2. Target Loss Count (Mean of loss counts per run)
    if all_loss_counts:
        mean_loss_count = np.mean(all_loss_counts)
        std_loss_count = np.std(all_loss_counts)
        print(f"Avg Target Loss Count: {mean_loss_count:.2f} times (Std: {std_loss_count:.2f})")
    else:
        print("Avg Target Loss Count: N/A")

    # 3. Reacquisition Time (Mean of all individual loss durations)
    if all_reacquisition_times:
        mean_reacquire_time = np.mean(all_reacquisition_times)
        std_reacquire_time = np.std(all_reacquisition_times)
        print(f"Avg Reacquisition Time: {mean_reacquire_time:.3f} s (Std: {std_reacquire_time:.3f} s)")
    else:
        print("Avg Reacquisition Time: N/A (Target was never lost or no valid data)")
    
    print("---------------------------------------------------\n")
    # --- END: New Statistics Printing ---


    # 2. Plot 1: Absolute Error vs. Time
    try:
        plt.figure(figsize=(12, 6))
        #for i in range(len(df)):
        #	plt.plot(df[i]['timestamp'], df[i]['error_xy'],color=algo_colors[algo_name],alpha=0.5)
        upper_bound = dfAvg['error_xy'] + dfAvg['error_xy_std']
        lower_bound = dfAvg['error_xy'] - dfAvg['error_xy_std']	
        plt.plot(dfAvg['timestamp'], dfAvg['error_xy'],color=algo_colors[algo_name])
        plt.fill_between(dfAvg['timestamp'], lower_bound, upper_bound, color=algo_colors[algo_name], alpha=0.5, label='Standard Deviation')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (m)')
        #plt.title(f'Pepper Tracking Error (X-Y Plane) vs. Time for Algorithm {algo_name}')
        #plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{outputFolder}/tracking_error_vs_time_{algo_name}.png')
        plt.close() # Close the plot to free up memory
        print(f"Successfully saved '{outputFolder}/tracking_error_vs_time_{algo_name}.png'")
    except Exception as e:
        print(f"Error creating error vs. time plot: {e}")

    # 3. Plot 2: X-Y Path Map
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot paths
        ax.plot(dfAvg['person_x'], dfAvg['person_y'], label='Person Path', color='gray', linestyle='-')
        ax.plot(dfAvg['pepper_x'], dfAvg['pepper_y'], label=f'{algo_name}', color=algo_colors[algo_name], linestyle='-')
        
        # --- Plotting Standard Deviation as a "tube" ---
        # (Using the PolyCollection method from our previous chat)
        
        # --- Y-Deviation Polygons ---
        x = dfAvg['pepper_x'].values
        y = dfAvg['pepper_y'].values
        y_std = dfAvg['pepper_y_std'].values
        y_upper_points = np.stack([x, y + y_std], axis=1)
        y_lower_points = np.stack([x, y - y_std], axis=1)
        
        y_quads = []
        for i in range(len(x) - 1):
            quad = [y_upper_points[i], y_upper_points[i+1], y_lower_points[i+1], y_lower_points[i]]
            y_quads.append(quad)
        y_std_collection = PolyCollection(y_quads, facecolor=algo_colors[algo_name], alpha=0.3, edgecolors='none')
        ax.add_collection(y_std_collection)

        # --- X-Deviation Polygons ---
        x_std = dfAvg['pepper_x_std'].values
        x_upper_points = np.stack([x + x_std, y], axis=1)
        x_lower_points = np.stack([x - x_std, y], axis=1)
        
        x_quads = []
        for i in range(len(x) - 1):
            quad = [x_upper_points[i], x_upper_points[i+1], x_lower_points[i+1], x_lower_points[i]]
            x_quads.append(quad)
        x_std_collection = PolyCollection(x_quads, facecolor=algo_colors[algo_name], alpha=0.3, edgecolors='none')
        ax.add_collection(x_std_collection)
        
        # Plot dashed lines for individual runs
        for i in range(len(df)):
        	ax.plot(df[i]['pepper_x'], df[i]['pepper_y'], color=algo_colors[algo_name], alpha=0.5, linestyle='--')	

        # Plot start points
        ax.plot(dfAvg['person_x'].iloc[0], dfAvg['person_y'].iloc[0], marker='o', color='gray', markersize=10, label='Person Start')
        ax.plot(dfAvg['pepper_x'].iloc[0], dfAvg['pepper_y'].iloc[0], marker='o', color=algo_colors[algo_name], markersize=10, label=f'{algo_name} Start')
        
        # Plot end points
        ax.plot(dfAvg['person_x'].iloc[-1], dfAvg['person_y'].iloc[-1], marker='x', color='gray', markersize=10, label='Person End')
        ax.plot(dfAvg['pepper_x'].iloc[-1], dfAvg['pepper_y'].iloc[-1], marker='x', color=algo_colors[algo_name], markersize=10, label=f'{algo_name} End')
        
        # --- Create custom legend ---
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, label='Person Path'),
            Line2D([0], [0], color=algo_colors[algo_name], lw=2, label=f'{algo_name} Mean Path'),
            Line2D([0], [0], color=algo_colors[algo_name], lw=2, linestyle='--', alpha=0.5, label=f'{algo_name} Individual Runs'),
            Polygon([[0,0]], facecolor=algo_colors[algo_name], alpha=0.3, label='Standard Deviation (X & Y)'),
            Line2D([0], [0], marker='o', color='gray', label='Person Start', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='x', color='gray', label='Person End', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='o', color=algo_colors[algo_name], label=f'{algo_name} Start', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='x', color=algo_colors[algo_name], label=f'{algo_name} End', markersize=10, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='best')

        ax.set_xlabel('X Coordinate (m)',fontsize=16)
        ax.set_ylabel('Y Coordinate (m)',fontsize=16)
        ax.set_title(f'Person vs. Pepper Path (X-Y Plane) Average for Algorithm {algo_name}')
        ax.grid(True)
        ax.axis('equal') # Ensure aspect ratio is equal so paths are not distorted
        plt.tight_layout()
        plt.savefig(f'{outputFolder}/tracking_path_xy_{algo_name}.png')
        plt.close() # Close the plot
        print(f"Successfully saved '{outputFolder}/tracking_path_xy_{algo_name}.png'")
    except Exception as e:
        print(f"Error creating path map plot: {e}")
        
def create_tracking_plots_avg_compare(file_names,algo_names):
    '''
    Loads tracking data from a CSV file and generates two plots:
    1. Absolute X-Y error vs. time.
    2. A map of the X-Y paths for the person and Pepper.
    
    Plots are saved as 'tracking_error_vs_time.png' and 'tracking_path_xy.png'.
    '''
    algo_colors = {"NaiveP":"red","Waypoint":"orange","Breadcrumbs":"green","BreadcrumbsTimedPop":"blue","HeadSwivel":"purple"}
    # Load the dataset
    dfAvgList = []
    for j in range(len(file_names)):
        df = []
        for i in range(len(file_names[j])):
        	try:
        		dftemp = pd.read_csv(file_names[j][i])
        		df.append(dftemp)
        	except FileNotFoundError:
        		print(f"Error: The file '{file_names[j][i]}' was not found.")
        		return
        	except Exception as e:
        		print(f"Error loading file: {e}")
        		return
        	required_cols = ['timestamp', 'person_x', 'person_y', 'pepper_x', 'pepper_y']
        	if not all(col in dftemp.columns for col in required_cols):
        		print(f"Error: The CSV must contain the following columns: {', '.join(required_cols)}")
        		return
        
        if not df:
            print(f"No data for {algo_names[j]}")
            continue

        dfAvg = df[0].copy(deep=True)
        dfAvg['person_x'] = 0.0
        dfAvg['person_y'] = 0.0
        dfAvg['pepper_x'] = 0.0
        dfAvg['pepper_y'] = 0.0
        dfAvg['error_xy'] = 0.0
        dfAvg['person_x_std'] = 0.0
        dfAvg['person_y_std'] = 0.0
        dfAvg['pepper_x_std'] = 0.0
        dfAvg['pepper_y_std'] = 0.0
        dfAvg['error_xy_std'] = 0.0
        # --- NEW: Initialize new cumulative columns ---
        dfAvg['cumulative_reacquire_time'] = 0.0
        dfAvg['cumulative_loss_count'] = 0.0
        dfAvg['cumulative_reacquire_time_std'] = 0.0
        dfAvg['cumulative_loss_count_std'] = 0.0
        
        for k in range(len(df)): # Renamed inner loop variable
            dftemp = df[k] # Get reference to dataframe for this run
            dftemp['error_xy'] = np.sqrt((dftemp['person_x'] - dftemp['pepper_x'])**2 + (dftemp['person_y'] - dftemp['pepper_y'])**2)
            
            # --- NEW: Calculate cumulative stats for this run ---
            # Assumes data is already sorted by timestamp in the log file
            if 'target_acquired' in dftemp.columns:
                dftemp['time_diff'] = dftemp['timestamp'].diff().fillna(0)
                dftemp['is_lost'] = ~dftemp['target_acquired']
                
                # Cumulative Reacquisition Time
                dftemp['lost_time'] = dftemp['time_diff'] * dftemp['is_lost']
                dftemp['cumulative_reacquire_time'] = dftemp['lost_time'].cumsum()
                
                # Cumulative Loss Count
                dftemp['loss_event'] = (dftemp['is_lost'].shift(1) == False) & (dftemp['is_lost'] == True)
                dftemp['cumulative_loss_count'] = dftemp['loss_event'].cumsum()
            else:
                # Fill with 0 if column is missing
                dftemp['cumulative_reacquire_time'] = 0.0
                dftemp['cumulative_loss_count'] = 0
            
            # --- Add to average ---
            dfAvg['person_x'] = dfAvg['person_x'] + dftemp['person_x']
            dfAvg['person_y'] = dfAvg['person_y'] + dftemp['person_y']
            dfAvg['pepper_x'] = dfAvg['pepper_x'] + dftemp['pepper_x']
            dfAvg['pepper_y'] = dfAvg['pepper_y'] + dftemp['pepper_y']
            dfAvg['error_xy'] = dfAvg['error_xy'] + dftemp['error_xy']
            dfAvg['cumulative_reacquire_time'] = dfAvg['cumulative_reacquire_time'] + dftemp['cumulative_reacquire_time']
            dfAvg['cumulative_loss_count'] = dfAvg['cumulative_loss_count'] + dftemp['cumulative_loss_count']
        
        # --- Normalize averages ---
        dfAvg['person_x'] = dfAvg['person_x'] / len(df)
        dfAvg['person_y'] = dfAvg['person_y'] / len(df)
        dfAvg['pepper_x'] = dfAvg['pepper_x'] / len(df)
        dfAvg['pepper_y'] = dfAvg['pepper_y'] / len(df)
        dfAvg['error_xy'] = dfAvg['error_xy'] / len(df)
        dfAvg['cumulative_reacquire_time'] = dfAvg['cumulative_reacquire_time'] / len(df)
        dfAvg['cumulative_loss_count'] = dfAvg['cumulative_loss_count'] / len(df)
        
        n_runs = len(df)
        for k in range(len(df)): # Renamed inner loop variable
            dfAvg['person_x_std'] = dfAvg['person_x_std'] + (df[k]['person_x'] - dfAvg['person_x'])**2
            dfAvg['person_y_std'] = dfAvg['person_y_std'] + (df[k]['person_y'] - dfAvg['person_y'])**2
            dfAvg['pepper_x_std'] = dfAvg['pepper_x_std'] + (df[k]['pepper_x'] - dfAvg['pepper_x'])**2
            dfAvg['pepper_y_std'] = dfAvg['pepper_y_std'] + (df[k]['pepper_y'] - dfAvg['pepper_y'])**2
            dfAvg['error_xy_std'] = dfAvg['error_xy_std'] + (df[k]['error_xy'] - dfAvg['error_xy'])**2
            dfAvg['cumulative_reacquire_time_std'] = dfAvg['cumulative_reacquire_time_std'] + (df[k]['cumulative_reacquire_time'] - dfAvg['cumulative_reacquire_time'])**2
            dfAvg['cumulative_loss_count_std'] = dfAvg['cumulative_loss_count_std'] + (df[k]['cumulative_loss_count'] - dfAvg['cumulative_loss_count'])**2
    
        if n_runs > 1:
            dfAvg['person_x_std'] = (dfAvg['person_x_std'] / (n_runs - 1))**0.5
            dfAvg['person_y_std'] = (dfAvg['person_y_std'] / (n_runs - 1))**0.5
            dfAvg['pepper_x_std'] = (dfAvg['pepper_x_std'] / (n_runs - 1))**0.5
            dfAvg['pepper_y_std'] = (dfAvg['pepper_y_std'] / (n_runs - 1))**0.5
            dfAvg['error_xy_std'] = (dfAvg['error_xy_std'] / (n_runs - 1))**0.5
            dfAvg['cumulative_reacquire_time_std'] = (dfAvg['cumulative_reacquire_time_std'] / (n_runs - 1))**0.5
            dfAvg['cumulative_loss_count_std'] = (dfAvg['cumulative_loss_count_std'] / (n_runs - 1))**0.5
        else:
            dfAvg['person_x_std'].values[:] = 0
            dfAvg['person_y_std'].values[:] = 0
            dfAvg['pepper_x_std'].values[:] = 0
            dfAvg['pepper_y_std'].values[:] = 0
            dfAvg['error_xy_std'].values[:] = 0
            dfAvg['cumulative_reacquire_time_std'].values[:] = 0
            dfAvg['cumulative_loss_count_std'].values[:] = 0

        dfAvgList.append(dfAvg)
    
    if not dfAvgList:
        print("No data to plot for avg_compare.")
        return

    # --- Plot 1: Absolute Error vs. Time ---
    try:
        plt.figure(figsize=(12, 6))
        for i in range(len(dfAvgList)):
            dfAvg = dfAvgList[i]
            algo_name = algo_names[i]
            upper_bound = dfAvg['error_xy'] + dfAvg['error_xy_std']
            lower_bound = dfAvg['error_xy'] - dfAvg['error_xy_std']	
            plt.plot(dfAvg['timestamp'], dfAvg['error_xy'],color=algo_colors[algo_name],label=f'{algo_name} Mean')
            plt.fill_between(dfAvg['timestamp'], lower_bound, upper_bound, color=algo_colors[algo_name], alpha=0.1, label=f'{algo_name} SD')	
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (m)')
        plt.title(f'Pepper Tracking Error (X-Y Plane) vs. Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'tracking_error_vs_time_compare.png')
        plt.close() # Close the plot to free up memory
        print("Successfully saved 'tracking_error_vs_time_compare.png'")
    except Exception as e:
        print(f"Error creating error vs. time plot: {e}")

    # --- NEW PLOT 1: Cumulative Loss Count vs. Time ---
    try:
        plt.figure(figsize=(12, 6))
        for i in range(len(dfAvgList)):
            dfAvg = dfAvgList[i]
            algo_name = algo_names[i]
            upper_bound = dfAvg['cumulative_loss_count'] + dfAvg['cumulative_loss_count_std']
            lower_bound = dfAvg['cumulative_loss_count'] - dfAvg['cumulative_loss_count_std']	
            plt.plot(dfAvg['timestamp'], dfAvg['cumulative_loss_count'],color=algo_colors[algo_name],label=f'{algo_name} Mean')
            #plt.fill_between(dfAvg['timestamp'], lower_bound, upper_bound, color=algo_colors[algo_name], alpha=0.1, label=f'{algo_name} SD')	
        plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('Cumulative Loss Count',fontsize=16)
        #plt.title(f'Cumulative Target Loss vs. Time')
        #plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'tracking_loss_count_vs_time_compare.png')
        plt.close() # Close the plot
        print("Successfully saved 'tracking_loss_count_vs_time_compare.png'")
    except Exception as e:
        print(f"Error creating cumulative loss count plot: {e}")

    # --- NEW PLOT 2: Cumulative Reacquisition Time vs. Time ---
    try:
        plt.figure(figsize=(12, 6))
        for i in range(len(dfAvgList)):
            dfAvg = dfAvgList[i]
            algo_name = algo_names[i]
            upper_bound = dfAvg['cumulative_reacquire_time'] + dfAvg['cumulative_reacquire_time_std']
            lower_bound = dfAvg['cumulative_reacquire_time'] - dfAvg['cumulative_reacquire_time_std']	
            plt.plot(dfAvg['timestamp'], dfAvg['cumulative_reacquire_time'],color=algo_colors[algo_name],label=f'{algo_name} Mean')
            #plt.fill_between(dfAvg['timestamp'], lower_bound, upper_bound, color=algo_colors[algo_name], alpha=0.1, label=f'{algo_name} SD')	
        plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('Cumulative Reacquisition Time (s)',fontsize=16)
        #plt.title(f'Cumulative Reacquisition Time vs. Time')
        #plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'tracking_reacquire_time_vs_time_compare.png')
        plt.close() # Close the plot
        print("Successfully saved 'tracking_reacquire_time_vs_time_compare.png'")
    except Exception as e:
        print(f"Error creating cumulative reacquisition time plot: {e}")

    # --- Plot 2: X-Y Path Map ---
    try:
    	plt.figure(figsize=(10, 10))
    	# Plot paths
    	plt.plot(dfAvgList[0]['person_x'], dfAvgList[0]['person_y'], label='Person Path', color='gray', linestyle='-')
    	for i in range(len(dfAvgList)):
    		plt.plot(dfAvgList[i]['pepper_x'], dfAvgList[i]['pepper_y'], color=algo_colors[algo_names[i]], linestyle='-', label=f'{algo_names[i]}')
    	# Plot start points
    	plt.plot(dfAvgList[0]['person_x'].iloc[0], dfAvgList[0]['person_y'].iloc[0], marker='o', color='gray', markersize=10, label='Person Start')
    	for i in range(len(dfAvgList)):
    		plt.plot(dfAvgList[i]['pepper_x'].iloc[0], dfAvgList[i]['pepper_y'].iloc[0], marker='o', color=algo_colors[algo_names[i]], markersize=10, label=f'{algo_names[i]} Start')
    	# Plot end points
    	plt.plot(dfAvgList[0]['person_x'].iloc[-1], dfAvgList[0]['person_y'].iloc[-1], marker='x', color='gray', markersize=10, label='Person End')
    	for i in range(len(dfAvgList)):
    		plt.plot(dfAvgList[i]['pepper_x'].iloc[-1], dfAvgList[i]['pepper_y'].iloc[-1], marker='x', color=algo_colors[algo_names[i]], markersize=10, label=f'{algo_names[i]} End')
    	plt.xlabel('X Coordinate (m)')
    	plt.ylabel('Y Coordinate (m)')
    	plt.title(f'Person vs. Pepper Path (X-Y Plane)')
    	plt.legend()
    	plt.grid(True)
    	plt.axis('equal') # Ensure aspect ratio is equal so paths are not distorted
    	plt.tight_layout()
    	plt.savefig(f'tracking_path_xy_compare.png')
    	plt.close() # Close the plot
    	print("Successfully saved 'tracking_path_xy_compare.png'")
    except Exception as e:
    	print(f"Error creating path map plot: {e}")	

if __name__ == '__main__':
    # --- IMPORTANT ---
    # This block finds all .csv files in subfolders defined in 'paths'
    # and runs the averaging and statistics functions on them.
    
    # Define the folders and corresponding algorithm names
    paths = ["following/1_naiveP","following/2_waypoint","following/3_averagePosBreadcrumbs","following/4_breadcrumbPopTimer","following/5_headTracking"]
    algo_names = ["NaiveP","Waypoint","Breadcrumbs","BreadcrumbsTimedPop","HeadSwivel"]
    
    print("Starting script run...")
    for i in range(len(paths)):
	    folder_path = paths[i]
	    # Get a list of all CSV files in the specified folder
	    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
	    
	    if csv_files:
	        print(f"Found {len(csv_files)} files for algorithm '{algo_names[i]}'.")
	        # Create output folder if it doesn't exist
	        os.makedirs(folder_path, exist_ok=True)
	        # Pass the folder path itself as the outputFolder
	        create_tracking_plots_avg(csv_files, algo_names[i], folder_path)
	    else:
	        print(f"No CSV files found in '{folder_path}' for algorithm '{algo_names[i]}'.")
    
    # --- Code for avg_compare (requires multiple algo folders) ---
    print("\n--- Running Comparison Plots ---")
    try:
        file_names = [glob.glob(os.path.join('following/1_naiveP', "*.csv")),
                      glob.glob(os.path.join('following/2_waypoint', "*.csv")),
                      glob.glob(os.path.join('following/3_averagePosBreadcrumbs', "*.csv")),
                      glob.glob(os.path.join('following/4_breadcrumbPopTimer', "*.csv")),
                      glob.glob(os.path.join('following/5_headTracking', "*.csv"))]
        
        # Filter out empty lists to avoid errors if a folder is missing
        valid_files = [f for f in file_names if f]
        valid_indices = [i for i, f in enumerate(file_names) if f]
        valid_algo_names = [algo_names[i] for i in valid_indices]

        if len(valid_files) > 1:
            print(f"Found data for {len(valid_algo_names)} algorithms, running comparison.")
            create_tracking_plots_avg_compare(valid_files, valid_algo_names)
        else:
            print("Not enough algorithm data found to run comparison plot.")

    except Exception as e:
        print(f"Could not run avg_compare: {e}")

    print("Script run finished.")
