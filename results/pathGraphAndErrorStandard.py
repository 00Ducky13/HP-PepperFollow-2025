import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import glob
import os
from math import sqrt

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
        plt.plot(df[1]['pepper_x'].iloc[0], df[1]['pepper_y'].iloc[0], marker='o', color='orange', markersize=10, label='Waypoint Start')

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
    # 1. Calculate Absolute Error (Euclidean distance in X-Y plane)
    # The error is the straight-line distance between (person_x, person_y) 
    # and (pepper_x, pepper_y)
    for j in range(len(df)):
        df[j]['error_xy'] = np.sqrt((df[j]['person_x'] - df[j]['pepper_x'])**2 + (df[j]['person_y'] - df[j]['pepper_y'])**2)
        dfAvg['person_x'] = dfAvg['person_x'] + df[j]['person_x']
        dfAvg['person_y'] = dfAvg['person_y'] + df[j]['person_y']
        dfAvg['pepper_x'] = dfAvg['pepper_x'] + df[j]['pepper_x']
        dfAvg['pepper_y'] = dfAvg['pepper_y'] + df[j]['pepper_y']
        dfAvg['error_xy'] = dfAvg['error_xy'] + df[j]['error_xy']
        
    

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
    
    
    dfAvg['person_x_std'] = (dfAvg['person_x_std'] / (len(df)-1))**0.5
    dfAvg['person_y_std'] = (dfAvg['person_y_std'] / (len(df)-1))**0.5
    dfAvg['pepper_x_std'] = (dfAvg['pepper_x_std'] / (len(df)-1))**0.5
    dfAvg['pepper_y_std'] = (dfAvg['pepper_y_std'] / (len(df)-1))**0.5
    dfAvg['error_xy_std'] = (dfAvg['error_xy_std'] / (len(df)-1))**0.5
    # 2. Plot 1: Absolute Error vs. Time
    try:
        plt.figure(figsize=(12, 6))
        #for i in range(len(df)):
        #	plt.plot(df[i]['timestamp'], df[i]['error_xy'],color=algo_colors[algo_name],alpha=0.5)
        upper_bound = dfAvg['error_xy'] + dfAvg['error_xy_std']
        lower_bound = dfAvg['error_xy'] - dfAvg['error_xy_std']	
        plt.plot(dfAvg['timestamp'], dfAvg['error_xy'],color=algo_colors[algo_name],label="Mean")
        plt.fill_between(dfAvg['timestamp'], lower_bound, upper_bound, color=algo_colors[algo_name], alpha=0.5, label='Standard Deviation')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (m)')
        plt.title(f'Pepper Tracking Error (X-Y Plane) vs. Time for Algorithm {algo_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{outputFolder}/tracking_error_vs_time_{algo_name}.png')
        plt.close() # Close the plot to free up memory
        print("Successfully saved 'tracking_error_vs_time.png'")
    except Exception as e:
        print(f"Error creating error vs. time plot: {e}")

    # 3. Plot 2: X-Y Path Map
    try:
        plt.figure(figsize=(10, 10))
        ax = plt.gca() # Get current axes

        # --- Extract data as numpy arrays ---
        x = dfAvg['pepper_x'].values
        y = dfAvg['pepper_y'].values
        x_std = dfAvg['pepper_x_std'].values
        y_std = dfAvg['pepper_y_std'].values

        # --- 1. Create the Y-Deviation Polygons ---
        y_upper_points = np.stack([x, y + y_std], axis=1)
        y_lower_points = np.stack([x, y - y_std], axis=1)
    
        # Create a list of small quads
        y_quads = []
        for i in range(len(x) - 1):
            quad = [
                y_upper_points[i],
                y_upper_points[i+1],
                y_lower_points[i+1],
                y_lower_points[i]
            ]
            y_quads.append(quad)

        # --- 2. Create the X-Deviation Polygons ---
        x_upper_points = np.stack([x + x_std, y], axis=1)
        x_lower_points = np.stack([x - x_std, y], axis=1)

        # Create a list of small quads
        x_quads = []
        for i in range(len(x) - 1):
            quad = [
                x_upper_points[i],
                x_upper_points[i+1],
                x_lower_points[i+1],
                x_lower_points[i]
            ]
            x_quads.append(quad)

        # --- 3. Add collections to the plot ---
        # `edgecolors='none'` is important to avoid lines on every quad
        y_std_collection = PolyCollection(y_quads, 
                                      facecolor=algo_colors[algo_name], 
                                      alpha=0.3, 
                                      edgecolors='none')
        ax.add_collection(y_std_collection)
    
        x_std_collection = PolyCollection(x_quads, 
                                      facecolor=algo_colors[algo_name], 
                                      alpha=0.3, 
                                      edgecolors='none')
        # Adding this collection will overlay the Y-dev, making the
        # intersecting areas slightly darker, which is a good effect.
        ax.add_collection(x_std_collection)

        # --- 3. Plot the mean paths ON TOP of the polygons ---
        plt.plot(dfAvg['person_x'], dfAvg['person_y'], label='Person Path', color='gray', linestyle='-')
        plt.plot(x, y, label=f'{algo_name}', color=algo_colors[algo_name], linestyle='-')
        #for i in range(len(df)):
        #	plt.plot(df[i]['pepper_x'], df[i]['pepper_y'], color=algo_colors[algo_name], alpha=0.5, linestyle='--')	

        # Plot start points
        plt.plot(dfAvg['person_x'].iloc[0], dfAvg['person_y'].iloc[0], marker='o', color='gray', markersize=10, label='Person Start')
        plt.plot(dfAvg['pepper_x'].iloc[0], dfAvg['pepper_y'].iloc[0], marker='o', color=algo_colors[algo_name], markersize=10, label=f'{algo_name} Start')
        #for i in range(len(df)):
        #	plt.plot(df[i]['pepper_x'].iloc[0], df[i]['pepper_y'].iloc[0], marker='o', color=algo_colors[algo_name], alpha=0.5, markersize=10)

        # Plot end points
        plt.plot(dfAvg['person_x'].iloc[-1], dfAvg['person_y'].iloc[0], marker='x', color='gray', markersize=10, label='Person End')
        plt.plot(dfAvg['pepper_x'].iloc[-1], dfAvg['pepper_y'].iloc[0], marker='x', color=algo_colors[algo_name], markersize=10, label=f'{algo_name} End')
        #for i in range(len(df)):
        #	plt.plot(df[i]['pepper_x'].iloc[-1], df[i]['pepper_y'].iloc[0], marker='x', color=algo_colors[algo_name], alpha=0.5, markersize=10)

        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title(f'Person vs. Pepper Path (X-Y Plane) Average for Algorithm {algo_name}')
        plt.legend(loc='center right')
        plt.grid(True)
        plt.axis('equal') # Ensure aspect ratio is equal so paths are not distorted
        plt.tight_layout()
        plt.savefig(f'{outputFolder}/tracking_path_xy_{algo_name}.png')
        plt.close() # Close the plot
        print("Successfully saved 'tracking_path_xy.png'")
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
        		print(f"Error: The file '{file_names[i]}' was not found.")
        		return
        	except Exception as e:
        		print(f"Error loading file: {e}")
        		return
        	required_cols = ['timestamp', 'person_x', 'person_y', 'pepper_x', 'pepper_y']
        	if not all(col in dftemp.columns for col in required_cols):
        		print(f"Error: The CSV must contain the following columns: {', '.join(required_cols)}")
        		return
        dfAvg = df[0].copy(deep=True)
        dfAvg['person_x'] = 0.0
        dfAvg['person_y'] = 0.0
        dfAvg['pepper_x'] = 0.0
        dfAvg['pepper_y'] = 0.0
        dfAvg['error_xy'] = 0.0
        for j in range(len(df)):
        	df[j]['error_xy'] = np.sqrt((df[j]['person_x'] - df[j]['pepper_x'])**2 + (df[j]['person_y'] - df[j]['pepper_y'])**2)
        	dfAvg['person_x'] = dfAvg['person_x'] + df[j]['person_x']
        	dfAvg['person_y'] = dfAvg['person_y'] + df[j]['person_y']
        	dfAvg['pepper_x'] = dfAvg['pepper_x'] + df[j]['pepper_x']
        	dfAvg['pepper_y'] = dfAvg['pepper_y'] + df[j]['pepper_y']
        	dfAvg['error_xy'] = dfAvg['error_xy'] + df[j]['error_xy']
        dfAvg['person_x'] = dfAvg['person_x'] / len(df)
        dfAvg['person_y'] = dfAvg['person_y'] / len(df)
        dfAvg['pepper_x'] = dfAvg['pepper_x'] / len(df)
        dfAvg['pepper_y'] = dfAvg['pepper_y'] / len(df)
        dfAvg['error_xy'] = dfAvg['error_xy'] / len(df)
        dfAvgList.append(dfAvg)
    # 2. Plot 1: Absolute Error vs. Time
    try:
        plt.figure(figsize=(12, 6))
        for i in range(len(dfAvgList)):
        	plt.plot(dfAvgList[i]['timestamp'], dfAvgList[i]['error_xy'],color=algo_colors[algo_names[i]],label=f'{algo_names[i]}')	
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (m)')
        plt.title(f'Pepper Tracking Error (X-Y Plane) vs. Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'tracking_error_vs_time.png')
        plt.close() # Close the plot to free up memory
        print("Successfully saved 'tracking_error_vs_time.png'")
    except Exception as e:
        print(f"Error creating error vs. time plot: {e}")

    # 3. Plot 2: X-Y Path Map
    try:
    	plt.figure(figsize=(10, 10))
    	# Plot paths
    	plt.plot(dfAvgList[0]['person_x'], dfAvgList[0]['person_y'], label='Person Path', color='gray', linestyle='--')
    	for i in range(len(dfAvgList)):
    		plt.plot(dfAvgList[i]['pepper_x'], dfAvgList[i]['pepper_y'], color=algo_colors[algo_names[i]], linestyle='-', label=f'{algo_names[i]}')
    	# Plot start points
    	plt.plot(dfAvgList[0]['person_x'].iloc[0], dfAvgList[0]['person_y'].iloc[0], marker='o', color='gray', markersize=10, label='Person Start')
    	for i in range(len(dfAvgList)):
    		plt.plot(dfAvgList[i]['pepper_x'].iloc[0], dfAvgList[i]['pepper_y'].iloc[0], marker='o', color=algo_colors[algo_names[i]], alpha=0.5, markersize=10,label=f'{algo_names[i]} Start')
    	# Plot end points
    	plt.plot(dfAvgList[0]['person_x'].iloc[-1], dfAvgList[0]['person_y'].iloc[0], marker='x', color='gray', markersize=10, label='Person End')
    	for i in range(len(dfAvgList)):
    		plt.plot(dfAvgList[i]['pepper_x'].iloc[-1], dfAvgList[i]['pepper_y'].iloc[-1], marker='x', color=algo_colors[algo_names[i]], alpha=0.5, markersize=10,label=f'{algo_names[i]} End')
    	plt.xlabel('X Coordinate (m)')
    	plt.ylabel('Y Coordinate (m)')
    	plt.title(f'Person vs. Pepper Path (X-Y Plane)')
    	plt.legend()
    	plt.grid(True)
    	plt.axis('equal') # Ensure aspect ratio is equal so paths are not distorted
    	plt.tight_layout()
    	plt.savefig(f'tracking_path_xy.png')
    	plt.close() # Close the plot
    	print("Successfully saved 'tracking_path_xy.png'")
    except Exception as e:
    	print(f"Error creating path map plot: {e}")	

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Replace 'your_file_name.csv' with the actual name of your CSV file
    # For this example, we'll use the one you provided:
    #input_files = ["following/1_naiveP/halfCoveredMaze/tracking_log_20251106_200951.csv","following/6_reidNN/halfCoveredMaze/tracking_log_20251106_192309.csv"]
    
    # Run the function
    #create_tracking_plots_compare(input_files)
    paths = ["following/1_naiveP","following/2_waypoint","following/3_averagePosBreadcrumbs","following/4_breadcrumbPopTimer","following/5_headTracking"]
    algo_names = ["NaiveP","Waypoint","Breadcrumbs","BreadcrumbsTimedPop","HeadSwivel"]
    for i in range(len(paths)):
	    folder_path = paths[i]  # Replace with the actual path to your folder
	    # Get a list of all CSV files in the specified folder
	    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
	    create_tracking_plots_avg(csv_files,algo_names[i],folder_path)
    
    
    #file_names = [glob.glob(os.path.join('following/1_naiveP', "*.csv")),glob.glob(os.path.join('following/2_waypoint', "*.csv")),glob.glob(os.path.join('following/3_averagePosBreadcrumbs', "*.csv")),glob.glob(os.path.join('following/4_breadcrumbPopTimer', "*.csv")),glob.glob(os.path.join('following/5_headTracking', "*.csv"))]
    #algo_names = ['NaiveP','Waypoint','Breadcrumbs','BreadcrumbsTimedPop','HeadSwivel']
    #create_tracking_plots_avg_compare(file_names,algo_names)
    
    #file_names = ["following/1_naiveP/halfCoveredMaze/tracking_log_20251106_200951.csv","following/2_waypoint/halfCoveredMaze/tracking_log_20251107_181753.csv"]
    #create_tracking_plots_compare(file_names)
