import math
from math import exp, sqrt
import json
import numpy as np
from preprocess import jsonl_to_data, accel_to_rich_one
import matplotlib.pyplot as plt


def jsonl_to_data_2(filename, start_time, end_time):
    times = []
    richters = []

    with open(f"{filename}.jsonl", 'r') as file:
        lines = file.readlines()

    # Process first line
    line0 = lines[0]
    json_data = json.loads(line0)
    # Append time, richter, and acceleration
    t = math.log(json_data['cloud_t']-start_time)
    times.append(t)
    richter = accel_to_rich_one(np.array(json_data["total_acceleration"]).max())
    richters.append(richter)
   
    # Process the rest of the lines in pairs
    for i in range(1, len(lines)):
        line2 = lines[i]
        line1 = lines[i - 1]
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        # Append time, richter, and acceleration
        t = math.log(json_data_2['cloud_t']-json_data_1['cloud_t'])
        times.append(t)
        richter = accel_to_rich_one(np.array(json_data_2["total_acceleration"]).max())
        richters.append(richter)
       
    times.append(math.log(end_time - json_data_2['cloud_t']))

    # Get average values after, and subtract them from relevant values
    log_avg_interval = math.log(sum(times) / len(times))
    times = [t - log_avg_interval for t in times]

    return times, richters

'''
IDEA:
x axis is the timeframe
each row in the data file is a different event
- dot size based on magnitude, arrow pointing in direction of acceleration
'''

def preprocess_vis(file, start, end):
    '''
    Function to create visual of preprocessed data
    time is on the x axis
    inputs:
        file: string containing the path to the file
    '''
    times, mags = jsonl_to_data_2(file, start, end)

    mags = [mag**3 for mag in mags]

    # plotting
    plt.figure(figsize=(10, 2))
    #plt.plot([min(times), max(times)], [0, 1], linestyle='--', color='white') 

    # Plot circles for earthquakes
    for i in range(len(mags)):
        plt.scatter(times[i], 0, s=20*mags[i], color='red', alpha=0.1)  # Adjust size based on magnitude

    # Customize your plot
    plt.xlabel('Time')
    plt.title('Tremor Events on Timescale')
    plt.xlim(min(times), max(times))  # Add a buffer to x-axis limits for better visualization

    plt.ylim(-1, 1)  # Set y-axis limits to hide the y-axis
    plt.yticks([])  # Remove y-axis ticks
    plt.grid(False)  # Remove gridlines

    # Show the plot
    plt.grid(True)
    plt.show()
        
preprocess_vis('big_data/device005_preprocessed', 1514782800, 1546318800)