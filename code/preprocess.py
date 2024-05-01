import json
import numpy as np
import os
import math


#SEE THAT THIS HAS BEEN CHANGED FROM THE ONE IN PREPROCESS: take in a single accelaration value, 
# get a single richter value
'''
NOTE: May want to change the ratio computation to more closely fit the log function 
'''
def accel_to_rich_one(accel):
    g = accel / 980.665
    mercalli_split = [.000464, .00175, .00297, .0276, .062, .115, .215, .401, .747, 1.39]
    ratios = g / next((mval for mval in mercalli_split if g < mval), mercalli_split[-1])
    mercalli_id = np.digitize(g, mercalli_split) + 1
    mercalli_richter = {1:1, 2:3, 3:3.5, 4:4, 5:4.5, 6:5, 7:5.5, 8:6, 9:6.5, 10:7, 11:7.5, 12:8}
    richter_val = mercalli_richter[mercalli_id]
    richter_val += ratios
    return richter_val

#Take a jsonl file, and for each line create data of the following format:
#data[1]=richter - avg, data[2]=accelaration matrix where the first row is the x accel, second row the y, 
#and third row the z, and where we are subtracting the average of all accelaration values
#  data[0] = log(t_i-t_{i-1})-(log_avg interval time)
def jsonl_to_data(filename, start_time, end_time):
    data = []
    time_intervals = []
    total_accels = []
    richters = []

    with open(f"{filename}.jsonl", 'r') as file:
        lines = file.readlines()

    for i in range(1, len(lines)):
        line2 = lines[i]
        line1 = lines[i - 1]
        # Process the pair of lines
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        inter_time = json_data_2['cloud_t']-json_data_1['cloud_t']
        time_intervals.append(inter_time)
        total_accels.append(np.array([json_data_2['x'], json_data_2['y'], json_data_2['z']]))
        richters.append(accel_to_rich_one(np.array(json_data_2["total_acceleration"]).max()))
    log_avg_interval = math.log(sum(time_intervals) / len(time_intervals))
    average_accel = np.mean(total_accels)

    
    #Append first datapoint:
    line0 = lines[0]
    json_data = json.loads(line0)
    t = math.log(json_data['cloud_t']-start_time) - log_avg_interval
    richter = accel_to_rich_one(np.array(json_data["total_acceleration"]).max())
    accel_matrix = np.array([json_data['x'], json_data['y'], json_data['z']])
    richters.append(richter)
    richter_avg = np.average(richters)

    times = []
    richters = []
    accels = []
    times.append(t-start_time)
    richters.append(richter-richter_avg)
    accels.append(accel_matrix - average_accel)
    # data.append([t-start_time, richter-richter_avg, accel_matrix - average_accel])

    #For rest iterate through getting interval times
    for i in range(1, len(lines)):
        line2 = lines[i]
        line1 = lines[i - 1]
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        t = math.log(json_data_2['cloud_t']-json_data_1['cloud_t']) - log_avg_interval
        # print(np.array(json_data_2["total_acceleration"]).max())
        richter = accel_to_rich_one(np.array(json_data_2["total_acceleration"]).max())
        accel_matrix = np.array([json_data_2['x'], json_data_2['y'], json_data_2['z']])

        times.append(t)
        richters.append(richter-richter_avg)
        accels.append(accel_matrix - average_accel)
        # data.append([t, richter-richter_avg, accel_matrix - average_accel])
    # return data
    return times, richters, accels

#takes in a JSONL filename WIHTOUT suffix, sorts by "cloud_t" value
def sort_by_time(filename):
    with open(f"{filename}.jsonl", 'r') as file:
        lines = file.readlines()
    sorted_lines = sorted(lines, key=lambda line: json.loads(line)['cloud_t'])
    with open(f"{filename}.jsonl", 'w') as file:
        file.writelines(sorted_lines)

#filepath does not include suffix, MODIFIES THE FILE
def delete_within_x(filepath:str, num_secs:int):
    with open(f"{filepath}.jsonl", 'r') as file:
        lines = file.readlines()
    # Process each line
    prev_time = float('-inf')
    for line in lines:
        data = json.loads(line)
        curr_time = data['cloud_t']
        # Check if curr_time and prev_time are separated by num_mins minutes
        if curr_time - prev_time > (num_secs):
            # Write the line to a new JSONL file in the root directory
            with open(f"delete_within_{str(num_secs)}.jsonl", 'a') as output_file:
                output_file.write(line)
        prev_time = curr_time
    # Delete the original file
    os.remove(f"{filepath}.jsonl")
    # Rename the second file to be the first file
    os.rename(f"delete_within_{str(num_secs)}.jsonl", f"{filepath}.jsonl")

#TAKE IN WHOLE PATH AND PREPROCESS: add_total_accelaration, DataWithAccel to get all of a certain accelaration, then sort_by_time, then delete_within_x, 

#This goes through a path and gets all jsonl files, creates total accelaration, and adds them to new file if accelaration is 
#greater than accel. output doesn't include suffied
def add_total_and_select(path:str, output:str, accel:float): 
    # Get a list of all files in the directory
    for dirpath, dirnames, filenames in os.walk(path):
        # Process each JSONL file
        for filename in filenames:
            if filename.endswith('.jsonl'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                # Process each line
                for line in lines:
                    data = json.loads(line)
                    total_acceleration = np.sqrt((np.square(data['x'])) + (np.square(data['y'])) + (np.square(data['z']))).tolist()
                    # Add total acceleration info to the JSON object
                    data['total_acceleration'] = total_acceleration
                    # Check if 'total_acceleration' contains a value greater than 5
                    if any(x > accel for x in data['total_acceleration']):
                        # Add the total acceleration array to the JSON object
                        data['total_acceleration'] = data['total_acceleration']
                        # Write the updated JSON object back to the file
                        line = json.dumps(data)
                        # Write the line to a new JSONL file in the root directory
                        with open(f"{output}.jsonl", 'a') as output_file:
                            output_file.write(f"{line}\n")



def full_preprocess(path:str, output:str, accel:float, start_time: int, end_time: int):
    add_total_and_select(path, output, accel)
    sort_by_time(output)
    delete_within_x(output, 100)
    return jsonl_to_data(output, start_time, end_time)

print(jsonl_to_data('processed_2018_2'))
