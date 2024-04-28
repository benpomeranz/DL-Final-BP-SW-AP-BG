import json
import numpy as np
import os
import math


# Function to process a JSONL file
def greaterAccel(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        data = json.loads(line)
        if any(x > 5 for x in data['x']):
            maxAccelInFile(filename)
            return
def findAccel():
    # Get a list of all files in the directory
    # Process each JSONL file
    high_accels = []
    highest_accel = 0
    highest_file = ""
    for dirpath, dirnames, filenames in os.walk('data'):
        # Process each JSONL file
        for filename in filenames:
            if filename.endswith('.jsonl'):
               maxInFile = maxAccelInFile(os.path.join(dirpath, filename))
               if maxInFile>highest_accel and maxInFile!=201.342:
                   highest_accel = maxInFile
                   highest_file = os.path.join(dirpath, filename)
               high_accels.append(maxInFile)
               
    return highest_accel, highest_file

def maxAccelInFile(filename: str):
    # Open the JSONL file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize the maximum value
    max_value = float('-inf')

    # Process each line
    for line in lines:
        data = json.loads(line)

        # Update the maximum value
        for axis in ['x', 'y', 'z']:
            max_value = max(max_value, max(data[axis]))
    return max_value

#Takes in file at pathname, takes sqrt of of squared x y and z accelarations at each timestep to get total ground accelaration
def add_total_accelaration(pathname:str):
    # Get a list of all files in the directory
    for dirpath, dirnames, filenames in os.walk(pathname):
        # Process each JSONL file
        for filename in filenames:
            if filename.endswith('.jsonl'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                # Process each line
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    # Calculate total acceleration
                    total_acceleration = np.sqrt((np.square(data['x'])) + (np.square(data['y'])) + (np.square(data['z']))).tolist()
                    # Add total acceleration info to the JSON object
                    data['total_acceleration'] = total_acceleration
                    # Write the updated JSON object back to the file
                    lines[i] = json.dumps(data)
                # Write the updated lines back to the file
                with open(filepath, 'w') as file:
                    file.write('\n'.join(lines))
#Take in a path, walk through all jsonl files in said path, and for each line if said line's 'total_accelaration'
# list of values contains a value greater than accel, add that to a new jsonl file in the root directory
def data_with_accel(path:str, accel:float):
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
                    # Check if 'total_acceleration' contains a value greater than 5
                    if any(x > accel for x in data['total_acceleration']):
                        # Write the line to a new JSONL file in the root directory
                        with open(f"over{str(accel)}.jsonl", 'a') as output_file:
                            output_file.write(line)

#SEE THAT THIS HAS BEEN CHANGED FROM THE ONE IN PREPROCESS: take in a single accelaration value, 
# get a single richter value
def accel_to_rich_one(accel):
    g = accel / 980.665
    mercalli_split = [.000464, .00175, .00297, .0276, .062, .115, .215, .401, .747, 1.39]
    ratios = g / next((mval for mval in mercalli_split if g < mval), mercalli_split[-1])
    mercalli_id = np.digitize(g, mercalli_split) + 1
    mercalli_richter = {1:1, 2:3, 3:3.5, 4:4, 5:4.5, 6:5, 7:5.5, 8:6, 9:6.5, 10:7, 11:7.5, 12:8}
    richter_val = mercalli_richter[mercalli_id]
    richter_val += ratios
    # print(richter_vals)
    return richter_val

#Take a jsonl file, and for each line create data of the following format:
#data[0]=richter, data[2]=accelaration matrix where the first row is the x accel, second row the y, 
#and third row the z, data[1] = log(t_i-t_{i-1})-(log_avg interval time) (seconds since start of 2018)
def jsonl_to_data(filename):
    data = []
    time_intervals = []
    total_accels = []
    with open(f"{filename}.jsonl", 'r') as file:
        lines = file.readlines()
    for i in range(1, len(lines) - 1):
        line2 = lines[i]
        line1 = lines[i - 1]
        # Process the pair of lines
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        inter_time = json_data_2['cloud_t']-json_data_1['cloud_t']
        time_intervals.append(inter_time)
        total_accels.append(np.array([json_data_2['x'], json_data_2['y'], json_data_2['z']]))
    log_avg_interval = math.log(sum(time_intervals) / len(time_intervals))
    average_accel = np.mean(total_accels, axis=0)
    for i in range(1, len(lines) - 1):
        line2 = lines[i]
        line1 = lines[i - 1]
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        t = math.log(json_data_2['cloud_t']-json_data_1['cloud_t']) - log_avg_interval
        print(np.array(json_data_2["total_acceleration"]).max())
        richter = accel_to_rich_one(np.array(json_data_2["total_acceleration"]).max())
        accel_matrix = np.array([json_data_2['x'], json_data_2['y'], json_data_2['z']])
        print("=====================", (accel_matrix-total_accels).shape, "===========================")
        data.append([t, richter, accel_matrix - total_accels])
    return data

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
    prev_time = json.loads(lines[0])['cloud_t']
    for line in lines[1:]:
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


def full_preprocess(path:str, output:str, accel:float):
    add_total_and_select(path, output, accel)
    sort_by_time(output)
    delete_within_x(output, 100)
    return jsonl_to_data(output)


# full_preprocess("data_2018", "processed_2018_2", 1.7)

# with open("processed_2018_2.jsonl", 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         data = json.loads(line)
#         print(len(data['total_acceleration']))

jsonl_to_data('processed_2018_2')