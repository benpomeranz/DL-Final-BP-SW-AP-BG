import json
import numpy as np
import os



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
def addTotalAccelerationInfo(pathname:str):
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
def dataWithAccel(path:str, accel:float):
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

#SEE THAT THIS HAS BEEN CHANGED FROM THE ONE IN PREPROCESS
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
#data[0]=richter, data[1]=accelaration matrix where the first row is the x accel, second row the y, 
#and third row the z, data[2] = cloud_t-1514782800 (seconds since start of 2018)
def jsonl_to_data(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    for line in lines:
        json_data = json.loads(line)
        print(np.array(json_data["total_acceleration"]).max())
        richter = accel_to_rich_one(np.array(json_data["total_acceleration"]).max())
        accel_matrix = np.array([json_data['x'], json_data['y'], json_data['z']])
        cloud_t = json_data['cloud_t']-1514782800 #seconds since 2018
        data.append([richter, accel_matrix, cloud_t])
    return data

data = jsonl_to_data("over1.7.jsonl")
print(data)

