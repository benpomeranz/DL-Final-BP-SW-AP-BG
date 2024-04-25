import json
import numpy as np
import os


#Open a JSONL file and average the x, y, and z by accelaraiton
def normalizeAccel():
    # Open the JSONL file
    with open('data/2018/02/day=27/hour=17/00.jsonl', 'r') as file:
        lines = file.readlines()

    # Process each line
    for i, line in enumerate(lines):
        data = json.loads(line)

        # Normalize each array
        for axis in ['x', 'y', 'z']:
            arr = np.array(data[axis])
            avg = np.average(arr)
            data[axis] = (arr / avg).tolist()

        # Write the updated JSON object back to the file
        lines[i] = json.dumps(data)

    # Write the updated lines back to the file
    with open('data/2018/02/day=27/hour=17/05.jsonl', 'w') as file:
        file.write('\n'.join(lines))

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
               if maxInFile>highest_accel:
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

print(findAccel())