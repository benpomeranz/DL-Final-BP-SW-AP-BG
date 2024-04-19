import json 
import os
import pandas as pd
import matplotlib.pyplot as plt
import ast

def process_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            json_object = json.loads(line)
            data.append(json_object["x"])
    return data

data_list = []
for root, dirs, files in os.walk('2018'):
    for file in files:
        if file.endswith('.jsonl'):
            file_path = os.path.join(root, file)
            data = process_file(file_path)
            data_list.append(data)


# Load the CSV file into a DataFrame
df = pd.read_csv('my_data.csv')

# Define a function to parse strings into lists
def parse_list(s):
    if pd.isna(s):
        return []
    else:
        return ast.literal_eval(s)

# Apply the function to each element in the DataFrame
df = df.applymap(parse_list)

df = df.applymap(pd.Series)

# Plot the first two series
df.iloc[0].plot(label='Period 1')
df.iloc[1].plot(label='Period 2')
# Add more lines as needed
plt.legend()
plt.show()