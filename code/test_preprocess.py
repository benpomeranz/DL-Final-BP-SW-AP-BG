from preprocess import *
from pytest import *
import tensorflow as tf

def  test_preprocess():
    test_batch = full_preprocess("Scrap/test_data", "test_output", 1.7, 0, 11000)
    print(test_batch)
    assert len(test_batch) == 12
    for data in test_batch:
        print(data[0])
    #TODO: CHECK ON LOGS TO MAKE SURE THEY ARE CORRECT  - DONE!

    
def jsonl_to_data(filename, start_time, end_time):
    times = []
    richters = []
    accels = []

    with open(f"{filename}.jsonl", 'r') as file:
        lines = file.readlines()

    # Process first line
    line0 = lines[0]
    json_data = json.loads(line0)
    t = math.log(json_data['cloud_t']-start_time)
    times.append(t)
    richter = accel_to_rich_one(np.array(json_data["total_acceleration"]).max())
    richters.append(richter)
    accel_matrix = np.array([json_data['x'], json_data['y'], json_data['z']])
    accels.append(accel_matrix)

    # Process the rest of the lines in pairs
    for i in range(1, len(lines)):
        line2 = lines[i]
        line1 = lines[i - 1]
        # Process the pair of lines
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        t = math.log(json_data_2['cloud_t']-json_data_1['cloud_t'])
        times.append(t)
        richter = accel_to_rich_one(np.array(json_data_2["total_acceleration"]).max())
        richters.append(richter)
        accel_matrix = np.array([json_data_2['x'], json_data_2['y'], json_data_2['z']])
        accels.append(accel_matrix)
    times.append(math.log(end_time - json_data_2['cloud_t']))

    # Get average values after, and subtract them from relevant values
    log_avg_interval = math.log(sum(times) / len(times))
    richter_avg = np.average(richters)
    average_accel = np.mean(accels)

    times = [t - log_avg_interval for t in times]
    richters = [r - richter_avg for r in richters]
    accels = [a - average_accel for a in accels]

    return times, richters, accels

print(jsonl_to_data("Scrap/test_data/test_preprocess_data", 0, 10000))