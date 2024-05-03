from preprocess import *
from pytest import *
import tensorflow as tf

def test_preprocess():
    times, richters, accels = full_preprocess("Scrap/test_data", "test_output", 1.7, 0, 11000)
    # print(test_batch)
    assert len(times) == 13
    assert len(richters) == 12
    assert len(accels) == 12
    # for data in test_batch:
    #     print(data[0])
    #TODO: CHECK ON LOGS TO MAKE SURE THEY ARE CORRECT  - DONE!
def get_jsonl_files(directory):
    jsonl_files = []
    for month_dir in os.listdir(directory):
        month_path = os.path.join(directory, month_dir)
        if os.path.isdir(month_path):
            for day_dir in os.listdir(month_path):
                day_path = os.path.join(month_path, day_dir)
                if os.path.isdir(day_path):
                    for hour_dir in os.listdir(day_path):
                        hour_path = os.path.join(day_path, hour_dir)
                        if os.path.isdir(hour_path):
                            for filename in os.listdir(hour_path):
                                if filename.endswith('.jsonl'):
                                    jsonl_files.append(os.path.join(hour_path, filename))
    return jsonl_files

#TODO: Write a function to preprocess the data in code/big_data by iterating through all the files in the directory
def test_preprocess_big_data():
    for file in get_jsonl_files("code/big_data"):
        times, richters, accels = full_preprocess(file, "big_data_output", 1.7, 0, 11000)
        assert len(times) == 13
        assert len(richters) == 12
        assert len(accels) == 12

test_preprocess_big_data()