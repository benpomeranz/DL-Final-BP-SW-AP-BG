from preprocess import *
from pytest import *
import tensorflow as tf

def  test_preprocess():
    test_batch = full_preprocess("code/big_data/device002/month=03/day=21/hour=18/45.jsonl", "test_output", 1.7, 0, 11000)
    print(test_batch)
    assert len(test_batch) == 12
    for data in test_batch:
        print(data[0])
    #TODO: CHECK ON LOGS TO MAKE SURE THEY ARE CORRECT  - DONE!

