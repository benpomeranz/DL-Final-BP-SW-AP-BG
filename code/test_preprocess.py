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
