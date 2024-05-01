from preprocess import *
from pytest import *

def  test_preprocess():
    test_batch = full_preprocess("/Users/benpomeranz/Desktop/CS1470/DL-Final-BP-SW-AP-BG/test_data", "test_output", 1.7)
    print(test_batch)
    assert len(test_batch) == 11
   
    #TODO: CHECK ON LOGS TO MAKE SURE THEY ARE CORRECT
    
