import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from data import Dataset
from tqdm import tqdm
from preprocess import jsonl_to_data 

# from model import TPPModel
# nll takes in list of distributions, input to call
'''
'''
def train(model, intervals, magnitudes, accels, batch_size, has_accel, start_time, end_time):
    times, magnitude, accels = jsonl_to_data(data, 0, 0)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    batched_data = dataset.batch(batch_size=batch_size, drop_remainder=False)
    for batch in batched_data:
        for magnitude, times, accels in batch:
            with tf.GradientTape() as tape:
                pred = model(magnitude, times, accels, has_accel, training=True)
                loss = model.nll(pred, has_accel, start_time, end_time)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

import numpy as np
def test():
    data = np.random.randint(0, 5, (9, 3))
    # print(data)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    for el in dataset:
        print(el)
    batched_data = dataset.batch(batch_size=3, drop_remainder=False)
    for batch in batched_data:
        for x,y,z in batch:
            print(x,y,z)

test()