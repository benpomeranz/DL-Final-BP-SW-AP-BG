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
def train(model, times, magnitudes, accels, sequence_length, has_accel):
    # magnitudes: [batchsize(1) x sequence size x 1]
    # times: [batchsize(1) x sequence size + 1 x 1]
    # accels: [batchsize(1) x sequence size x 96]
    times = np.expand_dims(np.array(times).T, axis=[0,2])
    magnitudes = np.expand_dims(np.array(magnitudes).T, axis=[0,2])
    accels = np.expand_dims(np.array(accels), axis=[0])
    with tf.GradientTape() as tape:
        pred = model(times, magnitudes, accels, has_accel, training=True)
        loss = model.nll(pred, has_accel)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

import numpy as np
def test():
    t = []
    r = []
    a = []
    for i in range(6):
        t.append(i)
        r.append(2*i)
        n = np.random.randint(0, 11, (3,6))
        a.append(n.flatten())
    t.append(6)
    # print(a)
    print(a-np.mean(a))
    print("==================")
    print(a-np.mean(a, axis=0))
    t = np.expand_dims(np.array(t).T, axis=[0,2])
    r = np.expand_dims(np.array(r).T, axis=[0,2])
    a = np.expand_dims(np.array(a), axis=[0])
    print(t.shape, r.shape, a.shape)
test()