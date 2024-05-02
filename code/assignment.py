import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from data import Dataset
from tqdm import tqdm
from preprocess import jsonl_to_data 
from model_experimental import Recurrent

# from model import TPPModel
# nll takes in list of distributions, input to call
def train(model, times, magnitudes, accels, sequence_length, has_accel):
    # magnitudes: [batchsize(1) x sequence size x 1]
    # times: [batchsize(1) x sequence size + 1 x 1]
    # accels: [batchsize(1) x sequence size x 96]
    losses = []
    times = np.expand_dims(np.array(times).T, axis=[0,2])
    magnitudes = np.expand_dims(np.array(magnitudes).T, axis=[0,2])
    accels = np.expand_dims(np.array(accels), axis=[0])
    with tf.GradientTape() as tape:
        pred = model(times, magnitudes, accels, has_accel, training=True)
        loss = model.nll(pred, has_accel)
        losses.append(loss)
        print(f"Loss: {loss}")
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return losses

def main():
    times, magnitudes, accels = jsonl_to_data('processed_2018_2', 1514782800, 1546318800)
    print(len(magnitudes))
    # model = Recurrent()
    # for i in range(6):
    #     print(f"Epoch {i}")
    #     # train(model, times, magnitudes, accels, len(magnitudes), has_accel)

main()