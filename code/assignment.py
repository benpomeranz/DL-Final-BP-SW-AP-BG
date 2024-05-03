import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from data import Dataset
from tqdm import tqdm
import preprocess
from model_experimental import Recurrent
import visuaization

# from model import TPPModel
# nll takes in list of distributions, input to call
def train(model, times, magnitudes, accels, start_time, end_time, sequence_length, has_accel):
    # magnitudes: [batchsize(1) x sequence size x 1]
    # times: [batchsize(1) x sequence size + 1 x 1]
    # accels: [batchsize(1) x sequence size x 96]
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    losses = []
    times = np.expand_dims(np.array(times).T, axis=[0,2])
    magnitudes = np.expand_dims(np.array(magnitudes).T, axis=[0,2])
    accels = np.expand_dims(np.array(accels), axis=[0])
    with tf.GradientTape() as tape:
        pred = model(times, magnitudes, accels, has_accel=has_accel, training=True)
        loss = model.loss_function(pred, times[:, 1:, :], start_time, end_time)
        #print(f"Model Trainable Variables: {model.trainable_variables}")
        visuaization.save_distributions_images(pred, (start_time, end_time, 100), "output")
        losses.append(loss)
        print(f"Loss: {loss}")
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return tf.math.reduce_mean(losses, axis=0)

def main():
    start_time, end_time = preprocess.get_year_unix_times(2018)
    epochs = 10
    losses = []
    times, magnitudes, accels = preprocess.jsonl_to_data('big_data/device004_preprocessed', start_time, end_time)
    model = Recurrent()
    for i in range(epochs):
        print(f"Epoch {i}")
        loss = train(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=True)
        losses.append(loss)

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# TODO Write a function to plot the loss returned by  against the amount of epochs


main()