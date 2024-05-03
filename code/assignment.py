import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
# from data import Dataset
from tqdm import tqdm
import preprocess 
from preprocess import jsonl_to_data
from model_experimental import Recurrent
import visuaization

# from model import TPPModel
# nll takes in list of distributions, input to call
def train(model, times, magnitudes, accels, start_time, end_time, sequence_length, has_accel):
    # magnitudes: [batchsize(1) x sequence size x 1]
    # times: [batchsize(1) x sequence size + 1 x 1]
    # accels: [batchsize(1) x sequence size x 96]
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    losses = []
    times = np.expand_dims(np.array(times).T, axis=[0,2])
    magnitudes = np.expand_dims(np.array(magnitudes).T, axis=[0,2])
    accels = np.expand_dims(np.array(accels), axis=[0])
    with tf.GradientTape() as tape:
        pred = model(times, magnitudes, accels, has_accel=has_accel, training=True)
        loss = model.loss_function(pred, times[:, 1:, :], start_time, end_time)
        #print(f"Model Trainable Variables: {model.trainable_variables}")
        #visuaization.save_distributions_images(pred, (start_time, end_time, 100), "output")
        losses.append(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return losses, pred

def validate(model, times, magnitudes, accels, start_time, end_time, sequence_length, has_accel):
    losses = []
    times = np.expand_dims(np.array(times).T, axis=[0,2])
    magnitudes = np.expand_dims(np.array(magnitudes).T, axis=[0,2])
    accels = np.expand_dims(np.array(accels), axis=[0])
    pred = model(times, magnitudes, accels, has_accel=has_accel, training=False)
    loss = model.loss_function(pred, times[:, 1:, :], start_time, end_time)
    losses.append(loss)
    return losses, pred

def main():
    start_time, end_time = preprocess.get_year_unix_times(2018)
    epochs = 10
    model = Recurrent()
    training_losses = []
    validation_losses = []
    test_losses = []

    # Here is some code to loop through all of our training data and then run it against our validation data
    if os.path.exists('data/training'):
        # We now loop through all of our training data for the specified number of epochs
        for i in range(epochs):
            epoch_training_losses = []
            epoch_validation_losses = []
            for filename in os.listdir('data/training'):
                file_path = os.path.join('data/training', filename)
                if os.path.isfile(file_path):
                    times, magnitudes, accels = preprocess.jsonl_to_data(file_path, start_time, end_time)
                    losses, train_pred = train(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=False)
                    epoch_training_losses.append(losses)
            training_losses.append(tf.math.reduce_mean(epoch_training_losses))
            # We now loop through all of our validation data
            for filename in os.listdir('data/validation'):
                file_path = os.path.join('data/validation', filename)
                if os.path.isfile(file_path):
                    times, magnitudes, accels = preprocess.jsonl_to_data(file_path, start_time, end_time)
                    losses, valid_pred = validate(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=False)
                    epoch_validation_losses.append(losses)
            validation_losses.append(tf.math.reduce_mean(epoch_validation_losses))
            print(f"Epoch {i}, Training Loss: {training_losses[-1]}, Validation Loss: {validation_losses[-1]}")

    #visuaization.plot_weibull_mixture(train_pred)
    visuaization.plot_loss(training_losses)
    visuaization.plot_loss(validation_losses, "Validation")

    # Now we test our model on the test data
    for filename in os.listdir('data/testing'):
        file_path = os.path.join('data/testing', filename)
        if os.path.isfile(file_path):
            times, magnitudes, accels = preprocess.jsonl_to_data(file_path, start_time, end_time)
            losses, test_pred = validate(model, times, magnitudes, accels, start_time, end_time, len(magnitudes), has_accel=False)
            test_losses.append(tf.math.reduce_mean(losses))
            print(f"Test Loss: {test_losses[-1]}")

    visuaization.plot_loss(test_losses, "Testing")


if __name__ == "__main__":
    main()