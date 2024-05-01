import argparse
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.math import sigmoid
from tqdm import tqdm

from model import TPPModel
# nll takes in list of distributions, input to call

def train(model, data, has_accel):
    for magnitude, times, accels in data:
        with tf.GradientTape() as tape:
            pred = model.call(magnitude, times, accels, has_accel, training=True)
            loss = model.nll(pred, has_accel)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))