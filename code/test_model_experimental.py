import numpy as np
import tensorflow as tf
from model_experimental import Recurrent
import matplotlib.pyplot as plt
import matplotlib
import math
from code.visualization import save_distributions_images
from preprocess import full_preprocess
matplotlib.use('TkAgg')



def basic_test():
    print(tf.__version__)
    # Create an instance of your model
    model = Recurrent()

    # Generate some dummy input data
    features = np.random.randn(1, 20, 98)  # Shape: (batch_size, sequence_length, num_features)

    # Pass the input data through the model
    output = model(features)


    # Plot the distributions
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    dist = output[0]  # Select the first distribution in output
    x = np.arange(0, .1, 0.001)  # Generate a list of numbers between -1 and 1
    y = [dist.prob(i) for i in x]  # Reshape the input tensor to match the shape of the output tensor
    print(x, y)
    axs.plot(x, y)
    axs.set_title("Distribution 1")
    plt.tight_layout()
    plt.show()

def test_loss():
    batch_size = 10
    sequence_size = 20
    mags = 1.5 * np.random.randn(batch_size, sequence_size, 1) + 4
    time_intervals = 50 * np.random.rand(batch_size, sequence_size + 1, 1) + 100
    accs = 1.5 * np.random.randn(batch_size, sequence_size, 96) + 4
    inputs = tf.concat([mags, time_intervals[:, :-1, :], accs], axis=-1)
    model = Recurrent()
    output = model(time_intervals, mags, accs, has_accel=True, training=False)
    print(f"Output: {output}")
    loss = model.loss_function(output, time_intervals[:, 1:, :], 0, 1)



    print(f"Loss: {loss}")
    return loss

def test_train_model():
    # wont work anymore, change format of inputs
    batch_size = 1
    sequence_size = 20
    mags = 1.5 * np.random.randn(batch_size, sequence_size, 1) + 4
    time_intervals = 50 * np.random.rand(batch_size, sequence_size + 1, 1) + 100
    accs = 1.5 * np.random.randn(batch_size, sequence_size, 96) + 4
    inputs = tf.concat([mags, time_intervals[:, :-1, :], accs], axis=-1)
    model = Recurrent()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(10):
        with tf.GradientTape() as tape:
            output = model(inputs)
            loss = model.loss_function(output, time_intervals[:, 1:, :], 0, 1)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Epoch {epoch}, Loss: {loss}")
        save_distributions_images(output, (0, 10, 100), f"output{epoch}")

    return loss

# basic_test()
test_loss()
# full_preprocess("code/big_data/device002/month=03/day=21/hour=18/45.jsonl", "test_output", 1.7, 0, 11000)
# print(test_train_model())

# Check the shape of the output
# expected_shape = (10, 20)  # Replace with the expected shape of your output
# assert output.sample().shape == expected_shape, "Output shape does not match expected shape"