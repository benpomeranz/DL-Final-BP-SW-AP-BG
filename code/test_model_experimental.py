import numpy as np
import tensorflow as tf
from model_experimental import Recurrent
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Create an instance of your model
model = Recurrent()

# Generate some dummy input data
features = np.random.randn(1, 20, 98)  # Shape: (batch_size, sequence_length, num_features)

# Pass the input data through the model
output = model(features)


# Plot the distributions
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
dist = output[0]  # Select the first distribution in output
x = np.arange(0, .1, 0.0001)  # Generate a list of numbers between -1 and 1
y = [dist.prob(i) for i in x]  # Reshape the input tensor to match the shape of the output tensor
print(x, y)
axs.plot(x, y)
axs.set_title("Distribution 1")
plt.tight_layout()
plt.show()




# Check the shape of the output
# expected_shape = (10, 20)  # Replace with the expected shape of your output
# assert output.sample().shape == expected_shape, "Output shape does not match expected shape"