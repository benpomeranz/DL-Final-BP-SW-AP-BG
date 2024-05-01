import numpy as np
import tensorflow as tf
from model_experimental import Recurrent


print(tf.__version__)
# Create an instance of your model
model = Recurrent()

# Generate some dummy input data
features = np.random.randn(1, 20, 98)  # Shape: (batch_size, sequence_length, num_features)

# Pass the input data through the model
output = model(features)

print(output)

# Check the shape of the output
# expected_shape = (10, 20)  # Replace with the expected shape of your output
# assert output.sample().shape == expected_shape, "Output shape does not match expected shape"