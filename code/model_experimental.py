
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Concatenate
from tensorflow.math import exp, sqrt, square
import numpy as np
import tensorflow_probability as tfp


class Recurrent(tf.keras.Model):

    '''
    Create a recurrent TPP model w/ recurrent encoder

    Args:
        input_magnitude: if magnitude be used as model input ----------------- YES
        predict_magnitude: if model predict the magnitude? ------------------- YES
        num_extra_features: Number of extra features to use as input. 
        hidden_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
        '''
    
    def __init__(self, input_magnitude: bool = True, # to use magnitude as input
                 #predict_magnitude: bool = False, # output distribution, NOT magnitude
                 hidden_size: int = 32, # hidden state size
                 num_components: int = 32, # WHAT SHOULD THIS BE ????? I THOUGHT 3 ???
                 rnn_type: str = "GRU", # REMOVE??
                 dropout_proba: float = 0.5,
                 tau_mean: float = 1.0, # mean inter-event times in data
                 mag_mean: float = 0.0, # mean earthquake magnitude in data
                 richter_b: float = 1.0, # fixed b value of Gutenberg-Richter distribution
                 mag_completeness: float = 2.0, # magnitude completeness
                 learning_rate: float = 5e-2):
        
        # initialize model
        super().__init__()

        # set parameters
        self.input_magnitude = input_magnitude
        #self.predict_magnitude = predict_magnitude
        # self.num_extra_features = num_extra_features ---> REMOVED- do we need this?
        self.hidden_size = hidden_size
        self.num_components = num_components
 
        # set untrainables
        self.tau_mean = tf.constant(tau_mean, dtype=tf.float32)
        self.log_tau_mean = tf.math.log(self.tau_mean)
        self.mag_mean = tf.constant(mag_mean, dtype=tf.float32)
        self.richter_b = tf.constant(richter_b, dtype=tf.float32)
        self.mag_completeness = tf.constant(mag_completeness, dtype=tf.float32)
        
        # set learning rate
        self.learning_rate = learning_rate

        # RNN input features
        self.num_mag_params = 1 + int(self.input_magnitude) # (1 rate)
        self.hypernet_time = tf.keras.layers.Dense(self.num_mag_params) # MIGHT NOT NEED --> used for time distribution
        self.hypernet_mag = tf.keras.layers.Dense(self.num_mag_params) # MIGHT NOT NEED --> used for magnitude distribution
        
        # RNN defining
        self.num_rnn_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # magnitude features
        )

        # input size is num_rnn_inputs
        self.rnn = tf.keras.layers.GRU(units=hidden_size, return_sequences=True)
        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout_proba)

    # call function
    def call(self, features, has_accel=True, training=False):
        '''
            inputs: 
                magnitudes: array containing the magnitudes of events
                times: array containing the times of events
                accels: array containing the accelerations of events
                has_accel: boolean indicating if the acceleration is being used
        '''

        # concatenate all features
        features = np.array(features)

        # pass features into RNN
        rnn_output = self.rnn(features, training=training)

        # dropout layer for overfit prevention
        context = self.dropout(rnn_output, training=training)

        # Time distribution parameters
        weibull_params = self.hypernet_time(context)
        # TODO: Split time_params and create a mixture distribution
        # time_params = tf.split(time_params, num_or_size_splits=3, axis=-1)
        # time_params = tf.concat(time_params, axis=-1)
        # time_params = tf.reshape(time_params, [-1, 3, self.num_components])
        # time_params = tf.nn.softmax(time_params, axis=-1)
        # time_params = tf.split(time_params, num_or_size_splits=3, axis=-1)
        # time_params = [tf.squeeze(param, axis=-1) for param in time_params]
        scale, shape, weight_logits = tf.split(
        weibull_params,
        [self.num_components, self.num_components, self.num_components],
        dim=-1,
        )
        scale = tf.math.softplus(scale.clamp_min(-5.0))
        shape = tf.math.softplus(shape.clamp_min(-5.0))
        weight_logits = tf.math.log_softmax(weight_logits, dim=-1)
        component_dists = tfp.distributions.Weibull(shape, scale)
        mixture_dist = tfp.distributions.Categorical(logits=weight_logits)
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dists,
            )
