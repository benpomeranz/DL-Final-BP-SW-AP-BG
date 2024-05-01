import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Concatenate
from tensorflow.math import exp, sqrt, square
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


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
        self.num_time_params = 3 * self.num_components
        self.hypernet_time = tf.keras.layers.Dense(self.num_time_params) # MIGHT NOT NEED --> used for time distribution
        
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
        ## SHAPE OF OUTPUT (BATCH_SIZE, SEQUENCE_LENGTH, 32)
        # print(f"Shape of rnn_output: {rnn_output.shape}")

        context = self.dropout(rnn_output, training=training)
        print(f"Shape of context: {context.shape}")

        # Time distribution parameters
        time_params = self.hypernet_time(context)
        ## SHAPE OF OUTPUT (BATCH_SIZE, SEQUENCE_LENGTH, 32)
        scale, shape, weight_logits = tf.split(
        time_params,
        [self.num_components, self.num_components, self.num_components],
        axis=-1,
        )
        scale = tf.math.softplus(tf.clip_by_value(scale, -5.0, float('inf')))
        shape = tf.math.softplus(tf.clip_by_value(shape, -5.0, float('inf')))
        weight_logits = tf.math.log_softmax(weight_logits, axis=-1)
        component_dists = tfp.distributions.Weibull(shape, scale)
        mixture_dist = tfp.distributions.Categorical(logits=weight_logits)
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=component_dists,
            )



    def loss(self, distributions, intervals):
        '''
        Compute the negative log likelihood loss.

        Args:
            distributions: A batch of sequences of TensorFlow distributions.
            intervals: Shape (B, S) interval[]

        Returns:
            The negative log likelihood loss.
        '''
        log_like = distributions.log_prob(intervals.clamp_min(1e-10))
        neg_log_likelihood = -tf.reduce_sum(log_probs)
        return neg_log_likelihood