
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Concatenate
from tensorflow.math import exp, sqrt, square

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
                 predict_magnitude: bool = False, # output distribution, NOT magnitude
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
        self.predict_magnitude = predict_magnitude
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


        
        
        # take in size of 
        self.hypernet_mag = tf.keras.layers.Dense(self.num_mag_params)
        nn.Linear(context_size, self.num_mag_params)


        # define layers
        self.gru = tf.keras.layers.GRU(units=hidden_size, return_sequences=True)
        self.feed_forward = tf.keras.layers.Dense(3 * num_components)
