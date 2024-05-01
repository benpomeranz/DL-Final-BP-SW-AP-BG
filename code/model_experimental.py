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
        time_params = self.hypernet_time(context)
        # TODO: Split time_params and create a mixture distribution
        # time_params = tf.split(time_params, num_or_size_splits=3, axis=-1)
        # time_params = tf.concat(time_params, axis=-1)
        # time_params = tf.reshape(time_params, [-1, 3, self.num_components])
        # time_params = tf.nn.softmax(time_params, axis=-1)
        # time_params = tf.split(time_params, num_or_size_splits=3, axis=-1)
        # time_params = [tf.squeeze(param, axis=-1) for param in time_params]

        # Outputs as dictionary for now
        outputs = {
            'time_params': time_params,
            # 'magnitude_params': mag_params,  # Uncomment if magnitude is predicted
        }
        #return outputs

        hidden_states = self.rnn(batch)
        weibull_params = self.hypernet_time(hidden_states)
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
    
        #THIS NEEDS TO BE MODIFIED TO ALSO HAVE SURVIVAL PROBABILITY OF LAST EVENT UNTIL END OF TIME PERIOD

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
    
    def nll_loss(self, batch: eq.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        # output of the RNN
        context = self.get_context(batch)  # (B, L, C)
        
        # calculate times between events
        inter_time_dist = self.get_inter_time_dist(context)
        # WHY DOES BATCH HAVE AN INTER_TIMES ATTTRIBUTE (??)
        log_pdf = inter_time_dist.log_prob(batch.inter_times.clamp_min(1e-10))  # (B, L)
        log_like = (log_pdf * batch.mask).sum(-1)

        # Survival time from last event until t_end
        arange = torch.arange(batch.batch_size)
        # last item of the output sequence
        last_surv_context = context[arange, batch.end_idx, :]
        last_surv_dist = self.get_inter_time_dist(last_surv_context)
        last_log_surv = last_surv_dist.log_survival(
            batch.inter_times[arange, batch.end_idx]
        )

        # reshaping + updating
        log_like = log_like + last_log_surv.squeeze(-1)  # (B,)

        # Remove survival time from t_prev to t_nll_start
        
        if torch.any(batch.t_nll_start != batch.t_start):
            prev_surv_context = context[arange, batch.start_idx, :]
            prev_surv_dist = self.get_inter_time_dist(prev_surv_context)
            prev_surv_time = batch.inter_times[arange, batch.start_idx] - (
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_log_surv = prev_surv_dist.log_survival(prev_surv_time)
            log_like = log_like - prev_log_surv

        return -log_like / (batch.t_end - batch.t_nll_start)  # (B,)

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        # call the dense layer
        params = self.hypernet_time(context)
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
        scale, shape, weight_logits = torch.split(
            params,
            [self.num_components, self.num_components, self.num_components],
            dim=-1,
        )
        # softplus --> basically just a smooth ReLU
        scale = F.softplus(scale.clamp_min(-5.0))
        shape = F.softplus(shape.clamp_min(-5.0))
        
        weight_logits = F.log_softmax(weight_logits, dim=-1)
        # create weibull distribution
        component_dist = dist.Weibull(scale=scale, shape=shape)
        # mixtrue dist --> create a categorical distribution
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )
    

    def get_magnitude_dist(self, context):
        log_rate = self.hypernet_mag(context).squeeze(-1)  # (B, L)
        b = self.richter_b * torch.ones_like(log_rate)
        mag_min = self.mag_completeness * torch.ones_like(log_rate)
        return dist.GutenbergRichter(b=b, mag_min=mag_min)

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)
