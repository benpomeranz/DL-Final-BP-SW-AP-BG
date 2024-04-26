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
        context_size: Size of the RNN hidden state.
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
                 context_size: int = 32, # hidden state size
                 num_components: int = 32,
                 rnn_type: str = "GRU",
                 dropout_proba: float = 0.5,
                 tau_mean: float = 1.0, # mean inter-event times in data
                 mag_mean: float = 0.0, # mean earthquake magnitude in data
                 richter_b: float = 1.0, # fixed b value of Gutenberg-Richter distribution
                 mag_completeness: float = 2.0, # magnitude completeness
                 learning_rate: float = 5e-2,):
        
        # initialize model
        super().__init__()

        # set parameters
        self.input_magnitude = input_magnitude
        self.predict_magnitude = predict_magnitude
        # self.num_extra_features = num_extra_features ---> REMOVED- do we need this?
        self.context_size = context_size
        self.num_components = num_components

        # set untrainables
        self.tau_mean = tf.constant(tau_mean, dtype=tf.float32)
        self.log_tau_mean = tf.math.log(self.tau_mean)
        self.mag_mean = tf.constant(mag_mean, dtype=tf.float32)
        self.richter_b = tf.constant(richter_b, dtype=tf.float32)
        self.mag_completeness = tf.constant(mag_completeness, dtype=tf.float32)
        
        # set learning rate
        self.learning_rate = learning_rate

        # Create decoder


        # 
        # Decoder for the time distribution
        self.num_time_params = 3 * self.num_components
        self.hypernet_time = nn.Linear(context_size, self.num_time_params)

        # RNN input features
        if self.input_magnitude:
            # Decoder for magnitude
            self.num_mag_params = 1  # (1 rate)
            self.hypernet_mag = nn.Linear(context_size, self.num_mag_params)

        if rnn_type not in ["RNN", "GRU"]:
            raise ValueError(
                f"rnn_type must be one of ['RNN', 'GRU'] " f"(got {rnn_type})"
            )
        self.num_rnn_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # magnitude features
            + 0 if self.num_extra_features is None else self.num_extra_features
        )

        self.rnn = getattr(nn, rnn_type)(
            self.num_rnn_inputs, context_size, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_proba)

        
    def __init__(self, context_size: int):
        super(Recurrent, self).__init__()
        self.rnn = tf.keras.layers.SimpleRNN(context_size, return_sequences=True)
        self.flatten = Flatten()

    '''
    neural TPP model with recurrent encoder
    '''

"""Neural TPP model with an recurrent encoder.

    Args:
        input_magnitude: Should magnitude be used as model input?
        predict_magnitude: Should the model predict the magnitude?
        num_extra_features: Number of extra features to use as input.
        context_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
    """

    def __init__(
        self,
        input_magnitude: bool = True,
        predict_magnitude: bool = True,
        num_extra_features: Optional[int] = None,
        context_size: int = 32,
        num_components: int = 32,
        rnn_type: str = "GRU",
        dropout_proba: float = 0.5,
        tau_mean: float = 1.0,
        mag_mean: float = 0.0,
        richter_b: float = 1.0,
        mag_completeness: float = 2.0,
        learning_rate: float = 5e-2,
    ):
        super().__init__()
        self.input_magnitude = input_magnitude
        self.predict_magnitude = predict_magnitude
        self.num_extra_features = num_extra_features
        self.context_size = context_size
        self.num_components = num_components
        self.register_buffer("tau_mean", torch.tensor(tau_mean, dtype=torch.float32))
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("mag_mean", torch.tensor(mag_mean, dtype=torch.float32))
        self.register_buffer("richter_b", torch.tensor(richter_b, dtype=torch.float32))
        self.register_buffer(
            "mag_completeness", torch.tensor(mag_completeness, dtype=torch.float32)
        )
        self.learning_rate = learning_rate

        # Decoder for the time distribution
        self.num_time_params = 3 * self.num_components
        self.hypernet_time = nn.Linear(context_size, self.num_time_params)

        # RNN input features
        if self.input_magnitude:
            # Decoder for magnitude
            self.num_mag_params = 1  # (1 rate)
            self.hypernet_mag = nn.Linear(context_size, self.num_mag_params)

        if rnn_type not in ["RNN", "GRU"]:
            raise ValueError(
                f"rnn_type must be one of ['RNN', 'GRU'] " f"(got {rnn_type})"
            )
        self.num_rnn_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # magnitude features
            + 0 if self.num_extra_features is None else self.num_extra_features
        )

        self.rnn = getattr(nn, rnn_type)(
            self.num_rnn_inputs, context_size, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_proba)

class TPPModel(tf.keras.Model):

    def __init__(self):
        # intialize model
        super().__init__()
        self.save_model() #This is saving the entire model including weights and entire architectuire
        # included for consistent results
    
    def nll_loss(self, batch):
        '''
        Compute the negative log-likelihood for the event sequences
        Args:
            batch: batch of event sequences
        '''
        raise NotImplementedError

    def sample(self, batch_size: int, duration: float, t_start:float = 0.0, past_seq = None):
        """""
        Sample a batch of sequences from the TPP Model.

        Args:
            batch_Size: Number of sequences to generate
            duration: Length of the time interval on which the sequence is simulates

        Returns:
            batch: Batch of padded event sequences
        """

        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        '''
        Compute the training loss for the batch
        Args:
            batch: batch of event sequences
            batch_idx: index of the batch
        '''
        # calculate negative log loss of the batch 
        loss = self.nll_loss(batch).mean()

    def training_step(self, batch, batch_idx):
        loss = self.nll_loss(batch).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.nll_loss(batch).mean()
        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )

    def test_step(self, batch, batch_idx, dataset_idx=None):
        with torch.no_grad():
            loss = self.nll_loss(batch).mean()
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )

    def configure_optimizers(self):
        if hasattr(self, "learning_rate"):
            lr = self.learning_rate
        else:
            lr = 1e-2
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
