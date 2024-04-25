import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Concatenate
from tensorflow.math import exp, sqrt, square

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
    
    def evaluate_compensator(
        # TODO: Implement data.Sequence
        self, sequence: eq.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the compensator for the given sequence (used for plotting).

        Args:
            sequence: Sequence for which to evaluate the compensator.
            num_grid_points: Number of points between consecutive events on which to
                evaluate the compensator.

        Returns:
            grid: Times for which the intensity is evaluated,
                shape (seq_len * num_grid_points,)
            intensity: Values of the conditional intensity on times in grid,
                shape (seq_len * num_grid_points,)
        """
        raise NotImplementedError

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[eq.data.Sequence] = None,
    ) -> eq.data.Batch:
        """
        Sample a batch of sequences from the TPP model.

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the time interval on which the sequence is simulated.

        Returns:
            batch: Batch of padded event sequences.
        """
        raise NotImplementedError

  
    def evaluate_compensator(
        self, sequence: eq.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the compensator for the given sequence (used for plotting).

        Args:
            sequence: Sequence for which to evaluate the compensator.
            num_grid_points: Number of points between consecutive events on which to
                evaluate the compensator.

        Returns:
            grid: Times for which the intensity is evaluated,
                shape (seq_len * num_grid_points,)
            intensity: Values of the conditional intensity on times in grid,
                shape (seq_len * num_grid_points,)
        """
        raise NotImplementedError

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
