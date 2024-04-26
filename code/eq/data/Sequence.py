import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class Weibull(tfd.Distribution):
    def __init__(self, scale, shape, eps=1e-10, validate_args=False, allow_nan_stats=True):
        self.scale = tf.convert_to_tensor(scale)
        self.shape = tf.convert_to_tensor(shape)
        self.eps = eps
        parameters = dict(locals())
        super().__init__(
            dtype=self.scale.dtype,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name='Weibull'
        )
    
    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.scale), tf.shape(self.shape))

    def _batch_shape(self):
        return tf.broadcast_static_shape(self.scale.shape, self.shape.shape)
    
    def _log_hazard(self, x):
        """Compute the logarithm of the hazard function.

        The hazard function h(x) is defined as h(x) = p(x) / S(x), where p(x) is the
        probability density function (PDF) and S(x) is the survival function (SF)
        defined as S(x) = \int_{0}^{x} p(u) du.

        Args:
            x: Input.

        Returns:
            log_h: log h(x), same shape as the input x.
        """
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))  # ensure x > 0 for numerical stability
        return tf.math.log(self.scale) + tf.math.log(self.shape) + (self.shape - 1) * tf.math.log(x)
    
    def _log_survival(self, x):
        """Compute the logarithm of the survival function.

        The survival function S(x) corresponds to Pr(X >= x) and can be computed as
        S(x) = \int_{0}^{x} p(u) du, where p(x) is the PDF.

        Args:
            x: Input.

        Returns:
            log_S: log S(x), same shape as the input x.
        """
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))  # ensure x > 0 for numerical stability
        return -self.scale * tf.pow(x, self.shape)
    
    def _log_prob(self, x):
        return self._log_hazard(x) + self._log_survival(x)
    
    def _mean(self):
        log_lmbd = -self.shape ** -1 * tf.math.log(self.scale)
        return tf.exp(log_lmbd + tf.math.lgamma(1 + self.shape ** -1))
    
    def _sample_n(self, n, seed=None):
        shape = tf.concat([[n], self._batch_shape_tensor()], axis=0)
        z = tf.random.exponential(shape=shape, dtype=self.scale.dtype)
        samples = tf.pow(z * tf.math.reciprocal(self.scale) + self.eps, tf.math.reciprocal(self.shape))
        return samples
