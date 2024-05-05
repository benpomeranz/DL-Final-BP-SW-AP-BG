import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os

def save_distributions_images(distributions, time_range, output_dir):
    """
    Visualize and save the probability density function (PDF) and cumulative distribution function (CDF)
    for each distribution in each batch of a TensorFlow Probability MixtureSameFamily distribution.

    Args:
        distributions: A TensorFlow Probability distribution object (MixtureSameFamily) with batch_shape.
        time_range: A tuple (start, stop, num_points) defining the range and resolution of time points.
        output_dir: Directory to save the output images.

    Returns:
        Saves plots of the PDF and CDF of the distributions in each batch as images.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate time points
    times = np.linspace(*time_range)

    # Loop through each batch
    for batch_index in range(distributions.batch_shape[0]):
        plt.figure(figsize=(15, 7))

        # Loop through each distribution in the batch
        for i in range(distributions.batch_shape[1]):
            pdf = distributions[batch_index, i].prob(times).numpy()
            cdf = distributions[batch_index, i].cdf(times).numpy()

            # Plot PDF
            plt.subplot(2, 1, 1)
            plt.plot(times, pdf, label=f'PDF Dist {i+1}')
            plt.title(f'Batch {batch_index + 1} - Probability Density Functions')
            plt.xlabel('Time')
            plt.ylabel('Probability Density')
            plt.grid(True)

            # Plot CDF
            plt.subplot(2, 1, 2)
            plt.plot(times, cdf, label=f'CDF Dist {i+1}')
            plt.title(f'Batch {batch_index + 1} - Cumulative Distribution Functions')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Probability')
            plt.grid(True)

        plt.subplot(2, 1, 1)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'Batch_{batch_index+1}_Distributions.png'))
        plt.close()

# Example usage
# Assuming `predicted_distribution` is your model's output distribution
# save_distributions_images(predicted_distribution, (0, 100, 500), 'output_images')

# Plotting function to visualize the loss
def plot_loss(losses, type: str = "Training"):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label=f'{type} Loss')
    plt.title(f'Loss During {type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_weibull_mixture(mixture_dist, num_points=32, x_range=(0, 3)):
    assert isinstance(mixture_dist, tfp.distributions.MixtureSameFamily), "Input is not a MixtureSameFamily distribution"
    
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    weights = mixture_dist.mixture_distribution.probs_parameter().numpy()
    weibull_components = mixture_dist.components_distribution

    # Diagnostic print to understand the structure of weights
    print("Weights shape:", weights.shape)
    print("Weights content:", weights)

    plt.figure(figsize=(10, 6))
    for i in range(weibull_components.batch_shape_tensor()[0]):
        concentration = weibull_components.concentration[i].numpy()
        scale = weibull_components.scale[i].numpy()
        weibull_dist = tfp.distributions.Weibull(concentration=concentration, scale=scale)
        weibull_pdf = weibull_dist.prob(x_values)
        
        # Ensure weights[i] is a scalar
        if np.isscalar(weights[i]):
            weight = weights[i]
        else:
            print(f"Weight at index {i}:", weights[i])  # Diagnostic print
            # Correct indexing if needed based on the prints
            weight = weights[i].item() if weights[i].size == 1 else weights[i][0]  # Adjust based on actual structure
        
        plt.plot(x_values, weibull_pdf * weight, label=f'Component {i+1} (Weight {weight:.2f})')
    
    plt.title('Weibull Mixture Components')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

