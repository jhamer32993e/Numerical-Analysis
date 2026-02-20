import numpy as np
import matplotlib.pyplot as plt


# Two-layer Neural Network
def two_layer_network(x, biases, output_weights, output_bias=0):
    """
    Evaluates a two-layer neural network approximation.

    Parameters:
    x (float): The value at which to evaluate the network.
    biases (list or array): Biases for hidden layer
        These determine the knot points: x_i = -b_i
    output_weights (list or array) : Weights for output layer
    output_bias (float): Bias for output layer (default: 0)

    Returns:
    --------
    float: The output of the neural network.
    """
    f_nn = output_bias * np.ones_like(x)  # bias term
    for i in range(len(biases)):
        f_nn += output_weights[i] * np.maximum(0, x + biases[i])
    return f_nn


def plot_neural_network(
    biases, output_weights, output_bias=0, x_range=(0, np.pi), target_func=np.sin
):
    """
    Plot a two-layer neural network approximation.

    Parameters:
    biases (list or array): Biases for hidden layer
        These determine the knot points: x_i = -b_i
    output_weights (list or array) : Weights for output layer
    output_bias (float): Bias for output layer (default: 0)
    x_range : tuple
        (x_min, x_max) for plotting (default: (0, np.pi))
    target_func : function
        The target function to approximate (default: np.sin)

    Returns:
    float: Maximum absolute error
    """
    x = np.linspace(x_range[0], x_range[1], 500)

    # Compute neural network output
    f_nn = two_layer_network(x, biases, output_weights, output_bias)

    # Compute target
    f_target = target_func(x)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, f_target, "r-", linewidth=2, label="Target function")
    plt.plot(x, f_nn, "b--", linewidth=2, label="Neural network")

    # Mark knot points
    knots = [-b for b in biases]
    for knot in knots:
        if x_range[0] <= knot <= x_range[1]:
            plt.axvline(knot, color="gray", linestyle=":", alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural Network Approximation")
    plt.show()

    # Calculate and return error
    error = np.max(np.abs(f_target - f_nn))
    print(f"Maximum error: {error:.4f}")
    return error


# Example usage:
output_weights = [0.9, -0.5, -0.8, -0.5]
biases = [0, -np.pi / 4, -np.pi / 2, -3 * np.pi / 4]
plot_neural_network(biases, output_weights)


# Depth Comparison
# 2.27
def shallow_2_peak_network(x):
    biases = [0, -0.25, -0.5, -0.75, -1]
    output_weights = [4, -8, 8, -8, 4]
    return two_layer_network(x, biases, output_weights)


# Plot
plt.figure(figsize=(10, 6))
x = np.linspace(-0.1, 1.1, 500)
plt.plot(x, shallow_2_peak_network(x), "r-", lw=3)
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("2-Peak Sawtooth Function")
plt.show()


def g(x):
    """ Single triangular peak """
    biases = [0, -0.5, -1]
    output_weights = [2, -4, 2]
    return two_layer_network(x, biases, output_weights)

# Plot
plt.figure(figsize=(10, 6))
x = np.linspace(-0.1,1.1,500)
plt.plot(x, g(x), 'r-', lw=3)
plt.grid()
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Single peak function')
plt.show()

f = lambda x: g(g(x))

# Plot
plt.figure(figsize=(10, 6))
x = np.linspace(-0.1,1.1,500)
plt.plot(x, f(x), 'r-', lw=3)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('2-Peak Sawtooth Function')
plt.show()

h = lambda x: g(g(g(x)))

# Plot
plt.figure(figsize=(10, 6))
x = np.linspace(-0.1,1.1,500)
plt.plot(x, h(x), 'r-', lw=3)
plt.grid()
plt.xlabel('x')
plt.ylabel('h(x)')
plt.title('4-Peak Sawtooth Function')
plt.show()

# TODO: Create arrays for different numbers of peaks
peaks = [2, 4, 8, 16, 32, 64]
shallow_neurons = [5, 9, 17, 33, 65, 129]  # Fill in based on your formula
deep_neurons = [6, 9, 12, 15, 18, 21]     # Fill in based on your formula

# Plot
plt.figure(figsize=(8, 5))
plt.plot(peaks, shallow_neurons, 'bo-', label='Shallow Network', lw=2)
plt.plot(peaks, deep_neurons, 'mo-', label='Deep Network', lw=2)
plt.xlabel('Number of Peaks')
plt.ylabel('Number of Hidden Neurons')
plt.title('Scaling: Shallow vs Deep Networks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()