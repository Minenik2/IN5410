import math
from random import random
from typing import Any

import numpy as np
import pandas as pd

def generate_random_weights(n: int) -> list[float]:
    """Generate a list of random weights."""
    return [random() for _ in range(n)]

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    """Derivative of the sigmoid, given the neuron's input."""
    s = sigmoid(x)
    return s * (1 - s)


class NeuralNetwork:
    def __init__(self, w1: float, w2: float, w3: float, w4: float, w5: float, w6: float, alpha: float = 0.4) -> None:
        """Initialize weights"""
        self.w1, self.w2 = w1, w2
        self.w3, self.w4 = w3, w4
        self.w5, self.w6 = w5, w6
        self.alpha = alpha  # Learning rate

        self.sum_h1 = None
        self.sum_h2 = None
        self.out_h1 = None
        self.out_h2 = None
        self.sum_y1 = None
        self.delta_y1 = None
        self.delta_h1 = None
        self.delta_h2 = None
        self.out_y1 = None

        self.dE_dw1 = None
        self.dE_dw2 = None
        self.dE_dw3 = None
        self.dE_dw4 = None
        self.dE_dw5 = None
        self.dE_dw6 = None

    def compute_hidden_net(self, x1: float, x2: float) -> None:
        """Net inputs for hidden layer"""
        self.sum_h1 = self.w1 * x1 + self.w2 * x2
        self.sum_h2 = self.w3 * x1 + self.w4 * x2

    def activate_hidden(self) -> None:
        """Activations of hidden layer"""
        self.out_h1 = sigmoid(self.sum_h1)
        self.out_h2 = sigmoid(self.sum_h2)

    def compute_output_net(self) -> None:
        """Net input for output neuron"""
        self.sum_y1 = self.w5 * self.out_h1 + self.w6 * self.out_h2

    def activate_output(self) -> None:
        """Activation (prediction) of output neuron"""
        self.out_y1 = sigmoid(self.sum_y1)

    def compute_error(self, y: float) -> float:
        """Squared error"""
        return 0.5 * (y - self.out_y1) ** 2

    def compute_output_delta(self, y: float) -> None:
        """Error signal for the output neuron"""
        self.delta_y1 = (self.out_y1 - y) * sigmoid_derivative(self.sum_y1)

    def compute_hidden_deltas(self) -> None:
        """Error signals for hidden neurons"""
        self.delta_h1 = self.delta_y1 * self.w5 * sigmoid_derivative(self.sum_h1)
        self.delta_h2 = self.delta_y1 * self.w6 * sigmoid_derivative(self.sum_h2)

    def update_weights(self, x1: float, x2: float) -> None:
        """Update weights from hidden to output"""
        self.dE_dw5 = self.delta_y1 * self.out_h1
        self.w5 -= self.alpha * self.dE_dw5

        self.dE_dw6 = self.delta_y1 * self.out_h2
        self.w6 -= self.alpha * self.dE_dw6
        # Update weights from input to hidden
        self.dE_dw1 = self.delta_h1 * x1
        self.w1 -= self.alpha * self.dE_dw1

        self.dE_dw2 = self.delta_h1 * x2
        self.w2 -= self.alpha * self.dE_dw2

        self.dE_dw3 = self.delta_h2 * x1
        self.w3 -= self.alpha * self.dE_dw3

        self.dE_dw4 = self.delta_h2 * x2
        self.w4 -= self.alpha * self.dE_dw4

    def train_step(self, x1: float, x2: float, y: float) -> float:
        """Performs one forward + backward propagation step and updates weights."""
        # Forward propagation
        self.compute_hidden_net(x1, x2)
        self.activate_hidden()
        self.compute_output_net()
        self.activate_output()

        # Compute error
        error = self.compute_error(y)

        # Backward propagation
        self.compute_output_delta(y)
        self.compute_hidden_deltas()
        self.update_weights(x1, x2)

        return error

    def iterate(
            self,
            x1: float,
            x2: float,
            y: float,
            delta_error: float
    ) -> tuple[dict[int | Any, list[float]], dict[int | Any, float | Any], dict[int | Any, list[Any]], dict[
        int | Any, list[Any]], dict[int | Any, list[Any]]]:
        """Performs training iterations until the error is less than `delta_error`.
        Returns the error and weights for each iteration.
        """
        weights, error_dict = {}, {}  # stores error and weights from each iteration
        sums, outputs = {}, {}  # stores sums and outputs from each iteration
        derivatives = {}  # stores derivatives from each iteration

        num_iterations = 0

        weights[num_iterations] = [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6]

        error = self.train_step(x1, x2, y)
        error_dict[num_iterations] = error
        sums[num_iterations] = [self.sum_h1, self.sum_h2, self.sum_y1]
        outputs[num_iterations] = [self.out_h1, self.out_h2, self.out_y1]
        derivatives[num_iterations] = [self.dE_dw1, self.dE_dw2, self.dE_dw3, self.dE_dw4, self.dE_dw5, self.dE_dw6]

        while error > delta_error:
            error = self.train_step(x1, x2, y)
            num_iterations += 1
            error_dict[num_iterations] = error
            weights[num_iterations] = [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6]
            sums[num_iterations] = [self.sum_h1, self.sum_h2, self.sum_y1]
            outputs[num_iterations] = [self.out_h1, self.out_h2, self.out_y1]
            derivatives[num_iterations] = [self.dE_dw1, self.dE_dw2, self.dE_dw3, self.dE_dw4, self.dE_dw5, self.dE_dw6]
        return weights, error_dict, sums, outputs, derivatives


if __name__ == "__main__":
    """For random weights, please uncomment the 3 lines below"""
    # random_weights = generate_random_weights(6)
    # w1, w2, w3, w4, w5, w6 = random_weights
    # nn = NeuralNetwork(w1, w2, w3, w4, w5, w6, alpha=0.4)
    """For our report case, we are using these initial random weights"""
    nn = NeuralNetwork(
        w1=0.370000, w2=0.950000,
        w3=0.730000, w4=0.60000,
        w5=0.150000, w6=0.150000,
        alpha=0.4
    )
    x1, x2, target = 0.04, 0.20, 0.50

    weights, error_dict, sums, outputs, derivatives = nn.iterate(x1, x2, target, delta_error=0)

    print("Iter |  W1 | W2 | w3| w4| w5| w6|  Error")
    iterations = list(range(10)) + [len(weights) - 1]
    for i in iterations:
        error = error_dict[i]
        w1, w2, w3, w4, w5, w6 = weights[i]
        print(f"Iter {i}: {w1:.6f} | {w2:.6f} | {w3:.6f} | {w4:.6f} | {w5:.6f} | {w6:.6f} | {error:.6f}")

    print('------------Iteration 0-------------------------------')
    print('Sum for iteration 0:')
    print(sums[0])
    print('Outputs for iteration 0:')
    print(outputs[0])
    print('Derivatives for iteration 0:')
    print(derivatives[0])
    print('-------------------------------------------')

    print('------------Iteration 1-------------------------------')
    print('Sum for iteration 1:')
    print(sums[1])
    print('Outputs for iteration 1:')
    print(outputs[1])
    print('Derivatives for iteration 1:')
    print(derivatives[1])
    print('-------------------------------------------')

    print("Iter |  sum_h1 | out_h1 | sum_h2| out_h2| sum_y1| out_y1|  Error")
    for i in iterations:
        sum_h1, sum_h2, sum_y1 = sums[i]
        out_h1, out_h2, out_y1 = outputs[i]
        print(
            f"Iter {i}: {sum_h1:.6f} | {out_h1:.6f} | {sum_h2:.6f} | {out_h2:.6f} | {sum_y1:.6f} | {out_y1:.6f} | {error_dict[i]:.6f}")

    df = pd.DataFrame(error_dict.items(), columns=['Iteration', 'Error'])

    diff = df['Error'].diff()  # NaN, Δ1, Δ2, ...
    df['Trend'] = np.select(
        [diff.isna(), diff > 0, diff < 0],  # conditions
        ['N/A', 'Increasing', 'Decreasing'],
        default='Stable'
    )

    print(df.head(10).to_string(index=False))
    print('-------------------------------------------')
    print(df.tail(10).to_string(index=False))
