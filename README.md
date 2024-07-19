# Anti-Hebbian Network Simulation

## Overview
This code simulates an Anti-Hebbian network using input patterns to update the network's weights and thresholds over a number of training epochs. It includes functions to generate input patterns, train the network, and visualize the results. This is a straightforward implementation of the algorithm provided in the Foldiak(1990) paper.

## Dependencies
The following Python packages are required to run the code:
- `matplotlib`
- `numpy`
- `Pillow`


## Components

### Class: AntiHebbianNetwork
The `AntiHebbianNetwork` class initializes the network, simulates its behavior with given inputs, updates weights, and tunes thresholds.

#### Initialization
The class is initialized with the following parameters:
- `input_size`: Number of input units.
- `output_size`: Number of output units.
- `alpha`: Anti-Hebbian learning rate.
- `beta`: Hebbian learning rate.
- `gamma`: Threshold learning rate.
- `target_probability`: Target bit probability.
- `activation_steepness`: Lambda parameter for the sigmoid activation function.
- `thresholds`: Initial thresholds for output units (optional).

#### Methods
- `sigmoid(x)`: Computes the sigmoid activation function.
- `tune_thresholds(data, iterations)`: Tunes the thresholds using Foldiak's parameters.
- `simulate(inputs)`: Simulates the network with given inputs and returns output patterns.
- `update_weights(inputs, yij)`: Updates the weights based on inputs and outputs.
- `train(generate, epochs, plot)`: Trains the network over a number of epochs.
- `plot_result(input_shape, epoch, ax, which_ax)`: Plots the weights of each unit as a function of learning trials.


## Usage
To train the network with line patterns and visualize the results, run the script:

```python
if __name__ == "__main__":
    p = 1/8
    network = AntiHebbianNetwork(64, 16, 0.3, 0.1, 0.1, p, 10, thresholds=None)
    network.train(generate_line_patterns, epochs=1200)
```
Replace generate_line_patterns with generate_letter_patterns to use letter patterns instead or just run the version 1.4 for the letter patterns. Adjust the epochs parameter to control the number of training epochs.

## Output
The script will print the progress of the training and plot the results at specified intervals, showing the weights of each unit as a function of learning trials.

