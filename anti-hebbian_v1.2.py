import matplotlib.pyplot as plt
import numpy as np

class AntiHebbianNetwork:
    def __init__(self, input_size, output_size, alpha, beta, gamma, target_probability, activation_steepness, thresholds=None):
        self.M = input_size  # Number of input neurons
        self.N = output_size  # Number of output neurons
        self.alpha = alpha  # Anti-Hebbian learning rate
        self.beta = beta  # Hebbian learning rate
        self.gamma = gamma  # Threshold learning rate
        self.p = target_probability  # Target bit probability
        self.lamda = activation_steepness  # Steepness of the sigmoid activation function

        self.qij = np.random.randn(output_size, input_size)  # Feedforward weights initialized randomly
        self.wij = np.zeros((output_size, output_size))  # Feedback weights initialized to zero
        
        # Initialize thresholds
        if thresholds is None:
            self.ti = np.random.rand(output_size, 1) * np.sum(self.qij**2, axis=1).reshape(-1, 1)
        else:
            self.ti = thresholds

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-self.lamda * x))
    
    def simulate(self, input):
        """
        Simulate the network with the given inputs.
        
        Parameters:
        - input: Input data
        
        Returns:
        - yij: Outputs of the network
        """
        inputs = np.array([input]).reshape(8 * 8, 1)
        num_samples = inputs.shape[1]
        qxij = np.dot(self.qij, inputs)  # Feedforward input
        yij = np.zeros((self.N, num_samples))

        # Iteratively update the network output
        for _ in range(100):
            wyij = np.dot(self.wij, yij)  # Feedback input
            dydtij = qxij + wyij - self.ti  # Total input
            dydtij = self.sigmoid(dydtij)  # Apply activation function
            yij += 0.01 * dydtij  # Update output

        # Threshold the output
        yij[yij > 0.5] = 1
        yij[yij < 0.5] = 0
        return yij
    
    def update_weights(self, input):
        """
        Update the network weights based on the input.
        
        Parameters:
        - input: Input data
        """
        xi = input.reshape(-1, 1)
        yi = self.simulate(input)

        # Anti-Hebbian learning rule for feedback weights
        self.wij += -self.alpha * (np.outer(yi, yi.T) - self.p**2)
        np.fill_diagonal(self.wij, 0)
        self.wij[self.wij > 0] = 0

        # Hebbian learning rule for feedforward weights
        self.qij += self.beta * (np.outer(yi, xi.T) - yi * self.qij)

        # Update thresholds
        self.ti += self.gamma * (yi - self.p)

        return
    
    def plot_result(self, input_shape, epoch, ax=None, which_ax=0):
        """
        Plot the feedforward weights of each unit.
        
        Parameters:
        - input_shape: Shape of the input
        - epoch: Current training epoch
        - ax: Matplotlib axes object
        - which_ax: Index of the subplot to use
        """
        num_units = self.wij.shape[0]
        if ax is None:
            fig, ax = plt.subplots(4, 16, figsize=(12, 12))
            fig.suptitle(f"Weights of each unit as a function of learning trials")
        
        for i, __ in enumerate(ax.flat):
            if i < num_units:
                unit_weights = self.qij[i].reshape(input_shape)
                a = ax.flat[which_ax * 16 + i]
                im = a.imshow(unit_weights, cmap='binary', interpolation='nearest')
                a.set_title(f"{epoch/100}")
            a.axis('off')
        
        plt.tight_layout()
        return ax

    def train(self, inputs, epochs, plot=True):
        """
        Train the network.
        
        Parameters:
        - inputs: Input data
        - epochs: Number of training epochs
        - plot: Whether to plot the results
        """
        ax = None
        for epoch in range(epochs):
            input = inputs[epoch]
            self.update_weights(input)
            print(f"\rLoading: {epoch:.2f}/{epochs:.2f}", end="")
            if plot:
                plot_intervals = [0, int(epochs/2**2), int(epochs/2), epochs-1]
                if epoch in plot_intervals:
                    which_ax = plot_intervals.index(epoch)
                    ax = self.plot_result(input.shape, epoch, ax, which_ax)
        if plot:
            plt.show()

def generate_line_patterns(size=8, num_patterns=100):
    """
    Generate line patterns for the input data.
    
    Parameters:
    - size: Size of the pattern
    - num_patterns: Number of patterns to generate
    
    Returns:
    - patterns: Generated patterns
    """
    patterns = []
    for _ in range(num_patterns):
        pattern = np.zeros((size, size))
        for i in range(size):
            if np.random.rand() < 1/8:
                pattern[i, :] = 1
            if np.random.rand() < 1/8:
                pattern[:, i] = 1
        patterns.append(pattern)
    return np.array(patterns)

if __name__ == "__main__":
    # Generate initial line patterns
    inputs = generate_line_patterns(num_patterns=100)
    
    # Create and stabilize the network
    stable_network = AntiHebbianNetwork(64, 16, 0, 0, 0.2, 1/8, 10)
    stable_network.train(inputs, epochs=100, plot=False)
    thresholds = stable_network.ti

    # Train the network
    epochs = 50000
    inputs = generate_line_patterns(num_patterns=epochs)
    network = AntiHebbianNetwork(64, 16, 0.1, 0.02, 0.02, 1/8, 10, thresholds=thresholds)
    network.train(inputs, epochs=epochs)
