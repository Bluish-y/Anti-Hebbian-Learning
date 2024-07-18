import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import string

class AntiHebbianNetwork:
    def __init__(self, input_size, output_size, alpha, beta, gamma, target_probability, activation_steepness, thresholds=None):
        """
        Initialize an Anti-Hebbian Network with given parameters.

        Parameters:
        - input_size: Number of input units.
        - output_size: Number of output units.
        - alpha: Anti-Hebbian learning rate.
        - beta: Hebbian learning rate.
        - gamma: Threshold learning rate.
        - target_probability: Target bit probability.
        - activation_steepness: Lambda parameter for sigmoid activation function.
        - thresholds: Initial thresholds for output units.
        """
        self.M = input_size  # Input Size
        self.N = output_size  # Output Size
        self.alpha = alpha  # Anti-Hebbian learning rate
        self.beta = beta  # Hebbian Learning Rate
        self.gamma = gamma  # Threshold Learning Rate
        self.p = target_probability  # Target bit probability
        self.lamda = activation_steepness  # Lambda

        # Initialize weights
        self.qij = np.random.randn(output_size, input_size)  # Feedforward weights
        self.wij = np.zeros((output_size, output_size))  # Feedback weights

        # Initialize thresholds
        if thresholds is None:
            self.ti = np.ones(output_size).reshape(output_size, 1)
        else:
            self.ti = thresholds
        
        self.stability_iter = 100  # Iterations for stability in simulation

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x: Input to the sigmoid function.

        Returns:
        - Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-self.lamda * x))

    def tune_thresholds(self, data, iterations=100):
        """
        Tune thresholds using Foldiak's parameters.

        Parameters:
        - data: Input data for tuning thresholds.
        - iterations: Number of iterations for tuning.

        """
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = 0, 0, 0.1  # Foldiak's parameters
        for _ in range(iterations):
            outputs = self.simulate(data)
            self.update_weights(data, outputs)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def simulate(self, inputs):
        """
        Simulate the network with given inputs.

        Parameters:
        - inputs: Input patterns to the network.

        Returns:
        - Output patterns from the network.
        """
        num_samples = inputs.shape[1]
        qxij = np.dot(self.qij, inputs)  # qij_(N x M) inputs_(M x num_samples) = N x num_samples
        yij = np.zeros((self.N, num_samples))
        for _ in range(self.stability_iter):
            wyij = np.dot(self.wij, yij)  # wij_(NxN) yij_(Nxnum_samples) = N x num_samples
            dydtij = qxij + wyij - self.ti  # (N x num_samples)
            dydtij = self.sigmoid(dydtij)  # (N x num_samples)
            yij += 0.01 * dydtij
        yij[yij > 0.5] = 1
        return yij

    def update_weights(self, inputs, yij):
        """
        Update weights based on the inputs and outputs.

        Parameters:
        - inputs: Input patterns.
        - yij: Output patterns from the network.
        """
        yi = np.mean(yij, axis=1).reshape(-1, 1)  # (N x 1)
        yxyij = np.dot(yij, yij.T) / inputs.shape[1]  # (N x N)

        # Anti-Hebbian
        self.wij += -self.alpha * (yxyij - self.p**2)
        self.wij[self.wij > 0] = 0
        np.fill_diagonal(self.wij, 0)

        # Hebbian
        self.qij += self.beta * (np.dot(yij, inputs.T) / inputs.shape[1] - yi * self.qij)

        # Threshold
        self.ti += self.gamma * (yi - self.p)

    def train(self, generate, epochs, plot=True):
        """
        Train the network over epochs using the generate function for input patterns.

        Parameters:
        - generate: Function to generate input patterns.
        - epochs: Number of training epochs.
        - plot: Whether to plot the results.
        """
        ax = None
        inputs = generate(num_patterns=100, p=self.p)
        self.tune_thresholds(inputs)
        for epoch in range(epochs):
            inputs = generate(num_patterns=100, p=self.p)
            yij = self.simulate(inputs)
            self.update_weights(inputs, yij)

            print(f"\rLoading: {epoch:.2f}/{epochs:.2f}", end="")
            if plot:
                plot_intervals = [0, int(epochs/2**2), int(epochs/2), epochs-1]
                if epoch in plot_intervals:
                    which_ax = plot_intervals.index(epoch)
                    ax = self.plot_result((8, 8), epoch, ax, which_ax)

        if plot:
            plt.show()

    def plot_result(self, input_shape, epoch, ax=None, which_ax=0):
        """
        Plot the weights of each unit as a function of learning trials.

        Parameters:
        - input_shape: Shape of the input patterns.
        - epoch: Current epoch number.
        - ax: Existing plot axis.
        - which_ax: Index of the axis to plot.
        
        Returns:
        - Updated plot axis.
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
                # a.set_title(f"{epoch}")
            a.axis('off')
        
        plt.tight_layout()
        return ax

def generate_line_patterns(size=8, num_patterns=100, p=1/8):
    """
    Generate line patterns with given parameters.

    Parameters:
    - size: Size of the square pattern.
    - num_patterns: Number of patterns to generate.
    - p: Probability parameter.

    Returns:
    - Generated line patterns.
    """
    patterns = []
    for _ in range(num_patterns):
        pattern = np.zeros((size, size))
        for i in range(size):
            if np.random.rand() < p:
                pattern[i, :] = 1
            if np.random.rand() < p:
                pattern[:, i] = 1
        patterns.append(pattern)
    return np.array(patterns).reshape((num_patterns, (size * size))).T

def generate_letter_pattern(letter, font_path='/usr/share/fonts/truetype/tlwg/Garuda.ttf', font_size=15, image_size=(8, 15)):
    """
    Generate a binary pattern for a given letter using a specified font.

    Parameters:
    - letter: Letter to generate (single character).
    - font_path: Path to the TrueType font file.
    - font_size: Size of the font.
    - image_size: Size of the output image.

    Returns:
    - Binary pattern of the letter.
    """
    # Create a blank image with a white background
    image = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(image)
    
    # Load the font and calculate the size of the letter
    font = ImageFont.truetype(font_path, font_size)
    text_size = draw.textsize(letter, font=font)
    
    # Calculate the position to center the letter in the image
    position = ((image_size[1] - text_size[0]) // 2, (image_size[0] - text_size[1]) // 2)
    
    # Draw the letter on the image
    draw.text(position, letter, fill=255, font=font)
    
    # Convert the image to a numpy array and binarize it
    pattern = np.array(image)
    pattern = (pattern > 128).astype(int)
    
    return pattern

def generate_letter_patterns(num_patterns=100, p=1/8):
    """
    Generate binary patterns for random letters using a specified font.

    Parameters:
    - num_patterns: Number of patterns to generate.
    - p: Probability parameter.

    Returns:
    - Generated letter patterns.
    """
    patterns = []
    for _ in range(num_patterns):
        letter = random.choice(string.printable)
        pattern = generate_letter_pattern(letter)
        patterns.append(pattern)
    return np.array(patterns).reshape((num_patterns, (8 * 15))).T

if __name__ == "__main__":
    p = 1/8
    network = AntiHebbianNetwork(64, 16, 0.3, 0.1, 0.1, p, 10, thresholds=None)
    network.train(generate_line_patterns, epochs = 1200)    
