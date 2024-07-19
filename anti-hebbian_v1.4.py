import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import string
from sklearn.datasets import fetch_openml
from skimage.transform import resize

# Define the image size
im_size = (8, 8)

class AntiHebbianNetwork:
    def __init__(self, input_size, output_size, alpha, beta, gamma, target_probability, activation_steepness, thresholds=None):
        """
        Initialize the Anti-Hebbian Network with the given parameters.
        
        Parameters:
        - input_size: Size of the input layer
        - output_size: Size of the output layer
        - alpha: Anti-Hebbian learning rate
        - beta: Hebbian learning rate
        - gamma: Threshold learning rate
        - target_probability: Target bit probability
        - activation_steepness: Steepness of the activation function
        - thresholds: Initial thresholds (optional)
        """
        self.M = input_size  # Input Size
        self.N = output_size  # Output Size
        self.alpha = alpha  # Anti-Hebbian learning rate
        self.beta = beta  # Hebbian Learning Rate
        self.gamma = gamma  # Threshold Learning Rate
        self.p = target_probability  # Target bit probability
        self.lamda = activation_steepness  # Activation steepness

        self.qij = np.random.randn(output_size, input_size)  # Feedforward weights
        self.wij = np.zeros((output_size, output_size))  # Feedback weights

        if thresholds is None:
            self.ti = np.ones(output_size).reshape(output_size, 1)  # Initial thresholds
        else:
            self.ti = thresholds
        
        self.stability_iter = 100  # Number of iterations for stability

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-self.lamda * x))
    
    def tune_thresholds(self, data, iterations=100):
        """
        Tune thresholds using Foldiak's parameters.
        
        Parameters:
        - data: Input data
        - iterations: Number of iterations for tuning
        """
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = 0, 0, 0.1  # Foldiak's parameters
        for _ in range(iterations):
            outputs = self.simulate(data)
            self.update_weights(data, outputs)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def simulate(self, inputs):
        """
        Simulate the network with the given inputs.
        
        Parameters:
        - inputs: Input data
        
        Returns:
        - yij: Outputs of the network
        """
        num_samples = inputs.shape[1]
        qxij = np.dot(self.qij, inputs)  # Feedforward input
        yij = np.zeros((self.N, num_samples))

        for _ in range(self.stability_iter):
            wyij = np.dot(self.wij, yij)  # Feedback input
            dydtij = qxij + wyij - self.ti  # Total input
            dydtij = self.sigmoid(dydtij)  # Activation function
            yij += 0.01 * dydtij  # Update output

        yij[yij > 0.5] = 1
        yij[yij < 0.5] = 0
        return yij
    
    def update_weights(self, inputs, yij):
        """
        Update the weights of the network.
        
        Parameters:
        - inputs: Input data
        - yij: Outputs of the network
        """
        yi = np.mean(yij, axis=1).reshape(-1, 1)  # Mean output
        yxyij = np.dot(yij, yij.T) / inputs.shape[1]  # Correlation matrix

        # Anti-Hebbian update
        self.wij += -self.alpha * (yxyij - self.p**2)
        self.wij[self.wij > 0] = 0
        np.fill_diagonal(self.wij, 0)

        # Hebbian update
        self.qij += self.beta * (np.dot(yij, inputs.T) / inputs.shape[1] - yi * self.qij)

        # Threshold update
        self.ti += self.gamma * (yi - self.p)

    def train(self, generate, epochs, plot=True):
        """
        Train the network.
        
        Parameters:
        - generate: Function to generate input patterns
        - epochs: Number of training epochs
        - plot: Whether to plot the results
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
                plot_intervals = [0, int(epochs/4), int(epochs/2), epochs-1]
                if epoch in plot_intervals:
                    which_ax = plot_intervals.index(epoch)
                    ax = self.plot_result(im_size, epoch, ax, which_ax)
        
        if plot:
            plt.show()

    def plot_result(self, input_shape, epoch, ax=None, which_ax=0):
        """
        Plot the results of the network training.
        
        Parameters:
        - input_shape: Shape of the input patterns
        - epoch: Current epoch
        - ax: Axes for plotting (optional)
        - which_ax: Index of the subplot
        
        Returns:
        - ax: Updated axes
        """
        num_units = self.wij.shape[0]

        if ax is None:
            fig, ax = plt.subplots(4, 16, figsize=(12, 12))
            fig.suptitle("Weights of each unit as a function of learning trials")

        for i, __ in enumerate(ax.flat):
            if i < num_units:
                unit_weights = self.qij[i].reshape(input_shape)
                a = ax.flat[which_ax * 16 + i]
                im = a.imshow(unit_weights, cmap='binary', interpolation='nearest')
                a.set_title(f"{epoch}")
            a.axis('off')

        plt.tight_layout()
        return ax

def generate_mnist_patterns(size=8, num_patterns=100, p=1/8):
    """
    Generate patterns from the MNIST dataset.
    
    Parameters:
    - size: Size of the output patterns
    - num_patterns: Number of patterns to generate
    - p: Probability of each pixel being 1
    
    Returns:
    - patterns: Generated patterns
    """
    mnist = fetch_openml('mnist_784')
    patterns = []
    images = mnist.data.values
    labels = mnist.target.astype(int).values

    for _ in range(num_patterns):
        letters = "abcdefghijklmnopqrstuvwxyz"
        probabilities = [0.082, 0.015, 0.028, 0.043, 0.127, 0.022, 0.020, 0.061, 0.070, 0.002,
                         0.008, 0.040, 0.024, 0.067, 0.075, 0.019, 0.001, 0.060, 0.063, 0.091,
                         0.028, 0.010, 0.023, 0.001, 0.020, 0.001]

        # Generate a random character based on the probabilities
        letter = random.choices(letters, probabilities)[0]
        images_resized = get_and_resize_images_for_label(letter, images, labels)
        pattern = np.array(images_resized)
        pattern = (pattern > 128).astype(int)
        patterns.append(pattern)

    return np.array(patterns).reshape((num_patterns, im_size[0] * im_size[1])).T

def get_and_resize_images_for_label(label, images, labels, target_size=(8, 8)):
    """
    Resize images for the given label.
    
    Parameters:
    - label: Target label
    - images: Array of images
    - labels: Array of labels
    - target_size: Target size of the images
    
    Returns:
    - resized_images: Resized images
    """
    indices = np.where(labels == label)[0]
    filtered_images = images[indices]

    resized_images = np.zeros((filtered_images.shape[0], target_size[0], target_size[1]))

    for i in range(filtered_images.shape[0]):
        image = filtered_images[i].reshape(28, 28)  # MNIST images are 28x28
        resized_image = resize(image, target_size, anti_aliasing=True)
        resized_images[i] = resized_image

    return resized_images

def generate_line_patterns(size=8, num_patterns=100, p=1/8):
    """
    Generate line patterns.
    
    Parameters:
    - size: Size of the patterns
    - num_patterns: Number of patterns to generate
    - p: Probability of each line being 1
    
    Returns:
    - patterns: Generated patterns
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
    return np.array(patterns).reshape((num_patterns, size * size)).T

def generate_letter_pattern(letter, font_path='/usr/share/fonts/X11/c0648bt_.pfb', font_size=8, image_size=(8, 15)):
    """
    Generate a pattern for a given letter.
    
    Parameters:
    - letter: Target letter
    - font_path: Path to the font file
    - font_size: Size of the font
    - image_size: Size of the output image
    
    Returns:
    - pattern: Generated pattern
    """
    image = Image.new('L', (image_size[1], image_size[0]), 0)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_path, font_size)
    position = (1, -1)

    draw.text(position, letter, fill=255, font=font)
    pattern = np.array(image)
    pattern = (pattern > 128).astype(int)

    return pattern

def generate_letter_patterns(num_patterns=100, p=1/8):
    """
    Generate patterns for random letters.
    
    Parameters:
    - num_patterns: Number of patterns to generate
    - p: Probability of each pixel being 1
    
    Returns:
    - patterns: Generated patterns
    """
    patterns = []
    for _ in range(num_patterns):
        letters = "abcdefghijklmnopqrstuvwxyz"
        probabilities = [0.082, 0.015, 0.028, 0.043, 0.127, 0.022, 0.020, 0.061, 0.070, 0.002,
                         0.008, 0.040, 0.024, 0.067, 0.075, 0.019, 0.001, 0.060, 0.063, 0.091,
                         0.028, 0.010, 0.023, 0.001, 0.020, 0.001]

        letter = random.choices(letters, probabilities)[0]
        pattern = generate_letter_pattern(letter, image_size=im_size)
        patterns.append(pattern)

    return np.array(patterns).reshape((num_patterns, im_size[0] * im_size[1])).T

if __name__ == "__main__":
    p = 1/8
    network = AntiHebbianNetwork(64, 16, 0.3, 0.1, 0.1, p, 10, thresholds=None)
    network.train(generate_line_patterns, epochs=1200)
