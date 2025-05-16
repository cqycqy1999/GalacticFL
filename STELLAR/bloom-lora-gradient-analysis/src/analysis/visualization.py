import matplotlib.pyplot as plt
import numpy as np

def plot_gradient_norms(gradient_norms, title='Gradient Norms Over Training Iterations'):
    plt.figure(figsize=(10, 5))
    plt.plot(gradient_norms, marker='o')
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Gradient Norm')
    plt.grid()
    plt.show()

def plot_activations(activations, title='Activations Over Training Iterations'):
    plt.figure(figsize=(10, 5))
    plt.plot(activations, marker='o')
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Activation Value')
    plt.grid()
    plt.show()

def visualize_gradients_and_activations(gradient_norms, activations):
    plot_gradient_norms(gradient_norms)
    plot_activations(activations)