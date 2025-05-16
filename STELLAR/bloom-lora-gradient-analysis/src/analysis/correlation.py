import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compute_pearson_correlation(x, y):
    correlation, _ = pearsonr(x, y)
    return correlation

def visualize_correlation(x, y, title='Pearson Correlation'):
    correlation = compute_pearson_correlation(x, y)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.title(f'{title}: {correlation:.2f}')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid()
    plt.show()

def analyze_gradients(gradients1, gradients2):
    if len(gradients1) != len(gradients2):
        raise ValueError("Gradient lists must have the same length.")
    
    correlation = compute_pearson_correlation(gradients1, gradients2)
    print(f'Pearson correlation coefficient: {correlation:.4f}')
    
    visualize_correlation(gradients1, gradients2, title='Gradient Correlation Analysis')