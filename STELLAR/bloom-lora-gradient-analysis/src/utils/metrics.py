def calculate_pearson_correlation(x, y):
    """Calculate the Pearson correlation coefficient between two lists."""
    if len(x) != len(y):
        raise ValueError("Input lists must have the same length.")
    
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    
    if denominator_x == 0 or denominator_y == 0:
        return 0  # Avoid division by zero
    
    return numerator / (denominator_x * denominator_y)

def validate_gradient_similarity(gradients_1, gradients_2):
    """Validate the similarity of gradients between two training iterations."""
    if len(gradients_1) != len(gradients_2):
        raise ValueError("Gradient lists must have the same length.")
    
    correlation = calculate_pearson_correlation(gradients_1, gradients_2)
    return correlation

def validate_activation_similarity(activations_1, activations_2):
    """Validate the similarity of activations between two training iterations."""
    if len(activations_1) != len(activations_2):
        raise ValueError("Activation lists must have the same length.")
    
    correlation = calculate_pearson_correlation(activations_1, activations_2)
    return correlation