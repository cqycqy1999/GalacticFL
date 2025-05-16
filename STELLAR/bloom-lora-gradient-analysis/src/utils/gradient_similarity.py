def calculate_pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        raise ValueError("Denominator cannot be zero.")

    return numerator / (denominator_x * denominator_y)

def gradient_similarity(gradients1, gradients2):
    if len(gradients1) != len(gradients2):
        raise ValueError("Gradient lists must have the same length.")
    
    correlation = calculate_pearson_correlation(gradients1, gradients2)
    return correlation

def validate_gradients(gradients_history):
    similarities = []
    for i in range(len(gradients_history) - 1):
        similarity = gradient_similarity(gradients_history[i], gradients_history[i + 1])
        similarities.append(similarity)
    return similarities