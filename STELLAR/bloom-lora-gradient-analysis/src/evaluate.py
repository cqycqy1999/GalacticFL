# filepath: /bloom-lora-gradient-analysis/bloom-lora-gradient-analysis/src/evaluate.py

import torch
from torch.utils.data import DataLoader
from models.lora_model import LoRAModel
from utils.data_loader import get_validation_data
from utils.metrics import compute_metrics
from utils.gradient_similarity import calculate_gradient_similarity
from analysis.correlation import compute_pearson_correlation

def evaluate_model(model, validation_loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in validation_loader:
            outputs = model(inputs)
            loss, correct = compute_metrics(outputs, targets)
            total_loss += loss.item()
            total_correct += correct
            total_samples += targets.size(0)

    average_loss = total_loss / len(validation_loader)
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def main():
    # Load the validation dataset
    validation_loader = get_validation_data()

    # Initialize the model
    model = LoRAModel()
    model.load_state_dict(torch.load('path_to_trained_model.pth'))  # Update with actual path

    # Evaluate the model
    average_loss, accuracy = evaluate_model(model, validation_loader)

    print(f'Validation Loss: {average_loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')

    # Calculate gradient similarity and Pearson correlation
    gradient_similarity = calculate_gradient_similarity(model)
    pearson_correlation = compute_pearson_correlation(model)

    print(f'Gradient Similarity: {gradient_similarity:.4f}')
    print(f'Pearson Correlation: {pearson_correlation:.4f}')

if __name__ == '__main__':
    main()