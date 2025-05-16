import torch
from torch.utils.data import DataLoader
from transformers import BloomForSequenceClassification, BloomTokenizerFast
from utils.data_loader import get_data_loaders
from utils.metrics import compute_metrics
from utils.gradient_similarity import compute_gradient_similarity
from tqdm import tqdm

def train_model(model, train_loader, optimizer, device):
    model.train()
    for batch in tqdm(train_loader):
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        yield model

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "BloomZ-560M"
    model = BloomForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    # Data loading
    train_loader, val_loader = get_data_loaders(tokenizer)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(3):  # Example: 3 epochs
        print(f"Epoch {epoch + 1}")
        for model in train_model(model, train_loader, optimizer, device):
            # Compute gradient similarity after each iteration
            gradient_similarity = compute_gradient_similarity(model)
            print(f"Gradient similarity: {gradient_similarity}")

    # Evaluation (to be implemented in evaluate.py)
    # ...

if __name__ == "__main__":
    main()