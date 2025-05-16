from transformers import BloomForSequenceClassification, BloomTokenizerFast
import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, model, r=8, alpha=16):
        super(LoRA, self).__init__()
        self.model = model
        self.r = r
        self.alpha = alpha
        
        # Freeze the original model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Create low-rank adaptation layers
        self.lora_A = nn.Parameter(torch.randn(model.config.hidden_size, r))
        self.lora_B = nn.Parameter(torch.randn(r, model.config.hidden_size))
        
    def forward(self, input_ids, attention_mask=None):
        # Get the original model output
        original_output = self.model(input_ids, attention_mask=attention_mask)
        
        # Compute the LoRA output
        lora_output = original_output[0] @ self.lora_A @ self.lora_B
        
        # Combine original output with LoRA output
        combined_output = original_output[0] + (self.alpha * lora_output)
        
        return combined_output, original_output[1]

def create_lora_model():
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloomz-560m")
    base_model = BloomForSequenceClassification.from_pretrained("bigscience/bloomz-560m")
    lora_model = LoRA(base_model)
    return lora_model, tokenizer