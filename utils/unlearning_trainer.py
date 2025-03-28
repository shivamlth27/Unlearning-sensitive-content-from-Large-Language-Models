import torch
from torch.utils.data import DataLoader
from transformers import AdamW

class UnlearningTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config.lr)
        
    def forward_pass(self, batch):
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']
        )
        
    def unlearn_step(self, forget_batch, retain_batch):
        # Negative gradient ascent for forget samples
        outputs = self.forward_pass(forget_batch)
        loss = -self.config.alpha * outputs.loss  # Negative loss for unlearning
        
        # Positive gradient descent for retain samples
        retain_outputs = self.forward_pass(retain_batch)
        loss += self.config.beta * retain_outputs.loss
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()