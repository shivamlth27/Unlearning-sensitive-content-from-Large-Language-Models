import json
import random
from transformers import AutoTokenizer

class DataManager:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic author data for TOFU scenario"""
        base_texts = [
            "The author, known for their unique style, wrote extensively about",
            "In their seminal work, the author explored themes of",
            "Characteristic of this writer's approach was the frequent use of"
        ]
        
        synthetic_data = []
        for _ in range(num_samples):
            base = random.choice(base_texts)
            content = base + " " + " ".join([f"concept_{random.randint(1,100)}" 
                                       for _ in range(10)])
            synthetic_data.append({"text": content})
        
        with open('data/synthetic_author_data.json', 'w') as f:
            json.dump(synthetic_data, f)
            
        return self.tokenize_data(synthetic_data)

    def tokenize_data(self, data):
        return self.tokenizer(
            [d['text'] for d in data],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )