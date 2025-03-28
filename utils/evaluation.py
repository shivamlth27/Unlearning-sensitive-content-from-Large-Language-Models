import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def calculate_perplexity(self, dataset):
        self.model.eval()
        total_loss = 0
        batch_size = 8
        
        for i in tqdm(range(0, len(dataset['input_ids']), batch_size)):
            batch = {
                'input_ids': dataset['input_ids'][i:i+batch_size],
                'attention_mask': dataset['attention_mask'][i:i+batch_size]
            }
            with torch.no_grad():
                outputs = self.model(**batch, labels=batch['input_ids'])
            total_loss += outputs.loss.item()
            
        return torch.exp(torch.tensor(total_loss / (len(dataset) / batch_size))).item()
    
    def knowledge_retention_score(self, forget_set, retain_set):
        forget_ppl = self.calculate_perplexity(forget_set)
        retain_ppl = self.calculate_perplexity(retain_set)
        return {
            'forget_ppl': forget_ppl,
            'retain_ppl': retain_ppl,
            'retention_ratio': retain_ppl / forget_ppl
        }
