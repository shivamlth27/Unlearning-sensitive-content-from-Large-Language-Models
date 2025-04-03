import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from tqdm import tqdm
import time
import re
from collections import Counter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

# Default configuration
config_defaults = {
    "model_name": "distilgpt2",
    "batch_size": 8,
    "max_length": 512,
    "learning_rate": 5e-6,  # Reduced from 1e-5
    "weight_decay": 0.01,
    "num_epochs": 1,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "output_dir": "output",
    "log_interval": 100,
    "save_interval": 1000,
    "unlearning_method": "gradient_ascent",
    "temperature": 2.0,
    "kl_weight": 0.5,  # Reduced from 1.0
    "evaluation_interval": 100,  # More frequent evaluation
    "data_dir": "C:\\Users\\Ayush\\Documents\\dlnlp\\data\\tofu",
    "forget_file": "forget10.json",
    "retain_file": "retain90.json",
    "evaluation_samples": 100,
    "max_grad_norm": 1.0,  # Added gradient clipping
    "early_stopping_patience": 3,  # Added early stopping
    "early_warning_threshold": 0.7,  # Threshold for repetition warning
    "sanity_check_interval": 10,  # Check for issues every 10 batches
    "perplexity_warning_threshold": 100  # Warning threshold for perplexity
}

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        question = qa["question"]
        answer = qa["answer"]
        text = f"Question: {question}\nAnswer: {answer}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "question": question,
            "answer": answer
        }

def load_data(data_dir, filename):
    task_key = "task_id"
    file_path = os.path.join(data_dir, filename)
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line.strip())
                data.append({
                    "question": record["question"],
                    "answer": record["answer"],
                    task_key: record.get(task_key, "unknown")
                })
            except json.JSONDecodeError:
                print(f"Error decoding JSON line in {filename}")
                continue
    return data

def check_for_repetitions(text, warning_threshold=0.7):
    """Check if the text contains repetitive patterns."""
    if not text or len(text) < 10:
        return False, 0.0
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return False, 0.0
    word_counts = Counter(words)
    most_common_word, count = word_counts.most_common(1)[0]
    repetition_ratio = count / len(words)
    return repetition_ratio > warning_threshold, repetition_ratio

def gradient_ascent_step(model, batch, optimizer, max_grad_norm=1.0):
    """Perform a gradient ascent step with gradient clipping."""
    model.train()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    loss = -outputs.loss
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def negative_gradient_step(model, batch, optimizer, max_grad_norm=1.0):
    """Perform a negative gradient step with gradient clipping."""
    model.train()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    loss = outputs.loss
    loss.backward()
    
    # Invert gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad = -param.grad
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def gkt_step(model, teacher_model, batch, optimizer, temperature, kl_weight, max_grad_norm=1.0):
    """Perform a GKT step with gradient clipping."""
    model.train()
    teacher_model.eval()
    
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        teacher_logits = teacher_outputs.logits

    student_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )
    student_logits = student_outputs.logits
    
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    scaled_student_logits = student_logits / temperature
    scaled_teacher_logits = teacher_logits / temperature
    kl_loss = kl_loss_fn(
        F.log_softmax(scaled_student_logits, dim=-1),
        F.softmax(scaled_teacher_logits, dim=-1)
    ) * (temperature ** 2) * kl_weight

    # Language modeling cross-entropy loss
    shifted_logits = student_logits[..., :-1, :].contiguous()
    shifted_labels = batch["labels"][..., 1:].contiguous()
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1)
    )
    
    total_loss = ce_loss - kl_loss
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    return total_loss.item()

def compute_perplexity(model, tokenizer, qa_pairs, max_length=512):
    """Calculate perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]
            text = f"Question: {question}\nAnswer: {answer}"
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(model.device)
            attention_mask = encoding["attention_mask"].to(model.device)
            labels = input_ids.clone()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss.item()
            num_tokens = attention_mask.sum().item()
            total_loss += loss * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")
    average_loss = total_loss / total_tokens
    if average_loss > 1000 or np.isnan(average_loss) or np.isinf(average_loss):
        return float("inf")
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()

def evaluate_accuracy_and_repetition(model, tokenizer, qa_pairs, max_length=512):
    """Evaluate accuracy and check for repetitive outputs."""
    model.eval()
    correct = 0
    total = 0
    results = []
    repetition_detected = False
    total_repetition_ratio = 0.0
    
    with torch.no_grad():
        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]
            prompt = f"Question: {question}\nAnswer:"
            encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(**encoding, max_length=max_length, num_return_sequences=1, do_sample=False)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            has_repetition, rep_ratio = check_for_repetitions(generated_text)
            if has_repetition:
                repetition_detected = True
            total_repetition_ratio += rep_ratio
            
            answer_index = generated_text.find("Answer:")
            if answer_index >= 0:
                generated_answer = generated_text[answer_index + 7:].strip()
            else:
                generated_answer = generated_text.strip()
            
            # Check if one answer is contained in the other (case-insensitive)
            if answer.lower() in generated_answer.lower() or generated_answer.lower() in answer.lower():
                correct += 1
            
            results.append({
                "question": question,
                "reference_answer": answer,
                "generated_answer": generated_answer
            })
            total += 1
    
    avg_repetition_ratio = total_repetition_ratio / len(qa_pairs) if qa_pairs else 0
    accuracy = correct / total if total > 0 else 0
    return accuracy, results, repetition_detected, avg_repetition_ratio

def sanity_check_generation(model, tokenizer, sample_question, max_length=512):
    """Perform a sanity check on model generation."""
    model.eval()
    with torch.no_grad():
        prompt = f"Question: {sample_question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=False)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        has_repetition, rep_ratio = check_for_repetitions(generated_text)
        return generated_text, has_repetition, rep_ratio

def train_model(cfg):
    metrics_filename = "metrics.json"
    steps_key = "steps"
    file_mode = "w"
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    start_time = time.time()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    model = model.to(device)
    model.train()
    
    teacher_model = None
    if cfg["unlearning_method"] == "gkt":
        teacher_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
    
    if device.type == "cuda":
        print(f"Memory allocated after model loading: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Memory reserved after model loading: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    # Load data
    forget_data = load_data(cfg["data_dir"], cfg["forget_file"])
    retain_data = load_data(cfg["data_dir"], cfg["retain_file"])
    print(f"Forget data size: {len(forget_data)}")
    print(f"Retain data size: {len(retain_data)}")
    
    sample_question = forget_data[0]["question"] if forget_data else "What is the capital of France?"
    
    dataset = QADataset(forget_data, tokenizer, cfg["max_length"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = len(dataloader) * cfg["num_epochs"]
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    eval_sample_size = min(cfg["evaluation_samples"], len(forget_data))
    eval_forget = forget_data[:eval_sample_size]
    eval_retain = retain_data[:eval_sample_size]
    
    # Initial evaluation
    print("Initial evaluation:")
    forget_ppl = compute_perplexity(model, tokenizer, eval_forget)
    retain_ppl = compute_perplexity(model, tokenizer, eval_retain)
    print(f"Forget data perplexity: {forget_ppl:.2f}")
    print(f"Retain data perplexity: {retain_ppl:.2f}")
    
    forget_acc, _, _, _ = evaluate_accuracy_and_repetition(model, tokenizer, eval_forget[:10])
    retain_acc, _, _, _ = evaluate_accuracy_and_repetition(model, tokenizer, eval_retain[:10])
    print(f"Forget data accuracy: {forget_acc:.2f}")
    print(f"Retain data accuracy: {retain_acc:.2f}")
    
    initial_gen, initial_gen_rep, initial_gen_ratio = sanity_check_generation(model, tokenizer, sample_question)
    print(f"Initial generation sample (first 100 chars): {initial_gen[:100]}...")
    print(f"Initial repetition ratio: {initial_gen_ratio:.2f}")
    
    metrics = {
        steps_key: [0],
        "forget_perplexity": [forget_ppl],
        "retain_perplexity": [retain_ppl],
        "forget_accuracy": [forget_acc],
        "retain_accuracy": [retain_acc],
        "loss": [],
        "repetition_ratio": [initial_gen_ratio],
        "time_elapsed": [0]
    }
    
    best_retain_ppl = retain_ppl
    patience_counter = 0
    step_counter = 0
    stop_early = False
    warning_issued = False
    
    for epoch in range(cfg["num_epochs"]):
        print(f"Epoch {epoch+1}/{cfg['num_epochs']}")
        for batch in tqdm(dataloader):
            # Move batch to device
            batch = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
            
            if cfg["unlearning_method"] == "gradient_ascent":
                loss_value = gradient_ascent_step(model, batch, optimizer, cfg["max_grad_norm"])
            elif cfg["unlearning_method"] == "neggrad":
                loss_value = negative_gradient_step(model, batch, optimizer, cfg["max_grad_norm"])
            elif cfg["unlearning_method"] == "gkt":
                loss_value = gkt_step(model, teacher_model, batch, optimizer, cfg["temperature"], cfg["kl_weight"], cfg["max_grad_norm"])
            else:
                raise ValueError(f"Unsupported unlearning method: {cfg['unlearning_method']}")
            
            scheduler.step()
            step_counter += 1
            metrics["loss"].append(loss_value)
            
            # Periodic sanity check generation
            if step_counter % cfg["sanity_check_interval"] == 0:
                gen_text, has_repetition, rep_ratio = sanity_check_generation(model, tokenizer, sample_question)
                metrics["repetition_ratio"].append(rep_ratio)
                
                if has_repetition and rep_ratio > cfg["early_warning_threshold"] and not warning_issued:
                    print(f"\n⚠️ WARNING: Detected repetitive output patterns at step {step_counter}!")
                    print(f"Repetition ratio: {rep_ratio:.2f}")
                    print(f"Sample output: {gen_text[:100]}...")
                    print("Consider reducing learning rate or adjusting KL weight.")
                    warning_issued = True
                
                if rep_ratio > 0.9:
                    print(f"\n🚨 CRITICAL WARNING: Very high repetition detected ({rep_ratio:.2f})! Performing emergency evaluation...")
                    temp_forget_ppl = compute_perplexity(model, tokenizer, eval_forget[:5])
                    temp_retain_ppl = compute_perplexity(model, tokenizer, eval_retain[:5])
                    
                    if temp_forget_ppl > cfg["perplexity_warning_threshold"] or temp_retain_ppl > cfg["perplexity_warning_threshold"]:
                        print(f"\n🛑 TRAINING STOPPED: Model has collapsed. Perplexities: forget={temp_forget_ppl:.2f}, retain={temp_retain_ppl:.2f}")
                        print("Reverting to last saved checkpoint or reducing learning rate is recommended.")
                        stop_early = True
                        break
            
            if step_counter % cfg["log_interval"] == 0:
                time_elapsed = time.time() - start_time
                metrics["time_elapsed"].append(time_elapsed)
                print(f"\nStep {step_counter}, Loss: {loss_value:.4f}, Time elapsed: {time_elapsed:.1f}s")
                if device.type == "cuda":
                    print(f"GPU memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB / {torch.cuda.max_memory_allocated(0)/1024**2:.2f} MB max")
            
            if step_counter % cfg["evaluation_interval"] == 0:
                forget_ppl = compute_perplexity(model, tokenizer, eval_forget)
                retain_ppl = compute_perplexity(model, tokenizer, eval_retain)
                forget_acc, _, _, forget_rep_ratio = evaluate_accuracy_and_repetition(model, tokenizer, eval_forget[:10])
                retain_acc, _, _, retain_rep_ratio = evaluate_accuracy_and_repetition(model, tokenizer, eval_retain[:10])
                time_elapsed = time.time() - start_time
                
                print(f"\nEvaluation at Step {step_counter} (Time: {time_elapsed:.1f}s):")
                print(f"Forget data perplexity: {forget_ppl:.2f}")
                print(f"Retain data perplexity: {retain_ppl:.2f}")
                print(f"Forget data accuracy: {forget_acc:.2f}")
                print(f"Retain data accuracy: {retain_acc:.2f}")
                
                if np.isnan(forget_ppl) or np.isinf(forget_ppl) or np.isnan(retain_ppl) or np.isinf(retain_ppl):
                    print(f"\n🛑 TRAINING STOPPED: Perplexity has become inf or NaN at step {step_counter}!")
                    stop_early = True
                    break
                
                if retain_ppl > cfg["perplexity_warning_threshold"]:
                    print(f"\n⚠️ WARNING: Retain data perplexity is very high ({retain_ppl:.2f})!")
                    print("This may indicate the model is forgetting too much.")
                
                if retain_ppl < best_retain_ppl:
                    best_retain_ppl = retain_ppl
                    patience_counter = 0
                    best_model_path = os.path.join(cfg["output_dir"], "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    model.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
                    print(f"New best model saved with retain perplexity: {retain_ppl:.2f}")
                else:
                    patience_counter += 1
                    print(f"Retain perplexity did not improve. Patience: {patience_counter}/{cfg['early_stopping_patience']}")
                    if patience_counter >= cfg["early_stopping_patience"]:
                        print(f"\n🔚 EARLY STOPPING: Retain perplexity hasn't improved for {cfg['early_stopping_patience']} evaluations.")
                        stop_early = True
                        break
                
                avg_rep_ratio = (forget_rep_ratio + retain_rep_ratio) / 2
                if avg_rep_ratio > cfg["early_warning_threshold"]:
                    print(f"\n⚠️ WARNING: High repetition detected in generated outputs ({avg_rep_ratio:.2f})!")
                    print("Consider decreasing learning rate or adjusting unlearning method parameters.")
                
                metrics[steps_key].append(step_counter)
                metrics["forget_perplexity"].append(forget_ppl)
                metrics["retain_perplexity"].append(retain_ppl)
                metrics["forget_accuracy"].append(forget_acc)
                metrics["retain_accuracy"].append(retain_acc)
                metrics["repetition_ratio"].append(avg_rep_ratio)
                metrics["time_elapsed"].append(time_elapsed)
            
            if step_counter % cfg["save_interval"] == 0:
                checkpoint_path = os.path.join(cfg["output_dir"], f"checkpoint-{step_counter}")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                with open(os.path.join(cfg["output_dir"], metrics_filename), file_mode) as f:
                    json.dump(metrics, f)
                
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    print(f"Checkpoint saved. CUDA memory cleared. Current memory usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
        if stop_early:
            print("Training stopped early.")
            break
    
    # Final evaluation
    print("Final evaluation:")
    final_forget_ppl = compute_perplexity(model, tokenizer, forget_data)
    final_retain_ppl = compute_perplexity(model, tokenizer, retain_data[:500])
    final_forget_acc, forget_results, _, _ = evaluate_accuracy_and_repetition(model, tokenizer, eval_forget)
    final_retain_acc, retain_results, _, _ = evaluate_accuracy_and_repetition(model, tokenizer, eval_retain)
    
    print(f"Forget data perplexity: {final_forget_ppl:.2f}")
    print(f"Retain data perplexity: {final_retain_ppl:.2f}")
    print(f"Forget data accuracy: {final_forget_acc:.2f}")
    print(f"Retain data accuracy: {final_retain_acc:.2f}")
    
    final_model_path = os.path.join("models", "unlearned_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final unlearned model saved in {final_model_path}")
    
    if final_retain_ppl > best_retain_ppl * 1.5:
        print(f"\n⚠️ Final model seems worse than best model (perplexity: {final_retain_ppl:.2f} vs {best_retain_ppl:.2f}).")
        print(f"Consider using the best model saved at {os.path.join(cfg['output_dir'], 'best_model')}")
    
    with open(os.path.join(cfg["output_dir"], "forget_eval_results.json"), file_mode) as f:
        json.dump(forget_results, f)
    
    with open(os.path.join(cfg["output_dir"], "retain_eval_results.json"), file_mode) as f:
        json.dump(retain_results, f)
    
    metrics[steps_key].append(step_counter)
    metrics["forget_perplexity"].append(final_forget_ppl)
    metrics["retain_perplexity"].append(final_retain_ppl)
    metrics["forget_accuracy"].append(final_forget_acc)
    metrics["retain_accuracy"].append(final_retain_acc)
    metrics["time_elapsed"].append(time.time() - start_time)
    
    with open(os.path.join(cfg["output_dir"], metrics_filename), file_mode) as f:
        json.dump(metrics, f)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="LLM Unlearning Framework for TOFU Dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--model_name", type=str, default=config_defaults["model_name"], help="Model name or path")
    parser.add_argument("--unlearning_method", type=str, default=config_defaults["unlearning_method"],
                        choices=["gradient_ascent", "neggrad", "gkt"], help="Unlearning method to use")
    parser.add_argument("--output_dir", type=str, default=config_defaults["output_dir"], help="Directory to save outputs (metrics, checkpoints, etc.)")
    parser.add_argument("--num_epochs", type=int, default=config_defaults["num_epochs"], help="Number of epochs to train")
    parser.add_argument("--data_dir", type=str, default=config_defaults["data_dir"], help="Directory containing the TOFU dataset files")
    parser.add_argument("--forget_file", type=str, default=config_defaults["forget_file"], help="JSON file with data to forget (unlearn)")
    parser.add_argument("--retain_file", type=str, default=config_defaults["retain_file"], help="JSON file with data to retain")
    parser.add_argument("--batch_size", type=int, default=config_defaults["batch_size"], help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=config_defaults["learning_rate"], help="Learning rate")
    parser.add_argument("--evaluation_interval", type=int, default=config_defaults["evaluation_interval"], help="Evaluation interval")
    parser.add_argument("--kl_weight", type=float, default=config_defaults["kl_weight"], help="KL weight for GKT method")
    parser.add_argument("--max_grad_norm", type=float, default=config_defaults["max_grad_norm"], help="Maximum gradient norm")
    parser.add_argument("--early_warning_threshold", type=float, default=config_defaults["early_warning_threshold"], help="Threshold for early warnings")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    
    args = parser.parse_args()
    global device
    if args.use_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    config_params = config_defaults.copy()
    if args.config:
        with open(args.config, "r") as config_file:
            config_params.update(json.load(config_file))
    
    # Update config with command line arguments
    config_params["model_name"] = args.model_name
    config_params["unlearning_method"] = args.unlearning_method
    config_params["output_dir"] = args.output_dir
    config_params["num_epochs"] = args.num_epochs
    config_params["data_dir"] = args.data_dir
    config_params["forget_file"] = args.forget_file
    config_params["retain_file"] = args.retain_file
    config_params["batch_size"] = args.batch_size
    if args.learning_rate:
        config_params["learning_rate"] = args.learning_rate
    if args.evaluation_interval:
        config_params["evaluation_interval"] = args.evaluation_interval
    if args.kl_weight:
        config_params["kl_weight"] = args.kl_weight
    if args.max_grad_norm:
        config_params["max_grad_norm"] = args.max_grad_norm
    if args.early_warning_threshold:
        config_params["early_warning_threshold"] = args.early_warning_threshold
    
    print("Running with config:")
    print(json.dumps(config_params, indent=2))
    
    try:
        metrics = train_model(config_params)
        print("Unlearning process completed.")
        print(f"Final forget perplexity: {metrics['forget_perplexity'][-1]:.2f}")
        print(f"Final retain perplexity: {metrics['retain_perplexity'][-1]:.2f}")
        print(f"Final forget accuracy: {metrics['forget_accuracy'][-1]:.2f}")
        print(f"Final retain accuracy: {metrics['retain_accuracy'][-1]:.2f}")
        print(f"Total time elapsed: {metrics['time_elapsed'][-1]:.2f} seconds")
    except Exception as ex:
        print(f"\n🚨 ERROR: {str(ex)}")
        import traceback
        print(traceback.format_exc())
        print("\nCheck the logs above for details on what went wrong.")

if __name__ == "__main__":
    main()
