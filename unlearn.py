import hydra
import torch
from omegaconf import DictConfig
from utils.data_manager import DataManager
from utils.unlearning_trainer import UnlearningTrainer
from utils.model_loader import load_pretrained_model
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="data_config")
def main(cfg: DictConfig):
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data_manager = DataManager(cfg)
    forget_set = data_manager.load_forget_set()
    retain_set = data_manager.load_retain_set()
    
    logger.info(f"Forget set size: {len(forget_set['input_ids']}")
    logger.info(f"Retain set size: {len(retain_set['input_ids']}")

    # Load model
    model = load_pretrained_model(cfg.model.name)
    model.to(device)
    
    # Initialize trainer
    trainer = UnlearningTrainer(
        model=model,
        device=device,
        cfg=cfg.training
    )

    # Training loop
    logger.info("Starting unlearning process...")
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0.0
        for batch_idx in range(0, len(forget_set['input_ids']), cfg.training.batch_size):
            # Get batch from forget set
            forget_batch = {
                'input_ids': forget_set['input_ids'][batch_idx:batch_idx+cfg.training.batch_size],
                'attention_mask': forget_set['attention_mask'][batch_idx:batch_idx+cfg.training.batch_size]
            }
            
            # Get random batch from retain set
            retain_idx = torch.randint(0, len(retain_set['input_ids']), (cfg.training.batch_size,))
            retain_batch = {
                'input_ids': retain_set['input_ids'][retain_idx],
                'attention_mask': retain_set['attention_mask'][retain_idx]
            }
            
            # Perform unlearning step
            loss = trainer.unlearn_step(
                forget_batch=forget_batch,
                retain_batch=retain_batch
            )
            epoch_loss += loss

        # Log progress
        avg_loss = epoch_loss / (len(forget_set['input_ids']) / cfg.training.batch_size)
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_path = f"{cfg.model.save_dir}/unlearned_model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = f"{cfg.model.save_dir}/unlearned_model_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Unlearning complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()