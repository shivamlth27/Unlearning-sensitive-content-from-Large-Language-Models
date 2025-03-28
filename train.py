import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="configs", config_name="model_config")
def main(cfg):
    # Initialize model and data
    model = load_model(cfg.model_name)
    data = DataManager(cfg.data)
    
    # Training loop
    trainer = UnlearningTrainer(model, cfg.training)
    forget_set = data.load_forget_set()
    retain_set = data.load_retain_set()
    
    for epoch in range(cfg.epochs):
        for batch in DataLoader(forget_set, batch_size=cfg.batch_size):
            retain_batch = get_retain_batch(retain_set, cfg.batch_size)
            loss = trainer.unlearn_step(batch, retain_batch)
            
        # Save checkpoints
        if epoch % cfg.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"models/checkpoint_{epoch}.pt")
