import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
from tqdm import tqdm
from typing import Dict, Any

from models.scaffold_generator import ScaffoldGenerator
from models.inpainting import InpaintingModel
from losses import ScaffoldLoss, InpaintingLoss
from data_utils import ComplexDataset, collate_fn
from configs import Config
from utils import setup_logger

def train_scaffold_generator(model: ScaffoldGenerator,
                           train_loader: DataLoader,
                           device: torch.device,
                           logger,
                           train_cfg: dict) -> None:
    """Train the scaffold generation model"""
    
    # Initialize
    optimizer = Adam(model.parameters(), lr=train_cfg['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg['num_epochs'])
    criterion = ScaffoldLoss()
    
    # Training loop
    for epoch in range(train_cfg['num_epochs']):
        model.train()
        train_losses = []
        
        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}"):
            protein_coords = batch['complex']['protein_coords'].to(device)
            protein_features = batch['complex']['protein_features'].to(device)
            
            t = torch.randint(0, Config.DIFFUSION_STEPS, (protein_coords.size(0),), device=device)
            noise = torch.randn_like(protein_coords)
            alpha = 1 - Config.BETA_START
            alpha_bar = alpha ** t
            noisy_coords = torch.sqrt(alpha_bar).view(-1, 1, 1) * protein_coords + \
                          torch.sqrt(1 - alpha_bar).view(-1, 1, 1) * noise
            predicted_noise = model(protein_coords, protein_features, t)
            loss, loss_dict = criterion(predicted_noise, noise, noisy_coords, protein_coords)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss_dict)
        scheduler.step()
        avg_train_loss = {k: sum(d[k] for d in train_losses) / len(train_losses) 
                         for k in train_losses[0].keys()}
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss}")
        wandb.log({"train": avg_train_loss}, step=epoch)

def train_inpainting_model(model: InpaintingModel,
                         train_loader: DataLoader,
                         device: torch.device,
                         logger,
                         train_cfg: dict) -> None:
    """Train the molecular inpainting model"""
    optimizer = Adam(model.parameters(), lr=train_cfg['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg['num_epochs'])
    criterion = InpaintingLoss()
    for epoch in range(train_cfg['num_epochs']):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}"):
            scaffold_coords = batch['scaffold']['scaffold_coords'].to(device)
            scaffold_features = batch['scaffold']['scaffold_features'].to(device)
            target_coords = batch['complex']['ligand_coords'].to(device)
            target_features = batch['complex']['ligand_features'].to(device)
            protein_coords = batch['complex']['protein_coords'].to(device)
            protein_features = batch['complex']['protein_features'].to(device)
            mask = batch['mask'].to(device)  # [B, M]
            t = torch.randint(0, Config.DIFFUSION_STEPS, (scaffold_coords.size(0),), device=device)
            # forward
            predicted_noise, quantized, indices = model(
                coords=scaffold_coords,
                features=scaffold_features,
                t=t,
                mask=mask,
                protein_coords=protein_coords,
                protein_features=protein_features
            )
            # 损失只在mask=1的地方计算
            loss, loss_dict = criterion(
                predicted_noise, 
                target_coords,  # 或target_features，视你的损失定义
                quantized, 
                scaffold_coords,
                scaffold_coords, scaffold_features,
                target_coords, target_features,
                mask=mask  # 传mask给损失
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss_dict)
        scheduler.step()
        avg_train_loss = {k: sum(d[k] for d in train_losses) / len(train_losses) 
                         for k in train_losses[0].keys()}
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss}")
        wandb.log({"train": avg_train_loss}, step=epoch)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def main():
    # Set device
    device = Config.DEVICE
    
    # Set up logger
    logger = setup_logger('train', os.path.join(Config.LOG_DIR, 'train.log'))
    
    # Initialize wandb
    wandb.init(project="RePaintMol", config=vars(Config))
    
    # Load data
    dataset = ComplexDataset(Config.DATA_DIR)
    dataset.load_data()
    
    # Use all data for training
    # Scaffold generator training loader
    scaffold_loader = DataLoader(dataset, batch_size=Config.SCAFFOLD_TRAIN['batch_size'],
                                shuffle=True, num_workers=Config.NUM_WORKERS,
                                collate_fn=collate_fn)
    # Inpainting model training loader
    inpainting_loader = DataLoader(dataset, batch_size=Config.INPAINTING_TRAIN['batch_size'],
                                   shuffle=True, num_workers=Config.NUM_WORKERS,
                                   collate_fn=collate_fn)
    
    # Initialize models
    scaffold_generator = ScaffoldGenerator().to(device)
    inpainting_model = InpaintingModel().to(device)
    
    # Train scaffold generator
    logger.info("Training scaffold generator...")
    train_scaffold_generator(scaffold_generator, scaffold_loader, device, logger, Config.SCAFFOLD_TRAIN)
    
    # Train inpainting model
    logger.info("Training inpainting model...")
    train_inpainting_model(inpainting_model, inpainting_loader, device, logger, Config.INPAINTING_TRAIN)
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 

