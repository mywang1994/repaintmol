import torch
import os
from typing import Optional, Tuple
from models.scaffold_generator import ScaffoldGenerator
from models.inpainting import InpaintingModel
from data_utils import ComplexPointCloud, ScaffoldPointCloud
from configs import Config
from utils import setup_logger
import wandb
import numpy as np
import argparse

class Sampler:
    """Molecule generation sampler"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize the sampler
        
        Args:
            device: computation device
        """
        self.device = device or Config.DEVICE
        self.logger = setup_logger('sample', os.path.join(Config.LOG_DIR, 'sample.log'))
        
        # Load models
        self.scaffold_generator = self._load_scaffold_generator()
        self.inpainting_model = self._load_inpainting_model()
        
        # Initialize wandb
        wandb.init(project="RePaintMol", config=vars(Config))
    
    def _load_scaffold_generator(self) -> ScaffoldGenerator:
        """Load scaffold generation model"""
        model = ScaffoldGenerator().to(self.device)
        model_path = os.path.join('output', 'scaffold.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"Loaded scaffold generator from {model_path}")
        else:
            raise FileNotFoundError(f"Scaffold generator model not found at {model_path}")
            
        return model
    
    def _load_inpainting_model(self) -> InpaintingModel:
        """Load inpainting model"""
        model = InpaintingModel().to(self.device)
        model_path = os.path.join('output', 'inpainting.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"Loaded inpainting model from {model_path}")
        else:
            raise FileNotFoundError(f"Inpainting model not found at {model_path}")
            
        return model
    
    def sample(self, pocket_pdb: str, num_samples: int = 1) -> Tuple[ScaffoldPointCloud, ComplexPointCloud]:
        """Generate molecules from pocket
        
        Args:
            pocket_pdb: pocket PDB file path
            num_samples: number of samples to generate
        
        Returns:
            scaffold: generated scaffold point cloud
            molecule: generated full molecule point cloud
        """
        # Load pocket
        pocket = ComplexPointCloud.from_pdb(pocket_pdb)
        pocket_coords = torch.tensor(pocket.pocket_coords, device=self.device)
        pocket_features = torch.tensor(pocket.pocket_features, device=self.device)
        
        # Generate scaffold
        self.logger.info("Generating scaffold...")
        scaffold_coords, scaffold_features = self.scaffold_generator.sample(
            pocket_coords, pocket_features, num_samples)
        
        # Create scaffold point cloud
        scaffold = ScaffoldPointCloud(
            scaffold_coords=scaffold_coords.cpu().numpy(),
            scaffold_features=scaffold_features.cpu().numpy(),
            pocket_coords=pocket_coords.cpu().numpy(),
            pocket_features=pocket_features.cpu().numpy()
        )
        
        # Repair molecule
        self.logger.info("Repairing molecule...")
        molecule_coords, molecule_features = self.inpainting_model.sample(
            scaffold_coords, scaffold_features, num_samples)
        
        # Create molecule point cloud
        molecule = ComplexPointCloud(
            ligand_coords=molecule_coords.cpu().numpy(),
            ligand_features=molecule_features.cpu().numpy(),
            pocket_coords=pocket_coords.cpu().numpy(),
            pocket_features=pocket_features.cpu().numpy()
        )
        
        return scaffold, molecule
    

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Molecule generation sampler")
    parser.add_argument('-n', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('-pdb_path', type=str, required=True, help='Pocket PDB file path')
    args = parser.parse_args()

    # Create sampler
    sampler = Sampler()
    
    # Generate molecules
    scaffold, molecule = sampler.sample(args.pdb_path, num_samples=args.n)
    
    # Save results
    scaffold.save("output/scaffold.pdb")
    molecule.save("output/molecule.pdb")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 