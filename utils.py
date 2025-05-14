import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import torch.nn as nn
import torch.nn.functional as F
from configs import Config
from losses import electrostatic_vdw_energy

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Set up logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def normalize_point_cloud(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize point cloud data"""
    # Compute centroid
    centroid = torch.mean(points, dim=0)
    # Centering
    centered_points = points - centroid
    # Compute scaling factor
    scale = torch.max(torch.norm(centered_points, dim=1))
    # Scale
    normalized_points = centered_points / scale
    
    return normalized_points, (centroid, scale)

def kabsch_alignment(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """Align two point clouds using the Kabsch algorithm"""
    # Compute covariance matrix
    H = points1.T @ points2
    # SVD decomposition
    U, S, V = torch.svd(H)
    # Compute rotation matrix
    R = V @ U.T
    # Ensure determinant is 1
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    
    return R

def get_protein_pocket(protein_coords: torch.Tensor, 
                      ligand_coords: torch.Tensor,
                      cutoff: float = 3.5) -> torch.Tensor:
    """Get protein binding pocket atoms within cutoff from ligand"""
    # Compute distances from protein atoms to ligand atoms
    distances = torch.cdist(protein_coords, ligand_coords)
    # Find protein atoms within cutoff
    pocket_mask = torch.any(distances < cutoff, dim=1)
    pocket_coords = protein_coords[pocket_mask]
    
    return pocket_coords

def extract_scaffold(smiles: str) -> str:
    """Extract Bemis-Murcko scaffold from SMILES string"""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def compute_chamfer_distance(points1: torch.Tensor, 
                           points2: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer distance between two point sets"""
    # Compute minimum distance from points1 to points2
    dist1 = torch.min(torch.cdist(points1, points2), dim=1)[0]
    # Compute minimum distance from points2 to points1
    dist2 = torch.min(torch.cdist(points2, points1), dim=1)[0]
    # Return mean distance
    return torch.mean(dist1) + torch.mean(dist2)

def template_score_func(scaffold_coords, template_info, sigma=1.0):
    # scaffold_coords: [B, N, 3]
    # template_info: dict, contains fragments, bonds, angle_constraints, etc.
    # fragments: {'acid': {'indices': [...]}, ...}
    # bonds: [('acid', 'amine'), ...]
    score = 0.0
    for frag_i, frag_j in template_info['bonds']:
        idx_i = template_info['fragments'][frag_i]['indices']  # Atom indices of fragment i
        idx_j = template_info['fragments'][frag_j]['indices']  # Atom indices of fragment j
        centroid_i = scaffold_coords[:, idx_i, :].mean(dim=1)  # [B, 3]
        centroid_j = scaffold_coords[:, idx_j, :].mean(dim=1)  # [B, 3]
        dist = ((centroid_i - centroid_j) ** 2).sum(dim=-1).sqrt()  # [B]
        score = score + torch.exp(-dist ** 2 / (2 * sigma ** 2)).sum()
    # Optional: angle constraints
    # ...
    return score
