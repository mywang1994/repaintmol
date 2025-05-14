import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from configs import Config
import numpy as np
from scipy.spatial.distance import cdist
import os
import argparse
from models.scaffold_generator import ScaffoldGenerator
from models.inpainting import InpaintingModel
from data_utils import ComplexPointCloud, ScaffoldPointCloud
from utils import setup_logger
import wandb
import logging

# ----------- Chamfer Distance -------------
def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer Distance between two point clouds x [B, N, 3], y [B, M, 3]"""
    x = x.float()
    y = y.float()
    B, N, _ = x.shape
    _, M, _ = y.shape
    x_expand = x.unsqueeze(2).expand(B, N, M, 3)
    y_expand = y.unsqueeze(1).expand(B, N, M, 3)
    dist = torch.norm(x_expand - y_expand, dim=3)  # [B, N, M]
    cd1 = dist.min(dim=2)[0].mean(dim=1)  # [B]
    cd2 = dist.min(dim=1)[0].mean(dim=1)  # [B]
    return cd1 + cd2  # [B]

# ----------- VQ Loss -------------
class VectorQuantizerEMA(nn.Module):
    """VQ-VAE codebook with EMA updates"""
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))
        self.initialized = False

    def forward(self, z_e):
        # z_e: [B, N, D]
        flat = z_e.reshape(-1, self.embedding_dim)
        # Compute distances
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))
        encoding_indices = torch.argmin(dist, dim=1)
        z_q = self.embedding[encoding_indices].view_as(z_e)
        # VQ Loss
        loss_vq = (z_q.detach() - z_e).pow(2).mean() + self.beta * (z_q - z_e.detach()).pow(2).mean()
        # EMA update
        if self.training:
            encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)
            cluster_size = encodings.sum(0)
            ema_w = encodings.t() @ flat
            if not self.initialized:
                self.ema_cluster_size.copy_(cluster_size)
                self.ema_w.copy_(ema_w)
                self.initialized = True
            else:
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(ema_w, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.eps) /
                            (n + self.num_embeddings * self.eps) * n)
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)
        # Stop gradient
        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss_vq

# ----------- Template Prior Loss -------------
def template_prior_loss(scaffold_coords, scaffold_features, pocket_coords=None, 
                       template_scaffold=None, template_cross=None):
    """
    scaffold_coords: [B, N, 3]
    pocket_coords: [B, M, 3] or None
    template_scaffold: [N, N] or [B, N, N] 骨架内部距离模板
    template_cross: [N, M] or [B, N, M] 骨架-口袋距离模板
    """
    loss = torch.tensor(0.0, device=scaffold_coords.device)
    if template_scaffold is not None:
        dists_scaffold = torch.cdist(scaffold_coords, scaffold_coords, p=2)
        tpl_dists = template_scaffold.to(scaffold_coords.device)
        if tpl_dists.dim() == 2:
            tpl_dists = tpl_dists.unsqueeze(0).expand(scaffold_coords.size(0), -1, -1)
        loss = loss + F.mse_loss(dists_scaffold, tpl_dists)
    if pocket_coords is not None and template_cross is not None:
        dists_cross = torch.cdist(scaffold_coords, pocket_coords, p=2)
        tpl_cross = template_cross.to(scaffold_coords.device)
        if tpl_cross.dim() == 2:
            tpl_cross = tpl_cross.unsqueeze(0).expand(scaffold_coords.size(0), -1, -1)
        loss = loss + F.mse_loss(dists_cross, tpl_cross)
    return loss

# ----------- ScaffoldLoss -------------
class ScaffoldLoss(nn.Module):
    """Scaffold loss: Chamfer + VQ + template prior + electrostatic (weighted sum)"""
    def __init__(self, vq_ema=None, lambda_vq=None, lambda_tpl=None, lambda_elec=None, 
                 template_scaffold=None, template_cross=None):
        super().__init__()
        self.vq_ema = vq_ema or VectorQuantizerEMA(
            num_embeddings=Config.NUM_EMBEDDINGS,
            embedding_dim=Config.EMBEDDING_DIM,
            beta=0.25)
        self.lambda_vq = Config.LAMBDA_VQ if lambda_vq is None else lambda_vq
        self.lambda_tpl = Config.LAMBDA_TPL if lambda_tpl is None else lambda_tpl
        self.lambda_elec = Config.LAMBDA_ELEC if lambda_elec is None else lambda_elec
        self.template_scaffold = template_scaffold
        self.template_cross = template_cross

    def forward(self, x, y, encoder_out, 
                protein_coords=None, scaffold_features=None, protein_features=None, 
                template_scaffold=None, template_cross=None):
        """
        x: ground-truth point cloud [B, N, 3]
        y: reconstructed point cloud [B, M, 3]
        encoder_out: encoder output [B, M, D] (for VQ)
        protein_coords: [B, M2, 3]
        scaffold_features: [B, M, D]
        protein_features: [B, M2, D]
        template_scaffold: [N, N] or [B, N, N]
        template_cross: [N, M2] or [B, N, M2]
        """
        # Chamfer Distance
        chamfer = chamfer_distance(x, y).mean()
        # VQ Loss
        z_q, loss_vq = self.vq_ema(encoder_out)
        # Template prior loss（骨架-骨架+骨架-口袋）
        tpl_loss = template_prior_loss(
            y, encoder_out, protein_coords, 
            template_scaffold or self.template_scaffold, 
            template_cross or self.template_cross
        )
        # Electrostatic/vdw
        elec_loss = torch.tensor(0.0, device=y.device)
        if self.lambda_elec > 0 and protein_coords is not None and scaffold_features is not None and protein_features is not None:
            elec_loss = electrostatic_vdw_energy(
                y, protein_coords, scaffold_features, protein_features
            ).mean()
        # Total loss
        total_loss = chamfer + self.lambda_vq * loss_vq + self.lambda_tpl * tpl_loss + self.lambda_elec * elec_loss
        loss_dict = {
            'chamfer': chamfer.item(),
            'vq_loss': loss_vq.item(),
            'template_loss': tpl_loss.item(),
            'elec_loss': elec_loss.item(),
            'total_loss': total_loss.item()
        }
        return total_loss, loss_dict

def get_atom_charges_from_features(features, atom_type_list, charge_dict):
    """
    features: [B, N, D] or [N, D] (one-hot + ...)
    atom_type_list: list of atom type strings (e.g. Config.LIGAND_ATOM_TYPES)
    charge_dict: dict mapping atom type to charge (e.g. Config.ATOM_CHARGES)
    Returns: [B, N] or [N]
    """
    if features.dim() == 3:
        # [B, N, D]
        one_hot = features[..., :len(atom_type_list)]  # [B, N, T]
        idx = one_hot.argmax(dim=-1)  # [B, N]
        charges = torch.tensor([charge_dict[t] for t in atom_type_list], device=features.device)
        return charges[idx]
    else:
        # [N, D]
        one_hot = features[..., :len(atom_type_list)]  # [N, T]
        idx = one_hot.argmax(dim=-1)  # [N]
        charges = torch.tensor([charge_dict[t] for t in atom_type_list], device=features.device)
        return charges[idx]

def electrostatic_vdw_energy(x0, protein, ligand_features, protein_features, 
                             lambda_elec=1.0, lambda_vdw=1.0, epsilon=1.0, delta=1e-2, A=1.0, B=1.0):
    """
    Compute binding energy as sum of electrostatic and van der Waals terms.
    x0: [B, N_lig, 3]
    protein: [B, N_prot, 3]
    ligand_features: [B, N_lig, D] or [N_lig, D]
    protein_features: [B, N_prot, D] or [N_prot, D]
    Returns: [B] (energy per sample)
    """
    # Get charges
    ligand_charges = get_atom_charges_from_features(ligand_features, Config.LIGAND_ATOM_TYPES, Config.ATOM_CHARGES)  # [B, N_lig] or [N_lig]
    protein_charges = get_atom_charges_from_features(protein_features, Config.PROTEIN_ATOM_TYPES, Config.ATOM_CHARGES)  # [B, N_prot] or [N_prot]
    # Ensure batch
    if x0.dim() == 2:
        x0 = x0.unsqueeze(0)
        protein = protein.unsqueeze(0)
        ligand_charges = ligand_charges.unsqueeze(0)
        protein_charges = protein_charges.unsqueeze(0)
    dists = torch.cdist(x0, protein, p=2) + delta  # [B, N_lig, N_prot]
    # Electrostatic
    q_lig = ligand_charges.unsqueeze(-1)  # [B, N_lig, 1]
    q_prot = protein_charges.unsqueeze(1)  # [B, 1, N_prot]
    elec = (q_lig * q_prot / (epsilon * dists)).sum(dim=[1,2])  # [B]
    # van der Waals
    vdw = (A / dists**12 - B / dists**6).sum(dim=[1,2])  # [B]
    return lambda_elec * elec + lambda_vdw * vdw

class InpaintingLoss(nn.Module):
    """Inpainting loss: Denoising score matching + KL(VLB) + protein-guided binding loss + mask相关正则"""
    def __init__(self, lambda_kl=1.0, lambda_bind=1.0, lambda_mask_reg=0.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lambda_kl = lambda_kl
        self.lambda_bind = lambda_bind
        self.lambda_mask_reg = lambda_mask_reg

    def forward(self, predicted_noise, target_coords, quantized, 
                scaffold_coords, scaffold_coords2, scaffold_features,
                target_coords2, target_features, mask,
                protein_coords=None, protein_features=None):
        # 1. Denoising score matching（MSE，只在mask=1的地方）
        loss_denoise = self.mse(predicted_noise, target_coords)
        loss_denoise = (loss_denoise * mask.unsqueeze(-1)).sum() / mask.sum()

        # 2. KL(VLB) loss（可选，视你的扩散模型结构而定）
        loss_kl = torch.tensor(0.0, device=predicted_noise.device)  # 占位

        # 3. protein-guided binding loss（蛋白结合能引导）
        loss_bind = torch.tensor(0.0, device=predicted_noise.device)
        if protein_coords is not None and protein_features is not None:
            loss_bind = electrostatic_vdw_energy(
                target_coords2, protein_coords, target_features, protein_features
            ).mean()

        # 4. mask 区域的正则（可选，比如mask区域的预测不能偏离太多等）
        loss_mask_reg = torch.tensor(0.0, device=predicted_noise.device)
        if self.lambda_mask_reg > 0:
            # 例如：mask区域的 scaffold_coords 和 target_coords 的距离
            mask_diff = ((scaffold_coords - target_coords)**2).sum(dim=-1)
            loss_mask_reg = (mask_diff * mask).sum() / mask.sum()

        # 总损失
        total_loss = (loss_denoise +
                      self.lambda_kl * loss_kl +
                      self.lambda_bind * loss_bind +
                      self.lambda_mask_reg * loss_mask_reg)
        loss_dict = {
            'denoise': loss_denoise.item(),
            'kl': loss_kl.item() if isinstance(loss_kl, torch.Tensor) else float(loss_kl),
            'bind': loss_bind.item() if isinstance(loss_bind, torch.Tensor) else float(loss_bind),
            'mask_reg': loss_mask_reg.item() if isinstance(loss_mask_reg, torch.Tensor) else float(loss_mask_reg),
            'total': total_loss.item()
        }
        return total_loss, loss_dict 