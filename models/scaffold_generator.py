import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from configs import Config
from losses import electrostatic_vdw_energy
from egnn_pytorch import EGNN

def template_score_func(scaffold_coords, template_info, sigma=1.0):
    # scaffold_coords: [B, N, 3]
    # template_info: dict, 包含fragments, bonds, angle_constraints等
    # fragments: {'acid': {'indices': [...]}, ...}
    # bonds: [('acid', 'amine'), ...]
    score = 0.0
    for frag_i, frag_j in template_info['bonds']:
        idx_i = template_info['fragments'][frag_i]['indices']  # 片段i的原子索引
        idx_j = template_info['fragments'][frag_j]['indices']  # 片段j的原子索引
        centroid_i = scaffold_coords[:, idx_i, :].mean(dim=1)  # [B, 3]
        centroid_j = scaffold_coords[:, idx_j, :].mean(dim=1)  # [B, 3]
        dist = ((centroid_i - centroid_j) ** 2).sum(dim=-1).sqrt()  # [B]
        score = score + torch.exp(-dist ** 2 / (2 * sigma ** 2)).sum()
    # 可选：角度约束
    # ...
    return score

class ScaffoldGenerator(nn.Module):
    """Scaffold generation model using diffusion model with template/pocket guidance"""
    
    def __init__(self):
        super().__init__()
        
        class ResidualPocketEncoder(nn.Module):
            def __init__(self, in_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden_dim)
                self.relu1 = nn.ReLU()
                self.drop1 = nn.Dropout(0.1)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.relu2 = nn.ReLU()
                self.drop2 = nn.Dropout(0.1)
                self.fc3 = nn.Linear(hidden_dim, hidden_dim)
                self.relu3 = nn.ReLU()
            def forward(self, x):
                identity = x
                out = self.fc1(x)
                out = self.relu1(out)
                out = self.drop1(out)
                out = self.fc2(out)
                out = self.relu2(out)
                out = self.drop2(out)
                out = self.fc3(out)
                # 残差连接（如果输入输出维度一致）
                if identity.shape[-1] == out.shape[-1]:
                    out = out + identity
                out = self.relu3(out)
                return out

        self.pocket_encoder = ResidualPocketEncoder(3 + Config.ATOM_FEATURE_DIM, Config.HIDDEN_DIM)
        
        self.egnn = EGNN(
            feats_dim=Config.HIDDEN_DIM,  
            m_dim=0,  
            num_layers=6,
            norm_feats=True
        )
        
        class ResidualDecoder(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(hidden_dim, hidden_dim)
                self.relu1 = nn.ReLU()
                self.drop1 = nn.Dropout(0.1)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.relu2 = nn.ReLU()
                self.drop2 = nn.Dropout(0.1)
                self.fc3 = nn.Linear(hidden_dim, 3)
            def forward(self, x):
                identity = x
                out = self.fc1(x)
                out = self.relu1(out)
                out = self.drop1(out)
                out = self.fc2(out)
                out = self.relu2(out)
                out = self.drop2(out)
                # 残差连接（如果输入输出维度一致）
                if identity.shape[-1] == out.shape[-1]:
                    out = out + identity
                out = self.fc3(out)
                return out

        self.decoder_out = ResidualDecoder(Config.HIDDEN_DIM)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        )

        class ResidualFeatureDecoder(nn.Module):
            def __init__(self, feature_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(feature_dim, hidden_dim)
                self.relu1 = nn.ReLU()
                self.drop1 = nn.Dropout(0.1)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.relu2 = nn.ReLU()
                self.drop2 = nn.Dropout(0.1)
                self.fc3 = nn.Linear(hidden_dim, feature_dim)
            def forward(self, x):
                identity = x
                out = self.fc1(x)
                out = self.relu1(out)
                out = self.drop1(out)
                out = self.fc2(out)
                out = self.relu2(out)
                out = self.drop2(out)
                out = self.fc3(out)
                if identity.shape[-1] == out.shape[-1]:
                    out = out + identity
                return out

        self.feature_decoder = ResidualFeatureDecoder(Config.ATOM_FEATURE_DIM, Config.HIDDEN_DIM)
    
    def encode_pocket(self, pocket_coords: torch.Tensor,
                     pocket_features: torch.Tensor) -> torch.Tensor:
        """Encode pocket coordinates and features
        
        Args:
            pocket_coords: Pocket atom coordinates [B, N, 3]
            pocket_features: Pocket atom features [B, N, D]
            
        Returns:
            pocket_encoding: Encoded pocket representation [B, N, H]
        """
        # Concatenate coordinates and features
        pocket_input = torch.cat([pocket_coords, pocket_features], dim=-1)
        
        # Encode pocket
        pocket_encoding = self.pocket_encoder(pocket_input)
        
        return pocket_encoding
    
    def forward(self, pocket_coords: torch.Tensor,
                pocket_features: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            pocket_coords: Pocket atom coordinates [B, N, 3]
            pocket_features: Pocket atom features [B, N, D]
            t: Time step [B]
            
        Returns:
            predicted_noise: Predicted noise [B, M, 3]
        """
        # Encode pocket
        pocket_encoding = self.encode_pocket(pocket_coords, pocket_features)
        
        # Add time embedding
        t_embedding = self.time_embedding(t.unsqueeze(-1).float())
        t_embedding = t_embedding.unsqueeze(1).expand(-1, pocket_encoding.size(1), -1)
        pocket_encoding = pocket_encoding + t_embedding
        

        egnn_output, _ = self.egnn(
            h=pocket_encoding,
            x=pocket_coords
        )
        predicted_noise = self.decoder_out(egnn_output)
        
        return predicted_noise
    
    def sample(self, pocket_coords: torch.Tensor,
               pocket_features: torch.Tensor,
               num_samples: int = 1,
               use_template_guidance=Config.USE_TEMPLATE_GUIDANCE,
               use_pocket_guidance=Config.USE_POCKET_GUIDANCE,
               alpha_tpl=Config.ALPHA_TPL,
               alpha_pocket=Config.ALPHA_POCKET,
               template_info=None,
               protein_coords=None,
               protein_features=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample scaffold coordinates and features
        
        Args:
            pocket_coords: Pocket atom coordinates [B, N, 3]
            pocket_features: Pocket atom features [B, N, D]
            num_samples: Number of samples to generate
            use_template_guidance: Whether to use template guidance
            use_pocket_guidance: Whether to use pocket guidance
            alpha_tpl: Template guidance weight
            alpha_pocket: Pocket guidance weight
            template_info: Template information
            protein_coords: Protein atom coordinates [B, P, 3]
            protein_features: Protein atom features [B, P, D]
            
        Returns:
            scaffold_coords: Generated scaffold coordinates [B, M, 3]
            scaffold_features: Generated scaffold features [B, M, D]
        """
        # Initialize random noise
        batch_size = pocket_coords.size(0)
        scaffold_coords = torch.randn(batch_size, num_samples, 3, device=pocket_coords.device, requires_grad=True)
        scaffold_features = torch.randn(batch_size, num_samples, Config.ATOM_FEATURE_DIM,
                                      device=pocket_coords.device)
        
        # Reverse diffusion process
        for t in range(Config.DIFFUSION_STEPS - 1, -1, -1):
            # Create time tensor
            t_tensor = torch.full((batch_size,), t, device=pocket_coords.device)

            # 预测噪声
            scaffold_coords.requires_grad_(True)
            scaffold_features.requires_grad_(True)
            predicted_noise_coords = self(pocket_coords, pocket_features, t_tensor)  # [B, M, 3]
            predicted_noise_features = self.feature_decoder(scaffold_features)       # [B, M, D]

            # 模板引导
            if use_template_guidance and template_info is not None:
                tpl_score = template_score_func(scaffold_coords, template_info)
                grad_tpl = torch.autograd.grad(tpl_score, scaffold_coords, retain_graph=True, create_graph=True)[0]
                predicted_noise_coords = predicted_noise_coords + alpha_tpl * grad_tpl

            # 蛋白引导
            if use_pocket_guidance and protein_coords is not None and protein_features is not None:
                pocket_score = electrostatic_vdw_energy(scaffold_coords, protein_coords, scaffold_features, protein_features).sum()
                grad_coords, grad_features = torch.autograd.grad(
                    pocket_score, [scaffold_coords, scaffold_features], retain_graph=True, create_graph=True
                )
                predicted_noise_coords = predicted_noise_coords + alpha_pocket * grad_coords
                predicted_noise_features = predicted_noise_features + alpha_pocket * grad_features

            # 更新坐标
            alpha = 1 - Config.BETA_START
            alpha_bar = alpha ** t
            scaffold_coords = (scaffold_coords - 
                             (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise_coords) / \
                             torch.sqrt(alpha)
            # 更新特征
            scaffold_features = (scaffold_features - 
                                (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise_features) / \
                                torch.sqrt(alpha)

            # 加噪声
            if t > 0:
                noise = torch.randn_like(scaffold_coords)
                scaffold_coords = scaffold_coords + torch.sqrt(1 - alpha) * noise
                noise_feat = torch.randn_like(scaffold_features)
                scaffold_features = scaffold_features + torch.sqrt(1 - alpha) * noise_feat

        return scaffold_coords, scaffold_features 