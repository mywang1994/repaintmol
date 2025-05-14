import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from configs import Config
from losses import electrostatic_vdw_energy
import numpy as np
from egnn_pytorch import EGNN

class ResNetEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual  
            out = self.relu(out)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: [B, M, D]
        x = self.transformer(x)
        x = self.norm(x)
        x = self.dropout(x)
        residual = x
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        x = x + residual[..., :x.size(-1)] if residual.size(-1) == x.size(-1) else x  # 残差
        return x

class InpaintingModel(nn.Module):
    """
    Molecular inpainting model with mask, partial discrete diffusion, and protein-guided energy.
    Supports strict masking, one-hot discrete token diffusion, and protein context conditioning.
    """
    
    def __init__(self):
        super().__init__()
        
        # ResNet-style encoder
        self.resnet_encoder = ResNetEncoder(
            in_dim=3 + Config.ATOM_FEATURE_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=3
        )
        
        # EGNN for encoder
        self.encoder_egnn = EGNN(
            feats_dim=Config.HIDDEN_DIM,
            m_dim=0,
            num_layers=Config.NUM_LAYERS,
            norm_feats=True
        )
        
        # Discrete codebook for one-hot tokens
        self.codebook = nn.Parameter(
            torch.randn(Config.NUM_CODEBOOK_VECTORS, Config.HIDDEN_DIM)
        )
        
        # Transformer for decoder
        self.decoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=Config.HIDDEN_DIM + Config.HIDDEN_DIM,
                nhead=8,
                dim_feedforward=Config.HIDDEN_DIM * 2,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=4
        )
        
        # EGNN for decoder
        self.decoder_egnn = EGNN(
            feats_dim=Config.HIDDEN_DIM + Config.HIDDEN_DIM,
            m_dim=0,
            num_layers=Config.NUM_LAYERS,
            norm_feats=True
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        )
    
    def encode(self, coords: torch.Tensor,
               features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode coordinates and features into latent representation
        
        Args:
            coords: Atom coordinates [B, M, 3]
            features: Atom features [B, M, D]
            
        Returns:
            z: Latent representation [B, M, H]
            quantized: Quantized latent representation [B, M, H]
            indices: Indices of the quantized representation [B, M]
        """
        # Concatenate coordinates and features
        x = torch.cat([coords, features], dim=-1)
        
        # ResNet-style编码
        h = self.resnet_encoder(x)  # [B, M, H]
        
        # EGNN空间关系建模
        h, _ = self.encoder_egnn(h=h, x=coords)
        
        z = h
        
        # Vector quantization (one-hot token)
        flat_z = z.view(-1, z.size(-1))  # [B*M, H]
        codebook = self.codebook  # [K, H]
        dists = torch.cdist(flat_z, codebook)  # [B*M, K]
        indices = torch.argmin(dists, dim=-1)  # [B*M]
        one_hot = F.one_hot(indices, num_classes=codebook.size(0)).float()  # [B*M, K]
        quantized = torch.matmul(one_hot, codebook)  # [B*M, H]
        quantized = quantized.view(z.size())  # [B, M, H]
        
        return z, quantized, indices.view(z.size(0), z.size(1))
    
    def decode(self, z: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent representation into coordinates and features
        Args:
            z: Latent representation [B, M, 2H]
            coords: Atom coordinates [B, M, 3]
        Returns:
            coords: Decoded coordinates [B, M, 3]
            features: Decoded features [B, M, D]
        """
        z_t = self.decoder_transformer(z)
        h, x = self.decoder_egnn(h=z_t, x=coords)
        coords = x
        features = h[..., 3:]
        return coords, features
    
    def forward(self, coords: torch.Tensor,
                features: torch.Tensor,
                t: torch.Tensor,
                mask: torch.Tensor,
                protein_coords: Optional[torch.Tensor] = None,
                protein_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            coords: Atom coordinates [B, M, 3]
            features: Atom features [B, M, D]
            t: Time step [B]
            mask: Mask tensor [B, M] (1=to inpaint, 0=keep fixed)
            protein_coords: Protein coordinates [B, P, 3] (optional)
            protein_features: Protein features [B, P, D] (optional)
            
        Returns:
            predicted_noise: Predicted noise [B, M, H]
            quantized: Quantized latent representation [B, M, H]
            indices: Indices of the quantized representation [B, M]
        """
        z, quantized, indices = self.encode(coords, features)
        t_embedding = self.time_embedding(t.unsqueeze(-1).float())
        t_embedding = t_embedding.unsqueeze(1).expand(-1, z.size(1), -1)
        z = z + t_embedding
        if protein_features is not None:
            prot_ctx = protein_features.mean(dim=1, keepdim=True).expand(-1, z.size(1), -1)
        else:
            prot_ctx = torch.zeros(z.size(0), z.size(1), Config.HIDDEN_DIM, device=z.device)
        z_input = torch.cat([z, prot_ctx], dim=-1)  # [B, M, 2H]
        z_t = self.decoder_transformer(z_input)
        h, x = self.decoder_egnn(h=z_t, x=coords)
        predicted_noise = mask.unsqueeze(-1) * h
        quantized = mask.unsqueeze(-1) * quantized + (1 - mask.unsqueeze(-1)) * z
        return predicted_noise, quantized, indices
    
    def q_sample(self, indices, t, mask, transition_matrices):
        """
        Forward discrete diffusion: only mask=1 positions are noised.
        indices: [B, M] (int tokens)
        t: int (current timestep)
        mask: [B, M]
        transition_matrices: list of [K, K] matrices, len=Config.DIFFUSION_STEPS
        Returns: noised_indices [B, M]
        """
        B, M = indices.shape
        K = transition_matrices[0].shape[0]
        device = indices.device
        noised_indices = indices.clone()
        for b in range(B):
            for m in range(M):
                if mask[b, m] == 1:
                    # Sample new token according to Q_t
                    probs = transition_matrices[t][indices[b, m]].cpu().numpy()
                    new_token = np.random.choice(K, p=probs)
                    noised_indices[b, m] = new_token
        return noised_indices
    
    def sample(self, coords: torch.Tensor,
               features: torch.Tensor,
               mask: torch.Tensor,
               protein_coords: Optional[torch.Tensor] = None,
               protein_features: Optional[torch.Tensor] = None,
               num_samples: int = 1,
               lambda_bind: float = 0.4,
               transition_matrices: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample and repair molecules
        
        Args:
            coords: Input coordinates [B, M, 3]
            features: Input features [B, M, D]
            mask: Mask tensor [B, M] (1=to inpaint, 0=keep fixed)
            protein_coords: Protein coordinates [B, P, 3] (optional)
            protein_features: Protein features [B, P, D] (optional)
            num_samples: Number of samples to generate
            lambda_bind: Binding energy weight for protein-guided energy
            transition_matrices: List of [K, K] matrices for each t
            
        Returns:
            coords: Repaired coordinates [B, M, 3]
            features: Repaired features [B, M, D]
        """
        batch_size, M, _ = coords.shape
        device = coords.device
        _, _, indices = self.encode(coords, features)  # [B, M]
        K = self.codebook.size(0)
        t_max = Config.DIFFUSION_STEPS - 1
        if transition_matrices is not None:
            indices_t = self.q_sample(indices, t_max, mask, transition_matrices)
        else:
            indices_t = indices.clone()
            indices_t[mask == 1] = torch.randint(0, K, (mask == 1).sum().item(), device=device)
        z = torch.matmul(F.one_hot(indices_t, num_classes=K).float(), self.codebook)  # [B, M, H]
        for t in range(t_max, -1, -1):
            t_tensor = torch.full((batch_size,), t, device=device)
            z.requires_grad_(True)
            if protein_features is not None:
                prot_ctx = protein_features.mean(dim=1, keepdim=True).expand(-1, z.size(1), -1)
            else:
                prot_ctx = torch.zeros(z.size(0), z.size(1), Config.HIDDEN_DIM, device=z.device)
            z_input = torch.cat([z, prot_ctx], dim=-1)
            z_t = self.decoder_transformer(z_input)
            h, x = self.decoder_egnn(h=z_t, x=coords)
            predicted_noise = mask.unsqueeze(-1) * h
            if lambda_bind > 0 and protein_coords is not None and protein_features is not None:
                coords_pred, features_pred = self.decode(z_input, coords)
                bind_energy = electrostatic_vdw_energy(coords_pred, protein_coords, features_pred, protein_features).sum()
                grad_bind = torch.autograd.grad(bind_energy, z, retain_graph=True, create_graph=True)[0]
                predicted_noise = predicted_noise + lambda_bind * mask.unsqueeze(-1) * grad_bind
            alpha = 1 - Config.BETA_START
            alpha_bar = alpha ** t
            z = (z - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if t > 0:
                noise = torch.randn_like(z)
                z = z + mask.unsqueeze(-1) * torch.sqrt(1 - alpha) * noise
        coords_out, features_out = self.decode(z_input, coords)
        coords_out = mask.unsqueeze(-1) * coords_out + (1 - mask.unsqueeze(-1)) * coords
        features_out = mask.unsqueeze(-1) * features_out + (1 - mask.unsqueeze(-1)) * features
        return coords_out, features_out 