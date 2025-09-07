# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# SRA-enhanced DiT for molecular generation
# Adapted from SiT paper: Self-Representation Alignment
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers import use_fused_attn
import einops
from copy import deepcopy

from models import (
    modulate, CrossAttention, TimestepEmbedder, RMSNorm, SwiGLU,
    get_1d_sincos_pos_embed_from_grid
)


class ProjectionHead(nn.Module):
    """Lightweight projection head for SRA alignment"""
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_dim + out_dim) // 2
            
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class DiTBlockSRA(nn.Module):
    """
    DiT block with SRA capability - maintains your existing architecture
    while adding ability to extract intermediate representations
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cross_attn=0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        if cross_attn > 0:
            self.norm3 = RMSNorm(hidden_size, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, cross_attn, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(
            dim_in=hidden_size,
            dim_hidden=mlp_hidden_dim, 
            dim_out=hidden_size
        )
        
        self.factor = 9 if cross_attn > 0 else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * self.factor, bias=True)
        )

    def forward(self, x, c, y=None, pad_mask=None, return_intermediate=False):
        if self.factor == 9:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(self.factor, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(self.factor, dim=1)
            
        def modulate_rms(x, shift, scale):
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate_rms(self.norm1(x), shift_msa, scale_msa))
        
        # Store intermediate representation after self-attention
        if return_intermediate:
            intermediate_after_sa = x.clone()
        
        # Cross-attention (if applicable)
        if self.factor == 9:
            x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate_rms(self.norm3(x), shift_mca, scale_mca), y, pad_mask)
            
        # Store intermediate representation after cross-attention
        if return_intermediate:
            intermediate_after_ca = x.clone()
        
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate_rms(self.norm2(x), shift_mlp, scale_mlp))
        
        if return_intermediate:
            # Return the representation after cross-attention (or self-attention if no cross-attention)
            # This captures the most meaningful conditioning-aware representation
            if self.factor == 9:
                return x, intermediate_after_ca
            else:
                return x, intermediate_after_sa
        
        return x


class FinalLayerSRA(nn.Module):
    """Final layer with SRA support - keeps your existing architecture"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        def modulate_rms(x, shift, scale):
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            
        x = modulate_rms(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTSRA(nn.Module):
    """
    DiT model enhanced with Self-Representation Alignment
    Maintains all your existing functionality while adding SRA training capability
    """
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            learn_sigma=True, 
            cross_attn=256, 
            condition_dim=256,
            # SRA-specific parameters
            use_sra=True,
            sra_layer_student=4,  # Which layer to extract student representation from
            sra_layer_teacher=8,  # Which layer to extract teacher representation from
            sra_projection_dim=None,  # Projection head output dim (None = same as hidden_size)
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        
        # SRA configuration
        self.use_sra = use_sra
        self.sra_layer_student = sra_layer_student
        self.sra_layer_teacher = sra_layer_teacher
        self.depth = depth
        
        # Validate SRA layer indices
        assert 0 <= sra_layer_student < depth, f"Student layer {sra_layer_student} must be < depth {depth}"
        assert sra_layer_student < sra_layer_teacher < depth, f"Teacher layer {sra_layer_teacher} must be > student layer {sra_layer_student} and < depth {depth}"

        # Core architecture (unchanged from your implementation)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_linear = nn.Linear(condition_dim, cross_attn)
        self.y_to_hidden = nn.Linear(condition_dim, hidden_size)
        
        num_patches = input_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Blocks with SRA support
        self.blocks = nn.ModuleList([
            DiTBlockSRA(hidden_size, num_heads, mlp_ratio=mlp_ratio, cross_attn=cross_attn) 
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayerSRA(hidden_size, patch_size, self.out_channels)
        
        # SRA components
        if self.use_sra:
            if sra_projection_dim is None:
                sra_projection_dim = hidden_size
            self.sra_projection_head = ProjectionHead(hidden_size, sra_projection_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.input_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y=None, pad_mask=None, teacher_mode=False, teacher_timestep_offset=None):
        """
        Forward pass with optional SRA intermediate representation extraction
        
        Args:
            x: Input tensor
            t: Timestep
            y: Conditioning (RNA + image features)
            pad_mask: Padding mask for conditioning
            teacher_mode: If True, extract teacher representation
            teacher_timestep_offset: Offset for teacher timestep (for lower noise)
        """
        x = x.squeeze(-1).permute((0, 2, 1))
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        
        # Handle timestep for teacher mode (lower noise)
        if teacher_mode and teacher_timestep_offset is not None:
            t_effective = torch.clamp(t - teacher_timestep_offset, min=0)
        else:
            t_effective = t
            
        t_emb = self.t_embedder(t_effective)  # (N, D)

        if y is not None:   
            # Handle the stacked [control, treatment] features
            y_for_cross_attn = self.y_linear(y)  # Keep original for cross-attention
            y_pooled = y.mean(dim=1)  # Pool control+treatment: [B, 2, 192] -> [B, 192]
            y_pooled = self.y_to_hidden(y_pooled)  # Project to hidden_size: [B, 192] -> [B, 768]
        else:
            y_for_cross_attn = None
            y_pooled = torch.zeros_like(t_emb)  # Match timestep embedding dimensions

        c = t_emb + y_pooled  # Now both are [B, 768]

        # Forward through blocks with optional intermediate extraction
        sra_representation = None
        
        for i, block in enumerate(self.blocks):
            # Determine if we need intermediate representation from this layer
            extract_intermediate = self.use_sra and (
                (not teacher_mode and self.training and i == self.sra_layer_student) or
                (teacher_mode and i == self.sra_layer_teacher)
            )
            
            if extract_intermediate:
                x, intermediate = block(x, c, y_for_cross_attn, pad_mask, return_intermediate=True)
                sra_representation = intermediate
            else:
                x = block(x, c, y_for_cross_attn, pad_mask, return_intermediate=False)
        
        x = self.final_layer(x, c).permute((0, 2, 1)).unsqueeze(-1)  # (N, T, patch_size ** 2 * out_channels)
        
        return x, sra_representation

    def forward_with_cfg(self, x, t, y, pad_mask, cfg_scale):
        """
        Forward pass with classifier-free guidance (unchanged from your implementation)
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, _ = self.forward(combined, t, y, pad_mask)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class EMAManager:
    """
    Manages EMA teacher network for SRA training
    """
    def __init__(self, student_model, ema_decay=0.9999):
        self.ema_decay = ema_decay
        self.teacher_model = deepcopy(student_model).to(next(student_model.parameters()).device)
        
        # Freeze teacher parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Initialize teacher with student weights
        self.update_teacher(student_model, decay=0.0)
    
    def update_teacher(self, student_model, decay=None):
        """Update teacher model using EMA of student weights"""
        if decay is None:
            decay = self.ema_decay
            
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), 
                student_model.parameters()
            ):
                teacher_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)
    
    def get_teacher(self):
        return self.teacher_model


def compute_sra_loss(student_repr, teacher_repr, projection_head, distance_type='mse'):
    """
    Compute Self-Representation Alignment loss
    
    Args:
        student_repr: [B, N, D] student representation
        teacher_repr: [B, N, D] teacher representation  
        projection_head: projection network for student representation
        distance_type: 'mse' or 'cosine'
    """
    if student_repr is None or teacher_repr is None:
        return torch.tensor(0.0, device=student_repr.device if student_repr is not None else teacher_repr.device)
    
    # Project student representation
    student_projected = projection_head(student_repr)  # [B, N, D]
    
    # Compute patch-wise distance
    if distance_type == 'mse':
        # MSE loss between projected student and teacher representations
        loss = F.mse_loss(student_projected, teacher_repr.detach(), reduction='none')
        loss = loss.mean(dim=-1)  # Average over feature dimension
        loss = loss.mean()  # Average over patches and batch
    elif distance_type == 'cosine':
        # Cosine similarity loss
        student_norm = F.normalize(student_projected, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_repr.detach(), p=2, dim=-1)
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)  # [B, N]
        loss = (1 - cosine_sim).mean()  # Convert similarity to loss
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")
    
    return loss


# Model factory functions (maintain compatibility with your existing code)
def LDMolSRA(**kwargs):
    return DiTSRA(depth=12, hidden_size=768, patch_size=1, num_heads=16, **kwargs)

def DiT_XL_2_SRA(**kwargs):
    return DiTSRA(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4_SRA(**kwargs):
    return DiTSRA(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_L_2_SRA(**kwargs):
    return DiTSRA(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_B_2_SRA(**kwargs):
    return DiTSRA(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

DiT_SRA_models = {
    'LDMolSRA': LDMolSRA,
    'DiT-XL/2-SRA': DiT_XL_2_SRA, 
    'DiT-XL/4-SRA': DiT_XL_4_SRA,
    'DiT-L/2-SRA': DiT_L_2_SRA,
    'DiT-B/2-SRA': DiT_B_2_SRA,
}