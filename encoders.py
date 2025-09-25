import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ResidualBlock(nn.Module):
    """Residual block with normalization and dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        # Skip connection with projection if dimensions don't match
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.main_branch(x) + self.skip(x)


class GeneMultiHeadAttention(nn.Module):
    """Multi-head self-attention for genes to attend to each other."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections for all heads at once
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, num_genes, embed_dim]
        """
        B, N, D = x.shape  # B=batch, N=num_genes, D=embed_dim
        
        # Generate Q, K, V for all heads
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        return self.out_proj(out), attn_weights


class RNAEncoder(nn.Module):
    """
    Encoder for RNA expression data with real self-attention over genes.
    Genes can dynamically attend to each other based on expression context.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=256, 
                dropout=0.1, use_gene_relations=False, num_heads=4, relation_rank=25,
                gene_embed_dim=512, num_attention_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_gene_relations = use_gene_relations
        self.num_heads = num_heads
        self.relation_rank = relation_rank
        self.gene_embed_dim = gene_embed_dim
        
        # Project raw gene expressions to embedding space for attention
        self.gene_embedding = nn.Sequential(
            nn.Linear(1, gene_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(gene_embed_dim // 2, gene_embed_dim),
            nn.LayerNorm(gene_embed_dim)
        )

        self.gene_relation_projection = nn.Linear(gene_embed_dim, 1)
        
        # Multi-layer self-attention for genes
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GeneMultiHeadAttention(gene_embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(gene_embed_dim),
                'norm2': nn.LayerNorm(gene_embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(gene_embed_dim, gene_embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gene_embed_dim * 2, gene_embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_attention_layers)
        ])

        # Low-rank gene relations
        if use_gene_relations:
            self.gene_relation_net_base = nn.Sequential(
                nn.Linear(gene_embed_dim * input_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.gene_relation_factors_head = nn.Linear(256, 2 * input_dim * self.relation_rank)

        # Pooling to get sample-level representation
        self.pooling_type = 'attention'
        if self.pooling_type == 'attention':
            self.pooling_attention = nn.Sequential(
                nn.Linear(gene_embed_dim, 1),
                nn.Tanh()
            )
        
        # Final encoder layers
        pooled_dim = gene_embed_dim
        layers = []
        prev_dim = pooled_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.LayerNorm(prev_dim),
                ResidualBlock(prev_dim, hidden_dim, dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Final projection
        self.final_encoder = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def apply_gene_relations(self, x_attended):
        """Apply learned gene-gene relationships using low-rank factorization."""
        batch_size, num_genes, embed_dim = x_attended.shape

        # Flatten attended features for relation learning
        x_flat = x_attended.view(batch_size, -1)  # [B, num_genes * embed_dim]
        
        # Get cell-specific embedding from attended gene features
        cell_embedding_for_relations = self.gene_relation_net_base(x_flat)

        # Predict parameters for U and V factor matrices
        relation_factors_params = self.gene_relation_factors_head(cell_embedding_for_relations)

        # Reshape to get U [B, G, K] and V [B, K, G] matrices per cell
        U = relation_factors_params[:, :num_genes * self.relation_rank].view(
            batch_size, num_genes, self.relation_rank
        )
        V = relation_factors_params[:, num_genes * self.relation_rank:].view(
            batch_size, self.relation_rank, num_genes
        )

        # Apply transformation to mean-pooled gene features for relations
        gene_values = self.gene_relation_projection(x_attended).squeeze(-1)  # [B, G]
        gene_values_unsqueezed = gene_values.unsqueeze(1)  # [B, 1, G]
        temp = torch.bmm(gene_values_unsqueezed, U)  # [B, 1, K]
        gene_relations = torch.bmm(temp, V).squeeze(1)  # [B, G]
        
        # Apply relations back to attended features
        relation_weights = torch.sigmoid(gene_relations).unsqueeze(-1)  # [B, G, 1]
        return x_attended * (1 + 0.1 * relation_weights)

    def pool_gene_features(self, x):
        """Pool gene features to get sample-level representation."""
        if self.pooling_type == 'mean':
            return x.mean(dim=1)  # [B, embed_dim]
        elif self.pooling_type == 'max':
            return x.max(dim=1)[0]  # [B, embed_dim]
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attn_weights = self.pooling_attention(x)  # [B, N, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            return (x * attn_weights).sum(dim=1)  # [B, embed_dim]

    def forward(self, x):
        """
        Forward pass with optional KG enhancement.
        x: [batch_size, num_genes] - raw gene expressions
        """
        batch_size, num_genes = x.shape
        
        # Embed each gene expression individually
        x_reshaped = x.unsqueeze(-1)  # [B, G, 1]
        gene_embeds = self.gene_embedding(x_reshaped)  # [B, G, embed_dim]
        
        # Apply multi-layer self-attention between genes
        attention_weights_all = []
        x_attended = gene_embeds
        
        for layer in self.attention_layers:
            # Self-attention with residual connection
            attn_out, attn_weights = layer['attention'](x_attended)
            x_attended = layer['norm1'](x_attended + attn_out)
            
            # Feed-forward with residual connection  
            ffn_out = layer['ffn'](x_attended)
            x_attended = layer['norm2'](x_attended + ffn_out)
            
            attention_weights_all.append(attn_weights)
        
        # Apply gene relations if enabled
        if self.use_gene_relations:
            x_attended = self.apply_gene_relations(x_attended)
        
        # Pool to get sample-level representation
        pooled_features = self.pool_gene_features(x_attended)  # [B, embed_dim]
        
        # Pass through encoder layers
        encoded_features = self.encoder(pooled_features)
        
        # Final projection
        final_embeddings = self.final_encoder(encoded_features)
        
        return final_embeddings
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability."""
        with torch.no_grad():
            batch_size, num_genes = x.shape
            
            # Embed genes
            x_reshaped = x.unsqueeze(-1)
            gene_embeds = self.gene_embedding(x_reshaped)
            
            # Get attention weights from each layer
            attention_weights_all = []
            x_attended = gene_embeds
            
            for layer in self.attention_layers:
                attn_out, attn_weights = layer['attention'](x_attended)
                x_attended = layer['norm1'](x_attended + attn_out)
                ffn_out = layer['ffn'](x_attended)
                x_attended = layer['norm2'](x_attended + ffn_out)
                attention_weights_all.append(attn_weights.cpu())
            
            return attention_weights_all  # List of [B, num_heads, N, N] tensors


class GeneConditionCrossAttention(nn.Module):
    """Cross-attention between control and treatment gene representations."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate Q, K, V projections for cross-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False) 
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        query: [B, N, D] - genes from one condition (e.g., control)
        key_value: [B, N, D] - genes from other condition (e.g., treatment)
        """
        B, N, D = query.shape
        
        # Generate Q from query, K and V from key_value
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Cross-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        return self.out_proj(out), attn_weights


class PairedRNAEncoder(nn.Module):
    """
    Encoder for paired RNA expression data (control vs treatment) with explicit
    gene-to-gene cross-attention to learn which pre-treatment genes map to 
    which post-treatment genes for drug prediction.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=256, 
                dropout=0.1, num_heads=4, gene_embed_dim=512, 
                num_self_attention_layers=1, num_cross_attention_layers=1,
                use_bidirectional_cross_attn=True, enable_separate_outputs=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.gene_embed_dim = gene_embed_dim
        self.use_bidirectional_cross_attn = use_bidirectional_cross_attn
        
        # Shared gene embedding network for both conditions
        self.gene_embedding = nn.Sequential(
            nn.Linear(1, gene_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(gene_embed_dim // 2, gene_embed_dim),
            nn.LayerNorm(gene_embed_dim)
        )
        
        # Self-attention layers for each condition (applied separately)
        self.self_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GeneMultiHeadAttention(gene_embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(gene_embed_dim),
                'norm2': nn.LayerNorm(gene_embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(gene_embed_dim, gene_embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gene_embed_dim * 2, gene_embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_self_attention_layers)
        ])
        
        # Cross-attention layers between conditions
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'control_to_treatment': GeneConditionCrossAttention(gene_embed_dim, num_heads, dropout),
                'treatment_to_control': GeneConditionCrossAttention(gene_embed_dim, num_heads, dropout) if use_bidirectional_cross_attn else None,
                'norm_control': nn.LayerNorm(gene_embed_dim),
                'norm_treatment': nn.LayerNorm(gene_embed_dim),
                'ffn_control': nn.Sequential(
                    nn.Linear(gene_embed_dim, gene_embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gene_embed_dim * 2, gene_embed_dim),
                    nn.Dropout(dropout)
                ),
                'ffn_treatment': nn.Sequential(
                    nn.Linear(gene_embed_dim, gene_embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gene_embed_dim * 2, gene_embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_cross_attention_layers)
        ])
        
        # Pooling for each condition
        self.pooling_attention = nn.Sequential(
            nn.Linear(gene_embed_dim, 1),
            nn.Tanh()
        )
        
        # Final encoder layers for combined representation
        pooled_dim = gene_embed_dim * 2  # Both control and treatment pooled features
        layers = []
        prev_dim = pooled_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.LayerNorm(prev_dim),
                ResidualBlock(prev_dim, hidden_dim, dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Final projection
        self.final_encoder = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

         # Add separate projectors if needed
        if enable_separate_outputs:
            self.control_projector = nn.Sequential(
                nn.LayerNorm(gene_embed_dim),
                nn.Linear(gene_embed_dim, output_dim // 2),
                nn.Dropout(dropout),
                nn.LayerNorm(output_dim // 2)
            )
            self.treatment_projector = nn.Sequential(
                nn.LayerNorm(gene_embed_dim),
                nn.Linear(gene_embed_dim, output_dim // 2),
                nn.Dropout(dropout),
                nn.LayerNorm(output_dim // 2)
            )
        else:
            self.control_projector = None
            self.treatment_projector = None

    def apply_self_attention(self, x):
        """Apply self-attention within condition."""
        for layer in self.self_attention_layers:
            attn_out, _ = layer['attention'](x)
            x = layer['norm1'](x + attn_out)
            
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        return x

    def apply_cross_attention(self, control_genes, treatment_genes):
        """Apply cross-attention between conditions."""
        cross_attn_weights = []
        
        for layer in self.cross_attention_layers:
            # Control genes attend to treatment genes
            control_cross_out, control_cross_weights = layer['control_to_treatment'](control_genes, treatment_genes)
            control_genes = layer['norm_control'](control_genes + control_cross_out)
            control_ffn_out = layer['ffn_control'](control_genes)
            control_genes = layer['norm_control'](control_genes + control_ffn_out)
            
            # Treatment genes attend to control genes (if bidirectional)
            if self.use_bidirectional_cross_attn and layer['treatment_to_control'] is not None:
                treatment_cross_out, treatment_cross_weights = layer['treatment_to_control'](treatment_genes, control_genes)
                treatment_genes = layer['norm_treatment'](treatment_genes + treatment_cross_out)
                treatment_ffn_out = layer['ffn_treatment'](treatment_genes)
                treatment_genes = layer['norm_treatment'](treatment_genes + treatment_ffn_out)
                
                cross_attn_weights.append({
                    'control_to_treatment': control_cross_weights,
                    'treatment_to_control': treatment_cross_weights
                })
            else:
                cross_attn_weights.append({
                    'control_to_treatment': control_cross_weights
                })
        
        return control_genes, treatment_genes, cross_attn_weights

    def pool_gene_features(self, x):
        """Pool gene features to get condition-level representation."""
        attn_weights = self.pooling_attention(x)  # [B, N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        return (x * attn_weights).sum(dim=1)  # [B, embed_dim]

    def forward(self, control_rna, treatment_rna, return_separate=True):
        """
        Forward pass with gene-to-gene cross-attention.
        
        Args:
            control_rna: [batch_size, num_genes] - control condition gene expressions
            treatment_rna: [batch_size, num_genes] - treatment condition gene expressions
            return_separate: If True, return separate control and treatment representations
        
        Returns:
            If return_separate=False: final_embeddings: [batch_size, output_dim] - combined representation
            If return_separate=True: (control_features, treatment_features) each [batch_size, output_dim//2]
        """
        batch_size, num_genes = control_rna.shape
        
        # Embed genes for both conditions
        control_reshaped = control_rna.unsqueeze(-1)  # [B, G, 1]
        treatment_reshaped = treatment_rna.unsqueeze(-1)  # [B, G, 1]
        
        control_gene_embeds = self.gene_embedding(control_reshaped)  # [B, G, embed_dim]
        treatment_gene_embeds = self.gene_embedding(treatment_reshaped)  # [B, G, embed_dim]
        
        # Apply self-attention within each condition
        control_self_attended = self.apply_self_attention(control_gene_embeds)
        treatment_self_attended = self.apply_self_attention(treatment_gene_embeds)
        
        # Apply cross-attention between conditions
        control_cross_attended, treatment_cross_attended, cross_attn_weights = self.apply_cross_attention(
            control_self_attended, treatment_self_attended
        )
        
        # Pool each condition to get sample-level representations
        control_pooled = self.pool_gene_features(control_cross_attended)  # [B, embed_dim]
        treatment_pooled = self.pool_gene_features(treatment_cross_attended)  # [B, embed_dim]
        
        if return_separate:
            # Project to desired output dimensions
            control_features = self.control_projector(control_pooled)
            treatment_features = self.treatment_projector(treatment_pooled)
            return control_features, treatment_features
        else:
            # Combine both representations (original behavior)
            combined_features = torch.cat([control_pooled, treatment_pooled], dim=-1)  # [B, 2*embed_dim]
            
            # Pass through encoder layers
            encoded_features = self.encoder(combined_features)
            
            # Final projection
            final_embeddings = self.final_encoder(encoded_features)
            
            return final_embeddings
            
    def get_cross_attention_weights(self, control_rna, treatment_rna, gene_names=None):
        """
        Get cross-attention weights for interpretability - shows which control genes attend to which treatment genes.
        
        Args:
            control_rna: [batch_size, num_genes] 
            treatment_rna: [batch_size, num_genes]
            gene_names: List of gene names/IDs for interpretation (optional)
            
        Returns:
            Dictionary with attention weights and interpretation utilities
        """
        with torch.no_grad():
            batch_size, num_genes = control_rna.shape
            
            # Embed genes
            control_reshaped = control_rna.unsqueeze(-1)
            treatment_reshaped = treatment_rna.unsqueeze(-1)
            
            control_gene_embeds = self.gene_embedding(control_reshaped)
            treatment_gene_embeds = self.gene_embedding(treatment_reshaped)
            
            # Apply self-attention
            control_self_attended = self.apply_self_attention(control_gene_embeds)
            treatment_self_attended = self.apply_self_attention(treatment_gene_embeds)
            
            # Get cross-attention weights
            _, _, cross_attn_weights = self.apply_cross_attention(control_self_attended, treatment_self_attended)
            
            # Process attention weights for interpretability
            result = {
                'raw_weights': cross_attn_weights,
                'gene_names': gene_names,
                'num_genes': num_genes,
                'batch_size': batch_size
            }
            
            # Aggregate attention weights across heads and layers
            aggregated_weights = self._aggregate_attention_weights(cross_attn_weights)
            result['aggregated_weights'] = aggregated_weights
            
            # Find most important gene pairs
            top_gene_pairs = self._find_top_gene_pairs(aggregated_weights, gene_names, top_k=20)
            result['top_gene_pairs'] = top_gene_pairs
            
            return result
    
    def _aggregate_attention_weights(self, cross_attn_weights):
        """Aggregate attention weights across layers and heads."""
        aggregated = {}
        
        for direction in ['control_to_treatment', 'treatment_to_control']:
            if direction in cross_attn_weights[0]:
                # Stack weights from all layers: [num_layers, batch, num_heads, num_genes, num_genes]
                stacked_weights = torch.stack([layer[direction] for layer in cross_attn_weights])
                
                # Average across layers and heads: [batch, num_genes, num_genes]
                avg_weights = stacked_weights.mean(dim=[0, 2]).cpu()
                aggregated[direction] = avg_weights
                
        return aggregated
    
    def _find_top_gene_pairs(self, aggregated_weights, gene_names=None, top_k=20):
        """Find the most important gene pairs for drug identification, separating same-gene and cross-gene relationships."""
        results = {}
        
        for direction, weights in aggregated_weights.items():
            batch_results = []
            
            for batch_idx in range(weights.shape[0]):
                sample_weights = weights[batch_idx]  # [num_genes, num_genes]
                
                # Separate diagonal (same-gene) and off-diagonal (cross-gene) weights
                diagonal_mask = torch.eye(sample_weights.shape[0], dtype=torch.bool)
                off_diagonal_mask = ~diagonal_mask
                
                # Get same-gene relationships (diagonal)
                diagonal_weights = sample_weights[diagonal_mask]
                diagonal_indices = torch.arange(sample_weights.shape[0])
                top_diagonal_indices = torch.topk(diagonal_weights, min(top_k//2, len(diagonal_weights))).indices
                
                # Get cross-gene relationships (off-diagonal) 
                off_diagonal_weights = sample_weights[off_diagonal_mask]
                off_diagonal_coords = torch.nonzero(off_diagonal_mask, as_tuple=False)
                top_off_diagonal_indices = torch.topk(off_diagonal_weights, min(top_k//2, len(off_diagonal_weights))).indices
                
                # Process same-gene pairs (corrected for drug identification)
                same_gene_pairs = []
                for idx in top_diagonal_indices:
                    gene_idx = diagonal_indices[idx].item()
                    weight_value = diagonal_weights[idx].item()
                    
                    if gene_names is not None:
                        gene_name = gene_names[gene_idx]
                        if direction == 'control_to_treatment':
                            pair_info = {
                                'control_gene': gene_name,
                                'treatment_gene': gene_name,
                                'attention_weight': weight_value,
                                'is_same_gene': True,
                                'interpretation': f"{gene_name} change pattern",
                                'biological_meaning': "This gene's change pattern is diagnostic for drug identification"
                            }
                        else:
                            pair_info = {
                                'treatment_gene': gene_name,
                                'control_gene': gene_name,
                                'attention_weight': weight_value,
                                'is_same_gene': True,
                                'interpretation': f"{gene_name} response signature",
                                'biological_meaning': "This gene's response signature helps identify the drug"
                            }
                    else:
                        if direction == 'control_to_treatment':
                            pair_info = {
                                'control_gene_idx': gene_idx,
                                'treatment_gene_idx': gene_idx,
                                'attention_weight': weight_value,
                                'is_same_gene': True,
                                'interpretation': f"Gene_{gene_idx} change pattern"
                            }
                        else:
                            pair_info = {
                                'treatment_gene_idx': gene_idx,
                                'control_gene_idx': gene_idx,
                                'attention_weight': weight_value,
                                'is_same_gene': True,
                                'interpretation': f"Gene_{gene_idx} response signature"
                            }
                    
                    same_gene_pairs.append(pair_info)
                
                # Process cross-gene pairs (corrected for drug identification)
                cross_gene_pairs = []
                for idx in top_off_diagonal_indices:
                    coord_idx = off_diagonal_coords[idx]
                    i, j = coord_idx[0].item(), coord_idx[1].item()
                    weight_value = off_diagonal_weights[idx].item()
                    
                    if gene_names is not None:
                        if direction == 'control_to_treatment':
                            pair_info = {
                                'control_gene': gene_names[i],
                                'treatment_gene': gene_names[j],
                                'attention_weight': weight_value,
                                'is_same_gene': False,
                                'interpretation': f"{gene_names[i]} baseline ↔ {gene_names[j]} response",
                                'biological_meaning': "This cross-gene pattern is a distinctive drug signature"
                            }
                        else:
                            pair_info = {
                                'treatment_gene': gene_names[i],
                                'control_gene': gene_names[j],
                                'attention_weight': weight_value,
                                'is_same_gene': False,
                                'interpretation': f"{gene_names[i]} response ↔ {gene_names[j]} baseline",
                                'biological_meaning': "This cross-gene relationship helps distinguish drugs"
                            }
                    else:
                        if direction == 'control_to_treatment':
                            pair_info = {
                                'control_gene_idx': i,
                                'treatment_gene_idx': j,
                                'attention_weight': weight_value,
                                'is_same_gene': False,
                                'interpretation': f"Gene_{i} baseline ↔ Gene_{j} response"
                            }
                        else:
                            pair_info = {
                                'treatment_gene_idx': i,
                                'control_gene_idx': j,
                                'attention_weight': weight_value,
                                'is_same_gene': False,
                                'interpretation': f"Gene_{i} response ↔ Gene_{j} baseline"
                            }
                    
                    cross_gene_pairs.append(pair_info)
                
                # Combine and sort all pairs by attention weight
                all_pairs = same_gene_pairs + cross_gene_pairs
                all_pairs.sort(key=lambda x: x['attention_weight'], reverse=True)
                
                batch_results.append({
                    'all_pairs': all_pairs[:top_k],
                    'same_gene_pairs': same_gene_pairs,
                    'cross_gene_pairs': cross_gene_pairs
                })
            
            results[direction] = batch_results
            
        return results


class ResBlock(nn.Module):
    """ResNet-style residual block for image processing"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels or stride > 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out


class ImageEncoder(nn.Module):
    """Much better replacement for your basic CNN encoder"""
    def __init__(self, img_channels=4, output_dim=256):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual layers (like ResNet)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)      # [B, 64, H/4, W/4]
        x = self.layer1(x)    # [B, 64, H/4, W/4]
        x = self.layer2(x)    # [B, 128, H/8, W/8]
        x = self.layer3(x)    # [B, 256, H/16, W/16]
        x = self.layer4(x)    # [B, 512, H/32, W/32]
        
        x = self.global_pool(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        return self.head(x)  # [B, output_dim]


def dual_rna_image_encoder_separate(control_images, treatment_images, control_rna, treatment_rna, 
                                   image_encoder, rna_encoder, device, has_rna=True, has_imaging=True):
    """
    Encode paired control and treatment data maintaining separate representations.
    Now supports single modality training by handling None encoders.
    
    This approach:
    - Learns gene-to-gene mappings via cross-attention (if RNA available)
    - Maintains separate control and treatment feature vectors
    - Handles single modality cases by using zero tensors for missing modalities
    - Avoids creating dummy encoders for efficiency
    
    Args:
        control_images: Control images tensor
        treatment_images: Treatment images tensor  
        control_rna: Control RNA tensor
        treatment_rna: Treatment RNA tensor
        image_encoder: Image encoder model (can be None for RNA-only)
        rna_encoder: RNA encoder model (can be None for image-only) 
        device: Device to run on
        has_rna: Whether RNA data is available
        has_imaging: Whether imaging data is available
    
    Output: Stack of [control_features, treatment_features] where each has been enhanced by cross-attention
    """
    
    # Encode images if available, otherwise use zero tensors
    if has_imaging and image_encoder is not None:
        control_img_features = image_encoder(control_images.to(device))
        treatment_img_features = image_encoder(treatment_images.to(device))
    else:
        # Create zero tensors with expected image feature dimensions
        batch_size = control_images.shape[0] if control_images is not None else control_rna.shape[0]
        img_feature_dim = 128  # Default image feature dimension
        control_img_features = torch.zeros(batch_size, img_feature_dim, device=device)
        treatment_img_features = torch.zeros(batch_size, img_feature_dim, device=device)
    
    # Encode RNA if available, otherwise use zero tensors  
    if has_rna and rna_encoder is not None:
        underlying_rna_encoder = rna_encoder.module if hasattr(rna_encoder, 'module') else rna_encoder
        
        if hasattr(underlying_rna_encoder, 'apply_cross_attention'):
            # Use PairedRNAEncoder to get separate cross-attended representations
            control_rna_features, treatment_rna_features = rna_encoder(
                control_rna.to(device), treatment_rna.to(device), return_separate=True
            )
        else:
            # Fallback to old RNAEncoder (no cross-attention)
            control_rna_features = rna_encoder(control_rna.to(device))
            treatment_rna_features = rna_encoder(treatment_rna.to(device))
    else:
        # Create zero tensors with expected RNA feature dimensions
        batch_size = control_rna.shape[0] if control_rna is not None else control_images.shape[0]
        rna_feature_dim = 64  # Default RNA feature dimension (or 128 for paired encoder)
        control_rna_features = torch.zeros(batch_size, rna_feature_dim, device=device)
        treatment_rna_features = torch.zeros(batch_size, rna_feature_dim, device=device)
    
    # Concatenate image and RNA features for each condition
    control_features = torch.cat([control_img_features, control_rna_features], dim=-1)
    treatment_features = torch.cat([treatment_img_features, treatment_rna_features], dim=-1)
    
    # Stack to maintain separate identities: [batch, 2, feature_dim]
    combined_features = torch.stack([control_features, treatment_features], dim=1)
    attention_mask = torch.ones(combined_features.size(0), 2, dtype=torch.bool, device=device)
    
    return combined_features, attention_mask

