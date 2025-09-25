#!/usr/bin/env python3
"""
Retrieval accuracy evaluation for drug generation model.
Tests whether the learned conditioning representations cluster same compounds together.
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from pathlib import Path
import json
from tqdm import tqdm
import time

# Import your model components
from models_sra import ReT_SRA_models
from encoders import ImageEncoder, RNAEncoder, PairedRNAEncoder
from encoders import dual_rna_image_encoder_separate as dual_rna_image_encoder
from dataloaders.dataset_gdp import create_gdp_dataloaders
from utils import load_checkpoint_and_resume_specified

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluate retrieval accuracy of learned drug representations."""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.embeddings = None
        self.compound_labels = None
        self.metadata = None
        
    def load_model_from_checkpoint(self, checkpoint_path, args):
        """Load trained encoders from checkpoint."""
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Initialize encoders (same as training script)
        has_rna = args.gene_count_matrix_path is not None
        has_imaging = args.image_json_path is not None
        
        image_encoder = None
        rna_encoder = None
        
        if has_imaging:
            image_encoder = ImageEncoder(img_channels=4, output_dim=128).to(self.device)
            image_encoder.eval()
            
        if has_rna:
            if args.paired_rna_encoder:
                # Get gene count to determine input dim
                gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)
                input_dim = gene_count_matrix.shape[0]
                rna_encoder = PairedRNAEncoder(
                    input_dim=input_dim,
                    output_dim=128,
                    dropout=0.1,
                    num_heads=4,
                    gene_embed_dim=512,
                    num_self_attention_layers=1,
                    num_cross_attention_layers=2,
                    use_bidirectional_cross_attn=True
                ).to(self.device)
            else:
                gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)
                input_dim = gene_count_matrix.shape[0]
                rna_encoder = RNAEncoder(input_dim=input_dim, output_dim=64, dropout=0.1).to(self.device)
            rna_encoder.eval()
        
        # Load checkpoint weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if image_encoder is not None and "image_encoder" in checkpoint:
                image_encoder.load_state_dict(checkpoint["image_encoder"])
                logger.info("Loaded image encoder weights")
                
            if rna_encoder is not None and "rna_encoder" in checkpoint:
                rna_encoder.load_state_dict(checkpoint["rna_encoder"])
                logger.info("Loaded RNA encoder weights")
                
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        return image_encoder, rna_encoder, has_rna, has_imaging
    
    def extract_embeddings(self, dataloader, image_encoder, rna_encoder, has_rna, has_imaging, max_samples=None):
        """Extract conditioning embeddings from dataset."""
        logger.info("Extracting embeddings...")
        
        all_embeddings = []
        all_compounds = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                if max_samples and len(all_embeddings) * dataloader.batch_size >= max_samples:
                    break
                    
                try:
                    # Get the conditioning vectors (same as training)
                    control_imgs = batch['control_images']
                    treatment_imgs = batch['treatment_images']
                    control_rna = batch['control_transcriptomics']
                    treatment_rna = batch['treatment_transcriptomics']

                    # Extract embeddings using the same encoder as training
                    y, pad_mask = dual_rna_image_encoder(
                        control_imgs, treatment_imgs, control_rna, treatment_rna,
                        image_encoder, rna_encoder, self.device, has_rna, has_imaging
                    )
                    
                    # Flatten to get 192-dim vectors: [batch, 2, feature_dim] -> [batch, 192]
                    batch_embeddings = y.view(y.shape[0], -1).cpu()  # [batch, 192]
                    
                    # Store embeddings and metadata
                    all_embeddings.append(batch_embeddings)
                    all_compounds.extend(batch['compound_name'])  # Fixed: singular not plural
                    
                    # Store additional metadata if available - extract cell lines from conditioning_info
                    cell_lines = [info['cell_line'] for info in batch['conditioning_info']]
                    batch_metadata = {
                        'compound_names': batch['compound_name'],  # Fixed: use available key
                        'cell_lines': cell_lines  # Extract from conditioning_info
                    }
                    all_metadata.append(batch_metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch {batch_idx}: {e}")
                    continue
        
        # Concatenate all embeddings
        self.embeddings = torch.cat(all_embeddings, dim=0)  # [total_samples, 192]
        self.compound_labels = all_compounds
        self.metadata = all_metadata
        
        logger.info(f"Extracted {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
        logger.info(f"Found {len(set(self.compound_labels))} unique compounds")
        
        return self.embeddings, self.compound_labels
    
    def compute_similarity_matrix(self, embeddings=None):
        """Compute cosine similarity matrix between all embeddings."""
        if embeddings is None:
            embeddings = self.embeddings
            
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        return similarity_matrix
    
    def evaluate_retrieval(self, k_values=[1, 3, 5, 10], similarity_threshold=0.8):
        """
        Evaluate retrieval accuracy.
        
        Args:
            k_values: List of k values for Precision@K and Hit@K
            similarity_threshold: Threshold for considering embeddings similar
        """
        logger.info(f"Computing similarity matrix for {len(self.embeddings)} samples...")
        similarity_matrix = self.compute_similarity_matrix()
        
        # Group samples by compound
        compound_to_indices = defaultdict(list)
        for idx, compound in enumerate(self.compound_labels):
            compound_to_indices[compound].append(idx)
        
        # Filter out compounds with only 1 sample (can't do retrieval)
        valid_compounds = {k: v for k, v in compound_to_indices.items() if len(v) > 1}
        logger.info(f"Evaluating on {len(valid_compounds)} compounds with >1 sample")
        
        # Initialize metrics
        metrics = {}
        for k in k_values:
            metrics[f'precision_at_{k}'] = []
            metrics[f'hit_at_{k}'] = []
        metrics['mrr'] = []
        metrics['similarity_scores'] = []
        
        total_queries = 0
        
        # Evaluate each sample as a query
        for compound, indices in tqdm(valid_compounds.items(), desc="Evaluating retrieval"):
            for query_idx in indices:
                # Get other samples from same compound as ground truth
                ground_truth_indices = [idx for idx in indices if idx != query_idx]
                
                # Get similarity scores for this query
                query_similarities = similarity_matrix[query_idx]
                
                # Sort by similarity (excluding self)
                query_similarities[query_idx] = -float('inf')  # Exclude self
                sorted_indices = torch.argsort(query_similarities, descending=True)
                
                # Evaluate at different k values
                for k in k_values:
                    top_k_indices = sorted_indices[:k].tolist()
                    
                    # Count how many retrieved samples have same compound
                    correct_retrievals = sum(1 for idx in top_k_indices if idx in ground_truth_indices)
                    
                    # Precision@K
                    precision = correct_retrievals / k
                    metrics[f'precision_at_{k}'].append(precision)
                    
                    # Hit@K (whether at least one correct retrieval)
                    hit = 1.0 if correct_retrievals > 0 else 0.0
                    metrics[f'hit_at_{k}'].append(hit)
                
                # Mean Reciprocal Rank
                reciprocal_rank = 0.0
                for rank, idx in enumerate(sorted_indices.tolist(), 1):
                    if idx in ground_truth_indices:
                        reciprocal_rank = 1.0 / rank
                        break
                metrics['mrr'].append(reciprocal_rank)
                
                # Store similarity scores to same compound
                same_compound_sims = [query_similarities[idx].item() for idx in ground_truth_indices]
                metrics['similarity_scores'].extend(same_compound_sims)
                
                total_queries += 1
        
        # Compute average metrics
        results = {}
        for k in k_values:
            results[f'Precision@{k}'] = np.mean(metrics[f'precision_at_{k}'])
            results[f'Hit@{k}'] = np.mean(metrics[f'hit_at_{k}'])
        
        results['MRR'] = np.mean(metrics['mrr'])
        results['Avg_Similarity_Same_Compound'] = np.mean(metrics['similarity_scores'])
        results['Total_Queries'] = total_queries
        results['Unique_Compounds'] = len(valid_compounds)
        
        return results
    
    def analyze_compound_clusters(self, top_k=5):
        """Analyze how well compounds cluster together."""
        logger.info("Analyzing compound clustering...")
        
        similarity_matrix = self.compute_similarity_matrix()
        
        compound_analysis = {}
        compound_to_indices = defaultdict(list)
        for idx, compound in enumerate(self.compound_labels):
            compound_to_indices[compound].append(idx)
        
        for compound, indices in compound_to_indices.items():
            if len(indices) < 2:
                continue
                
            # Compute intra-compound similarities
            intra_sims = []
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sim = similarity_matrix[indices[i], indices[j]].item()
                    intra_sims.append(sim)
            
            # Compute inter-compound similarities (to other compounds)
            inter_sims = []
            for idx in indices:
                for other_compound, other_indices in compound_to_indices.items():
                    if other_compound != compound:
                        for other_idx in other_indices:
                            sim = similarity_matrix[idx, other_idx].item()
                            inter_sims.append(sim)
            
            compound_analysis[compound] = {
                'sample_count': len(indices),
                'avg_intra_similarity': np.mean(intra_sims),
                'avg_inter_similarity': np.mean(inter_sims),
                'separation_score': np.mean(intra_sims) - np.mean(inter_sims)
            }
        
        return compound_analysis
    
    def save_results(self, results, compound_analysis, output_path):
        """Save evaluation results."""
        output_data = {
            'retrieval_metrics': results,
            'compound_analysis': compound_analysis,
            'evaluation_info': {
                'total_samples': len(self.embeddings),
                'embedding_dim': self.embeddings.shape[1],
                'unique_compounds': len(set(self.compound_labels)),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy of drug generation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--gene-count-matrix-path", type=str, required=True)
    parser.add_argument("--image-json-path", type=str, default=None)
    parser.add_argument("--drug-data-path", type=str, required=True)
    parser.add_argument("--raw-drug-csv-path", type=str, required=True)
    parser.add_argument("--paired-rna-encoder", action="store_true", help="Use paired RNA encoder")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate (for quick testing)")
    parser.add_argument("--output-dir", type=str, default="retrieval_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--debug-mode", action="store_true", help="Enable debug mode for smaller dataset")
    parser.add_argument("--debug-samples", type=int, default=1000, help="Number of samples in debug mode")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(device=args.device)
    
    # Load model
    image_encoder, rna_encoder, has_rna, has_imaging = evaluator.load_model_from_checkpoint(args.checkpoint, args)
    
    # Create dataloader (using same function as training)
    logger.info("Creating dataloader...")
    
    # Load gene count matrix
    gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)
    
    # Create dataloader
    from dataloaders.dataset_gdp import create_gdp_dataloaders
    train_loader, test_loader = create_gdp_dataloaders(
        metadata_control=None,
        metadata_drug=None,
        gene_count_matrix=gene_count_matrix,
        image_json_path=args.image_json_path,
        drug_data_path=args.drug_data_path,
        raw_drug_csv_path=args.raw_drug_csv_path,
        use_highly_variable_genes=True,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent evaluation
        num_workers=4,
        compound_name_label='compound',
        debug_mode=args.debug_mode,
        debug_samples=args.debug_samples,
        random_state=42,
        split_train_test=True,
        test_size=0.2,
    )
    
    # Extract embeddings from training set (where model learned representations)
    embeddings, compound_labels = evaluator.extract_embeddings(
        train_loader, image_encoder, rna_encoder, has_rna, has_imaging, 
        max_samples=args.max_samples
    )
    
    # Evaluate retrieval
    logger.info("Evaluating retrieval accuracy...")
    results = evaluator.evaluate_retrieval(k_values=[1, 3, 5, 10])
    
    # Print results
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:30s}: {value:.4f}")
        else:
            print(f"{metric:30s}: {value}")
    
    # Analyze compound clusters
    compound_analysis = evaluator.analyze_compound_clusters()
    
    # Print top and bottom performing compounds
    sorted_compounds = sorted(compound_analysis.items(), 
                            key=lambda x: x[1]['separation_score'], reverse=True)
    
    print(f"\nTOP 10 BEST CLUSTERED COMPOUNDS:")
    print("-" * 60)
    for compound, analysis in sorted_compounds[:10]:
        print(f"{compound:20s}: sep_score={analysis['separation_score']:.3f}, "
              f"intra_sim={analysis['avg_intra_similarity']:.3f}, "
              f"samples={analysis['sample_count']}")
    
    print(f"\nTOP 10 WORST CLUSTERED COMPOUNDS:")
    print("-" * 60)
    for compound, analysis in sorted_compounds[-10:]:
        print(f"{compound:20s}: sep_score={analysis['separation_score']:.3f}, "
              f"intra_sim={analysis['avg_intra_similarity']:.3f}, "
              f"samples={analysis['sample_count']}")
    
    # Save results
    output_file = os.path.join(args.output_dir, "retrieval_evaluation.json")
    evaluator.save_results(results, compound_analysis, output_file)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()