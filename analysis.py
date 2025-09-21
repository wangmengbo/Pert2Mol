from absl import logging
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from rdkit import RDLogger
import re
import multiprocessing as mp
from functools import partial
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


def analyze_gene_importance_for_drug_identification(model, rna_encoder, batch, gene_names=None, device='cuda'):
    """
    Analyze which gene pairs are most important for drug IDENTIFICATION during inference.
    
    Args:
        model: Your trained drug generation model
        rna_encoder: PairedRNAEncoder instance
        batch: Data batch containing control/treatment RNA and target drug
        gene_names: List of gene names for interpretation
        device: Device to run on
        
    Returns:
        Dictionary with gene importance analysis for drug identification
    """
    model.eval()
    rna_encoder.eval()
    
    with torch.no_grad():
        control_rna = batch['control_transcriptomics'].to(device)
        treatment_rna = batch['treatment_transcriptomics'].to(device)
        target_smiles = batch['target_smiles']
        
        # Get attention weights and analysis
        attention_analysis = rna_encoder.get_cross_attention_weights(
            control_rna, treatment_rna, gene_names=gene_names
        )
        
        results = {
            'attention_analysis': attention_analysis,
            'target_smiles': target_smiles,
            'control_rna': control_rna.cpu(),
            'treatment_rna': treatment_rna.cpu(),
            'gene_names': gene_names
        }
        
        return results


def print_gene_importance_report_for_drug_identification(results, sample_idx=0):
    """
    Print a human-readable report for drug identification task.
    """
    attention_analysis = results['attention_analysis']
    target_smiles = results['target_smiles']
    
    print(f"\n=== Drug Identification Gene Analysis ===")
    print(f"Target SMILES: {target_smiles[sample_idx] if isinstance(target_smiles, list) else target_smiles}")
    
    # Analyze control → treatment direction
    if 'control_to_treatment' in attention_analysis['top_gene_pairs']:
        sample_data = attention_analysis['top_gene_pairs']['control_to_treatment'][sample_idx]
        
        print(f"\n--- Most Diagnostic Gene Signatures for Drug Identification ---")
        
        # Same-gene relationships (individual gene change patterns)
        same_gene_pairs = sample_data['same_gene_pairs']
        if same_gene_pairs:
            print(f"\n🧬 Individual Gene Change Signatures:")
            print("   (Gene changes that are characteristic of this specific drug)")
            for i, pair in enumerate(same_gene_pairs[:5], 1):
                print(f"   {i:2d}. {pair['interpretation']:<40} (diagnostic value: {pair['attention_weight']:.4f})")
                if 'biological_meaning' in pair:
                    print(f"       → {pair['biological_meaning']}")
        
        # Cross-gene relationships (multi-gene interaction patterns)  
        cross_gene_pairs = sample_data['cross_gene_pairs']
        if cross_gene_pairs:
            print(f"\n🔗 Multi-Gene Interaction Signatures:")
            print("   (Cross-gene patterns that distinguish this drug from others)")
            for i, pair in enumerate(cross_gene_pairs[:5], 1):
                print(f"   {i:2d}. {pair['interpretation']:<50} (diagnostic value: {pair['attention_weight']:.4f})")
                if 'biological_meaning' in pair:
                    print(f"       → {pair['biological_meaning']}")


def analyze_drug_diagnostic_features(control_rna, treatment_rna, attention_weights, gene_names=None, sample_idx=0):
    """
    Analyze which gene change patterns are most diagnostic for drug identification.
    """
    # Calculate gene expression changes
    gene_changes = treatment_rna[sample_idx] - control_rna[sample_idx]
    
    # Get top attention weights for same genes
    sample_data = attention_weights['top_gene_pairs']['control_to_treatment'][sample_idx]
    same_gene_pairs = sample_data['same_gene_pairs']
    
    print(f"\n=== Drug-Diagnostic Gene Changes ===")
    print("Genes whose change patterns are most diagnostic for identifying this drug:\n")
    
    # Sort genes by diagnostic strength
    diagnostic_genes = []
    for pair in same_gene_pairs[:15]:
        if gene_names:
            gene_name = pair['control_gene']
            if gene_name in gene_names:
                gene_idx = gene_names.index(gene_name)
            else:
                continue
        else:
            gene_idx = pair['control_gene_idx']
            gene_name = f"Gene_{gene_idx}"
        
        baseline_expr = control_rna[sample_idx, gene_idx].item()
        response_expr = treatment_rna[sample_idx, gene_idx].item()
        expression_change = gene_changes[gene_idx].item()
        diagnostic_strength = pair['attention_weight']
        
        diagnostic_genes.append({
            'gene_name': gene_name,
            'baseline': baseline_expr,
            'response': response_expr,
            'change': expression_change,
            'diagnostic_strength': diagnostic_strength
        })
    
    # Display results sorted by diagnostic strength
    for i, gene_info in enumerate(diagnostic_genes, 1):
        # Calculate fold change for better interpretation
        if gene_info['baseline'] != 0:
            fold_change = gene_info['response'] / gene_info['baseline']
            fold_change_str = f"{fold_change:.2f}x"
        else:
            fold_change_str = "N/A"
        
        # Categorize diagnostic strength
        strength = gene_info['diagnostic_strength']
        if strength > 0.7:
            diagnostic_level = "🔴 Highly diagnostic"
        elif strength > 0.5:
            diagnostic_level = "🟡 Moderately diagnostic" 
        else:
            diagnostic_level = "🟢 Weakly diagnostic"
        
        # Categorize change magnitude
        change_mag = abs(gene_info['change'])
        if change_mag > 2.0:
            change_category = "Large change"
        elif change_mag > 0.5:
            change_category = "Moderate change"
        else:
            change_category = "Small change"
        
        print(f"{i:2d}. {gene_info['gene_name']:15s} | {diagnostic_level:20s} | {change_category:14s}")
        print(f"    Diagnostic strength: {strength:.3f} | Change: {gene_info['change']:+7.3f} ({fold_change_str})")
        print(f"    Expression: {gene_info['baseline']:6.2f} → {gene_info['response']:6.2f}")
        print()
    
    print(f"Interpretation:")
    print(f"• High diagnostic strength = This gene's change pattern is characteristic of the target drug")
    print(f"• The model uses these specific gene change signatures to distinguish drugs")
    print(f"• Large changes with high diagnostic strength = Key drug mechanism markers")


def interpret_attention_patterns_for_drug_identification(attention_analysis, target_drug=None, sample_idx=0):
    """
    Provide interpretation of attention patterns for drug identification task.
    """
    sample_data = attention_analysis['top_gene_pairs']['control_to_treatment'][sample_idx]
    
    print(f"\n=== Drug Identification Interpretation ===")
    if target_drug:
        print(f"Target Drug: {target_drug}")
    
    # Analyze same-gene vs cross-gene attention distribution
    same_gene_weights = [p['attention_weight'] for p in sample_data['same_gene_pairs']]
    cross_gene_weights = [p['attention_weight'] for p in sample_data['cross_gene_pairs']]
    
    if same_gene_weights and cross_gene_weights:
        avg_same_gene = sum(same_gene_weights) / len(same_gene_weights)
        avg_cross_gene = sum(cross_gene_weights) / len(cross_gene_weights)
        
        print(f"Average same-gene attention: {avg_same_gene:.3f}")
        print(f"Average cross-gene attention: {avg_cross_gene:.3f}")
        
        if avg_same_gene > avg_cross_gene * 1.5:
            print("🎯 Drug Signature: Individual gene change patterns are most diagnostic")
            print("   This drug is identified by characteristic changes in specific genes")
            print("   Each gene's baseline→response pattern is a distinctive fingerprint")
        elif avg_cross_gene > avg_same_gene * 1.2:
            print("🕸️ Drug Signature: Complex cross-gene relationships are most diagnostic") 
            print("   This drug is identified by relationships between different genes")
            print("   The drug's signature involves coordinated multi-gene patterns")
        else:
            print("⚖️ Drug Signature: Both direct changes and gene relationships matter")
            print("   This drug is identified by both individual gene changes and gene interactions")
    
    # Analyze diagnostic gene signatures
    top_same_genes = sample_data['same_gene_pairs'][:3]
    top_cross_pairs = sample_data['cross_gene_pairs'][:3]
    
    if top_same_genes:
        print(f"\nMost diagnostic individual gene changes:")
        for pair in top_same_genes:
            print(f"  • {pair['interpretation']} (diagnostic strength: {pair['attention_weight']:.3f})")
            print(f"    → This gene's change pattern is characteristic of the target drug")
    
    if top_cross_pairs:
        print(f"\nMost diagnostic cross-gene signatures:")
        for pair in top_cross_pairs:
            print(f"  • {pair['interpretation']} (diagnostic strength: {pair['attention_weight']:.3f})")
            print(f"    → This gene relationship pattern distinguishes the target drug")


def compare_drug_signatures(results_list, gene_names=None, top_k=5):
    """
    Compare drug signatures across multiple samples to identify common patterns.
    
    Args:
        results_list: List of results from analyze_gene_importance_for_drug_identification
        gene_names: List of gene names
        top_k: Number of top gene pairs to analyze
    """
    print(f"\n=== Drug Signature Comparison ===")
    
    # Collect all gene pairs across samples
    all_same_gene_pairs = {}
    all_cross_gene_pairs = {}
    
    for i, results in enumerate(results_list):
        target_smiles = results['target_smiles']
        if isinstance(target_smiles, list):
            target_smiles = target_smiles[0]
        
        print(f"\nSample {i+1}: {target_smiles}")
        
        attention_analysis = results['attention_analysis']
        sample_data = attention_analysis['top_gene_pairs']['control_to_treatment'][0]
        
        # Collect same-gene pairs
        for pair in sample_data['same_gene_pairs'][:top_k]:
            gene_key = pair['interpretation']
            if gene_key not in all_same_gene_pairs:
                all_same_gene_pairs[gene_key] = []
            all_same_gene_pairs[gene_key].append({
                'sample': i,
                'smiles': target_smiles,
                'weight': pair['attention_weight']
            })
        
        # Collect cross-gene pairs
        for pair in sample_data['cross_gene_pairs'][:top_k]:
            gene_key = pair['interpretation']
            if gene_key not in all_cross_gene_pairs:
                all_cross_gene_pairs[gene_key] = []
            all_cross_gene_pairs[gene_key].append({
                'sample': i,
                'smiles': target_smiles,
                'weight': pair['attention_weight']
            })
    
    # Find common patterns
    print(f"\n--- Common Same-Gene Patterns ---")
    common_same_gene = {k: v for k, v in all_same_gene_pairs.items() if len(v) > 1}
    if common_same_gene:
        for gene_pattern, samples in sorted(common_same_gene.items(), 
                                           key=lambda x: len(x[1]), reverse=True)[:5]:
            avg_weight = sum(s['weight'] for s in samples) / len(samples)
            print(f"{gene_pattern}: appears in {len(samples)} samples (avg strength: {avg_weight:.3f})")
            for sample in samples:
                print(f"  Sample {sample['sample']}: {sample['smiles'][:20]}... (strength: {sample['weight']:.3f})")
    else:
        print("No common same-gene patterns found across samples")
    
    print(f"\n--- Common Cross-Gene Patterns ---")
    common_cross_gene = {k: v for k, v in all_cross_gene_pairs.items() if len(v) > 1}
    if common_cross_gene:
        for gene_pattern, samples in sorted(common_cross_gene.items(), 
                                           key=lambda x: len(x[1]), reverse=True)[:5]:
            avg_weight = sum(s['weight'] for s in samples) / len(samples)
            print(f"{gene_pattern}: appears in {len(samples)} samples (avg strength: {avg_weight:.3f})")
            for sample in samples:
                print(f"  Sample {sample['sample']}: {sample['smiles'][:20]}... (strength: {sample['weight']:.3f})")
    else:
        print("No common cross-gene patterns found across samples")


def run_complete_drug_identification_analysis(results, sample_idx=0):
    """
    Run the complete analysis workflow for drug identification task.
    """
    # 1. Print main drug identification report
    print_gene_importance_report_for_drug_identification(results, sample_idx)
    
    # 2. Analyze diagnostic gene changes
    analyze_drug_diagnostic_features(
        results['control_rna'], 
        results['treatment_rna'],
        results['attention_analysis'],
        results['gene_names'],
        sample_idx
    )
    
    # 3. Provide drug identification interpretation
    interpret_attention_patterns_for_drug_identification(
        results['attention_analysis'], 
        target_drug=results['target_smiles'][sample_idx] if isinstance(results['target_smiles'], list) else results['target_smiles'],
        sample_idx=sample_idx
    )


def run_batch_drug_identification_analysis(model, rna_encoder, data_loader, gene_names=None, max_batches=5):
    """
    Run drug identification analysis on multiple batches and compare patterns.
    """
    all_results = []
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
            
        print(f"\n{'='*60}")
        print(f"Analyzing batch {batch_idx + 1}/{max_batches}")
        print(f"{'='*60}")
        
        # Get gene importance analysis for this batch
        results = analyze_gene_importance_for_drug_identification(
            model, rna_encoder, batch, gene_names=gene_names
        )
        
        # Analyze first sample in batch
        run_complete_drug_identification_analysis(results, sample_idx=0)
        
        all_results.append(results)
    
    # Compare signatures across all samples
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"Cross-Sample Drug Signature Analysis")
        print(f"{'='*60}")
        compare_drug_signatures(all_results, gene_names=gene_names)
    
    return all_results


# Utility functions for loading gene names (same as before)
def load_gene_names(gene_count_matrix_path):
    """
    Load gene names from your data. Adapt this to your data format.
    """
    import pandas as pd
    
    # Assuming your gene count matrix has gene names as column names or index
    gene_df = pd.read_parquet(gene_count_matrix_path)
    
    # If genes are columns:
    if hasattr(gene_df, 'columns'):
        gene_names = list(gene_df.columns)
    # If genes are in index:
    elif hasattr(gene_df, 'index'):
        gene_names = list(gene_df.index)
    else:
        gene_names = [f"Gene_{i}" for i in range(len(gene_df))]
    
    return gene_names


# Example usage:
"""
# Load gene names
gene_names = load_gene_names(args.gene_count_matrix_path)

# Run analysis on test data
results_list = run_batch_drug_identification_analysis(
    model, rna_encoder, test_loader, gene_names=gene_names, max_batches=3
)

# Or analyze a single batch
results = analyze_gene_importance_for_drug_identification(
    model, rna_encoder, batch, gene_names=gene_names
)
run_complete_drug_identification_analysis(results, sample_idx=0)
"""