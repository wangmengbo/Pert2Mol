import os
import json
import pickle
import pandas as pd
import numpy as np
import argparse
import time
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import hashlib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import (Descriptors, rdMolDescriptors, AllChem, Scaffolds, MACCSkeys,
    Descriptors, rdMolDescriptors, Crippen, Lipinski)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer
from rdkit import DataStructs
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
from typing import List, Dict, Tuple

from utils import AE_SMILES_encoder
from encoders import dual_rna_image_encoder_separate as dual_rna_image_encoder
from evaluation_metrics import calculate_mode_specific_metrics

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_three_section_summary(comprehensive_data, args):
    """Create comprehensive summary report with ALL calculated metrics and std deviations"""
    
    metadata = comprehensive_data['metadata']
    results_by_mode = comprehensive_data['results_by_mode']
    mode_metrics = comprehensive_data['mode_metrics']
    
    output_txt = os.path.join(args.output_dir, f"{args.output_prefix}_three_mode_summary.txt")
    
    def format_metric_with_std(metrics_dict, mean_key, std_key=None):
        """Helper function to format meanÂ±std"""
        if std_key is None:
            std_key = mean_key.replace('_mean', '_std')
        
        mean_val = metrics_dict.get(mean_key, 0)
        std_val = metrics_dict.get(std_key, None)
        
        if std_val is not None:
            return f"{mean_val:.3f} Â± {std_val:.3f}"
        else:
            return f"{mean_val:.3f}"
    
    with open(output_txt, 'w') as f:
        f.write("COMPREHENSIVE THREE-MODE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Global seed: {args.global_seed}\n")
        f.write(f"Evaluation time: {metadata['evaluation_time']:.1f}s\n")
        f.write(f"Retrieval Top-K: {args.retrieval_top_k}\n")
        f.write(f"Generation steps: {args.generation_steps}\n\n")
        
        modes_enabled = metadata['modes_enabled']
        f.write("MODES ENABLED:\n")
        f.write(f"  Pure Generation: {modes_enabled['pure_generation']}\n")
        f.write(f"  Conventional Retrieval: {modes_enabled['conventional_retrieval']}\n")
        f.write(f"  Retrieval by Generation: {modes_enabled['retrieval_by_generation']}\n\n")
        
        # SECTION 1: PURE GENERATION MODE - COMPREHENSIVE
        f.write("=" * 80 + "\n")
        f.write("SECTION 1: PURE GENERATION MODE - COMPREHENSIVE METRICS\n")
        f.write("=" * 80 + "\n")
        f.write("Methodology: Generate novel molecules from biological conditions\n")
        f.write("Purpose: Evaluate model's ability to create new drug candidates\n\n")
        
        if results_by_mode.get('pure_generation') and modes_enabled['pure_generation']:
            gen_results = results_by_mode['pure_generation']
            gen_metrics = mode_metrics.get('pure_generation', {})
            
            f.write(f"Total samples: {len(gen_results)}\n")
            valid_molecules = sum(1 for r in gen_results if r.get('is_valid'))
            f.write(f"Valid molecules: {valid_molecules}\n")
            f.write(f"Validity rate: {valid_molecules/len(gen_results)*100:.1f}%\n\n")
            
            # COMPREHENSIVE METRICS (NEW)
            if 'comprehensive_metrics' in gen_metrics:
                comp_metrics = gen_metrics['comprehensive_metrics']
                f.write("COMPREHENSIVE MOLECULAR GENERATION METRICS:\n")
                f.write(f"  Validity: {comp_metrics.get('validity', 0):.3f}\n")
                f.write(f"  Uniqueness: {comp_metrics.get('uniqueness', 0):.3f}\n")
                f.write(f"  Novelty: {comp_metrics.get('novelty', 0):.3f}\n")
                
                if comp_metrics.get('fcd') is not None:
                    f.write(f"  FrÃ©chet ChemNet Distance (FCD): {comp_metrics['fcd']:.3f}\n")
                
                # SNN metrics with std
                if 'snn_mean' in comp_metrics:
                    f.write(f"  Nearest Neighbor Similarity: {format_metric_with_std(comp_metrics, 'snn_mean', 'snn_std')}\n")
                
                # Fragment similarity
                if 'fragment_similarity' in comp_metrics:
                    frag_sim = comp_metrics['fragment_similarity']
                    f.write(f"  Fragment Similarity: {frag_sim.get('fragment_similarity', 0):.3f}\n")
                    f.write(f"  Fragment Recovery: {frag_sim.get('fragment_recovery', 0):.3f}\n")
                
                f.write("\n")
            
            # DRUG-LIKENESS METRICS (ENHANCED)
            if 'comprehensive_metrics' in gen_metrics and 'drug_likeness' in gen_metrics['comprehensive_metrics']:
                drug_metrics = gen_metrics['comprehensive_metrics']['drug_likeness']
                f.write("DRUG-LIKENESS METRICS:\n")
                f.write(f"  QED Score: {format_metric_with_std(drug_metrics, 'qed_mean', 'qed_std')}\n")
                f.write(f"  Molecular Weight: {format_metric_with_std(drug_metrics, 'mw_mean', 'mw_std')}\n")
                f.write(f"  LogP: {format_metric_with_std(drug_metrics, 'logp_mean', 'logp_std')}\n")
                f.write(f"  TPSA: {format_metric_with_std(drug_metrics, 'tpsa_mean', 'tpsa_std')}\n")
                f.write(f"  Lipinski Violations: {drug_metrics.get('lipinski_violations_mean', 0):.2f}\n")
                f.write(f"  Lipinski Compliant Fraction: {drug_metrics.get('lipinski_compliant_fraction', 0):.3f}\n")
                f.write(f"  Drug-like Fraction: {drug_metrics.get('drug_like_fraction', 0):.3f}\n\n")
            
            # SCAFFOLD METRICS (ENHANCED)
            if 'comprehensive_metrics' in gen_metrics and 'scaffold_metrics' in gen_metrics['comprehensive_metrics']:
                scaffold_metrics = gen_metrics['comprehensive_metrics']['scaffold_metrics']
                f.write("SCAFFOLD ANALYSIS:\n")
                f.write(f"  Unique Scaffolds: {scaffold_metrics.get('unique_scaffolds', 0)}\n")
                f.write(f"  Scaffold Diversity: {scaffold_metrics.get('scaffold_diversity', 0):.3f}\n")
                f.write(f"  Scaffold Novelty: {scaffold_metrics.get('scaffold_novelty', 0):.3f}\n")
                f.write(f"  Scaffold Recovery: {scaffold_metrics.get('scaffold_recovery', 0):.3f}\n\n")
            
            # DIVERSITY METRICS (ENHANCED)
            if 'diversity_metrics' in gen_metrics:
                diversity = gen_metrics['diversity_metrics']
                f.write("DIVERSITY METRICS:\n")
                f.write(f"  Internal Diversity (Fingerprint): {diversity.get('internal_diversity', 0):.3f}\n")
                f.write(f"  Scaffold Diversity: {diversity.get('scaffold_diversity', 0):.3f}\n")
                f.write(f"  Functional Group Diversity: {diversity.get('functional_group_diversity', 0):.3f}\n")
                f.write(f"  Property-based Diversity: {diversity.get('property_diversity', 0):.3f}\n")
                f.write(f"  Atom Type Diversity: {diversity.get('atom_type_diversity', 0):.3f}\n")
                f.write(f"  Unique Scaffolds: {diversity.get('num_unique_scaffolds', 0)}\n")
                f.write(f"  Unique Molecules: {diversity.get('num_unique_molecules', 0)}\n\n")
            
            # ENHANCED DIVERSITY METRICS (from comprehensive_metrics)
            if 'comprehensive_metrics' in gen_metrics and 'diversity' in gen_metrics['comprehensive_metrics']:
                comp_diversity = gen_metrics['comprehensive_metrics']['diversity']
                f.write("ENHANCED DIVERSITY METRICS:\n")
                f.write(f"  Molecular Weight Diversity (std): {comp_diversity.get('molecular_weight_diversity', 0):.3f}\n")
                f.write(f"  LogP Diversity (std): {comp_diversity.get('logp_diversity', 0):.3f}\n")
                f.write(f"  Ring System Diversity (std): {comp_diversity.get('ring_diversity', 0):.3f}\n\n")
            
            # DISTRIBUTION METRICS (NEW)
            if 'comprehensive_metrics' in gen_metrics and 'distribution_metrics' in gen_metrics['comprehensive_metrics']:
                dist_metrics = gen_metrics['comprehensive_metrics']['distribution_metrics']
                f.write("DISTRIBUTION MATCHING (KL Divergence - lower is better):\n")
                f.write(f"  Molecular Weight KL: {dist_metrics.get('mw_kl_div', float('inf')):.3f}\n")
                f.write(f"  LogP KL Divergence: {dist_metrics.get('logp_kl_div', float('inf')):.3f}\n")
                f.write(f"  TPSA KL Divergence: {dist_metrics.get('tpsa_kl_div', float('inf')):.3f}\n\n")
            
            # COVERAGE METRICS (NEW)
            if 'comprehensive_metrics' in gen_metrics and 'coverage' in gen_metrics['comprehensive_metrics']:
                coverage = gen_metrics['comprehensive_metrics']['coverage']
                f.write("CHEMICAL SPACE COVERAGE:\n")
                f.write(f"  Coverage (Tanimoto â‰¥0.7): {coverage.get('coverage', 0):.3f}\n\n")
            
            # GENERATION QUALITY (EXISTING)
            if 'generation_quality' in gen_metrics:
                quality = gen_metrics['generation_quality']
                f.write("DETAILED GENERATION QUALITY:\n")
                
                # Synthetic accessibility with std
                if 'synthetic_accessibility' in quality:
                    sa = quality['synthetic_accessibility']
                    f.write(f"  Synthetic Accessibility: {format_metric_with_std(sa, 'mean', 'std')}\n")
                    f.write(f"  Easy Synthesis Rate (SAâ‰¤3): {sa.get('easy_synthesis_rate', 0):.3f}\n")
                
                # Molecular complexity with std
                if 'molecular_complexity' in quality:
                    complexity = quality['molecular_complexity']
                    f.write(f"  Molecular Complexity (BertzCT): {format_metric_with_std(complexity, 'mean', 'std')}\n")
                
                f.write(f"  PAINS Alerts Rate: {quality.get('pains_alerts', 0):.3f}\n")
                
                # Functional groups
                if 'functional_groups' in quality:
                    f.write("  Functional Group Frequencies:\n")
                    for fg, freq in quality['functional_groups'].items():
                        f.write(f"    {fg}: {freq:.3f}\n")
                
                f.write("\n")
            
            # TARGET SIMILARITY (with std)
            if 'target_similarity' in gen_metrics:
                similarity = gen_metrics['target_similarity']
                f.write("TARGET SIMILARITY ANALYSIS:\n")
                f.write(f"  Mean Tanimoto Similarity: {format_metric_with_std(similarity, 'mean', 'std')}\n")
                f.write(f"  High Similarity Rate (â‰¥0.7): {similarity.get('high_similarity_rate', 0):.3f}\n\n")
        else:
            f.write("Pure generation mode was not enabled or no results available.\n\n")
        
        # SECTION 2: CONVENTIONAL RETRIEVAL MODE - ENHANCED
        f.write("=" * 80 + "\n")
        f.write("SECTION 2: CONVENTIONAL RETRIEVAL MODE - DETAILED METRICS\n")
        f.write("=" * 80 + "\n")
        f.write("Methodology: Find similar biological conditions in training data\n")
        f.write("Purpose: Baseline drug repurposing via biological similarity\n\n")
        
        if results_by_mode.get('conventional_retrieval') and modes_enabled['conventional_retrieval']:
            ret_results = results_by_mode['conventional_retrieval']
            ret_metrics = mode_metrics.get('conventional_retrieval', {})
            
            f.write(f"Total samples: {len(ret_results)}\n")
            f.write(f"Retrieval database size: {len(comprehensive_data['training_data'].get('smiles', []))}\n\n")
            
            if 'retrieval_accuracy' in ret_metrics:
                accuracy = ret_metrics['retrieval_accuracy']
                f.write("RETRIEVAL PERFORMANCE SUMMARY:\n")
                f.write(f"  SMILES Top-{args.retrieval_top_k} Accuracy: {accuracy.get('smiles_top_k_accuracy', 0):.3f}\n")
                f.write(f"  Compound Top-{args.retrieval_top_k} Accuracy: {accuracy.get('compound_top_k_accuracy', 0):.3f}\n")
                f.write(f"  Mean Reciprocal Rank: {accuracy.get('mean_reciprocal_rank', 0):.3f}\n\n")
            
            if 'progressive_metrics' in ret_metrics:
                prog = ret_metrics['progressive_metrics']
                f.write("PROGRESSIVE RETRIEVAL ACCURACY:\n")
                for k in range(1, min(args.retrieval_top_k + 1, 11)):  # Show up to top-10 or retrieval_top_k
                    if f'top_{k}_accuracy' in prog:
                        f.write(f"  Top-{k} accuracy: {prog[f'top_{k}_accuracy']:.3f}\n")
                f.write("\n")
            
            # Add retrieval confidence analysis if available
            confidences = [r.get('confidence', 0) for r in ret_results if r.get('confidence') is not None]
            if confidences:
                f.write("RETRIEVAL CONFIDENCE ANALYSIS:\n")
                f.write(f"  Mean Confidence: {np.mean(confidences):.3f} Â± {np.std(confidences):.3f}\n")
                f.write(f"  High Confidence Rate (>0.8): {sum(1 for c in confidences if c > 0.8) / len(confidences):.3f}\n\n")
        else:
            f.write("Conventional retrieval mode was not enabled or no results available.\n\n")
        
        # SECTION 3: RETRIEVAL BY GENERATION MODE - ENHANCED
        f.write("=" * 80 + "\n")
        f.write("SECTION 3: RETRIEVAL BY GENERATION MODE - DETAILED METRICS\n")
        f.write("=" * 80 + "\n")
        f.write("Methodology: Generate drug embeddings from biological data, find similar drugs\n")
        f.write("Purpose: Leverage model's biologicalâ†’drug mapping for functional similarity\n\n")
        
        if results_by_mode.get('retrieval_by_generation') and modes_enabled['retrieval_by_generation']:
            rbg_results = results_by_mode['retrieval_by_generation']
            rbg_metrics = mode_metrics.get('retrieval_by_generation', {})
            
            f.write(f"Total samples: {len(rbg_results)}\n")
            f.write(f"Drug database size: {len(comprehensive_data['training_data'].get('smiles', []))}\n\n")
            
            if 'retrieval_accuracy' in rbg_metrics:
                accuracy = rbg_metrics['retrieval_accuracy']
                f.write("RETRIEVAL BY GENERATION PERFORMANCE:\n")
                f.write(f"  SMILES Top-{args.retrieval_by_generation_top_k} Accuracy: {accuracy.get('smiles_top_k_accuracy', 0):.3f}\n")
                f.write(f"  Compound Top-{args.retrieval_by_generation_top_k} Accuracy: {accuracy.get('compound_top_k_accuracy', 0):.3f}\n")
                f.write(f"  Mean Reciprocal Rank: {accuracy.get('mean_reciprocal_rank', 0):.3f}\n\n")
            
            if 'progressive_metrics' in rbg_metrics:
                prog = rbg_metrics['progressive_metrics']
                f.write("PROGRESSIVE RETRIEVAL BY GENERATION ACCURACY:\n")
                for k in range(1, min(args.retrieval_by_generation_top_k + 1, 11)):
                    if f'top_{k}_accuracy' in prog:
                        f.write(f"  Top-{k} accuracy: {prog[f'top_{k}_accuracy']:.3f}\n")
                f.write("\n")
            
            # Add embedding similarity analysis if available
            confidences = [r.get('confidence', 0) for r in rbg_results if r.get('confidence') is not None]
            if confidences:
                f.write("EMBEDDING SIMILARITY ANALYSIS:\n")
                f.write(f"  Mean Embedding Similarity: {np.mean(confidences):.3f} Â± {np.std(confidences):.3f}\n")
                f.write(f"  High Similarity Rate (>0.8): {sum(1 for c in confidences if c > 0.8) / len(confidences):.3f}\n\n")
            
            # Report saved embedding files
            embedding_files = metadata.get('embedding_files', {})
            f.write("SAVED INTERMEDIATE RESULTS:\n")
            for file_type, filename in embedding_files.items():
                if filename:
                    f.write(f"  {file_type}: {filename}\n")
            f.write("\n")
        else:
            f.write("Retrieval by generation mode was not enabled or no results available.\n\n")
        
        # SECTION 4: GROUND TRUTH BASELINE ANALYSIS
        f.write("=" * 80 + "\n")
        f.write("SECTION 4: GROUND TRUTH BASELINE CHARACTERISTICS\n")
        f.write("=" * 80 + "\n")
        f.write("Methodology: Analyze intrinsic properties of target molecules\n")
        f.write("Purpose: Establish baseline characteristics for comparison\n\n")
        
        if 'ground_truth_baseline' in comprehensive_data:
            gt_baseline = comprehensive_data['ground_truth_baseline']
            gt_data = comprehensive_data.get('ground_truth_data', {})
            
            f.write(f"Total ground truth molecules: {len(gt_data.get('smiles', []))}\n")
            f.write(f"Valid ground truth molecules: {int(gt_baseline.get('validity', 0) * len(gt_data.get('smiles', [])))}\n")
            f.write(f"Validity rate: {gt_baseline.get('validity', 0)*100:.1f}%\n")
            f.write(f"Uniqueness: {gt_baseline.get('uniqueness', 0):.3f}\n\n")
            
            # Drug-likeness baseline
            f.write("GROUND TRUTH DRUG-LIKENESS CHARACTERISTICS:\n")
            f.write(f"  QED Score: {format_metric_with_std(gt_baseline, 'qed_mean', 'qed_std')}\n")
            f.write(f"  QED Median: {gt_baseline.get('qed_median', 0):.3f}\n")
            f.write(f"  Drug-like Fraction (QEDâ‰¥0.5): {gt_baseline.get('drug_like_fraction', 0):.3f}\n")
            f.write(f"  Molecular Weight: {format_metric_with_std(gt_baseline, 'mw_mean', 'mw_std')}\n")
            f.write(f"  LogP: {format_metric_with_std(gt_baseline, 'logp_mean', 'logp_std')}\n")
            f.write(f"  TPSA: {format_metric_with_std(gt_baseline, 'tpsa_mean', 'tpsa_std')}\n")
            f.write(f"  Lipinski Violations: {gt_baseline.get('lipinski_violations_mean', 0):.2f}\n")
            f.write(f"  Lipinski Compliant Fraction: {gt_baseline.get('lipinski_compliant_fraction', 0):.3f}\n\n")
            
            # Structural characteristics
            f.write("GROUND TRUTH STRUCTURAL CHARACTERISTICS:\n")
            f.write(f"  Heavy Atoms: {format_metric_with_std(gt_baseline, 'heavy_atoms_mean', 'heavy_atoms_std')}\n")
            f.write(f"  Aromatic Rings: {format_metric_with_std(gt_baseline, 'aromatic_rings_mean', 'aromatic_rings_std')}\n")
            f.write(f"  Rotatable Bonds: {format_metric_with_std(gt_baseline, 'rotatable_bonds_mean', 'rotatable_bonds_std')}\n")
            f.write(f"  Synthetic Accessibility: {format_metric_with_std(gt_baseline, 'sa_score_mean', 'sa_score_std')}\n")
            f.write(f"  Easy Synthesis Fraction (SAâ‰¤3): {gt_baseline.get('easy_synthesis_fraction', 0):.3f}\n")
            f.write(f"  Molecular Complexity: {format_metric_with_std(gt_baseline, 'complexity_mean', 'complexity_std')}\n\n")
            
            # Diversity characteristics
            if 'diversity_metrics' in gt_baseline:
                gt_diversity = gt_baseline['diversity_metrics']
                f.write("GROUND TRUTH DIVERSITY CHARACTERISTICS:\n")
                f.write(f"  Internal Diversity: {gt_diversity.get('internal_diversity', 0):.3f}\n")
                f.write(f"  Scaffold Diversity: {gt_diversity.get('scaffold_diversity', 0):.3f}\n")
                f.write(f"  Unique Scaffolds: {gt_diversity.get('num_unique_scaffolds', 0)}\n")
                f.write(f"  Property Diversity: {gt_diversity.get('property_diversity', 0):.3f}\n\n")
            
            # Scaffold analysis
            if 'scaffold_analysis' in gt_baseline:
                scaffold_analysis = gt_baseline['scaffold_analysis']
                f.write("GROUND TRUTH SCAFFOLD ANALYSIS:\n")
                f.write(f"  Total Scaffolds: {scaffold_analysis.get('total_scaffolds', 0)}\n")
                f.write(f"  Unique Scaffolds: {scaffold_analysis.get('unique_scaffolds', 0)}\n")
                f.write(f"  Scaffold Diversity: {scaffold_analysis.get('scaffold_diversity', 0):.3f}\n\n")
            
            # Property ranges for reference
            f.write("GROUND TRUTH PROPERTY RANGES (for reference):\n")
            for prop in ['mw', 'logp', 'tpsa', 'hba', 'hbd']:
                range_key = f'{prop}_range'
                if range_key in gt_baseline and gt_baseline[range_key]:
                    min_val, max_val = gt_baseline[range_key]
                    f.write(f"  {prop.upper()}: [{min_val:.2f}, {max_val:.2f}]\n")
            f.write("\n")
        else:
            f.write("Ground truth baseline analysis not available.\n\n")
            
        # ENHANCED COMPARATIVE SUMMARY
        f.write("=" * 80 + "\n")
        f.write("COMPARATIVE ANALYSIS WITH BASELINE\n")
        f.write("=" * 80 + "\n\n")
        
        if 'ground_truth_baseline' in comprehensive_data:
            f.write("GENERATED vs GROUND TRUTH COMPARISON:\n")
            f.write(f"{'Property':<25} {'Generated':<20} {'Ground Truth':<20} {'Ratio':<15}\n")
            f.write("-" * 80 + "\n")
            
            # Compare key metrics if available
            gt_baseline = comprehensive_data['ground_truth_baseline']
            
            if results_by_mode.get('pure_generation') and modes_enabled['pure_generation']:
                gen_metrics = mode_metrics.get('pure_generation', {})
                
                # QED comparison
                if ('comprehensive_metrics' in gen_metrics and 
                    'drug_likeness' in gen_metrics['comprehensive_metrics'] and
                    'qed_mean' in gt_baseline):
                    
                    gen_qed = gen_metrics['comprehensive_metrics']['drug_likeness'].get('qed_mean', 0)
                    gt_qed = gt_baseline.get('qed_mean', 0)
                    ratio = gen_qed / gt_qed if gt_qed > 0 else 0
                    
                    f.write(f"{'QED Score':<25} {gen_qed:<20.3f} {gt_qed:<20.3f} {ratio:<15.3f}\n")
                
                # Validity comparison
                if 'comprehensive_metrics' in gen_metrics and 'validity' in gt_baseline:
                    gen_validity = gen_metrics['comprehensive_metrics'].get('validity', 0)
                    gt_validity = gt_baseline.get('validity', 0)
                    ratio = gen_validity / gt_validity if gt_validity > 0 else 0
                    
                    f.write(f"{'Validity':<25} {gen_validity:<20.3f} {gt_validity:<20.3f} {ratio:<15.3f}\n")
                
                # Diversity comparison
                if ('diversity_metrics' in gen_metrics and 
                    'diversity_metrics' in gt_baseline):
                    
                    gen_div = gen_metrics['diversity_metrics'].get('internal_diversity', 0)
                    gt_div = gt_baseline['diversity_metrics'].get('internal_diversity', 0)
                    ratio = gen_div / gt_div if gt_div > 0 else 0
                    
                    f.write(f"{'Internal Diversity':<25} {gen_div:<20.3f} {gt_div:<20.3f} {ratio:<15.3f}\n")
            
            f.write("\n")
            
            # Add interpretation
            f.write("BASELINE COMPARISON INSIGHTS:\n")
            f.write("  â€¢ Values >1.0 indicate generated molecules exceed ground truth characteristics\n")
            f.write("  â€¢ Values <1.0 indicate generated molecules fall short of ground truth\n")
            f.write("  â€¢ Ideal ratios depend on application: drug discovery may prefer higher drug-likeness\n\n")
    
        # Mode comparison table
        f.write("MODE PERFORMANCE COMPARISON:\n")
        f.write(f"{'Mode':<25} {'Valid Results':<15} {'Key Metric':<20} {'Value':<15}\n")
        f.write("-" * 75 + "\n")
        
        for mode_name, display_name in [
            ('pure_generation', 'Pure Generation'),
            ('conventional_retrieval', 'Conventional Retrieval'),
            ('retrieval_by_generation', 'Retrieval by Generation')
        ]:
            if results_by_mode.get(mode_name) and modes_enabled.get(mode_name.replace('_', '_')):
                results = results_by_mode[mode_name]
                valid_count = sum(1 for r in results if r.get('is_valid'))
                total_count = len(results)
                valid_str = f"{valid_count}/{total_count}"
                
                if mode_name == 'pure_generation' and 'target_similarity' in mode_metrics.get(mode_name, {}):
                    sim = mode_metrics[mode_name]['target_similarity']['mean']
                    key_metric = "Target Similarity"
                    value = f"{sim:.3f}"
                elif mode_name in ['conventional_retrieval', 'retrieval_by_generation'] and 'retrieval_accuracy' in mode_metrics.get(mode_name, {}):
                    acc = mode_metrics[mode_name]['retrieval_accuracy']['smiles_top_k_accuracy']
                    if mode_name == 'retrieval_by_generation':
                        key_metric = f"Top-{args.retrieval_by_generation_top_k} Acc"
                    else:
                        key_metric = f"Top-{args.retrieval_top_k} Acc"
                    value = f"{acc:.3f}"
                else:
                    key_metric = "N/A"
                    value = "N/A"
                
                f.write(f"{display_name:<25} {valid_str:<15} {key_metric:<20} {value:<15}\n")
            else:
                f.write(f"{display_name:<25} {'Not Enabled':<15} {'N/A':<20} {'N/A':<15}\n")
        
        f.write("\nKEY INSIGHTS AND RECOMMENDATIONS:\n")
        
        # Enhanced insights
        if (results_by_mode.get('conventional_retrieval') and results_by_mode.get('retrieval_by_generation') and 
            modes_enabled.get('conventional_retrieval') and modes_enabled.get('retrieval_by_generation')):
            
            conv_acc = mode_metrics.get('conventional_retrieval', {}).get('retrieval_accuracy', {}).get('smiles_top_k_accuracy', 0)
            rbg_acc = mode_metrics.get('retrieval_by_generation', {}).get('retrieval_accuracy', {}).get('smiles_top_k_accuracy', 0)
            
            f.write(f"  â€¢ Retrieval Mode Comparison:\n")
            if rbg_acc > conv_acc:
                improvement = ((rbg_acc - conv_acc) / conv_acc * 100) if conv_acc > 0 else float('inf')
                f.write(f"    - Retrieval by generation outperforms conventional retrieval\n")
                f.write(f"      ({rbg_acc:.3f} vs {conv_acc:.3f}, +{improvement:.1f}% improvement)\n")
                f.write(f"    - Model successfully learned meaningful biologicalâ†’drug mappings\n")
            elif conv_acc > rbg_acc:
                f.write(f"    - Conventional retrieval outperforms retrieval by generation\n")
                f.write(f"      ({conv_acc:.3f} vs {rbg_acc:.3f})\n")
                f.write(f"    - Direct biological similarity more reliable than learned mappings\n")
            else:
                f.write(f"    - Both retrieval modes perform similarly ({conv_acc:.3f})\n")
        
        # Generation insights
        if results_by_mode.get('pure_generation') and modes_enabled.get('pure_generation'):
            gen_metrics = mode_metrics.get('pure_generation', {})
            
            f.write(f"  â€¢ Generation Quality Assessment:\n")
            
            # Validity assessment
            if 'comprehensive_metrics' in gen_metrics:
                validity = gen_metrics['comprehensive_metrics'].get('validity', 0)
                if validity > 0.8:
                    f.write(f"    - High validity rate ({validity:.3f}) indicates good molecular generation\n")
                elif validity > 0.5:
                    f.write(f"    - Moderate validity rate ({validity:.3f}) suggests room for improvement\n")
                else:
                    f.write(f"    - Low validity rate ({validity:.3f}) indicates generation issues\n")
            
            # Novelty vs similarity tradeoff
            if 'target_similarity' in gen_metrics and 'comprehensive_metrics' in gen_metrics:
                sim_mean = gen_metrics['target_similarity']['mean']
                novelty = gen_metrics['comprehensive_metrics'].get('novelty', 0)
                
                f.write(f"    - Novelty-Similarity Balance:\n")
                f.write(f"      Novelty: {novelty:.3f}, Target Similarity: {sim_mean:.3f}\n")
                
                if novelty > 0.8 and sim_mean < 0.3:
                    f.write(f"      â†’ High novelty but low target similarity (explore vs exploit)\n")
                elif novelty < 0.5 and sim_mean > 0.7:
                    f.write(f"      â†’ Low novelty but high target similarity (may be memorizing)\n")
                else:
                    f.write(f"      â†’ Balanced exploration-exploitation trade-off\n")
        
        f.write(f"\n  â€¢ Caching and Efficiency:\n")
        f.write(f"    - Intermediate results saved for faster re-evaluation\n")
        f.write(f"    - Cache directory: {args.cache_dir}\n")
        f.write(f"    - Use --recompute-metrics-only for metric-only updates\n")
        
        f.write(f"\nEvaluation completed in {metadata['evaluation_time']:.1f}s\n")
        f.write(f"Report generated on {metadata['timestamp']}\n")
        f.write(f"Cache key: {get_evaluation_cache_key(args)}\n")
    
    return output_txt


def save_molecular_properties_cache(smiles_list, cache_key, cache_dir="./cache"):
    """Save computed molecular properties to avoid recomputation"""
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"molecular_props_{cache_key}.pkl")
    
    logger.info(f"Computing and caching molecular properties for {len(smiles_list)} molecules...")
    
    properties_data = {}
    fingerprints_data = {}
    
    for i, smiles in enumerate(tqdm(smiles_list, desc="Computing molecular properties")):
        try:
            # Comprehensive properties
            properties_data[smiles] = calculate_comprehensive_molecular_properties(smiles)
            
            # Fingerprints for similarity calculations
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                # Convert to numpy array for storage
                fp_array = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(morgan_fp, fp_array)
                fingerprints_data[smiles] = fp_array
            
        except Exception as e:
            logger.warning(f"Failed to compute properties for {smiles}: {e}")
            properties_data[smiles] = None
            fingerprints_data[smiles] = None
    
    cache_data = {
        'properties': properties_data,
        'fingerprints': fingerprints_data,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_molecules': len(smiles_list)
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    logger.info(f"Molecular properties cached: {cache_file}")
    return cache_file


def load_molecular_properties_cache(cache_key, cache_dir="./cache"):
    """Load cached molecular properties"""
    
    cache_file = os.path.join(cache_dir, f"molecular_props_{cache_key}.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        logger.info(f"Loaded cached molecular properties: {cache_data['total_molecules']} molecules")
        return cache_data
        
    except Exception as e:
        logger.warning(f"Failed to load molecular properties cache: {e}")
        return None


def calculate_comprehensive_molecular_properties(smiles):
    """Extended molecular property calculation with error handling."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        Chem.SanitizeMol(mol)
        props = {}
        
        # Basic properties
        props['MW'] = Descriptors.MolWt(mol)
        props['LogP'] = Descriptors.MolLogP(mol)
        props['HBA'] = Descriptors.NumHAcceptors(mol)
        props['HBD'] = Descriptors.NumHDonors(mol)
        props['TPSA'] = Descriptors.TPSA(mol)
        props['RotBonds'] = Descriptors.NumRotatableBonds(mol)
        props['AromaticRings'] = Descriptors.NumAromaticRings(mol)
        props['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
        props['FractionCsp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
        props['QED'] = Descriptors.qed(mol)
        
        # Additional descriptors
        props['BertzCT'] = Descriptors.BertzCT(mol)  # Complexity
        props['MolMR'] = Descriptors.MolMR(mol)  # Molar refractivity
        props['NumRings'] = Descriptors.RingCount(mol)
        props['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
        props['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        props['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        
        # Synthetic accessibility
        props['SA_Score'] = sascorer.calculateScore(mol)
        
        # Drug-likeness indicators
        props['Lipinski_violations'] = sum([
            props['MW'] > 500,
            props['LogP'] > 5,
            props['HBA'] > 10,
            props['HBD'] > 5
        ])
        
        # Veber's rule
        props['Veber_violations'] = sum([
            props['RotBonds'] > 10,
            props['TPSA'] > 140
        ])
        
        return props
        
    except Exception as e:
        logger.debug(f"Failed to process SMILES {smiles}: {e}")
        return None


def get_cache_key(args, train_loader, checkpoint_path):
    """Generate a unique cache key based on model and data parameters"""
    
    # Get checkpoint file hash
    ckpt_hash = ""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt_hash = hashlib.md5(f.read()).hexdigest()[:8]
    
    # Create cache key from relevant parameters
    cache_params = {
        'dataset': args.dataset,
        'checkpoint_hash': ckpt_hash,
        'batch_size': args.batch_size,
        'gene_count_matrix_path': args.gene_count_matrix_path,
        'paired_rna_encoder': args.paired_rna_encoder,
        'total_train_batches': len(train_loader),
        'random_state': args.random_state,
    }
    
    # Create deterministic hash
    cache_str = str(sorted(cache_params.items()))
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:12]
    
    return f"training_features_{args.dataset}_{cache_hash}"


def save_training_features_cache(cache_key, training_data, cache_dir="./cache"):
    """Save training features to cache file"""
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    logger.info(f"Saving training features cache to {cache_file}")
    
    # Add metadata to cache
    cache_data = {
        'training_data': training_data,
        'cache_version': '1.0',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(training_data['smiles']),
        'total_batches': len(training_data['biological_features']) if 'biological_features' in training_data else 0
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # Save a human-readable summary
    summary_file = os.path.join(cache_dir, f"{cache_key}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Training Features Cache Summary\n")
        f.write(f"==============================\n")
        f.write(f"Cache Key: {cache_key}\n")
        f.write(f"Created: {cache_data['timestamp']}\n")
        f.write(f"Total Samples: {cache_data['total_samples']}\n")
        f.write(f"Total Batches: {cache_data['total_batches']}\n")
        f.write(f"Cache Version: {cache_data['cache_version']}\n")
    
    logger.info(f"Cache saved successfully: {cache_data['total_samples']} samples")
    return cache_file


def load_training_features_cache(cache_key, cache_dir="./cache"):
    """Load training features from cache file"""
    
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        logger.info(f"Loading training features cache from {cache_file}")
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache structure
        if 'training_data' not in cache_data:
            logger.warning("Invalid cache file structure, will recompute")
            return None
        
        training_data = cache_data['training_data']
        
        # Validate training data structure (more flexible for different modes)
        required_keys = ['smiles', 'compound_names']
        if not all(key in training_data for key in required_keys):
            logger.warning("Invalid training data structure in cache, will recompute")
            return None
        
        logger.info(f"Cache loaded successfully: {len(training_data['smiles'])} samples from {cache_data.get('timestamp', 'unknown time')}")
        return training_data
        
    except Exception as e:
        logger.warning(f"Failed to load cache file {cache_file}: {e}")
        logger.info("Will recompute training features")
        return None


def collect_training_data_with_biological_features_cached(args, train_loader, image_encoder, rna_encoder, device, 
    max_batches=None, max_training_portion=None, force_recompute=False, cache_dir="./cache",
    has_rna=True, has_imaging=True):
    """Collect training data with biological features using caching"""
    
    # Generate cache key
    cache_key = get_cache_key(args, train_loader, args.ckpt)
    
    # Try to load from cache first (unless forced to recompute)
    if not force_recompute:
        cached_data = load_training_features_cache(cache_key, cache_dir)
        if cached_data is not None:
            # Check if cached data has enough samples for current request
            cached_samples = len(cached_data['smiles'])
            
            # Calculate required samples
            total_train_batches = len(train_loader)
            if max_training_portion is not None:
                required_batches = int(total_train_batches * max_training_portion)
                required_samples = required_batches * args.batch_size
            elif max_batches is not None:
                required_samples = max_batches * args.batch_size
            else:
                required_samples = cached_samples  # Use all cached data
            
            if cached_samples >= required_samples:
                logger.info(f"Using cached data: {cached_samples} samples available, {required_samples} requested")
                
                # Trim cached data to requested amount if needed
                if required_samples < cached_samples:
                    trimmed_data = {
                        'smiles': cached_data['smiles'][:required_samples],
                        'compound_names': cached_data['compound_names'][:required_samples],
                    }
                    
                    # Only trim biological features if they exist (for conventional retrieval)
                    if 'biological_features' in cached_data:
                        trimmed_data['biological_features'] = []
                        samples_so_far = 0
                        for batch_features in cached_data['biological_features']:
                            batch_size = batch_features.shape[0]
                            if samples_so_far + batch_size <= required_samples:
                                trimmed_data['biological_features'].append(batch_features)
                                samples_so_far += batch_size
                            else:
                                # Partial batch
                                remaining = required_samples - samples_so_far
                                if remaining > 0:
                                    trimmed_data['biological_features'].append(batch_features[:remaining])
                                break
                    
                    logger.info(f"Trimmed cached data to {len(trimmed_data['smiles'])} samples")
                    return trimmed_data
                else:
                    return cached_data
            else:
                logger.info(f"Cached data insufficient: {cached_samples} available, {required_samples} required")
                logger.info("Will recompute with more data")
    
    # Cache miss or insufficient data - compute features
    logger.info("Computing training features (this will take time but only needs to be done once)")
    
    # Compute features using the original function
    training_data = collect_training_data_with_biological_features_original(
        args, train_loader, image_encoder, rna_encoder, device, 
        max_batches, max_training_portion, has_rna=has_rna, has_imaging=has_imaging
    )
    
    # Save to cache for future use
    try:
        save_training_features_cache(cache_key, training_data, cache_dir)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
        logger.info("Continuing without caching")
    
    return training_data


def collect_training_data_with_biological_features_original(args, train_loader, image_encoder, rna_encoder, device, 
    max_batches=None, max_training_portion=None, has_rna=True, has_imaging=True):
    """Original function renamed for caching system"""
    
    training_data = {
        'smiles': [],
        'compound_names': [],
    }
    
    # Only add biological_features if we actually need them (for conventional retrieval)
    need_biological_features = getattr(args, 'run_conventional_retrieval', False)
    if need_biological_features:
        training_data['biological_features'] = []
    
    # Save additional metadata for comprehensive analysis
    training_data['metadata'] = {
        'has_rna': has_rna,
        'has_imaging': has_imaging,
        'need_biological_features': need_biological_features,
        'max_batches': max_batches,
        'max_training_portion': max_training_portion
    }
    
    logger.info("Collecting training data with biological features...")
    
    total_train_batches = len(train_loader)
    
    # Use single argument priority: max_training_portion > max_batches > use all data
    if max_training_portion is not None:
        actual_max_batches = int(total_train_batches * max_training_portion)
        logger.info(f"Using {actual_max_batches}/{total_train_batches} training batches "
                    f"(portion: {max_training_portion:.1%})")
    elif max_batches is not None:
        actual_max_batches = min(max_batches, total_train_batches)
        logger.info(f"Using {actual_max_batches}/{total_train_batches} training batches "
                    f"(max_batches: {max_batches})")
    else:
        actual_max_batches = total_train_batches
        logger.info(f"Using all {actual_max_batches} training batches")

    # Process with optimized batch size for encoding
    encoding_batch_size = min(16, args.batch_size) if need_biological_features else args.batch_size
    if need_biological_features:
        logger.info(f"Using encoding batch size: {encoding_batch_size}")

    processed_batches = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Collecting training data")):
        if batch_idx >= actual_max_batches:
            break
        
        batch_size = len(batch['target_smiles'])
        
        # Always collect SMILES and compound names
        training_data['smiles'].extend(batch['target_smiles'])
        training_data['compound_names'].extend(batch['compound_name'])
        
        # Only compute biological features if needed
        if need_biological_features:
            # Process batch in smaller chunks for memory efficiency
            all_y_features = []
            
            for start_idx in range(0, batch_size, encoding_batch_size):
                end_idx = min(start_idx + encoding_batch_size, batch_size)
                
                # Extract sub-batch
                control_imgs = batch['control_images'][start_idx:end_idx]
                treatment_imgs = batch['treatment_images'][start_idx:end_idx]
                control_rna = batch['control_transcriptomics'][start_idx:end_idx]
                treatment_rna = batch['treatment_transcriptomics'][start_idx:end_idx]
                
                # Encode biological features for sub-batch
                with torch.no_grad():
                    if args.legacy_dummy_tensors:
                        # Use the old approach with dummy tensors
                        y_features, pad_mask = dual_rna_image_encoder(
                            control_imgs, treatment_imgs, 
                            control_rna, treatment_rna,
                            image_encoder, rna_encoder, device
                        )
                    else:
                        # Use the new approach
                        y_features, pad_mask = dual_rna_image_encoder(
                            control_imgs, treatment_imgs, 
                            control_rna, treatment_rna,
                            image_encoder, rna_encoder, device, has_rna, has_imaging
                        )
                    all_y_features.append(y_features.cpu())
            
            # Concatenate all sub-batch features
            batch_y_features = torch.cat(all_y_features, dim=0)
            training_data['biological_features'].append(batch_y_features)

            feature_batch_size = batch_y_features.shape[0]

            if batch_size != feature_batch_size:
                logger.warning(f"MISMATCH: batch samples={batch_size}, features={feature_batch_size}")
            if len(batch['target_smiles']) != feature_batch_size:
                logger.warning(f"MISMATCH: batch SMILES={len(batch['target_smiles'])}, features={feature_batch_size}")
            if len(batch['compound_name']) != feature_batch_size:
                logger.warning(f"MISMATCH: batch compound names={len(batch['compound_name'])}, features={feature_batch_size}")
            logger.info(f"batch_idx={batch_idx}: len(batch['target_smiles'])={len(batch['target_smiles'])}, len(batch['compound_name'])={len(batch['compound_name'])}, batch_y_features.shape={batch_y_features.shape}")
            
        processed_batches += 1

    logger.info(f"Collected {len(training_data['smiles'])} training samples" + 
               (" with biological features" if need_biological_features else ""))
    return training_data


def clear_training_features_cache(cache_dir="./cache"):
    """Clear all cached training features"""
    
    if not os.path.exists(cache_dir):
        logger.info("No cache directory found")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('training_features_') or f.startswith('molecular_props_')]
    
    if not cache_files:
        logger.info("No cache files found")
        return
    
    for cache_file in cache_files:
        cache_path = os.path.join(cache_dir, cache_file)
        try:
            os.remove(cache_path)
            logger.info(f"Removed cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to remove {cache_file}: {e}")
    
    logger.info(f"Cache cleanup complete")


def get_evaluation_cache_key(args):
    """Generate cache key for evaluation results"""
    
    cache_params = {
        'dataset': args.dataset,
        'model': args.model,
        'ckpt': args.ckpt,
        'generation_steps': args.generation_steps,
        'num_samples_per_condition': args.num_samples_per_condition,
        'retrieval_top_k': args.retrieval_top_k,
        'retrieval_by_generation_top_k': args.retrieval_by_generation_top_k,
        'global_seed': args.global_seed,
        'random_state': args.random_state,
        'max_test_samples': args.max_test_samples,
        'eval_portion': args.eval_portion,
        'max_eval_batches': args.max_eval_batches,
        # Add mode flags to cache key
        'run_generation': getattr(args, 'run_generation', False),
        'run_conventional_retrieval': getattr(args, 'run_conventional_retrieval', False),
        'run_retrieval_by_generation': getattr(args, 'run_retrieval_by_generation', False),
    }
    
    # Create deterministic hash
    cache_str = str(sorted(cache_params.items()))
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:12]
    
    return f"evaluation_results_{args.dataset}_{cache_hash}"


def save_evaluation_results_cache(cache_key, results_data, cache_dir="./cache"):
    """Enhanced save evaluation results with more intermediate data"""
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    logger.info(f"Saving comprehensive evaluation results cache to {cache_file}")
    
    # Add metadata to cache
    cache_data = {
        'results_data': results_data,
        'cache_version': '3.0',  # Updated version for enhanced caching
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(results_data.get('all_results_by_mode', {}).get('pure_generation', [])),
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # Save enhanced summary
    summary_file = os.path.join(cache_dir, f"{cache_key}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Enhanced Evaluation Results Cache Summary\n")
        f.write(f"========================================\n")
        f.write(f"Cache Key: {cache_key}\n")
        f.write(f"Created: {cache_data['timestamp']}\n")
        f.write(f"Total Samples: {cache_data['total_samples']}\n")
        f.write(f"Cache Version: {cache_data['cache_version']}\n")
        
        # List what's cached
        f.write(f"\nCached Components:\n")
        for key in results_data.keys():
            if key == 'all_results_by_mode':
                f.write(f"  - {key}:\n")
                for mode, results in results_data[key].items():
                    f.write(f"    - {mode}: {len(results)} samples\n")
            elif key == 'metadata':
                f.write(f"  - {key}: evaluation configuration\n")
            elif key == 'training_data':
                f.write(f"  - {key}: {len(results_data[key].get('smiles', []))} training samples\n")
            else:
                f.write(f"  - {key}\n")
        
        # List embedding files if available
        embedding_files = results_data.get('metadata', {}).get('embedding_files', {})
        if any(embedding_files.values()):
            f.write(f"\nSaved Intermediate Files:\n")
            for file_type, filename in embedding_files.items():
                if filename:
                    f.write(f"  - {file_type}: {filename}\n")
    
    logger.info(f"Enhanced evaluation cache saved successfully")
    return cache_file


def load_evaluation_results_cache(cache_key, cache_dir="./cache"):
    """Load evaluation results from cache"""
    
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        logger.info(f"Loading evaluation results cache from {cache_file}")
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache structure
        if 'results_data' not in cache_data:
            logger.warning("Invalid cache file structure, will recompute")
            return None
        
        results_data = cache_data['results_data']
        cache_version = cache_data.get('cache_version', '1.0')
        
        logger.info(f"Cache loaded successfully from {cache_data.get('timestamp', 'unknown time')} (version: {cache_version})")
        return results_data
        
    except Exception as e:
        logger.warning(f"Failed to load cache file {cache_file}: {e}")
        logger.info("Will recompute evaluation results")
        return None


def clear_evaluation_results_cache(cache_dir="./cache"):
    """Clear all cached evaluation results"""
    
    if not os.path.exists(cache_dir):
        logger.info("No cache directory found")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('evaluation_results_')]
    
    if not cache_files:
        logger.info("No evaluation cache files found")
        return
    
    for cache_file in cache_files:
        cache_path = os.path.join(cache_dir, cache_file)
        try:
            os.remove(cache_path)
            logger.info(f"Removed cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to remove {cache_file}: {e}")
    
    logger.info(f"Evaluation cache cleanup complete")


def recompute_metrics_from_cache(cache_key, cache_dir="./cache", output_dir="./evaluation_results", output_prefix="recomputed"):
    """Standalone function to recompute metrics from cached results"""
    
    # Load cached results
    cached_results = load_evaluation_results_cache(cache_key, cache_dir)
    if cached_results is None:
        logger.error(f"No cached results found for key: {cache_key}")
        return None
    
    logger.info("Recomputing metrics from cached evaluation results...")
    
    # Recompute metrics with current metric functions
    mode_metrics = calculate_mode_specific_metrics(
        cached_results['all_results_by_mode'], 
        cached_results['training_data']
    )
    
    # Update metrics
    cached_results['mode_metrics'] = mode_metrics
    cached_results['results_by_mode'] = cached_results['all_results_by_mode']  # For compatibility
    
    # Update timestamp
    cached_results['metadata']['evaluation_time'] = 0.1  # Metrics recomputation is fast
    cached_results['metadata']['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    cached_results['metadata']['recomputed_from_cache'] = True
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    output_json = os.path.join(output_dir, f"{output_prefix}_comprehensive.json")
    with open(output_json, 'w') as f:
        json.dump(cached_results, f, indent=2, default=str)
    
    # Create summary (create a dummy args object for the function)
    class DummyArgs:
        def __init__(self, cached_results):
            metadata = cached_results['metadata']
            self.output_dir = output_dir
            self.output_prefix = output_prefix
            self.dataset = metadata.get('dataset', 'unknown')
            self.global_seed = metadata.get('global_seed', 42)
            self.retrieval_top_k = metadata.get('retrieval_top_k', 5)
            self.retrieval_by_generation_top_k = metadata.get('retrieval_by_generation_top_k', 20)
            self.generation_steps = metadata.get('generation_steps', 50)
            self.cache_dir = cache_dir
    
    dummy_args = DummyArgs(cached_results)
    summary_file = create_three_section_summary(cached_results, dummy_args)
    
    logger.info(f"Metrics recomputed and saved:")
    logger.info(f"  JSON: {output_json}")
    logger.info(f"  Summary: {summary_file}")
    
    return cached_results


def save_comprehensive_intermediate_results(args, comprehensive_data, all_intermediate_data):
    """Save all intermediate results for future reuse"""
    
    output_dir = args.output_dir
    prefix = args.output_prefix
    
    # 1. Save molecular fingerprints cache for generated molecules
    if args.run_generation:
        all_generated_smiles = []
        for mode_results in comprehensive_data['all_results_by_mode'].values():
            for result in mode_results:
                if result.get('all_samples'):
                    all_generated_smiles.extend(result['all_samples'])
                elif result.get('generated_smiles'):
                    all_generated_smiles.append(result['generated_smiles'])
        
        if all_generated_smiles:
            cache_key = f"generated_molecules_{prefix}"
            mol_props_cache = save_molecular_properties_cache(
                list(set(all_generated_smiles)), cache_key, args.cache_dir
            )
            logger.info(f"Saved molecular properties cache: {mol_props_cache}")
    
    # 2. Save biological features for each batch (if computed)
    if all_intermediate_data:
        bio_features_file = os.path.join(output_dir, f"{prefix}_batch_biological_features.npz")
        bio_features_data = {}
        
        for batch_idx, batch_data in enumerate(all_intermediate_data):
            if batch_data.get('batch_biological_features') is not None:
                bio_features_data[f'batch_{batch_idx}'] = batch_data['batch_biological_features']
        
        if bio_features_data:
            np.savez_compressed(bio_features_file, **bio_features_data)
            logger.info(f"Saved batch biological features: {bio_features_file}")
    
    # 3. Save detailed generation analysis
    if args.run_generation and all_intermediate_data:
        generation_analysis = {
            'total_batches': len(all_intermediate_data),
            'all_generated_samples': [],
            'validity_per_batch': [],
            'samples_per_condition': args.num_samples_per_condition
        }
        
        for batch_data in all_intermediate_data:
            if batch_data.get('generation_data'):
                gen_data = batch_data['generation_data']
                generation_analysis['all_generated_samples'].extend(gen_data.get('all_generated_per_condition', []))
        
        generation_file = os.path.join(output_dir, f"{prefix}_generation_analysis.json")
        with open(generation_file, 'w') as f:
            json.dump(generation_analysis, f, indent=2)
        logger.info(f"Saved generation analysis: {generation_file}")
    
    # 4. Save retrieval similarity matrices (if computed)
    # This could be added based on specific needs
    
    return {
        'molecular_properties_cached': args.run_generation,
        'biological_features_saved': len(all_intermediate_data) > 0,
        'generation_analysis_saved': args.run_generation and all_intermediate_data,
    }


def calculate_ground_truth_baseline_metrics(ground_truth_smiles: List[str]) -> Dict:
    """Calculate baseline metrics on ground truth molecules for comparison."""
    
    logger.info(f"Calculating baseline metrics on {len(ground_truth_smiles)} ground truth molecules...")
    
    # Use existing comprehensive metrics function
    from evaluation_metrics import calculate_comprehensive_generation_metrics
    
    # Calculate intrinsic properties (no reference needed)
    baseline_metrics = {}
    
    # 1. Basic molecular properties
    valid_molecules = []
    qed_scores = []
    molecular_properties = {
        'mw': [], 'logp': [], 'tpsa': [], 'hba': [], 'hbd': [], 
        'rotatable_bonds': [], 'aromatic_rings': [], 'heavy_atoms': []
    }
    lipinski_violations = []
    sa_scores = []
    complexity_scores = []
    pains_count = 0
    
    for smiles in ground_truth_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_molecules.append(smiles)
                
                # QED and basic properties
                qed_scores.append(Descriptors.qed(mol))
                molecular_properties['mw'].append(Descriptors.MolWt(mol))
                molecular_properties['logp'].append(Descriptors.MolLogP(mol))
                molecular_properties['tpsa'].append(Descriptors.TPSA(mol))
                molecular_properties['hba'].append(Descriptors.NumHAcceptors(mol))
                molecular_properties['hbd'].append(Descriptors.NumHDonors(mol))
                molecular_properties['rotatable_bonds'].append(Descriptors.NumRotatableBonds(mol))
                molecular_properties['aromatic_rings'].append(Descriptors.NumAromaticRings(mol))
                molecular_properties['heavy_atoms'].append(Descriptors.HeavyAtomCount(mol))
                
                # Lipinski violations
                mw = molecular_properties['mw'][-1]
                logp = molecular_properties['logp'][-1]
                hba = molecular_properties['hba'][-1]
                hbd = molecular_properties['hbd'][-1]
                
                violations = sum([mw > 500, logp > 5, hba > 10, hbd > 5])
                lipinski_violations.append(violations)
                
                # SA score and complexity
                sa_scores.append(sascorer.calculateScore(mol))
                complexity_scores.append(Descriptors.BertzCT(mol))
                
        except Exception as e:
            logger.debug(f"Failed to process ground truth SMILES {smiles}: {e}")
            continue
    
    # Calculate summary statistics
    baseline_metrics['validity'] = len(valid_molecules) / len(ground_truth_smiles) if ground_truth_smiles else 0
    baseline_metrics['uniqueness'] = len(set(valid_molecules)) / len(valid_molecules) if valid_molecules else 0
    
    # QED statistics
    if qed_scores:
        baseline_metrics['qed_mean'] = np.mean(qed_scores)
        baseline_metrics['qed_std'] = np.std(qed_scores)
        baseline_metrics['qed_median'] = np.median(qed_scores)
        baseline_metrics['drug_like_fraction'] = sum(1 for q in qed_scores if q >= 0.5) / len(qed_scores)
    
    # Molecular property statistics
    for prop, values in molecular_properties.items():
        if values:
            baseline_metrics[f'{prop}_mean'] = np.mean(values)
            baseline_metrics[f'{prop}_std'] = np.std(values)
            baseline_metrics[f'{prop}_median'] = np.median(values)
            baseline_metrics[f'{prop}_range'] = [np.min(values), np.max(values)]
    
    # Lipinski compliance
    if lipinski_violations:
        baseline_metrics['lipinski_violations_mean'] = np.mean(lipinski_violations)
        baseline_metrics['lipinski_compliant_fraction'] = sum(v == 0 for v in lipinski_violations) / len(lipinski_violations)
    
    # SA scores
    if sa_scores:
        baseline_metrics['sa_score_mean'] = np.mean(sa_scores)
        baseline_metrics['sa_score_std'] = np.std(sa_scores)
        baseline_metrics['easy_synthesis_fraction'] = sum(1 for s in sa_scores if s <= 3.0) / len(sa_scores)
    
    # Complexity
    if complexity_scores:
        baseline_metrics['complexity_mean'] = np.mean(complexity_scores)
        baseline_metrics['complexity_std'] = np.std(complexity_scores)
    
    # Diversity metrics using existing functions
    from evaluation_metrics import calculate_enhanced_diversity_metrics
    diversity_metrics = calculate_enhanced_diversity_metrics(valid_molecules)
    baseline_metrics['diversity_metrics'] = diversity_metrics
    
    # Scaffold analysis
    def get_murcko_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            return None
    
    scaffolds = [get_murcko_scaffold(smi) for smi in valid_molecules]
    scaffolds = [s for s in scaffolds if s is not None]
    unique_scaffolds = set(scaffolds)
    
    baseline_metrics['scaffold_analysis'] = {
        'total_scaffolds': len(scaffolds),
        'unique_scaffolds': len(unique_scaffolds),
        'scaffold_diversity': len(unique_scaffolds) / len(scaffolds) if scaffolds else 0
    }
    
    logger.info(f"Ground truth baseline calculated: {len(valid_molecules)} valid molecules")
    return baseline_metrics


def collect_ground_truth_molecules(eval_loader, eval_batches):
    """Collect all ground truth molecules from test set."""
    
    all_ground_truth_smiles = []
    all_ground_truth_compounds = []
    
    logger.info("Collecting ground truth molecules for baseline analysis...")
    
    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Collecting ground truth")):
        if batch_idx >= eval_batches:
            break
            
        all_ground_truth_smiles.extend(batch['target_smiles'])
        all_ground_truth_compounds.extend(batch['compound_name'])
    
    logger.info(f"Collected {len(all_ground_truth_smiles)} ground truth molecules")
    
    return {
        'smiles': all_ground_truth_smiles,
        'compound_names': all_ground_truth_compounds
    }


def save_metrics_to_csv(comprehensive_data, args):
    """Save key evaluation metrics to CSV file in the specified format."""
    # Extract metrics for pure generation mode (main focus)
    results_by_mode = comprehensive_data.get('results_by_mode', {})
    mode_metrics = comprehensive_data.get('mode_metrics', {})
    
    # Initialize row with default values
    csv_row = {
        'Dataset': args.dataset.upper(),
        'Method': args.output_prefix if args.output_prefix else 'Unknown',
        'Validity': 0.0,
        'Uniqueness': 0.0, 
        'Novelty': 0.0,
        'FCD': None,
        'QED': 0.0,
        'Molecular Weight': 0.0,
        'LogP': 0.0,
        'TPSA': 0.0,
        'Lipinski Compliant Fraction': 0.0,
        'Drug-like Fraction': 0.0,
        'Target Similarity': 0.0,
        'NN Similarity': 0.0,
        'Scaffold Diversity': 0.0,
        'Internal Diversity': 0.0,
        'SAS': 0.0,
        'MW KL': 0.0,
        'LogP KL': 0.0,
        'TPSA KL': 0.0
    }
    
    # Extract metrics from pure generation mode if available
    if 'pure_generation' in mode_metrics and mode_metrics['pure_generation']:
        gen_metrics = mode_metrics['pure_generation']
        
        # Comprehensive metrics
        if 'comprehensive_metrics' in gen_metrics:
            comp_metrics = gen_metrics['comprehensive_metrics']
            
            csv_row['Validity'] = comp_metrics.get('validity', 0.0)
            csv_row['Uniqueness'] = comp_metrics.get('uniqueness', 0.0)
            csv_row['Novelty'] = comp_metrics.get('novelty', 0.0)
            csv_row['FCD'] = comp_metrics.get('fcd', None)
            csv_row['NN Similarity'] = comp_metrics.get('snn_mean', 0.0)
            
            # Drug-likeness metrics
            if 'drug_likeness' in comp_metrics:
                drug_metrics = comp_metrics['drug_likeness']
                csv_row['QED'] = drug_metrics.get('qed_mean', 0.0)
                csv_row['Molecular Weight'] = drug_metrics.get('mw_mean', 0.0)
                csv_row['LogP'] = drug_metrics.get('logp_mean', 0.0)
                csv_row['TPSA'] = drug_metrics.get('tpsa_mean', 0.0)
                csv_row['Lipinski Compliant Fraction'] = drug_metrics.get('lipinski_compliant_fraction', 0.0)
                csv_row['Drug-like Fraction'] = drug_metrics.get('drug_like_fraction', 0.0)
            
            # Scaffold metrics
            if 'scaffold_metrics' in comp_metrics:
                scaffold_metrics = comp_metrics['scaffold_metrics']
                csv_row['Scaffold Diversity'] = scaffold_metrics.get('scaffold_diversity', 0.0)
            
            # Distribution metrics (KL divergences)
            if 'distribution_metrics' in comp_metrics:
                dist_metrics = comp_metrics['distribution_metrics']
                csv_row['MW KL'] = dist_metrics.get('mw_kl_div', 0.0)
                csv_row['LogP KL'] = dist_metrics.get('logp_kl_div', 0.0)
                csv_row['TPSA KL'] = dist_metrics.get('tpsa_kl_div', 0.0)
        
        # Target similarity
        if 'target_similarity' in gen_metrics:
            csv_row['Target Similarity'] = gen_metrics['target_similarity'].get('mean', 0.0)
        
        # Diversity metrics
        if 'diversity_metrics' in gen_metrics:
            diversity_metrics = gen_metrics['diversity_metrics']
            csv_row['Internal Diversity'] = diversity_metrics.get('internal_diversity', 0.0)
            # Use scaffold diversity from diversity_metrics if not found in comprehensive
            if csv_row['Scaffold Diversity'] == 0.0:
                csv_row['Scaffold Diversity'] = diversity_metrics.get('scaffold_diversity', 0.0)
        
        # Generation quality metrics
        if 'generation_quality' in gen_metrics:
            quality_metrics = gen_metrics['generation_quality']
            if 'synthetic_accessibility' in quality_metrics:
                csv_row['SAS'] = quality_metrics['synthetic_accessibility'].get('mean', 0.0)
    
    # Handle inf values and round to reasonable precision
    for key, value in csv_row.items():
        if isinstance(value, (int, float)):
            if value == float('inf') or value == float('-inf'):
                csv_row[key] = 999.999  # Replace inf with large number
            elif value is not None:
                csv_row[key] = round(value, 3)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame([csv_row])
    
    # Save to CSV file
    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved metrics to CSV: {csv_path}")
    
    # Also print the CSV row for verification
    logger.info("CSV Metrics Summary:")
    for key, value in csv_row.items():
        logger.info(f"  {key}: {value}")
    
    return csv_path


def append_metrics_to_master_csv(csv_row_data, master_csv_path="evaluation_master_results.csv"):
    """Append metrics to a master CSV file for comparing multiple runs."""
    
    import pandas as pd
    import os
    
    # Check if master file exists
    if os.path.exists(master_csv_path):
        # Load existing data
        existing_df = pd.read_csv(master_csv_path)
        # Append new row
        updated_df = pd.concat([existing_df, pd.DataFrame([csv_row_data])], ignore_index=True)
    else:
        # Create new master file
        updated_df = pd.DataFrame([csv_row_data])
    
    # Save updated master file
    updated_df.to_csv(master_csv_path, index=False)
    logger.info(f"Updated master CSV: {master_csv_path}")
    
    return master_csv_path

