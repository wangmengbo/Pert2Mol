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


def calculate_comprehensive_generation_metrics(generated_smiles: List[str], 
                                              reference_smiles: List[str],
                                              training_smiles: List[str] = None) -> Dict:
    """
    Calculate comprehensive molecular generation evaluation metrics following
    current state-of-the-art standards from recent papers.
    """
    
    metrics = {}
    
    # === BASIC VALIDITY METRICS ===
    valid_generated = []
    for smi in generated_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                valid_generated.append(canonical)
        except:
            continue
    
    metrics['validity'] = len(valid_generated) / len(generated_smiles) if generated_smiles else 0
    metrics['uniqueness'] = len(set(valid_generated)) / len(valid_generated) if valid_generated else 0
    
    # === NOVELTY (Critical Missing Metric) ===
    if training_smiles:
        training_set = set(training_smiles)
        novel_molecules = [smi for smi in valid_generated if smi not in training_set]
        metrics['novelty'] = len(novel_molecules) / len(valid_generated) if valid_generated else 0
    
    # === FRÉCHET CHEMNET DISTANCE (Gold Standard) ===
    # Note: Requires pre-trained ChemNet model
    try:
        from fcd import get_fcd, load_ref_model
        model = load_ref_model()
        metrics['fcd'] = get_fcd(reference_smiles, valid_generated, model)
    except ImportError:
        print("FCD calculation requires 'fcd' package. Install with: pip install FCD")
        metrics['fcd'] = None
    
    # === SIMILARITY METRICS ===
    if reference_smiles and valid_generated:
        # Nearest Neighbor Similarity (SNN)
        snn_scores = []
        for gen_smi in valid_generated:
            gen_mol = Chem.MolFromSmiles(gen_smi)
            if gen_mol is None:
                continue
            gen_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(gen_mol, 2)
            
            max_sim = 0
            for ref_smi in reference_smiles:
                ref_mol = Chem.MolFromSmiles(ref_smi)
                if ref_mol is None:
                    continue
                ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2)
                sim = DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
                max_sim = max(max_sim, sim)
            snn_scores.append(max_sim)
        
        metrics['snn_mean'] = np.mean(snn_scores) if snn_scores else 0
        metrics['snn_std'] = np.std(snn_scores) if snn_scores else 0
    
    # === FRAGMENT SIMILARITY ===
    metrics['fragment_similarity'] = calculate_fragment_similarity(valid_generated, reference_smiles)
    
    # === SCAFFOLD ANALYSIS ===
    metrics['scaffold_metrics'] = calculate_scaffold_metrics(valid_generated, reference_smiles)
    
    # === DRUG-LIKENESS METRICS ===
    metrics['drug_likeness'] = calculate_drug_likeness_metrics(valid_generated)
    
    # === DIVERSITY METRICS ===
    metrics['diversity'] = calculate_enhanced_diversity_metrics(valid_generated)
    
    # === DISTRIBUTION METRICS ===
    metrics['distribution_metrics'] = calculate_distribution_metrics(valid_generated, reference_smiles)
    
    # === COVERAGE METRICS ===
    metrics['coverage'] = calculate_coverage_metrics(valid_generated, reference_smiles)
    
    return metrics


def calculate_drug_likeness_metrics(smiles_list: List[str]) -> Dict:
    """Calculate comprehensive drug-likeness metrics."""
    
    drug_metrics = {
        'qed_mean': 0, 'qed_std': 0,
        'lipinski_violations_mean': 0,
        'molecular_weight_mean': 0, 'molecular_weight_std': 0,
        'logp_mean': 0, 'logp_std': 0,
        'tpsa_mean': 0, 'tpsa_std': 0,
        'lipinski_compliant_fraction': 0,
        'drug_like_fraction': 0
    }
    
    qed_scores = []
    lipinski_violations = []
    properties = {'mw': [], 'logp': [], 'tpsa': [], 'hbd': [], 'hba': []}
    
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
                
            # QED Score
            qed_scores.append(Descriptors.qed(mol))
            
            # Molecular properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            properties['mw'].append(mw)
            properties['logp'].append(logp)
            properties['tpsa'].append(tpsa)
            properties['hbd'].append(hbd)
            properties['hba'].append(hba)
            
            # Lipinski violations
            violations = sum([
                mw > 500,
                logp > 5,
                hbd > 5,
                hba > 10
            ])
            lipinski_violations.append(violations)
            
        except:
            continue
    
    if qed_scores:
        drug_metrics['qed_mean'] = np.mean(qed_scores)
        drug_metrics['qed_std'] = np.std(qed_scores)
    
    if lipinski_violations:
        drug_metrics['lipinski_violations_mean'] = np.mean(lipinski_violations)
        drug_metrics['lipinski_compliant_fraction'] = sum(v == 0 for v in lipinski_violations) / len(lipinski_violations)
    
    for prop, values in properties.items():
        if values:
            drug_metrics[f'{prop}_mean'] = np.mean(values)
            drug_metrics[f'{prop}_std'] = np.std(values)
    
    # Combined drug-likeness score
    if qed_scores and lipinski_violations:
        drug_like_count = sum(1 for qed, viol in zip(qed_scores, lipinski_violations) 
                             if qed >= 0.5 and viol <= 1)
        drug_metrics['drug_like_fraction'] = drug_like_count / len(qed_scores)
    
    return drug_metrics


def calculate_scaffold_metrics(generated_smiles: List[str], reference_smiles: List[str]) -> Dict:
    """Calculate Murcko scaffold-based metrics."""
    
    def get_murcko_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            return None
    
    gen_scaffolds = [get_murcko_scaffold(smi) for smi in generated_smiles]
    gen_scaffolds = [s for s in gen_scaffolds if s is not None]
    
    ref_scaffolds = [get_murcko_scaffold(smi) for smi in reference_smiles]
    ref_scaffolds = [s for s in ref_scaffolds if s is not None]
    
    gen_scaffold_set = set(gen_scaffolds)
    ref_scaffold_set = set(ref_scaffolds)
    
    scaffold_metrics = {
        'unique_scaffolds': len(gen_scaffold_set),
        'scaffold_diversity': len(gen_scaffold_set) / len(gen_scaffolds) if gen_scaffolds else 0,
        'scaffold_novelty': len(gen_scaffold_set - ref_scaffold_set) / len(gen_scaffold_set) if gen_scaffold_set else 0,
        'scaffold_recovery': len(gen_scaffold_set & ref_scaffold_set) / len(ref_scaffold_set) if ref_scaffold_set else 0
    }
    
    return scaffold_metrics


def calculate_fragment_similarity(generated_smiles: List[str], reference_smiles: List[str]) -> Dict:
    """Calculate fragment-based similarity metrics."""
    
    def get_fragments(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return set()
            # Get BRICS fragments
            frags = Chem.BRICS.BRICSDecompose(mol)
            return set(frags)
        except:
            return set()
    
    # Get all fragments
    gen_fragments = set()
    for smi in generated_smiles:
        gen_fragments.update(get_fragments(smi))
    
    ref_fragments = set()
    for smi in reference_smiles:
        ref_fragments.update(get_fragments(smi))
    
    if not gen_fragments or not ref_fragments:
        return {'fragment_similarity': 0, 'fragment_recovery': 0}
    
    overlap = len(gen_fragments & ref_fragments)
    
    return {
        'fragment_similarity': overlap / len(gen_fragments | ref_fragments),
        'fragment_recovery': overlap / len(ref_fragments)
    }


def calculate_distribution_metrics(generated_smiles: List[str], reference_smiles: List[str]) -> Dict:
    """Calculate distribution-level metrics including KL divergence."""
    
    def get_property_distribution(smiles_list, prop_func):
        props = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    props.append(prop_func(mol))
            except:
                continue
        return np.array(props)
    
    properties = {
        'mw': Descriptors.MolWt,
        'logp': Descriptors.MolLogP,
        'tpsa': Descriptors.TPSA
    }
    
    kl_divergences = {}
    
    for prop_name, prop_func in properties.items():
        gen_props = get_property_distribution(generated_smiles, prop_func)
        ref_props = get_property_distribution(reference_smiles, prop_func)
        
        if len(gen_props) > 0 and len(ref_props) > 0:
            # Calculate KL divergence using histograms
            try:
                # Create common bins
                all_props = np.concatenate([gen_props, ref_props])
                bins = np.linspace(all_props.min(), all_props.max(), 50)
                
                gen_hist, _ = np.histogram(gen_props, bins=bins, density=True)
                ref_hist, _ = np.histogram(ref_props, bins=bins, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                gen_hist = gen_hist + epsilon
                ref_hist = ref_hist + epsilon
                
                # Normalize
                gen_hist = gen_hist / gen_hist.sum()
                ref_hist = ref_hist / ref_hist.sum()
                
                # Calculate KL divergence
                kl_div = np.sum(gen_hist * np.log(gen_hist / ref_hist))
                kl_divergences[f'{prop_name}_kl_div'] = kl_div
                
            except:
                kl_divergences[f'{prop_name}_kl_div'] = float('inf')
        else:
            kl_divergences[f'{prop_name}_kl_div'] = float('inf')
    
    return kl_divergences


def calculate_coverage_metrics(generated_smiles: List[str], reference_smiles: List[str]) -> Dict:
    """Calculate how well generated molecules cover reference chemical space."""
    
    # Convert to fingerprints
    def smiles_to_fingerprint(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except:
            return None
    
    gen_fps = [smiles_to_fingerprint(smi) for smi in generated_smiles]
    gen_fps = [fp for fp in gen_fps if fp is not None]
    
    ref_fps = [smiles_to_fingerprint(smi) for smi in reference_smiles]
    ref_fps = [fp for fp in ref_fps if fp is not None]
    
    if not gen_fps or not ref_fps:
        return {'coverage': 0}
    
    # Calculate coverage using nearest neighbors
    covered_count = 0
    threshold = 0.7  # Similarity threshold for "coverage"
    
    for ref_fp in ref_fps:
        max_sim = 0
        for gen_fp in gen_fps:
            sim = DataStructs.TanimotoSimilarity(ref_fp, gen_fp)
            max_sim = max(max_sim, sim)
        
        if max_sim >= threshold:
            covered_count += 1
    
    coverage = covered_count / len(ref_fps)
    
    return {'coverage': coverage}


def create_three_section_summary(comprehensive_data, args):
    """Create summary report with three major sections for the three evaluation modes"""
    
    metadata = comprehensive_data['metadata']
    results_by_mode = comprehensive_data['results_by_mode']
    mode_metrics = comprehensive_data['mode_metrics']
    
    output_txt = os.path.join(args.output_dir, f"{args.output_prefix}_three_mode_summary.txt")
    
    with open(output_txt, 'w') as f:
        f.write("THREE-MODE COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
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
        
        # SECTION 1: PURE GENERATION MODE
        f.write("=" * 70 + "\n")
        f.write("SECTION 1: PURE GENERATION MODE\n")
        f.write("=" * 70 + "\n")
        f.write("Methodology: Generate novel molecules from biological conditions\n")
        f.write("Purpose: Evaluate model's ability to create new drug candidates\n\n")
        
        if results_by_mode.get('pure_generation') and modes_enabled['pure_generation']:
            gen_results = results_by_mode['pure_generation']
            gen_metrics = mode_metrics.get('pure_generation', {})
            
            f.write(f"Total samples: {len(gen_results)}\n")
            valid_molecules = sum(1 for r in gen_results if r.get('is_valid'))
            f.write(f"Valid molecules: {valid_molecules}\n")
            f.write(f"Validity rate: {valid_molecules/len(gen_results)*100:.1f}%\n\n")
            
            if 'generation_quality' in gen_metrics:
                quality = gen_metrics['generation_quality']
                f.write("GENERATION QUALITY METRICS:\n")
                f.write(f"  Validity: {quality.get('validity', 0):.3f}\n")
                f.write(f"  Uniqueness: {quality.get('uniqueness', 0):.3f}\n")
                
                if 'synthetic_accessibility' in quality:
                    sa = quality['synthetic_accessibility']
                    f.write(f"  Synthetic Accessibility (mean): {sa.get('mean', 0):.2f}\n")
                    f.write(f"  Easy synthesis rate (SA≤3): {sa.get('easy_synthesis_rate', 0):.3f}\n")
                
                f.write(f"  PAINS alerts rate: {quality.get('pains_alerts', 0):.3f}\n")
                f.write("\n")
            
            if 'diversity_metrics' in gen_metrics:
                diversity = gen_metrics['diversity_metrics']
                f.write("DIVERSITY METRICS:\n")
                f.write(f"  Internal diversity: {diversity.get('internal_diversity', 0):.3f}\n")
                f.write(f"  Scaffold diversity: {diversity.get('scaffold_diversity', 0):.3f}\n")
                f.write(f"  Unique scaffolds: {diversity.get('num_unique_scaffolds', 0)}\n\n")
            
            if 'target_similarity' in gen_metrics:
                similarity = gen_metrics['target_similarity']
                f.write("TARGET SIMILARITY:\n")
                f.write(f"  Mean Morgan similarity: {similarity.get('mean', 0):.3f}\n")
                f.write(f"  High similarity rate (≥0.7): {similarity.get('high_similarity_rate', 0):.3f}\n\n")
        else:
            f.write("Pure generation mode was not enabled or no results available.\n\n")
        
        # SECTION 2: CONVENTIONAL RETRIEVAL MODE
        f.write("=" * 70 + "\n")
        f.write("SECTION 2: CONVENTIONAL RETRIEVAL MODE\n")
        f.write("=" * 70 + "\n")
        f.write("Methodology: Find similar biological conditions in training data\n")
        f.write("Purpose: Baseline drug repurposing via biological similarity\n\n")
        
        if results_by_mode.get('conventional_retrieval') and modes_enabled['conventional_retrieval']:
            ret_results = results_by_mode['conventional_retrieval']
            ret_metrics = mode_metrics.get('conventional_retrieval', {})
            
            f.write(f"Total samples: {len(ret_results)}\n")
            f.write(f"Retrieval database size: {len(comprehensive_data['training_data'].get('smiles', []))}\n\n")
            
            if 'retrieval_accuracy' in ret_metrics:
                accuracy = ret_metrics['retrieval_accuracy']
                f.write("RETRIEVAL PERFORMANCE:\n")
                f.write(f"  SMILES Top-{args.retrieval_top_k} Accuracy: {accuracy.get('smiles_top_k_accuracy', 0):.3f}\n")
                f.write(f"  Compound Top-{args.retrieval_top_k} Accuracy: {accuracy.get('compound_top_k_accuracy', 0):.3f}\n")
                f.write(f"  Mean Reciprocal Rank: {accuracy.get('mean_reciprocal_rank', 0):.3f}\n\n")
            
            if 'progressive_metrics' in ret_metrics:
                prog = ret_metrics['progressive_metrics']
                f.write("PROGRESSIVE RETRIEVAL RATES:\n")
                for k in range(1, args.retrieval_top_k + 1):
                    if f'top_{k}_accuracy' in prog:
                        f.write(f"  Top-{k} accuracy: {prog[f'top_{k}_accuracy']:.3f}\n")
                f.write("\n")
        else:
            f.write("Conventional retrieval mode was not enabled or no results available.\n\n")
        
        # SECTION 3: RETRIEVAL BY GENERATION MODE
        f.write("=" * 70 + "\n")
        f.write("SECTION 3: RETRIEVAL BY GENERATION MODE\n")
        f.write("=" * 70 + "\n")
        f.write("Methodology: Generate drug embeddings from biological data, find similar drugs\n")
        f.write("Purpose: Leverage model's biological→drug mapping for functional similarity\n\n")
        
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
                f.write("PROGRESSIVE RETRIEVAL BY GENERATION RATES:\n")
                for k in range(1, args.retrieval_by_generation_top_k + 1):
                    if f'top_{k}_accuracy' in prog:
                        f.write(f"  Top-{k} accuracy: {prog[f'top_{k}_accuracy']:.3f}\n")
                f.write("\n")
        else:
            f.write("Retrieval by generation mode was not enabled or no results available.\n\n")
        
        # COMPARATIVE SUMMARY
        f.write("=" * 70 + "\n")
        f.write("COMPARATIVE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("MODE COMPARISON:\n")
        for mode_name, display_name in [
            ('pure_generation', 'Pure Generation'),
            ('conventional_retrieval', 'Conventional Retrieval'),
            ('retrieval_by_generation', 'Retrieval by Generation')
        ]:
            if results_by_mode.get(mode_name) and modes_enabled.get(mode_name.replace('_', '_')):
                results = results_by_mode[mode_name]
                valid_count = sum(1 for r in results if r.get('is_valid'))
                total_count = len(results)
                f.write(f"  {display_name}: {valid_count}/{total_count} valid results\n")
                
                if mode_name == 'pure_generation' and 'target_similarity' in mode_metrics.get(mode_name, {}):
                    sim = mode_metrics[mode_name]['target_similarity']['mean']
                    f.write(f"    Mean target similarity: {sim:.3f}\n")
                elif mode_name in ['conventional_retrieval', 'retrieval_by_generation'] and 'retrieval_accuracy' in mode_metrics.get(mode_name, {}):
                    acc = mode_metrics[mode_name]['retrieval_accuracy']['smiles_top_k_accuracy']
                    if mode_name == 'retrieval_by_generation':
                        f.write(f"    Top-{args.retrieval_by_generation_top_k} retrieval accuracy: {acc:.3f}\n")
                    else:
                        f.write(f"    Top-{args.retrieval_top_k} retrieval accuracy: {acc:.3f}\n")
            else:
                f.write(f"  {display_name}: Not enabled\n")
        
        f.write(f"\nKEY INSIGHTS:\n")
        
        # Compare retrieval modes if both enabled
        if (results_by_mode.get('conventional_retrieval') and results_by_mode.get('retrieval_by_generation') and 
            modes_enabled.get('conventional_retrieval') and modes_enabled.get('retrieval_by_generation')):
            
            conv_acc = mode_metrics.get('conventional_retrieval', {}).get('retrieval_accuracy', {}).get('smiles_top_k_accuracy', 0)
            rbg_acc = mode_metrics.get('retrieval_by_generation', {}).get('retrieval_accuracy', {}).get('smiles_top_k_accuracy', 0)
            
            if rbg_acc > conv_acc:
                f.write(f"  • Retrieval by generation outperforms conventional retrieval\n")
                f.write(f"    ({rbg_acc:.3f} vs {conv_acc:.3f}), suggesting model learned\n")
                f.write(f"    meaningful biological→drug mappings.\n")
            elif conv_acc > rbg_acc:
                f.write(f"  • Conventional retrieval outperforms retrieval by generation\n")
                f.write(f"    ({conv_acc:.3f} vs {rbg_acc:.3f}), suggesting direct biological\n")
                f.write(f"    similarity is more reliable than model-learned mappings.\n")
            else:
                f.write(f"  • Both retrieval modes perform similarly ({conv_acc:.3f}),\n")
                f.write(f"    indicating comparable effectiveness.\n")
        
        # Comment on generation novelty vs similarity tradeoff
        if results_by_mode.get('pure_generation') and modes_enabled.get('pure_generation'):
            gen_metrics = mode_metrics.get('pure_generation', {})
            if 'target_similarity' in gen_metrics:
                sim_mean = gen_metrics['target_similarity']['mean']
                if sim_mean < 0.3:
                    f.write(f"  • Generated molecules show high novelty (low target similarity)\n")
                elif sim_mean > 0.7:
                    f.write(f"  • Generated molecules are highly similar to targets\n")
                else:
                    f.write(f"  • Generated molecules balance novelty and target similarity\n")
        
        f.write(f"\nEvaluation completed in {metadata['evaluation_time']:.1f}s\n")
        f.write(f"Report generated on {metadata['timestamp']}\n")
    
    return output_txt


def calculate_mode_specific_metrics(results_by_mode, training_data):
    """Calculate metrics specific to each evaluation mode"""
    
    mode_metrics = {}
    
    for mode, results in results_by_mode.items():
        if not results:
            mode_metrics[mode] = {}
            continue
            
        if mode == 'pure_generation':
            # Generation-specific metrics
            all_generated = [r['generated_smiles'] for r in results if r.get('generated_smiles')]
            
            generation_metrics = calculate_additional_generation_metrics(all_generated)
            diversity_metrics = diversity_analysis(all_generated)
            
            # Calculate target similarity metrics
            similarities = []
            for r in results:
                target_smiles = r.get('target_smiles', '')
                generated_smiles = r.get('generated_smiles', '')
                
                if target_smiles and generated_smiles:
                    # Calculate Morgan fingerprint similarity
                    try:
                        target_mol = Chem.MolFromSmiles(target_smiles)
                        generated_mol = Chem.MolFromSmiles(generated_smiles)
                        
                        if target_mol is not None and generated_mol is not None:
                            target_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(target_mol, 2)
                            generated_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(generated_mol, 2)
                            similarity = DataStructs.TanimotoSimilarity(target_fp, generated_fp)
                            similarities.append(similarity)
                    except:
                        continue
            
            mode_metrics[mode] = {
                'generation_quality': generation_metrics,
                'diversity_metrics': diversity_metrics,
                'target_similarity': {
                    'mean': np.mean(similarities) if similarities else 0.0,
                    'std': np.std(similarities) if similarities else 0.0,
                    'high_similarity_rate': sum(1 for s in similarities if s >= 0.7) / len(similarities) if similarities else 0.0
                }
            }
            
        elif mode in ['conventional_retrieval', 'retrieval_by_generation']:
            # Retrieval-specific metrics
            total_samples = len(results)
            smiles_hits = sum(1 for r in results if r.get('smiles_in_top_k', False))
            compound_hits = sum(1 for r in results if r.get('compound_in_top_k', False))
            
            # Calculate MRR
            mrr_scores = []
            for r in results:
                rank = r.get('smiles_hit_rank')
                if rank:
                    mrr_scores.append(1.0 / rank)
                else:
                    mrr_scores.append(0.0)
            
            # Progressive metrics (top-1, top-2, top-3, etc.)
            progressive_metrics = {}
            for k in range(1, 11):  # Up to top-10
                k_hits = sum(1 for r in results if r.get('smiles_hit_rank') and r.get('smiles_hit_rank') <= k)
                progressive_metrics[f'top_{k}_accuracy'] = k_hits / total_samples if total_samples > 0 else 0
            
            mode_metrics[mode] = {
                'retrieval_accuracy': {
                    'smiles_top_k_accuracy': smiles_hits / total_samples if total_samples > 0 else 0,
                    'compound_top_k_accuracy': compound_hits / total_samples if total_samples > 0 else 0,
                    'mean_reciprocal_rank': np.mean(mrr_scores) if mrr_scores else 0,
                    'total_samples': total_samples
                },
                'progressive_metrics': progressive_metrics
            }
    
    return mode_metrics


def calculate_additional_generation_metrics(smiles_list: List[str]) -> Dict:
    """Calculate additional metrics commonly used in molecular generation evaluation."""
    
    metrics = {
        'validity': 0.0,
        'uniqueness': 0.0,
        'synthetic_accessibility': {'mean': 0.0, 'std': 0.0, 'easy_synthesis_rate': 0.0},
        'functional_groups': {},
        'ring_systems': {},
        'pharmacophore_features': {},
        'pains_alerts': 0.0,
        'molecular_complexity': {'mean': 0.0, 'std': 0.0}
    }
    
    if not smiles_list:
        return metrics
    
    # Remove duplicates for uniqueness calculation
    unique_smiles = list(set(smiles_list))
    metrics['uniqueness'] = len(unique_smiles) / len(smiles_list) if smiles_list else 0.0
    
    valid_mols = []
    sa_scores = []
    complexity_scores = []
    pains_count = 0
    
    # Functional group patterns (SMARTS)
    functional_groups = {
        'hydroxyl': '[OH]',
        'carbonyl': '[CX3]=[OX1]',
        'carboxyl': '[CX3](=O)[OX2H1]',
        'amine': '[NX3;H2,H1;!$(NC=O)]',
        'amide': '[NX3][CX3](=[OX1])[#6]',
        'ester': '[#6][CX3](=O)[OX2H0][#6]',
        'ether': '[OD2]([#6])[#6]',
        'aromatic_ring': 'c1ccccc1',
        'heterocycle': '[r5,r6,r7]',
        'halogen': '[F,Cl,Br,I]'
    }
    
    # Ring system patterns
    ring_patterns = {
        'benzene': 'c1ccccc1',
        'pyridine': 'c1ccncc1',
        'pyrimidine': 'c1cncnc1',
        'imidazole': 'c1c[nH]cn1',
        'furan': 'c1ccoc1',
        'thiophene': 'c1ccsc1',
        'cyclohexane': 'C1CCCCC1',
        'piperidine': 'C1CCNCC1'
    }
    
    # PAINS patterns (simplified - common problematic structures)
    pains_patterns = [
        'c1ccc2c(c1)oc1ccccc12',  # dibenzofuran
        'c1ccc2c(c1)sc1ccccc12',  # dibenzothiophene
        '[#6]1~[#6]~[#6]~[#6](~[#6]~[#6]~1)=[#6]~[#6]=[#6]',  # polyene
        'N=C1C=CC(=C([O-])C1)c1ccccc1'  # phenolic Mannich base
    ]
    
    functional_group_counts = {fg: 0 for fg in functional_groups.keys()}
    ring_system_counts = {rs: 0 for rs in ring_patterns.keys()}
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                
                # Synthetic accessibility score
                sa_score = sascorer.calculateScore(mol)
                sa_scores.append(sa_score)
                
                # Molecular complexity (BertzCT)
                complexity = Descriptors.BertzCT(mol)
                complexity_scores.append(complexity)
                
                # Functional group analysis
                for fg_name, pattern in functional_groups.items():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        functional_group_counts[fg_name] += 1
                
                # Ring system analysis
                for rs_name, pattern in ring_patterns.items():
                    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                    ring_system_counts[rs_name] += len(matches)
                
                # PAINS detection
                for pains_pattern in pains_patterns:
                    try:
                        if mol.HasSubstructMatch(Chem.MolFromSmarts(pains_pattern)):
                            pains_count += 1
                            break
                    except:
                        continue
                        
        except Exception as e:
            continue
    
    metrics['validity'] = len(valid_mols) / len(smiles_list) if smiles_list else 0.0
    
    if sa_scores:
        metrics['synthetic_accessibility'] = {
            'mean': np.mean(sa_scores),
            'std': np.std(sa_scores),
            'easy_synthesis_rate': sum(1 for score in sa_scores if score <= 3.0) / len(sa_scores)
        }
    
    if complexity_scores:
        metrics['molecular_complexity'] = {
            'mean': np.mean(complexity_scores),
            'std': np.std(complexity_scores)
        }
    
    # Convert counts to rates
    total_molecules = len(smiles_list)
    metrics['functional_groups'] = {
        fg: count / total_molecules for fg, count in functional_group_counts.items()
    }
    
    metrics['ring_systems'] = {
        rs: count / total_molecules for rs, count in ring_system_counts.items()
    }
    
    metrics['pains_alerts'] = pains_count / total_molecules if total_molecules > 0 else 0.0
    
    return metrics


def diversity_analysis(smiles_list):
    """Diversity analysis with multiple metrics."""
    if len(smiles_list) < 2:
        return {
            'internal_diversity': 0.0,
            'scaffold_diversity': 0.0,
            'property_diversity': 0.0,
            'num_unique_scaffolds': 0
        }
    
    # Fingerprint-based diversity
    fingerprints = []
    scaffolds = []
    properties = []
    
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Fingerprints
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
                fingerprints.append(fp)
                
                # Scaffolds
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
                scaffolds.append(scaffold_smiles)
                
                # Properties for property-based diversity
                props = calculate_comprehensive_molecular_properties(smi)
                if props:
                    prop_vector = [props['MW'], props['LogP'], props['TPSA'], props['QED']]
                    properties.append(prop_vector)
        except:
            continue
    
    results = {}
    
    # Fingerprint diversity
    if len(fingerprints) >= 2:
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(sim)
        results['internal_diversity'] = 1.0 - np.mean(similarities)
    else:
        results['internal_diversity'] = 0.0
    
    # Scaffold diversity
    unique_scaffolds = len(set(scaffolds))
    results['scaffold_diversity'] = unique_scaffolds / len(smiles_list)
    results['num_unique_scaffolds'] = unique_scaffolds
    
    # Property-based diversity
    if len(properties) >= 2:
        prop_matrix = np.array(properties)
        # Normalize properties
        scaler = StandardScaler()
        prop_matrix_norm = scaler.fit_transform(prop_matrix)
        distances = euclidean_distances(prop_matrix_norm)
        # Get upper triangle (excluding diagonal)
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        results['property_diversity'] = np.mean(upper_triangle)
    else:
        results['property_diversity'] = 0.0
    
    return results


def calculate_comprehensive_molecular_properties(smiles):
    """Extended molecular property calculation."""
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
        logger.info(f"Failed to process SMILES {smiles}: {e}")
        return None


# def collect_training_drugs_simple(args, train_loader):
#     """Collect unique drugs from training metadata without biological encoding"""
    
#     training_drugs = {
#         'compound_names': set(),
#         'smiles': set()
#     }
    
#     logger.info("Collecting unique training drugs...")
    
#     # Just iterate through a few batches to get all unique compounds
#     max_batches = 50  # Much smaller since we only need unique drugs
    
#     for batch_idx, batch in enumerate(tqdm(train_loader, desc="Collecting unique drugs")):
#         # Just collect drug information, no biological encoding
#         training_drugs['smiles'].update(batch['target_smiles'])
#         training_drugs['compound_names'].update(batch['compound_name'])
        
#         if batch_idx >= max_batches:
#             logger.info(f"Collected unique drugs from {max_batches} batches")
#             break
    
#     # Convert sets to lists
#     training_drugs['smiles'] = list(training_drugs['smiles'])
#     training_drugs['compound_names'] = list(training_drugs['compound_names'])
    
#     logger.info(f"Found {len(training_drugs['compound_names'])} unique compounds")
#     logger.info(f"Found {len(training_drugs['smiles'])} unique SMILES")
    
#     return training_drugs

