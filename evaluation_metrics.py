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
import faiss
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from utils import AE_SMILES_encoder
from encoders import dual_rna_image_encoder_separate as dual_rna_image_encoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    
    # === FRÃ‰CHET CHEMNET DISTANCE (Gold Standard) ===
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


def calculate_enhanced_diversity_metrics(generated_smiles: List[str]) -> Dict:
    """
    Calculate enhanced diversity metrics for generated molecules.
    
    Args:
        generated_smiles: List of generated SMILES strings
        
    Returns:
        Dict containing various diversity metrics
    """
    
    if len(generated_smiles) < 2:
        return {
            'internal_diversity': 0.0,
            'scaffold_diversity': 0.0,
            'functional_group_diversity': 0.0,
            'property_diversity': 0.0,
            'fingerprint_diversity': 0.0,
            'num_unique_molecules': len(set(generated_smiles)),
            'num_unique_scaffolds': 0,
            'molecular_weight_diversity': 0.0,
            'logp_diversity': 0.0,
            'ring_diversity': 0.0,
            'atom_type_diversity': 0.0
        }
    
    # Convert to molecules and calculate fingerprints
    molecules = []
    fingerprints = []
    scaffolds = []
    functional_groups = []
    properties = []
    
    # Functional group SMARTS patterns
    fg_patterns = {
        'hydroxyl': '[OH]',
        'carbonyl': '[CX3]=[OX1]',
        'carboxyl': '[CX3](=O)[OX2H1]', 
        'amine': '[NX3;H2,H1;!$(NC=O)]',
        'amide': '[NX3][CX3](=[OX1])[#6]',
        'ester': '[#6][CX3](=O)[OX2H0][#6]',
        'ether': '[OD2]([#6])[#6]',
        'aromatic': 'c1ccccc1',
        'halogen': '[F,Cl,Br,I]',
        'nitro': '[N+](=O)[O-]',
        'sulfur': '[S]',
        'phosphorus': '[P]'
    }
    
    for smiles in generated_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            molecules.append(mol)
            
            # Morgan fingerprints
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints.append(fp)
            
            # Murcko scaffolds
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
                scaffolds.append(scaffold_smiles)
            except:
                scaffolds.append("")
            
            # Functional groups
            fg_vector = []
            for pattern in fg_patterns.values():
                try:
                    matches = len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
                    fg_vector.append(matches)
                except:
                    fg_vector.append(0)
            functional_groups.append(fg_vector)
            
            # Molecular properties
            try:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                num_rings = Descriptors.RingCount(mol)
                num_rotatable = Descriptors.NumRotatableBonds(mol)
                properties.append([mw, logp, tpsa, num_rings, num_rotatable])
            except:
                properties.append([0, 0, 0, 0, 0])
                
        except Exception as e:
            continue
    
    if len(molecules) < 2:
        return {
            'internal_diversity': 0.0,
            'scaffold_diversity': 0.0,
            'functional_group_diversity': 0.0,
            'property_diversity': 0.0,
            'fingerprint_diversity': 0.0,
            'num_unique_molecules': len(set(generated_smiles)),
            'num_unique_scaffolds': 0,
            'molecular_weight_diversity': 0.0,
            'logp_diversity': 0.0,
            'ring_diversity': 0.0,
            'atom_type_diversity': 0.0
        }
    
    diversity_metrics = {}
    
    # 1. Fingerprint-based internal diversity (Tanimoto)
    tanimoto_similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            tanimoto_similarities.append(sim)
    
    diversity_metrics['internal_diversity'] = 1.0 - np.mean(tanimoto_similarities) if tanimoto_similarities else 0.0
    diversity_metrics['fingerprint_diversity'] = diversity_metrics['internal_diversity']
    
    # 2. Scaffold diversity
    unique_scaffolds = set(s for s in scaffolds if s)
    diversity_metrics['scaffold_diversity'] = len(unique_scaffolds) / len(generated_smiles) if generated_smiles else 0.0
    diversity_metrics['num_unique_scaffolds'] = len(unique_scaffolds)
    
    # 3. Functional group diversity
    if functional_groups:
        fg_array = np.array(functional_groups)
        # Calculate pairwise cosine distances
        from sklearn.metrics.pairwise import cosine_distances
        try:
            fg_distances = cosine_distances(fg_array)
            # Get upper triangle (excluding diagonal)
            upper_triangle_mask = np.triu(np.ones(fg_distances.shape, dtype=bool), k=1)
            fg_diversity_scores = fg_distances[upper_triangle_mask]
            diversity_metrics['functional_group_diversity'] = np.mean(fg_diversity_scores) if len(fg_diversity_scores) > 0 else 0.0
        except:
            diversity_metrics['functional_group_diversity'] = 0.0
    else:
        diversity_metrics['functional_group_diversity'] = 0.0
    
    # 4. Property-based diversity
    if properties:
        prop_array = np.array(properties)
        try:
            # Normalize properties to same scale
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            prop_array_norm = scaler.fit_transform(prop_array)
            
            # Calculate pairwise Euclidean distances
            prop_distances = euclidean_distances(prop_array_norm)
            upper_triangle_mask = np.triu(np.ones(prop_distances.shape, dtype=bool), k=1)
            prop_diversity_scores = prop_distances[upper_triangle_mask]
            diversity_metrics['property_diversity'] = np.mean(prop_diversity_scores) if len(prop_diversity_scores) > 0 else 0.0
            
            # Individual property diversities
            diversity_metrics['molecular_weight_diversity'] = np.std(prop_array[:, 0]) if len(prop_array) > 1 else 0.0
            diversity_metrics['logp_diversity'] = np.std(prop_array[:, 1]) if len(prop_array) > 1 else 0.0
            diversity_metrics['ring_diversity'] = np.std(prop_array[:, 3]) if len(prop_array) > 1 else 0.0
            
        except Exception as e:
            diversity_metrics['property_diversity'] = 0.0
            diversity_metrics['molecular_weight_diversity'] = 0.0
            diversity_metrics['logp_diversity'] = 0.0
            diversity_metrics['ring_diversity'] = 0.0
    else:
        diversity_metrics['property_diversity'] = 0.0
        diversity_metrics['molecular_weight_diversity'] = 0.0
        diversity_metrics['logp_diversity'] = 0.0
        diversity_metrics['ring_diversity'] = 0.0
    
    # 5. Atom type diversity
    atom_type_sets = []
    for mol in molecules:
        atom_types = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
        atom_type_sets.append(atom_types)
    
    if atom_type_sets:
        # Calculate Jaccard diversity (average pairwise Jaccard distance)
        jaccard_distances = []
        for i in range(len(atom_type_sets)):
            for j in range(i + 1, len(atom_type_sets)):
                set1, set2 = atom_type_sets[i], atom_type_sets[j]
                if len(set1) == 0 and len(set2) == 0:
                    jaccard_dist = 0.0
                else:
                    jaccard_sim = len(set1 & set2) / len(set1 | set2)
                    jaccard_dist = 1.0 - jaccard_sim
                jaccard_distances.append(jaccard_dist)
        
        diversity_metrics['atom_type_diversity'] = np.mean(jaccard_distances) if jaccard_distances else 0.0
    else:
        diversity_metrics['atom_type_diversity'] = 0.0
    
    # 6. Uniqueness
    diversity_metrics['num_unique_molecules'] = len(set(generated_smiles))
    diversity_metrics['uniqueness'] = len(set(generated_smiles)) / len(generated_smiles) if generated_smiles else 0.0
    
    return diversity_metrics


def calculate_mode_specific_metrics(results_by_mode, training_data):
    """Calculate metrics specific to each evaluation mode"""
    
    mode_metrics = {}
    
    for mode, results in results_by_mode.items():
        if not results:
            mode_metrics[mode] = {}
            continue
            
        if mode == 'pure_generation':
            # ENHANCED: Add comprehensive metrics
            all_generated = [r['generated_smiles'] for r in results if r.get('generated_smiles')]
            all_targets = [r['target_smiles'] for r in results if r.get('target_smiles')]
            training_smiles = training_data.get('smiles', [])
            
            # Get comprehensive metrics (NEW)
            try:
                comprehensive_metrics = calculate_comprehensive_generation_metrics(
                    generated_smiles=all_generated,
                    reference_smiles=all_targets,
                    training_smiles=training_smiles
                )
            except Exception as e:
                logger.warning(f"Comprehensive metrics failed: {e}")
                comprehensive_metrics = {}
            
            # Keep existing metrics for compatibility
            generation_metrics = calculate_additional_generation_metrics(all_generated)
            diversity_metrics = diversity_analysis(all_generated)
            
            # Calculate target similarity metrics
            similarities = []
            for r in results:
                target_smiles = r.get('target_smiles', '')
                generated_smiles = r.get('generated_smiles', '')
                
                if target_smiles and generated_smiles:
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
            
            # Combine all metrics
            mode_metrics[mode] = {
                'generation_quality': generation_metrics,
                'diversity_metrics': diversity_metrics,
                'comprehensive_metrics': comprehensive_metrics,  # NEW
                'target_similarity': {
                    'mean': np.mean(similarities) if similarities else 0.0,
                    'std': np.std(similarities) if similarities else 0.0,
                    'high_similarity_rate': sum(1 for s in similarities if s >= 0.7) / len(similarities) if similarities else 0.0
                },
                # Extract key metrics for logging (NEW)
                'key_comprehensive': {
                    'novelty': comprehensive_metrics.get('novelty', 0),
                    'fcd': comprehensive_metrics.get('fcd', None),
                    'drug_likeness_qed': comprehensive_metrics.get('drug_likeness', {}).get('qed_mean', 0),
                    'scaffold_diversity': comprehensive_metrics.get('scaffold_metrics', {}).get('scaffold_diversity', 0),
                } if comprehensive_metrics else {}
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

