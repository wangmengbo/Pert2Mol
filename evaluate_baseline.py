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
import random
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

from dataloaders.dataset_gdp import create_gdp_dataloaders, create_gdp_rna_dataloaders, create_gdp_image_dataloaders
from dataloaders.dataset_lincs_rna import create_lincs_rna_dataloaders
from dataloaders.dataset_cpgjump import create_cpgjump_dataloaders
from dataloaders.dataset_tahoe import create_tahoe_dataloaders
from dataloaders.dataloader import create_leak_free_dataloaders
from models import ReT_models  # Use standard models for baseline
from train_autoencoder import pert2mol_autoencoder
from utils import AE_SMILES_decoder, regexTokenizer
from encoders import ImageEncoder, RNAEncoder, PairedRNAEncoder
from encoders import dual_rna_image_encoder_separate as dual_rna_image_encoder
from diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType  # Use Gaussian diffusion for baseline
from utils import (collect_all_drugs_from_csv, AE_SMILES_encoder, set_seed,)
from evaluation_utils import (create_three_section_summary, clear_training_features_cache,  
    get_evaluation_cache_key, load_evaluation_results_cache, save_evaluation_results_cache, clear_evaluation_results_cache,
    collect_training_data_with_biological_features_cached, collect_training_data_with_biological_features_original,
    save_comprehensive_intermediate_results, calculate_ground_truth_baseline_metrics, collect_ground_truth_molecules,
    save_metrics_to_csv
)
from evaluation_metrics import calculate_mode_specific_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pure_generation_baseline(model, flow, ae_model, y_features, pad_mask, device, args):
    """Pure generation for baseline method using Gaussian diffusion"""
    
    batch_size = y_features.shape[0]
    shape = (batch_size, 64, 127, 1)
    
    all_generated_smiles = []
    
    # Generate multiple samples per condition
    for sample_idx in range(args.num_samples_per_condition):
        with torch.no_grad():
            # Use Gaussian diffusion p_sample_loop for baseline
            model_kwargs = dict(y=y_features, pad_mask=pad_mask)
            
            # Generate using the baseline's Gaussian diffusion approach
            x = flow.p_sample_loop(
                model, 
                shape, 
                device=device,
                progress=False,
                model_kwargs=model_kwargs
            )
            
            # Decode to SMILES
            x = x.squeeze(-1).permute((0, 2, 1))
            batch_generated = AE_SMILES_decoder(x, ae_model, stochastic=False, k=1)
            all_generated_smiles.extend(batch_generated)
    
    return all_generated_smiles


def run_conventional_retrieval_baseline(test_batch, training_data, y_features, args):
    """Conventional retrieval for baseline - same as main method"""
    
    retrieved_results = []
    target_smiles = test_batch['target_smiles']
    compound_names = test_batch['compound_name']
    batch_size = len(target_smiles)
    
    if not training_data.get('biological_features'):
        # Return empty results if no biological features
        for i in range(batch_size):
            retrieved_results.append({
                'top_k_drugs': [],
                'top_k_similarities': [],
                'smiles_hit_rank': None,
                'compound_hit_rank': None,
                'smiles_in_top_k': False,
                'compound_in_top_k': False,
            })
        return retrieved_results
    
    # Concatenate all training biological features  
    all_training_features = torch.cat(training_data['biological_features'], dim=0)
    
    total_features = all_training_features.shape[0]
    total_smiles = len(training_data['smiles'])
    total_compounds = len(training_data['compound_names'])

    # Calculate similarities in biological space using optimized tensor operations
    query_flat = y_features.flatten(1)  # [batch_size, features]
    training_flat = all_training_features.flatten(1)  # [N, features]
    
    # Normalize for cosine similarity and compute efficiently on GPU
    device = y_features.device
    query_norm = torch.nn.functional.normalize(query_flat, p=2, dim=1)
    training_norm = torch.nn.functional.normalize(training_flat.to(device), p=2, dim=1)
    
    # Vectorized cosine similarity computation
    similarities = torch.mm(query_norm, training_norm.t())  # [batch_size, N_training]
    
    # Get top-k similar conditions for each query using efficient torch.topk
    top_k_similarities, top_k_indices = torch.topk(similarities, args.retrieval_top_k, dim=1, largest=True)
    
    # Move to CPU for result processing
    top_k_similarities = top_k_similarities.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()
    
    # Process results
    for i in range(batch_size):
        target_smile = target_smiles[i]
        target_compound = compound_names[i]
        
        indices = top_k_indices[i]
        sims = top_k_similarities[i]
        
        # Get corresponding drugs from training data
        top_k_smiles = [training_data['smiles'][idx] for idx in indices]
        top_k_compounds = [training_data['compound_names'][idx] for idx in indices]
        
        # Check if target is in top-k
        smiles_hit_rank = None
        compound_hit_rank = None
        
        for rank, (retrieved_smiles, retrieved_compound) in enumerate(zip(top_k_smiles, top_k_compounds)):
            # Check for SMILES match
            try:
                target_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(target_smile), canonical=True)
                retrieved_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(retrieved_smiles), canonical=True)
                if target_canonical == retrieved_canonical and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            except:
                if target_smile == retrieved_smiles and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            
            # Check for compound name match
            if target_compound == retrieved_compound and compound_hit_rank is None:
                compound_hit_rank = rank + 1
            
            if smiles_hit_rank is not None and compound_hit_rank is not None:
                break
        
        retrieved_results.append({
            'top_k_drugs': top_k_smiles,
            'top_k_similarities': sims.tolist(),
            'smiles_hit_rank': smiles_hit_rank,
            'compound_hit_rank': compound_hit_rank,
            'smiles_in_top_k': smiles_hit_rank is not None,
            'compound_in_top_k': compound_hit_rank is not None,
        })
    
    return retrieved_results


def run_retrieval_by_generation_baseline(test_batch, training_data, model, flow, ae_model, 
                                        y_features, pad_mask, device, args):
    """Retrieval by generation for baseline method using Gaussian diffusion"""
    
    retrieved_results = []
    target_smiles = test_batch['target_smiles']
    compound_names = test_batch['compound_name']
    batch_size = len(target_smiles)
    
    # Step 1: Use trained model to generate drug embeddings from biological features
    with torch.no_grad():
        shape = (batch_size, 64, 127, 1)
        
        # Use Gaussian diffusion for baseline generation
        model_kwargs = dict(y=y_features, pad_mask=pad_mask)
        
        # Generate using baseline's Gaussian diffusion approach
        x = flow.p_sample_loop(
            model, 
            shape, 
            device=device,
            progress=False,
            model_kwargs=model_kwargs
        )
        
        # Generated drug latent representations
        generated_drug_latents = x.squeeze(-1).permute((0, 2, 1))
    
    # Step 2: Get embeddings for all drugs in the dataset using the autoencoder
    dataset_smiles = training_data['smiles']
    dataset_compound_names = training_data['compound_names']
    
    # Batch process dataset drugs to get their latent representations
    dataset_latents = []
    batch_size_train = 32
    
    with torch.no_grad():
        for i in range(0, len(dataset_smiles), batch_size_train):
            batch_smiles = dataset_smiles[i:i+batch_size_train]
            train_latents_batch = AE_SMILES_encoder(batch_smiles, ae_model).permute((0, 2, 1))
            dataset_latents.append(train_latents_batch)
        
        dataset_latents = torch.cat(dataset_latents, dim=0)
    
    # Step 3: OPTIMIZED similarity computation using vectorized operations
    # Flatten embeddings for similarity computation
    gen_flat = generated_drug_latents.flatten(1)      # [batch_size, seq_len*dim]
    dataset_flat = dataset_latents.flatten(1)         # [N_drugs, seq_len*dim]
    
    # Normalize for cosine similarity
    gen_norm = torch.nn.functional.normalize(gen_flat, p=2, dim=1)
    dataset_norm = torch.nn.functional.normalize(dataset_flat.to(device), p=2, dim=1)
    
    # Vectorized cosine similarity computation - much faster than loops!
    similarities = torch.mm(gen_norm, dataset_norm.t())  # [batch_size, N_drugs]
    
    # Find top-k for all queries at once
    top_k_similarities, top_k_indices = torch.topk(
        similarities, args.retrieval_by_generation_top_k, dim=1, largest=True
    )
    
    # Move to CPU for result processing
    top_k_similarities = top_k_similarities.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()
    
    # Step 4: Process results
    for i in range(batch_size):
        target_smile = target_smiles[i]
        target_compound = compound_names[i]
        
        indices = top_k_indices[i]
        sims = top_k_similarities[i]
        
        # Extract top-k drugs
        top_k_smiles = [dataset_smiles[idx] for idx in indices]
        top_k_compounds = [dataset_compound_names[idx] for idx in indices]
        
        # Check if target is in top-k
        smiles_hit_rank = None
        compound_hit_rank = None
        
        for rank, (retrieved_smiles, retrieved_compound) in enumerate(zip(top_k_smiles, top_k_compounds)):
            # Check for SMILES match
            try:
                target_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(target_smile), canonical=True)
                retrieved_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(retrieved_smiles), canonical=True)
                if target_canonical == retrieved_canonical and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            except:
                if target_smile == retrieved_smiles and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            
            # Check for compound name match
            if target_compound == retrieved_compound and compound_hit_rank is None:
                compound_hit_rank = rank + 1
            
            if smiles_hit_rank is not None and compound_hit_rank is not None:
                break
        
        retrieved_results.append({
            'top_k_drugs': top_k_smiles,
            'top_k_similarities': sims.tolist(),
            'smiles_hit_rank': smiles_hit_rank,
            'compound_hit_rank': compound_hit_rank,
            'smiles_in_top_k': smiles_hit_rank is not None,
            'compound_in_top_k': compound_hit_rank is not None,
        })
    
    # Return both retrieval results and generated embeddings
    return retrieved_results, generated_drug_latents.cpu().numpy(), dataset_latents.cpu().numpy()


def run_three_mode_evaluation_baseline(test_batch, training_data, model, flow, ae_model, 
                                      image_encoder, rna_encoder, y_features, pad_mask, device, args):
    """Enhanced three-mode evaluation for baseline with comprehensive intermediate result saving"""
    
    batch_size = len(test_batch['target_smiles'])
    results_by_mode = {'pure_generation': [], 'conventional_retrieval': [], 'retrieval_by_generation': []}
    
    # Save additional data for comprehensive analysis
    intermediate_data = {
        'batch_biological_features': y_features.cpu().numpy() if y_features is not None else None,
        'batch_metadata': {
            'batch_size': batch_size,
            'target_smiles': test_batch['target_smiles'],
            'compound_names': test_batch['compound_name'],
        }
    }
    
    # Mode 1: Pure Generation with enhanced data collection
    if args.run_generation:
        generated_smiles_list = run_pure_generation_baseline(model, flow, ae_model, y_features, pad_mask, device, args)
        
        # Save all generated samples, not just the best one
        samples_per_condition = args.num_samples_per_condition
        all_generated_per_condition = []
        
        for i in range(batch_size):
            # Get all samples for this condition
            condition_samples = []
            for sample_idx in range(samples_per_condition):
                idx = i * samples_per_condition + sample_idx
                if idx < len(generated_smiles_list):
                    condition_samples.append(generated_smiles_list[idx])
            
            all_generated_per_condition.append(condition_samples)
            
            # Process for validity and select best
            valid_samples = []
            for smiles in condition_samples:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        valid_samples.append((smiles, canonical, True))
                    else:
                        valid_samples.append((smiles, smiles, False))
                except:
                    valid_samples.append((smiles, smiles, False))
            
            # Choose best sample
            valid_only = [s for s in valid_samples if s[2]]
            if valid_only:
                best_sample = valid_only[0]
            else:
                best_sample = valid_samples[0] if valid_samples else ("", "", False)
            
            result = {
                'sample_id': i + 1,
                'method': 'pure_generation',
                'target_smiles': test_batch['target_smiles'][i],
                'compound_name': test_batch['compound_name'][i],
                'generated_smiles': best_sample[1],
                'is_valid': best_sample[2],
                'all_samples': condition_samples,  # Save all generated samples
                'valid_samples': [s[1] for s in valid_samples if s[2]],  # Save all valid samples
                'confidence': 0.8 if best_sample[2] else 0.1,
                'num_valid_samples': len(valid_only),
                'validity_rate': len(valid_only) / len(condition_samples) if condition_samples else 0
            }
            results_by_mode['pure_generation'].append(result)
        
        # Save additional generation data
        intermediate_data['generation_data'] = {
            'all_generated_per_condition': all_generated_per_condition,
            'total_generated': len(generated_smiles_list),
            'samples_per_condition': samples_per_condition
        }
    
    # Mode 2: Conventional Retrieval (unchanged)
    if args.run_conventional_retrieval:
        retrieval_results = run_conventional_retrieval_baseline(test_batch, training_data, y_features, args)
        
        for i in range(batch_size):
            retrieval_result = retrieval_results[i] if i < len(retrieval_results) else {}
            
            # Get best retrieved drug
            if retrieval_result.get('top_k_drugs'):
                best_retrieved = retrieval_result['top_k_drugs'][0]
                confidence = retrieval_result['top_k_similarities'][0] if retrieval_result['top_k_similarities'] else 0.0
                is_valid = True
            else:
                best_retrieved = ""
                confidence = 0.0
                is_valid = False
            
            result = {
                'sample_id': i + 1,
                'method': 'conventional_retrieval',
                'target_smiles': test_batch['target_smiles'][i],
                'compound_name': test_batch['compound_name'][i],
                'generated_smiles': best_retrieved,
                'is_valid': is_valid,
                'confidence': float(confidence),
            }
            result.update(retrieval_result)
            results_by_mode['conventional_retrieval'].append(result)
    
    # Mode 3: Retrieval by Generation with enhanced embedding storage
    generated_embeddings = None
    reference_embeddings = None
    
    if args.run_retrieval_by_generation:
        retrieval_by_gen_results, generated_embeddings, reference_embeddings = run_retrieval_by_generation_baseline(
            test_batch, training_data, model, flow, ae_model, y_features, pad_mask, device, args
        )
        
        for i in range(batch_size):
            retrieval_by_gen_result = retrieval_by_gen_results[i] if i < len(retrieval_by_gen_results) else {}
            
            if retrieval_by_gen_result.get('top_k_drugs'):
                best_model_retrieved = retrieval_by_gen_result['top_k_drugs'][0]
                confidence = retrieval_by_gen_result['top_k_similarities'][0] if retrieval_by_gen_result['top_k_similarities'] else 0.0
                is_valid = True
            else:
                best_model_retrieved = ""
                confidence = 0.0
                is_valid = False
            
            result = {
                'sample_id': i + 1,
                'method': 'retrieval_by_generation',
                'target_smiles': test_batch['target_smiles'][i],
                'compound_name': test_batch['compound_name'][i],
                'generated_smiles': best_model_retrieved,
                'is_valid': is_valid,
                'confidence': float(confidence),
            }
            result.update(retrieval_by_gen_result)
            results_by_mode['retrieval_by_generation'].append(result)
        
        # Save embedding data
        intermediate_data['embedding_data'] = {
            'generated_embeddings_shape': generated_embeddings.shape if generated_embeddings is not None else None,
            'reference_embeddings_shape': reference_embeddings.shape if reference_embeddings is not None else None,
        }
    
    return results_by_mode, generated_embeddings, reference_embeddings, intermediate_data


def detect_modalities_baseline(args, gene_count_matrix):
    """Detect available modalities for evaluation - same as main method"""
    has_rna = gene_count_matrix is not None and len(gene_count_matrix) > 0
    has_imaging = args.image_json_path is not None and os.path.exists(args.image_json_path)
    
    if not has_rna and not has_imaging:
        raise ValueError("At least one modality (RNA or imaging) must be available")
    
    modality_info = []
    if has_rna:
        modality_info.append("RNA")
    if has_imaging:
        modality_info.append("Imaging")
        
    logger.info(f"Detected modalities for evaluation: {' + '.join(modality_info)}")
    return has_rna, has_imaging


def run_independent_evaluation_baseline(args):
    """Enhanced evaluation for baseline method with comprehensive intermediate result saving"""
    
    logger.info("Setting up enhanced three-mode evaluation for BASELINE method with comprehensive caching...")
    
    # Set up device and seed
    set_seed(args.seed)
    logger.info(f"NumPy random test: {np.random.randint(0, 1000)}")
    logger.info(f"Python random test: {random.randint(0, 1000)}")
    logger.info(f"PyTorch random test: {torch.randint(0, 1000, (1,)).item()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}, seed: {args.seed}")
    
    # Check for cached results first
    cache_key = get_evaluation_cache_key(args)
    
    if not args.force_recompute_cache and not args.no_cache:
        cached_results = load_evaluation_results_cache(cache_key, args.cache_dir)
        if cached_results is not None:
            logger.info("Using cached evaluation results. Recomputing only metrics...")
            
            # Recompute metrics with potentially updated metric functions
            mode_metrics = calculate_mode_specific_metrics(
                cached_results['all_results_by_mode'], 
                cached_results['training_data']
            )
            
            # Update metrics in cached results
            cached_results['mode_metrics'] = mode_metrics
            
            return cached_results

    # Validate evaluation modes
    if not any([args.run_generation, args.run_conventional_retrieval, args.run_retrieval_by_generation]):
        logger.warning("No evaluation mode specified. Enabling conventional retrieval mode by default.")
        args.run_conventional_retrieval = True
    
    # Load gene count matrix and set up data loaders (existing logic)
    if args.gene_count_matrix_path is not None:
        gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)
        logger.info(f"Loaded gene count matrix: {gene_count_matrix.shape}")
    else:
        gene_count_matrix = None
    
    has_rna, has_imaging = detect_modalities_baseline(args, gene_count_matrix)
    logger.info(f"Using modalities - RNA: {has_rna}, Imaging: {has_imaging}")

    # Set up data loader function based on dataset type
    if args.dataset == "gdp":
        create_data_loader = create_gdp_dataloaders
    elif args.dataset == "lincs_rna":
        create_data_loader = create_lincs_rna_dataloaders
        if gene_count_matrix is not None:
            gene_count_matrix = gene_count_matrix.T
    elif args.dataset == "cpgjump":
        create_data_loader = create_cpgjump_dataloaders
    elif args.dataset == "tahoe":
        create_data_loader = create_tahoe_dataloaders
        if gene_count_matrix is not None:
            gene_count_matrix = gene_count_matrix.T
    elif args.dataset == "other":
        if args.transpose_gene_count_matrix and gene_count_matrix is not None:
            logger.info("Transposing gene count matrix as per --transpose-gene-count-matrix flag.")
            gene_count_matrix = gene_count_matrix.T
        create_data_loader = create_leak_free_dataloaders
        if args.stratify_by is None:
            args.stratify_by = [args.compound_name_label]
            if args.cell_type_label is not None:
                args.stratify_by.append(args.cell_type_label)
        logger.info(f"Stratifying train/test split by {args.stratify_by}")
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")
    
    # Load metadata if provided
    metadata_control = metadata_drug = None
    if args.metadata_control_path is not None:
        metadata_control = pd.read_csv(args.metadata_control_path)
    if args.metadata_drug_path is not None:
        metadata_drug = pd.read_csv(args.metadata_drug_path)
    
    # Create data loaders
    train_loader, test_loader = create_data_loader(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug, 
        gene_count_matrix=gene_count_matrix,
        image_json_path=args.image_json_path,
        drug_data_path=args.drug_data_path,
        raw_drug_csv_path=args.raw_drug_csv_path,
        batch_size=args.batch_size,
        shuffle=True,
        shuffle_test=True,
        compound_name_label=args.compound_name_label,
        exclude_cell_lines=args.exclude_cell_lines,
        exclude_drugs=args.exclude_drugs,
        debug_mode=args.debug_mode,
        debug_samples=args.debug_samples,
        include_cell_lines=args.include_cell_lines,
        include_drugs=args.include_drugs,
        random_state=args.seed,
        split_train_test=True,
        test_size=args.test_size,
        return_datasets=False,
    )
    
    eval_loader = test_loader
    total_test_batches = len(eval_loader)
    
    # Calculate evaluation scope - cleaner logic
    if args.max_test_samples is not None:
        eval_batches = min(
            (args.max_test_samples + args.batch_size - 1) // args.batch_size,  # Ceiling division
            total_test_batches
        )
        logger.info(f"Evaluating up to {args.max_test_samples} test samples: {eval_batches}/{total_test_batches} batches")
    elif args.eval_portion is not None:
        eval_batches = max(1, int(total_test_batches * args.eval_portion))
        logger.info(f"Evaluating {args.eval_portion*100:.1f}% of test data: {eval_batches}/{total_test_batches} batches")
    elif args.max_eval_batches is not None:
        eval_batches = min(args.max_eval_batches, total_test_batches)
        logger.info(f"Evaluating {eval_batches}/{total_test_batches} batches (max_eval_batches: {args.max_eval_batches})")
    else:
        eval_batches = total_test_batches
        logger.info(f"Evaluating all {eval_batches} test batches")
    
    # Collect ground truth molecules for baseline analysis
    ground_truth_data = collect_ground_truth_molecules(eval_loader, eval_batches)

    # Load models if needed
    model = image_encoder = rna_encoder = ae_model = flow = None
    
    if args.run_generation or args.run_retrieval_by_generation or args.run_conventional_retrieval:
        logger.info("Loading BASELINE models for evaluation...")
        
        # Create ReT model for baseline (using standard models, not SRA)
        if args.run_generation or args.run_retrieval_by_generation:
            latent_size = 127
            in_channels = 64
            cross_attn = 192
            condition_dim = 192
            
            # Use standard ReT model for baseline
            model = ReT_models[args.model](
                input_size=latent_size,
                in_channels=in_channels,
                cross_attn=cross_attn,
                condition_dim=condition_dim,
                learn_sigma=False
            ).to(device)
            logger.info(f"Using baseline model: {args.model}")
            
            # Load checkpoint
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

            model.load_state_dict(checkpoint['model'], strict=False)
            model.eval()
            logger.info(f"Loaded baseline ReT model from {args.ckpt}")
        else:
            # For conventional retrieval only, we still need encoders
            checkpoint = torch.load(args.ckpt, map_location='cpu')
        
        image_encoder = None
        rna_encoder = None

        if has_imaging:
            image_encoder = ImageEncoder(img_channels=4, output_dim=128).to(device)
            image_encoder.load_state_dict(checkpoint['image_encoder'], strict=True)
            image_encoder.eval()
            logger.info("Loaded ImageEncoder for baseline evaluation")
        else:
            logger.info('Skipping ImageEncoder - no imaging data available')

        if has_rna:
            if args.paired_rna_encoder:
                rna_encoder = PairedRNAEncoder(
                    input_dim=gene_count_matrix.shape[0] if gene_count_matrix is not None else 2000, 
                    output_dim=128, 
                    dropout=0.1, 
                    num_heads=4, 
                    gene_embed_dim=512, 
                    num_self_attention_layers=1, 
                    num_cross_attention_layers=2,
                    use_bidirectional_cross_attn=True
                ).to(device)
            else:
                rna_encoder = RNAEncoder(
                    input_dim=gene_count_matrix.shape[0] if gene_count_matrix is not None else 2000,
                    output_dim=64,
                    dropout=0.1
                ).to(device)
            
            rna_encoder.load_state_dict(checkpoint['rna_encoder'], strict=True) 
            rna_encoder.eval()
            logger.info("Loaded RNAEncoder for baseline evaluation")
        else:
            logger.info('Skipping RNAEncoder - no RNA data available')
        
        # Setup autoencoder (needed for generation and retrieval by generation)
        if args.run_generation or args.run_retrieval_by_generation:
            ae_config = {
                'bert_config_decoder': './config_decoder.json',
                'bert_config_encoder': './config_encoder.json',
                'embed_dim': 256,
            }
            tokenizer = regexTokenizer(vocab_path='./dataloaders/vocab_bpe_300_sc.txt', max_len=127)
            ae_model = pert2mol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer, use_linear=True)
            
            if args.vae:
                ae_checkpoint = torch.load(args.vae, map_location='cpu')
                try:
                    ae_state_dict = ae_checkpoint['model']
                except:
                    ae_state_dict = ae_checkpoint['state_dict']
                ae_model.load_state_dict(ae_state_dict, strict=False)
            
            for param in ae_model.parameters():
                param.requires_grad = False
            ae_model = ae_model.to(device)
            ae_model.eval()
            
            # Setup Gaussian diffusion for baseline generation
            if args.run_generation:
                betas = get_named_beta_schedule("linear", num_diffusion_timesteps=1000)
                flow = GaussianDiffusion(
                    betas=betas,
                    model_mean_type=ModelMeanType.EPSILON,  # Model predicts noise
                    model_var_type=ModelVarType.FIXED_SMALL,
                    loss_type=LossType.MSE
                )
                logger.info("Using Gaussian diffusion for baseline generation")
        
        logger.info("Baseline models loaded successfully")
    
    # Collect training data based on enabled modes
    training_data = {'smiles': [], 'compound_names': []}
    
    # Priority: conventional retrieval needs biological features, others need drug lists
    if args.run_conventional_retrieval:
        # Need biological features for conventional retrieval
        if args.no_cache:
            # Use original function without caching
            training_data = collect_training_data_with_biological_features_original(
                args, train_loader, image_encoder, rna_encoder, device,
                max_batches=args.max_training_batches,
                max_training_portion=args.max_training_portion
            )
        else:
            # Use cached version
            training_data = collect_training_data_with_biological_features_cached(
                args, train_loader, image_encoder, rna_encoder, device,
                max_batches=args.max_training_batches,
                max_training_portion=args.max_training_portion,
                force_recompute=args.force_recompute_cache,
                cache_dir=args.cache_dir,
                has_rna=has_rna,
                has_imaging=has_imaging
            )
    elif args.run_retrieval_by_generation:
        # MUST use comprehensive drug list from CSV for retrieval by generation
        if args.raw_drug_csv_path and os.path.exists(args.raw_drug_csv_path):
            training_drugs_from_csv = collect_all_drugs_from_csv(
                args.raw_drug_csv_path, 
                compound_name_label=args.compound_name_label,
                smiles_label='canonical_smiles'
            )
            training_data = {
                'smiles': training_drugs_from_csv['smiles'],
                'compound_names': training_drugs_from_csv['compound_names']
            }
            logger.info(f"Loaded {len(training_data['compound_names'])} unique compounds for retrieval by generation")
        else:
            raise ValueError("--raw-drug-csv-path is required for retrieval by generation mode")
    else:
        # Minimal data collection for generation mode only
        logger.info("Collecting minimal training data for novelty evaluation...")
        max_training_batches = 20 if args.debug_mode else 50
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Collecting training SMILES")):
            training_data['smiles'].extend(batch['target_smiles'])
            if batch_idx >= max_training_batches:
                break
    
    # Enhanced evaluation loop
    all_results_by_mode = {'pure_generation': [], 'conventional_retrieval': [], 'retrieval_by_generation': []}
    all_intermediate_data = []  # Store intermediate data from each batch
    
    # For retrieval by generation: collect embeddings and metadata
    all_generated_embeddings = []
    all_sample_metadata = []
    reference_drug_embeddings = None
    
    eval_start_time = time.time()
    processed_samples = 0
    
    logger.info(f"Starting enhanced three-mode evaluation on BASELINE test data...")
    logger.info(f"Modes enabled: Generation={args.run_generation}, Conventional Retrieval={args.run_conventional_retrieval}, Retrieval by Generation={args.run_retrieval_by_generation}")
    
    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating test data")):
        if batch_idx >= eval_batches:
            break
        
        # Additional check for max_test_samples
        if args.max_test_samples is not None and args.max_test_samples > 0 and processed_samples >= args.max_test_samples:
            break
        
        # Encode biological features
        with torch.no_grad():
            if args.legacy_dummy_tensors:
                y_features, pad_mask = dual_rna_image_encoder(
                    batch['control_images'], batch['treatment_images'], 
                    batch['control_transcriptomics'], batch['treatment_transcriptomics'],
                    image_encoder, rna_encoder, device
                )
            else:
                y_features, pad_mask = dual_rna_image_encoder(
                    batch['control_images'], batch['treatment_images'], 
                    batch['control_transcriptomics'], batch['treatment_transcriptomics'],
                    image_encoder, rna_encoder, device, has_rna, has_imaging
                )
        
        # Run enhanced multi-mode evaluation for this batch using baseline
        batch_results, generated_embeddings, reference_embeddings, intermediate_data = run_three_mode_evaluation_baseline(
            batch, training_data, model, flow, ae_model, 
            image_encoder, rna_encoder, y_features, pad_mask, device, args
        )
        
        # Store intermediate data
        all_intermediate_data.append(intermediate_data)
        
        # Store reference drug embeddings (only need to do this once)
        if args.run_retrieval_by_generation and reference_drug_embeddings is None and reference_embeddings is not None:
            reference_drug_embeddings = reference_embeddings
        
        # Accumulate generated embeddings for retrieval by generation
        if args.run_retrieval_by_generation and generated_embeddings is not None:
            all_generated_embeddings.append(generated_embeddings)
        
        # Accumulate results
        for mode, results in batch_results.items():
            all_results_by_mode[mode].extend(results)
        
        # Collect sample metadata for retrieval by generation mode
        if args.run_retrieval_by_generation:
            for i in range(len(batch['target_smiles'])):
                sample_metadata = {
                    'batch_idx': batch_idx,
                    'sample_idx_in_batch': i,
                    'target_smiles': batch['target_smiles'][i],
                    'compound_name': batch['compound_name'][i],
                    'global_sample_idx': len(all_sample_metadata)
                }
                # Add any other batch fields you want to preserve
                for key in batch.keys():
                    if key not in ['control_images', 'treatment_images', 'control_transcriptomics', 'treatment_transcriptomics']:
                        if isinstance(batch[key], list) and i < len(batch[key]):
                            sample_metadata[key] = batch[key][i]
                
                all_sample_metadata.append(sample_metadata)
        
        processed_samples += len(batch['target_smiles'])
    
    eval_time = time.time() - eval_start_time
    logger.info(f"Baseline evaluation completed in {eval_time:.1f}s")
    logger.info(f"Processed {processed_samples} test samples")
    
    # Calculate mode-specific metrics
    mode_metrics = calculate_mode_specific_metrics(all_results_by_mode, training_data)
    
    # Save embeddings and metadata for retrieval by generation mode
    if args.run_retrieval_by_generation:
        # Concatenate all generated embeddings
        if all_generated_embeddings:
            all_generated_embeddings_matrix = np.vstack(all_generated_embeddings)
            
            # Save generated drug embeddings (one row per test sample)
            embeddings_path = os.path.join(args.output_dir, f"{args.output_prefix}_generated_embeddings.npy")
            np.save(embeddings_path, all_generated_embeddings_matrix)
            logger.info(f"Saved generated drug embeddings: {embeddings_path} (shape: {all_generated_embeddings_matrix.shape})")
        
        # Save reference drug embeddings (one row per reference drug, in CSV order)
        if reference_drug_embeddings is not None:
            reference_embeddings_path = os.path.join(args.output_dir, f"{args.output_prefix}_reference_embeddings.npy")
            np.save(reference_embeddings_path, reference_drug_embeddings)
            logger.info(f"Saved reference drug embeddings: {reference_embeddings_path} (shape: {reference_drug_embeddings.shape})")
        
        # Save sample metadata (in corresponding order to generated embeddings)
        if all_sample_metadata:
            metadata_df = pd.DataFrame(all_sample_metadata)
            metadata_path = os.path.join(args.output_dir, f"{args.output_prefix}_sample_metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            logger.info(f"Saved sample metadata: {metadata_path} ({len(metadata_df)} samples)")
        
        # Save reference drug metadata (in corresponding order to reference embeddings)
        if training_data.get('smiles') and training_data.get('compound_names'):
            reference_metadata = pd.DataFrame({
                'smiles': training_data['smiles'],
                'compound_name': training_data['compound_names'],
                'reference_drug_idx': range(len(training_data['smiles']))
            })
            reference_metadata_path = os.path.join(args.output_dir, f"{args.output_prefix}_reference_metadata.csv")
            reference_metadata.to_csv(reference_metadata_path, index=False)
            logger.info(f"Saved reference drug metadata: {reference_metadata_path} ({len(reference_metadata)} drugs)")
    
    # Create metadata
    metadata = {
        'evaluation_mode': 'three_mode_comprehensive_enhanced_baseline',
        'baseline_method': True,
        'diffusion_type': 'gaussian',
        'modes_enabled': {
            'pure_generation': args.run_generation,
            'conventional_retrieval': args.run_conventional_retrieval,
            'retrieval_by_generation': args.run_retrieval_by_generation
        },
        'dataset': args.dataset,
        'retrieval_top_k': args.retrieval_top_k,
        'retrieval_by_generation_top_k': args.retrieval_by_generation_top_k if args.run_retrieval_by_generation else None,
        'eval_portion': args.eval_portion,
        'max_eval_batches': args.max_eval_batches,
        'max_test_samples': args.max_test_samples,
        'total_test_batches': total_test_batches,
        'evaluated_batches': eval_batches,
        'processed_samples': processed_samples,
        'global_seed': args.seed,
        'random_state': args.seed,
        'evaluation_time': eval_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'intermediate_data_batches': len(all_intermediate_data),
        'embedding_files': {
            'generated_embeddings': f"{args.output_prefix}_generated_embeddings.npy" if args.run_retrieval_by_generation else None,
            'reference_embeddings': f"{args.output_prefix}_reference_embeddings.npy" if args.run_retrieval_by_generation else None,
            'sample_metadata': f"{args.output_prefix}_sample_metadata.csv" if args.run_retrieval_by_generation else None,
            'reference_metadata': f"{args.output_prefix}_reference_metadata.csv" if args.run_retrieval_by_generation else None,
            'biological_features': f"{args.output_prefix}_batch_biological_features.npz",
            'generation_analysis': f"{args.output_prefix}_generation_analysis.json" if args.run_generation else None,
        }
    }
    
    # Package comprehensive results for caching AND returning
    comprehensive_data = {
        'metadata': metadata,
        'training_data': training_data,
        'all_results_by_mode': all_results_by_mode,
        'mode_metrics': mode_metrics,
        'results_by_mode': all_results_by_mode,  # For backward compatibility
        # Enhanced cache data
        'all_intermediate_data': all_intermediate_data,
        'all_generated_embeddings': all_generated_embeddings if args.run_retrieval_by_generation else None,
        'all_sample_metadata': all_sample_metadata if args.run_retrieval_by_generation else None,
        'reference_drug_embeddings': reference_drug_embeddings if args.run_retrieval_by_generation else None,
    }
    
    # Save comprehensive intermediate results
    intermediate_save_info = save_comprehensive_intermediate_results(args, comprehensive_data, all_intermediate_data)
    comprehensive_data['intermediate_save_info'] = intermediate_save_info

    # Calculate and save ground truth baseline metrics
    try:
        ground_truth_baseline = calculate_ground_truth_baseline_metrics(ground_truth_data['smiles'])
        comprehensive_data['ground_truth_baseline'] = ground_truth_baseline
        logger.info(f"Ground truth baseline: {len(ground_truth_data['smiles'])} molecules")
    except Exception as e:
        logger.warning(f"Failed to calculate ground truth baseline: {e}")
        comprehensive_data['ground_truth_baseline'] = {}
    finally:
        comprehensive_data['ground_truth_data'] = ground_truth_data

    # Save to cache (unless disabled)
    if not args.no_cache:
        try:
            save_evaluation_results_cache(cache_key, comprehensive_data, args.cache_dir)
            logger.info(f"Comprehensive baseline evaluation results cached with key: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save evaluation cache: {e}")
    
    return comprehensive_data


def main():
    parser = argparse.ArgumentParser(description="Three-mode comprehensive evaluation of BASELINE drug generation models")
    
    # Three evaluation modes
    parser.add_argument("--run-generation", action="store_true", 
                       help="Run pure generation mode")
    parser.add_argument("--run-conventional-retrieval", action="store_true",
                       help="Run conventional retrieval mode (biological similarity-based)")
    parser.add_argument("--run-retrieval-by-generation", action="store_true",
                       help="Run retrieval by generation mode (model-based drug mapping)")
    
    parser.add_argument("--legacy-dummy-tensors", action="store_true",
                   help="Use legacy dummy tensors for missing modalities (for older checkpoints)")

    # Generation parameters
    parser.add_argument("--generation-steps", type=int, default=50, 
                       help="Number of diffusion steps for generation")
    parser.add_argument("--num-samples-per-condition", type=int, default=3,
                       help="Number of samples to generate per biological condition")
    
    # Retrieval parameters
    parser.add_argument("--retrieval-top-k", type=int, default=5,
                       help="Top-k drugs to retrieve for conventional retrieval evaluation")
    parser.add_argument("--retrieval-by-generation-top-k", type=int, default=20,
                       help="Top-k drugs to retrieve for retrieval by generation evaluation")
    
    # Evaluation scope
    parser.add_argument("--max-training-portion", type=float, default=None,
                   help="Maximum portion of training data to use for retrieval evaluation (0.0-1.0)")
    parser.add_argument("--max-training-batches", type=int, default=None, 
                    help="Maximum number of training batches to use for collecting training data")

    parser.add_argument("--max-test-samples", type=int, default=None, 
                    help="Maximum number of test samples to evaluate (0 = no limit)")
    parser.add_argument("--eval-portion", type=float, default=None,
                       help="Portion of test data to evaluate (0.0-1.0)")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                       help="Maximum number of batches to evaluate (0 = no limit)")

    # Model and checkpoint arguments
    parser.add_argument("--model", type=str, choices=list(ReT_models.keys()), 
                       default="pert2mol", help="Model architecture")
    parser.add_argument("--ckpt", type=str, default=None,
                       help="Path to model checkpoint (required for generation/retrieval-by-generation)")
    parser.add_argument("--vae", type=str, 
                       default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/dataloaders/checkpoint_autoencoder.ckpt",
                       help="Path to autoencoder checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=["gdp", "lincs_rna", "cpgjump", "tahoe", "other"], default="gdp")
    parser.add_argument("--image-json-path", type=str, default=None)
    parser.add_argument("--drug-data-path", type=str, default=None)
    parser.add_argument("--raw-drug-csv-path", type=str, default=None)
    parser.add_argument("--metadata-control-path", type=str, default=None)
    parser.add_argument("--metadata-drug-path", type=str, default=None)
    parser.add_argument("--gene-count-matrix-path", type=str, default=None)
    parser.add_argument("--compound-name-label", type=str, default="compound")
    parser.add_argument("--cell-type-label", type=str, default="cell_line")
    parser.add_argument("--stratify-by", type=str, nargs='+', default=None)
    parser.add_argument("--transpose-gene-count-matrix", action="store_true", default=False)
    parser.add_argument("--paired-rna-encoder", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)

    # Data filtering arguments
    parser.add_argument("--exclude-cell-lines", type=str, nargs='+', default=None)
    parser.add_argument("--exclude-drugs", type=str, nargs='+', default=None)
    parser.add_argument("--include-cell-lines", type=str, nargs='+', default=None)
    parser.add_argument("--include-drugs", type=str, nargs='+', default=None)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--debug-samples", type=int, default=100)
    
    # Generation parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--global-seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./evaluation_results_baseline")
    parser.add_argument("--verbose", action="store_true")
    
    # Cache control arguments
    parser.add_argument("--cache-dir", type=str, default=None,
                    help="Directory to store all cache files (biological features + evaluation results)")
    parser.add_argument("--force-recompute-cache", action="store_true",
                    help="Force recomputation of both biological features and evaluation results")
    parser.add_argument("--clear-cache", action="store_true",
                    help="Clear all cached files and exit")
    parser.add_argument("--no-cache", action="store_true",
                    help="Disable all caching (always recompute everything)")
    parser.add_argument("--recompute-metrics-only", action="store_true",
                    help="Only recompute metrics from cached results (fastest option)")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    
    if args.clear_cache:
        clear_training_features_cache(args.cache_dir)
        clear_evaluation_results_cache(args.cache_dir)
        logger.info("All caches cleared. Exiting.")
        return

    if args.recompute_metrics_only:
        args.force_recompute_cache = False
        args.no_cache = False

    # Validate that at least one mode is enabled
    if not any([args.run_generation, args.run_conventional_retrieval, args.run_retrieval_by_generation]):
        logger.warning("No evaluation mode specified. Enabling conventional retrieval mode by default.")
        args.run_conventional_retrieval = True
    
    # Validate checkpoint requirement
    if any([args.run_generation, args.run_retrieval_by_generation, args.run_conventional_retrieval]) and args.ckpt is None:
        raise ValueError("Must specify --ckpt when using any evaluation mode (all modes require trained encoders)")
  
    # Create output directory
    if args.output_prefix is not None:
        args.output_dir = os.path.join(args.output_dir, args.output_prefix)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.cache_dir is None:
        args.cache_dir = args.output_dir
    else:
        os.makedirs(args.cache_dir, exist_ok=True)

    logger.info(f"Starting BASELINE three-mode evaluation:")
    logger.info(f"  Pure Generation: {args.run_generation}")
    logger.info(f"  Conventional Retrieval: {args.run_conventional_retrieval}")
    logger.info(f"  Retrieval by Generation: {args.run_retrieval_by_generation}")

    if args.output_prefix == "evaluation":
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output_prefix = f"{args.dataset}_baseline_{args.seed}_{timestamp}"

    logger.info(f"Starting BASELINE evaluation with settings:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Eval portion: {args.eval_portion*100:.1f}%" if args.eval_portion is not None else "  Eval portion: all data")
    logger.info(f"  Max batches: {args.max_eval_batches}" if args.max_eval_batches is not None else "  Max batches: unlimited")
    logger.info(f"  Retrieval Top-K: {args.retrieval_top_k}")
    logger.info(f"  Generation steps: {args.generation_steps}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {args.output_dir}")
    
    # Run three-mode evaluation for baseline
    comprehensive_data = run_independent_evaluation_baseline(args)
    
    # Save comprehensive results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_json = os.path.join(args.output_dir, f"{args.output_prefix}_comprehensive.json")

    with open(output_json, 'w') as f:
        json.dump(comprehensive_data, f, indent=2, default=str)
    
    # Create three-section summary
    summary_file = create_three_section_summary(comprehensive_data, args)
    
    csv_file = save_metrics_to_csv(comprehensive_data, args)

    # Print final summary
    metadata = comprehensive_data['metadata']
    results_by_mode = comprehensive_data['results_by_mode']
    mode_metrics = comprehensive_data['mode_metrics']
    
    logger.info(f"\n{'='*80}")
    logger.info("BASELINE THREE-MODE EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Evaluation time: {metadata['evaluation_time']:.1f}s")
    
    # Report results for each enabled mode
    for mode_name, display_name in [
        ('pure_generation', 'Pure Generation'),
        ('conventional_retrieval', 'Conventional Retrieval'), 
        ('retrieval_by_generation', 'Retrieval by Generation')
    ]:
        if results_by_mode.get(mode_name):
            results = results_by_mode[mode_name]
            valid_count = sum(1 for r in results if r.get('is_valid'))
            total_count = len(results)
            logger.info(f"{display_name}: {valid_count}/{total_count} valid results")
            
            if mode_name == 'pure_generation':
                # Basic comprehensive metrics
                if 'key_comprehensive' in mode_metrics.get(mode_name, {}):
                    key_comp = mode_metrics[mode_name]['key_comprehensive']
                    if key_comp:
                        logger.info(f"  Novelty: {key_comp.get('novelty', 0):.3f}")
                        if key_comp.get('fcd') is not None:
                            logger.info(f"  FCD: {key_comp.get('fcd', 0):.3f}")
                        logger.info(f"  QED: {key_comp.get('drug_likeness_qed', 0):.3f}")
                        logger.info(f"  Scaffold Diversity: {key_comp.get('scaffold_diversity', 0):.3f}")
                
                # Add validity and uniqueness logging
                if 'comprehensive_metrics' in mode_metrics.get(mode_name, {}):
                    comp_metrics = mode_metrics[mode_name]['comprehensive_metrics']
                    if comp_metrics:
                        logger.info(f"  Validity: {comp_metrics.get('validity', 0):.3f}")
                        logger.info(f"  Uniqueness: {comp_metrics.get('uniqueness', 0):.3f}")
                
                # Target similarity (Tanimoto coefficients)
                if 'target_similarity' in mode_metrics.get(mode_name, {}):
                    target_sim = mode_metrics[mode_name]['target_similarity']
                    logger.info(f"  Mean Target Similarity (Tanimoto): {target_sim.get('mean', 0):.3f}")

            elif mode_name in ['conventional_retrieval', 'retrieval_by_generation'] and 'retrieval_accuracy' in mode_metrics.get(mode_name, {}):
                acc = mode_metrics[mode_name]['retrieval_accuracy']['smiles_top_k_accuracy']
                if mode_name == 'retrieval_by_generation':
                    logger.info(f"  Top-{args.retrieval_by_generation_top_k} accuracy: {acc:.3f}")
                else:
                    logger.info(f"  Top-{args.retrieval_top_k} accuracy: {acc:.3f}")

    logger.info(f"\nResults saved:")
    logger.info(f"  Comprehensive: {output_json}")
    logger.info(f"  Three-section summary: {summary_file}")
    logger.info(f"  CSV metrics: {csv_file}")


if __name__ == "__main__":
    main()