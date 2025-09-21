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

from dataloaders.dataset_gdp import create_gdp_dataloaders, create_gdp_rna_dataloaders, create_gdp_image_dataloaders
from dataloaders.dataset_lincs_rna import create_lincs_rna_dataloaders
from dataloaders.dataset_cpgjump import create_cpgjump_dataloaders
from dataloaders.dataset_tahoe import create_tahoe_dataloaders
from dataloaders.dataloader import create_leak_free_dataloaders
from models import ReT_models
from models_sra import ReT_SRA_models
from train_autoencoder import pert2mol_autoencoder
from utils import AE_SMILES_decoder, regexTokenizer
from encoders import ImageEncoder, RNAEncoder, PairedRNAEncoder
from encoders import dual_rna_image_encoder_separate as dual_rna_image_encoder
from diffusion.rectified_flow import create_rectified_flow
from utils import (standardize_smiles_batch_with_stats, standardize_smiles_single,
    collect_all_drugs_from_csv, AE_SMILES_encoder)
from evaluation_utils import (calculate_comprehensive_generation_metrics, calculate_drug_likeness_metrics, 
    calculate_scaffold_metrics, calculate_fragment_similarity, calculate_distribution_metrics, 
    calculate_coverage_metrics, create_three_section_summary, calculate_mode_specific_metrics, 
    calculate_additional_generation_metrics, diversity_analysis, calculate_comprehensive_molecular_properties
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pure_generation(model, flow, ae_model, y_features, pad_mask, device, args):
    """Pure generation: generate new molecules without retrieval"""
    
    batch_size = y_features.shape[0]
    shape = (batch_size, 64, 127, 1)
    
    all_generated_smiles = []
    
    # Generate multiple samples per condition
    for sample_idx in range(args.num_samples_per_condition):
        with torch.no_grad():
            x = torch.randn(*shape, device=device)
            dt = 1.0 / args.generation_steps
            
            for i in range(args.generation_steps):
                t = torch.full((batch_size,), i * dt, device=device)
                t_discrete = (t * 999).long()
                
                velocity = model(x, t_discrete, y=y_features, pad_mask=pad_mask)
                if isinstance(velocity, tuple):
                    velocity = velocity[0]
                
                if velocity.shape[1] == 2 * x.shape[1]:
                    velocity, _ = torch.split(velocity, x.shape[1], dim=1)
                
                x = x + dt * velocity
            
            # Decode to SMILES
            x = x.squeeze(-1).permute((0, 2, 1))
            batch_generated = AE_SMILES_decoder(x, ae_model, stochastic=False, k=1)
            all_generated_smiles.extend(batch_generated)
    
    return all_generated_smiles


def run_conventional_retrieval(test_batch, training_data, y_features, args):
    """Conventional retrieval: biological similarity-based matching to find drugs"""
    
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
    
    # Calculate similarities in biological space
    query_flat = y_features.flatten(1)  # [batch_size, features]
    training_flat = all_training_features.flatten(1)  # [N, features]
    
    # Use cosine similarity
    similarities = cosine_similarity(query_flat.cpu(), training_flat.cpu())
    
    # Get top-k similar conditions for each query
    for i in range(batch_size):
        target_smile = target_smiles[i]
        target_compound = compound_names[i]
        
        sample_similarities = similarities[i]
        top_indices = sample_similarities.argsort()[-args.retrieval_top_k:][::-1]
        
        # Get corresponding drugs from training data
        top_k_smiles = [training_data['smiles'][idx] for idx in top_indices]
        top_k_similarities = [sample_similarities[idx] for idx in top_indices]
        top_k_compounds = [training_data['compound_names'][idx] for idx in top_indices]
        
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
            'top_k_similarities': top_k_similarities,
            'smiles_hit_rank': smiles_hit_rank,
            'compound_hit_rank': compound_hit_rank,
            'smiles_in_top_k': smiles_hit_rank is not None,
            'compound_in_top_k': compound_hit_rank is not None,
        })
    
    return retrieved_results


def run_retrieval_by_generation(test_batch, training_data, model, flow, ae_model, 
                                y_features, pad_mask, device, args):
    """Retrieval by generation: use model to generate drug embeddings, then find similar drugs in dataset"""
    
    retrieved_results = []
    target_smiles = test_batch['target_smiles']
    compound_names = test_batch['compound_name']
    batch_size = len(target_smiles)
    
    # Step 1: Use trained model to generate drug embeddings from biological features
    with torch.no_grad():
        shape = (batch_size, 64, 127, 1)
        x = torch.randn(*shape, device=device)
        dt = 1.0 / args.generation_steps
        
        # Run diffusion process to get drug latent representations
        for i in range(args.generation_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            t_discrete = (t * 999).long()
            
            velocity = model(x, t_discrete, y=y_features, pad_mask=pad_mask)
            if isinstance(velocity, tuple):
                velocity = velocity[0]
            
            if velocity.shape[1] == 2 * x.shape[1]:
                velocity, _ = torch.split(velocity, x.shape[1], dim=1)
            
            x = x + dt * velocity
        
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
    
    # Step 3: Compare generated latents to dataset drug latents
    for i in range(batch_size):
        generated_latent = generated_drug_latents[i]
        target_smile = target_smiles[i]
        target_compound = compound_names[i]
        
        # Calculate similarities in drug latent space
        similarities = []
        
        for j, dataset_latent in enumerate(dataset_latents):
            gen_flat = generated_latent.flatten()
            dataset_flat = dataset_latent.flatten()
            
            similarity = torch.cosine_similarity(gen_flat.unsqueeze(0), dataset_flat.unsqueeze(0))
            similarities.append((j, similarity.item()))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = similarities[:args.retrieval_by_generation_top_k]
        
        # Extract top-k drugs
        top_k_smiles = [dataset_smiles[idx] for idx, _ in top_k_indices]
        top_k_similarities = [sim for _, sim in top_k_indices]
        top_k_compounds = [dataset_compound_names[idx] for idx, _ in top_k_indices]
        
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
            'top_k_similarities': top_k_similarities,
            'smiles_hit_rank': smiles_hit_rank,
            'compound_hit_rank': compound_hit_rank,
            'smiles_in_top_k': smiles_hit_rank is not None,
            'compound_in_top_k': compound_hit_rank is not None,
        })
    
    # Return both retrieval results and generated embeddings
    return retrieved_results, generated_drug_latents.cpu().numpy(), dataset_latents.cpu().numpy()


def run_three_mode_evaluation(test_batch, training_data, model, flow, ae_model, 
                             image_encoder, rna_encoder, y_features, pad_mask, device, args):
    """Run evaluation in all three modes: pure generation, conventional retrieval, and retrieval by generation"""
    
    batch_size = len(test_batch['target_smiles'])
    results_by_mode = {'pure_generation': [], 'conventional_retrieval': [], 'retrieval_by_generation': []}
    
    # Mode 1: Pure Generation
    if args.run_generation:
        # logger.info("Running pure generation mode...")
        generated_smiles_list = run_pure_generation(model, flow, ae_model, y_features, pad_mask, device, args)
        
        # Process generation results
        samples_per_condition = args.num_samples_per_condition
        for i in range(batch_size):
            # Get all samples for this condition
            condition_samples = []
            for sample_idx in range(samples_per_condition):
                idx = i * samples_per_condition + sample_idx
                if idx < len(generated_smiles_list):
                    condition_samples.append(generated_smiles_list[idx])
            
            # Select best sample (by validity first, then random)
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
                best_sample = valid_only[0]  # Take first valid
            else:
                best_sample = valid_samples[0] if valid_samples else ("", "", False)
            
            results_by_mode['pure_generation'].append({
                'sample_id': i + 1,
                'method': 'pure_generation',
                'target_smiles': test_batch['target_smiles'][i],
                'compound_name': test_batch['compound_name'][i],
                'generated_smiles': best_sample[1],
                'is_valid': best_sample[2],
                'all_samples': condition_samples,
                'confidence': 0.8 if best_sample[2] else 0.1
            })
    
    # Mode 2: Conventional Retrieval (Biological Similarity)
    if args.run_conventional_retrieval:
        # logger.info("Running conventional retrieval mode...")
        retrieval_results = run_conventional_retrieval(test_batch, training_data, y_features, args)
        
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
            result.update(retrieval_result)  # Add retrieval-specific metrics
            results_by_mode['conventional_retrieval'].append(result)
    
    # Mode 3: Retrieval by Generation (Model-based Drug Mapping)
    generated_embeddings = None
    reference_embeddings = None
    
    if args.run_retrieval_by_generation:
        # logger.info("Running retrieval by generation mode...")
        retrieval_by_gen_results, generated_embeddings, reference_embeddings = run_retrieval_by_generation(
            test_batch, training_data, model, flow, ae_model, y_features, pad_mask, device, args
        )
        
        for i in range(batch_size):
            retrieval_by_gen_result = retrieval_by_gen_results[i] if i < len(retrieval_by_gen_results) else {}
            
            # Get best result from model-based retrieval
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
            result.update(retrieval_by_gen_result)  # Add retrieval-specific metrics
            results_by_mode['retrieval_by_generation'].append(result)
    
    return results_by_mode, generated_embeddings, reference_embeddings


def run_independent_evaluation(args):
    """Run evaluation with three distinct modes: pure generation, conventional retrieval, and retrieval by generation."""
    
    logger.info("Setting up three-mode evaluation...")
    
    # Set up device and seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    logger.info(f"Using device: {device}, seed: {args.global_seed}")
    
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
        random_state=args.random_state,
        split_train_test=True,
        test_size=args.test_size,
        return_datasets=False,
    )
    
    eval_loader = test_loader
    total_test_batches = len(eval_loader)
    
    # Calculate evaluation scope
    if args.eval_portion < 1.0:
        eval_batches = max(1, int(total_test_batches * args.eval_portion))
        logger.info(f"Evaluating {args.eval_portion*100:.1f}% of test data: {eval_batches}/{total_test_batches} batches")
    else:
        eval_batches = min(args.max_eval_batches, total_test_batches) if args.max_eval_batches > 0 else total_test_batches
        logger.info(f"Evaluating {eval_batches}/{total_test_batches} batches")
    
    # Load models if needed
    model = image_encoder = rna_encoder = ae_model = flow = None
    
    if args.run_generation or args.run_retrieval_by_generation or args.run_conventional_retrieval:
        logger.info("Loading models for evaluation...")
        
        # Create ReT model (needed for generation and retrieval by generation)
        if args.run_generation or args.run_retrieval_by_generation:
            latent_size = 127
            in_channels = 64
            cross_attn = 192
            condition_dim = 192
            
            try:
                model = ReT_SRA_models[args.model](
                    input_size=latent_size,
                    in_channels=in_channels,
                    cross_attn=cross_attn,
                    condition_dim=condition_dim
                ).to(device)
                logger.info(f"Using SRA model: {args.model}")
            except KeyError:
                model = ReT_models[args.model](
                    input_size=latent_size,
                    in_channels=in_channels,
                    cross_attn=cross_attn,
                    condition_dim=condition_dim
                ).to(device)
                logger.info(f"Using standard model: {args.model}")
            
            # Load checkpoint
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            model.eval()
            logger.info(f"Loaded ReT model from {args.ckpt}")
        else:
            # For conventional retrieval only, we still need encoders
            checkpoint = torch.load(args.ckpt, map_location='cpu')
        
        # Setup encoders (needed for all modes that use biological features)
        image_encoder = ImageEncoder(img_channels=4, output_dim=128).to(device)
        image_encoder.load_state_dict(checkpoint['image_encoder'], strict=True)
        image_encoder.eval()
        
        # Setup RNA encoder
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
            
            # Setup rectified flow for generation
            if args.run_generation:
                flow = create_rectified_flow(num_timesteps=1000)
        
        logger.info("Models loaded successfully")
    
    # Collect training data based on enabled modes
    training_data = {'smiles': [], 'compound_names': []}
    
    # Priority: conventional retrieval needs biological features, others need drug lists
    if args.run_conventional_retrieval:
        # Need biological features for conventional retrieval
        training_data = collect_training_data_with_biological_features(
            args, train_loader, image_encoder, rna_encoder, device,
            max_batches=args.max_training_batches
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
    
    # Run three-mode evaluation on test data
    all_results_by_mode = {'pure_generation': [], 'conventional_retrieval': [], 'retrieval_by_generation': []}
    
    # For retrieval by generation: collect embeddings and metadata
    all_generated_embeddings = []
    all_sample_metadata = []
    reference_drug_embeddings = None
    
    eval_start_time = time.time()
    
    logger.info(f"Starting three-mode evaluation on test data...")
    logger.info(f"Modes enabled: Generation={args.run_generation}, Conventional Retrieval={args.run_conventional_retrieval}, Retrieval by Generation={args.run_retrieval_by_generation}")
    
    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating test data")):
        if batch_idx >= eval_batches:
            break
        
        # Encode biological features
        with torch.no_grad():
            y_features, pad_mask = dual_rna_image_encoder(
                batch['control_images'], batch['treatment_images'], 
                batch['control_transcriptomics'], batch['treatment_transcriptomics'],
                image_encoder, rna_encoder, device
            )
        
        # Run multi-mode evaluation for this batch
        batch_results, generated_embeddings, reference_embeddings = run_three_mode_evaluation(
            batch, training_data, model, flow, ae_model, 
            image_encoder, rna_encoder, y_features, pad_mask, device, args
        )
        
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
    
    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_time:.1f}s")
    
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
        'evaluation_mode': 'three_mode_comprehensive',
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
        'total_test_batches': total_test_batches,
        'evaluated_batches': eval_batches,
        'global_seed': args.global_seed,
        'random_state': args.random_state,
        'evaluation_time': eval_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'embedding_files': {
            'generated_embeddings': f"{args.output_prefix}_generated_embeddings.npy" if args.run_retrieval_by_generation else None,
            'reference_embeddings': f"{args.output_prefix}_reference_embeddings.npy" if args.run_retrieval_by_generation else None,
            'sample_metadata': f"{args.output_prefix}_sample_metadata.csv" if args.run_retrieval_by_generation else None,
            'reference_metadata': f"{args.output_prefix}_reference_metadata.csv" if args.run_retrieval_by_generation else None
        }
    }
    
    return {
        'metadata': metadata,
        'training_data': training_data,
        'results_by_mode': all_results_by_mode,
        'mode_metrics': mode_metrics
    }


def collect_training_data_with_biological_features(args, train_loader, image_encoder, rna_encoder, device, 
    max_batches=50):
    """Collect training data with biological features for conventional retrieval"""
    
    training_data = {
        'smiles': [],
        'compound_names': [],
        'biological_features': []
    }
    
    logger.info("Collecting training data with biological features...")
    
    if max_batches <= 0:
        max_batches = len(train_loader)
        logger.info(f"max_batches <= 0, using all {max_batches} batches from training data")

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Collecting training data")):
        if batch_idx >= max_batches:
            break
            
        control_imgs = batch['control_images']
        treatment_imgs = batch['treatment_images']
        control_rna = batch['control_transcriptomics']
        treatment_rna = batch['treatment_transcriptomics']
        target_smiles = batch['target_smiles']
        compound_names = batch['compound_name']
        
        # Encode biological features
        with torch.no_grad():
            y_features, pad_mask = dual_rna_image_encoder(
                control_imgs, treatment_imgs, control_rna, treatment_rna,
                image_encoder, rna_encoder, device
            )
        
        # Store data
        training_data['smiles'].extend(target_smiles)
        training_data['compound_names'].extend(compound_names)
        training_data['biological_features'].append(y_features.cpu())
    
    logger.info(f"Collected {len(training_data['smiles'])} training samples with biological features")
    return training_data

        
def main():
    parser = argparse.ArgumentParser(description="Three-mode comprehensive evaluation of drug generation models")
    
    # Three evaluation modes
    parser.add_argument("--run-generation", action="store_true", 
                       help="Run pure generation mode")
    parser.add_argument("--run-conventional-retrieval", action="store_true",
                       help="Run conventional retrieval mode (biological similarity-based)")
    parser.add_argument("--run-retrieval-by-generation", action="store_true",
                       help="Run retrieval by generation mode (model-based drug mapping)")
    
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
    parser.add_argument("--eval-portion", type=float, default=1.0,
                       help="Portion of test data to evaluate (0.0-1.0)")
    parser.add_argument("--max-eval-batches", type=int, default=0,
                       help="Maximum number of batches to evaluate (0 = no limit)")
    
    # Model and checkpoint arguments
    parser.add_argument("--model", type=str, choices=list(ReT_models.keys()) + list(ReT_SRA_models.keys()), 
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
    parser.add_argument("--max-training-batches", type=int, default=50, 
                       help="Maximum number of training batches to use for collecting training data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--global-seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./evaluation_results")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
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

    logger.info(f"Starting three-mode evaluation:")
    logger.info(f"  Pure Generation: {args.run_generation}")
    logger.info(f"  Conventional Retrieval: {args.run_conventional_retrieval}")
    logger.info(f"  Retrieval by Generation: {args.run_retrieval_by_generation}")

    if args.output_prefix == "evaluation":
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output_prefix = f"{args.dataset}_{args.random_state}_{args.global_seed}_{timestamp}"

    logger.info(f"Starting evaluation with settings:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Eval portion: {args.eval_portion*100:.1f}%")
    logger.info(f"  Max batches: {args.max_eval_batches if args.max_eval_batches > 0 else 'unlimited'}")
    logger.info(f"  Retrieval Top-K: {args.retrieval_top_k}")
    logger.info(f"  Generation steps: {args.generation_steps}")
    logger.info(f"  Seed: {args.global_seed}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info(f"  Output: {args.output_dir}")
    
    # Run three-mode evaluation
    comprehensive_data = run_independent_evaluation(args)
    
    # Save comprehensive results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_json = os.path.join(args.output_dir, f"{args.output_prefix}_comprehensive.json")

    with open(output_json, 'w') as f:
        json.dump(comprehensive_data, f, indent=2, default=str)
    
    # Create three-section summary
    summary_file = create_three_section_summary(comprehensive_data, args)
    
    # Print final summary
    metadata = comprehensive_data['metadata']
    results_by_mode = comprehensive_data['results_by_mode']
    mode_metrics = comprehensive_data['mode_metrics']
    
    logger.info(f"\n{'='*80}")
    logger.info("THREE-MODE EVALUATION COMPLETE")
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
            
            # Mode-specific metrics
            if mode_name == 'pure_generation' and 'target_similarity' in mode_metrics.get(mode_name, {}):
                sim = mode_metrics[mode_name]['target_similarity']['mean']
                logger.info(f"  Mean target similarity: {sim:.3f}")
            elif mode_name in ['conventional_retrieval', 'retrieval_by_generation'] and 'retrieval_accuracy' in mode_metrics.get(mode_name, {}):
                acc = mode_metrics[mode_name]['retrieval_accuracy']['smiles_top_k_accuracy']
                if mode_name == 'retrieval_by_generation':
                    logger.info(f"  Top-{args.retrieval_by_generation_top_k} accuracy: {acc:.3f}")
                else:
                    logger.info(f"  Top-{args.retrieval_top_k} accuracy: {acc:.3f}")

    logger.info(f"\nResults saved:")
    logger.info(f"  Comprehensive: {output_json}")
    logger.info(f"  Three-section summary: {summary_file}")


if __name__ == "__main__":
    main()

