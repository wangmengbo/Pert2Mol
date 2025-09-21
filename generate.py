import os
import torch
import torch.distributed as dist
from models import ReT_models
from models_sra import ReT_SRA_models
from dataloaders.download import find_model
from diffusion.rectified_flow import create_rectified_flow
import argparse
import pandas as pd
import numpy as np
import json
import pickle
from train_autoencoder import pert2mol_autoencoder
from utils import AE_SMILES_decoder, regexTokenizer
from encoders import ImageEncoder, RNAEncoder, PairedRNAEncoder
from encoders import dual_rna_image_encoder_separate as dual_rna_image_encoder
from dataloaders.dataset_gdp import create_gdp_dataloaders
from dataloaders.dataset_lincs_rna import create_lincs_rna_dataloaders
from dataloaders.dataset_cpgjump import create_cpgjump_dataloaders
from dataloaders.dataset_tahoe import create_tahoe_dataloaders
from dataloaders.dataloader import create_leak_free_dataloaders
from rdkit import Chem
import time
from tqdm import tqdm
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')


@torch.no_grad()
def sample_with_cfg(model, flow, shape, y_full, pad_mask,
                   cfg_scale_rna=2.0, cfg_scale_image=2.0,
                   num_steps=50, device=None):
    """Sample with compositional CFG using both modality baselines."""
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]
    x = torch.randn(*shape, device=device)
    dt = 1.0 / num_steps
    
    # Pre-compute conditioning variants
    y_rna_only = y_full.clone()
    y_rna_only[:, :, :128] = 0  # Zero out image features
    
    y_image_only = y_full.clone() 
    y_image_only[:, :, 128:] = 0  # Zero out RNA features
    
    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt, device=device)
        t_discrete = (t * 999).long()
        
        with torch.no_grad():
            # Batch all predictions for efficiency
            y_batch = torch.cat([y_full, y_rna_only, y_image_only], dim=0)
            pad_mask_batch = pad_mask.repeat(3, 1)
            x_batch = x.repeat(3, 1, 1, 1)
            
            # Handle both ReT and ReT_SRA models
            velocity_batch = model(x_batch, t_discrete.repeat(3), y=y_batch, pad_mask=pad_mask_batch)
            
            # Handle tuple return from SRA models
            if isinstance(velocity_batch, tuple):
                velocity_batch = velocity_batch[0]
            
            if velocity_batch.shape[1] == 2 * x.shape[1]:
                velocity_batch, _ = torch.split(velocity_batch, x.shape[1], dim=1)
            
            # Split predictions
            cond_velocity, rna_velocity, image_velocity = torch.chunk(velocity_batch, 3, dim=0)
            
            # Compositional CFG: guide each modality toward full conditioning
            velocity = (rna_velocity + 
                       cfg_scale_rna * (cond_velocity - rna_velocity) +
                       image_velocity + 
                       cfg_scale_image * (cond_velocity - image_velocity)) / 2
                
        x = x + dt * velocity 
    return x


def exact_retrieval(query_features, training_data, top_k=5, similarity_metric='cosine'):
    """Find most similar biological conditions and return corresponding molecules."""
    
    if not training_data['biological_features']:
        return [], []
    
    # Concatenate all training features
    all_training_features = torch.cat(training_data['biological_features'], dim=0)
    
    # Calculate similarities
    query_flat = query_features.flatten(1)  # [B, features]
    training_flat = all_training_features.flatten(1)  # [N, features]
    
    if similarity_metric == 'cosine':
        similarities = cosine_similarity(query_flat.cpu(), training_flat.cpu())
    else:  # euclidean
        distances = torch.cdist(query_flat, training_flat)
        similarities = 1 / (1 + distances.cpu())
    
    # Get top-k similar conditions for each query
    retrieved_smiles = []
    similarity_scores = []
    
    for i in range(query_features.shape[0]):
        sample_similarities = similarities[i]
        top_indices = sample_similarities.argsort()[-top_k:][::-1]
        
        sample_smiles = [training_data['smiles'][idx] for idx in top_indices]
        sample_scores = [sample_similarities[idx] for idx in top_indices]
        
        retrieved_smiles.append(sample_smiles)
        similarity_scores.append(sample_scores)
    
    return retrieved_smiles, similarity_scores


def estimate_generation_confidence(model, flow, x_final, y_features, pad_mask, device):
    """Estimate confidence of generated molecules using multiple metrics."""
    
    with torch.no_grad():
        # 1. Velocity consistency - check if we're at equilibrium
        t_final = torch.ones(x_final.shape[0], device=device)
        t_discrete = (t_final * 999).long()
        final_velocity = model(x_final, t_discrete, y=y_features, pad_mask=pad_mask)
        
        # Handle tuple return from SRA models
        if isinstance(final_velocity, tuple):
            final_velocity = final_velocity[0]
            
        if final_velocity.shape[1] == 2 * x_final.shape[1]:
            final_velocity, _ = torch.split(final_velocity, x_final.shape[1], dim=1)
        
        velocity_magnitude = torch.norm(final_velocity.flatten(1), dim=1)
        velocity_confidence = torch.exp(-velocity_magnitude)  # Lower velocity = higher confidence
        
        # 2. Sampling consistency - generate multiple times and check agreement
        consistency_samples = []
        for _ in range(3):
            sample = sample_with_cfg(model, flow, x_final.shape, y_features, pad_mask, 
                                   num_steps=20, device=device)  # Fewer steps for speed
            consistency_samples.append(sample)
        
        # Measure variance across samples
        sample_stack = torch.stack(consistency_samples)
        sample_variance = torch.var(sample_stack, dim=0).flatten(1).mean(1)
        consistency_confidence = torch.exp(-sample_variance)
        
        # Combine confidences
        overall_confidence = (velocity_confidence + consistency_confidence) / 2
        
    return overall_confidence.cpu()


def basic_validity_check(smiles):
    """Basic SMILES validity check using RDKit."""
    try:
        if not smiles or smiles == "":
            return False, ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, ""
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical
    except:
        return False, ""


@torch.no_grad()
def main(args):
    """Streamlined molecule generation with updated data loading."""
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Inference requires GPU"
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(0)
    print(f"Using device: {device}, seed: {args.global_seed}")

    if args.ckpt is None:
        raise ValueError("Please specify checkpoint path with --ckpt")

    # Load gene count matrix
    if args.gene_count_matrix_path is not None:
        gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)
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
        # Validate paths for custom dataset
        for path_arg in ["image_json_path", "gene_count_matrix_path"]:
            if args.__dict__.get(path_arg) is not None:
                if not os.path.exists(args.__dict__[path_arg]):
                    print(f"Warning: Cannot find {args.__dict__[path_arg]}. Setting to None.")
                    args.__dict__[path_arg] = None
        
        if args.image_json_path is None and args.gene_count_matrix_path is None:
            raise ValueError("At least one of image_json_path or gene_count_matrix_path must be provided for 'other' dataset")
        
        if args.transpose_gene_count_matrix and gene_count_matrix is not None:
            print("Transposing gene count matrix as per --transpose-gene-count-matrix flag.")
            gene_count_matrix = gene_count_matrix.T
        
        create_data_loader = create_leak_free_dataloaders

        # Set up stratification for custom dataset
        if args.stratify_by is None:
            args.stratify_by = [args.compound_name_label]
            if args.cell_type_label is not None:
                args.stratify_by.append(args.cell_type_label)
        
        print(f"Stratifying train/test split by {args.stratify_by}")
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    print(f"Using data loader: {create_data_loader}")

    # Load metadata if provided
    metadata_control = metadata_drug = None
    if args.metadata_control_path is not None:
        metadata_control = pd.read_csv(args.metadata_control_path)
    
    if args.metadata_drug_path is not None:
        metadata_drug = pd.read_csv(args.metadata_drug_path)

    # Create ReT model - handle both ReT and ReT_SRA models
    latent_size = 127
    in_channels = 64
    cross_attn = 192
    condition_dim = 192
    
    # Try ReT_SRA models first, then fall back to regular ReT models
    try:
        model = ReT_SRA_models[args.model](
            input_size=latent_size,
            in_channels=in_channels,
            cross_attn=cross_attn,
            condition_dim=condition_dim
        ).to(device)
        print(f"Using SRA model: {args.model}")
    except KeyError:
        model = ReT_models[args.model](
            input_size=latent_size,
            in_channels=in_channels,
            cross_attn=cross_attn,
            condition_dim=condition_dim
        ).to(device)
        print(f"Using standard model: {args.model}")

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)  # Use strict=False for SRA compatibility
    model.eval()
    print(f"Loaded ReT model from {args.ckpt}")

    # Setup encoders
    image_encoder = ImageEncoder(img_channels=4, output_dim=128).to(device)
    image_encoder.load_state_dict(checkpoint['image_encoder'], strict=True)
    image_encoder.eval()
    
    # Setup RNA encoder (handle paired vs regular)
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
    print("Loaded RNA and Image encoders")

    # Setup autoencoder
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
    del ae_model.text_encoder
    ae_model = ae_model.to(device)
    ae_model.eval()
    print("Loaded autoencoder")

    # Setup rectified flow
    flow = create_rectified_flow(num_timesteps=1000)

    # Create dataloader with same logic as training
    if args.dataset in ["gdp", "lincs_rna", "cpgjump", "tahoe"]:
        # Use dataset-specific loaders that return train/test splits
        if args.use_test_split:
            train_loader, test_loader = create_data_loader(
                metadata_control=metadata_control,
                metadata_drug=metadata_drug, 
                gene_count_matrix=gene_count_matrix,
                image_json_path=args.image_json_path,
                drug_data_path=args.drug_data_path,
                raw_drug_csv_path=args.raw_drug_csv_path,
                batch_size=args.batch_size,
                shuffle=True,
                compound_name_label=args.compound_name_label,
                exclude_cell_lines=args.exclude_cell_lines,
                exclude_drugs=args.exclude_drugs,
                debug_mode=args.debug_mode,
                debug_samples=args.debug_samples,
                debug_cell_lines=args.debug_cell_lines,
                debug_drugs=args.debug_drugs,
                seed=args.global_seed,
                split_train_test=True,
                return_datasets=False,
            )
            loader = test_loader  # Use test split for generation
        else:
            # Use single loader (no splitting)
            loader = create_data_loader(
                metadata_control=metadata_control,
                metadata_drug=metadata_drug, 
                gene_count_matrix=gene_count_matrix,
                image_json_path=args.image_json_path,
                drug_data_path=args.drug_data_path,
                raw_drug_csv_path=args.raw_drug_csv_path,
                batch_size=args.batch_size,
                shuffle=True,
                compound_name_label=args.compound_name_label,
                exclude_cell_lines=args.exclude_cell_lines,
                exclude_drugs=args.exclude_drugs,
                debug_mode=args.debug_mode,
                debug_samples=args.debug_samples,
                debug_cell_lines=args.debug_cell_lines,
                debug_drugs=args.debug_drugs,
                seed=args.global_seed,
                split_train_test=False,
            )
    else:
        # Use leak-free dataloaders for custom dataset
        if args.use_test_split:
            train_loader, test_loader = create_data_loader(
                metadata_control=metadata_control,
                drug_data_path=args.drug_data_path,
                raw_drug_csv_path=args.raw_drug_csv_path,
                metadata_rna=metadata_drug,
                metadata_imaging=None,  # Assuming no imaging for custom datasets
                gene_count_matrix=gene_count_matrix,
                image_json_path=args.image_json_path,
                batch_size=args.batch_size,
                shuffle=True,
                stratify_by=args.stratify_by,
                compound_name_label=args.compound_name_label,
                cell_type_label=args.cell_type_label,
                exclude_cell_lines=args.exclude_cell_lines,
                exclude_drugs=args.exclude_drugs,
                debug_mode=args.debug_mode,
                debug_samples=args.debug_samples,
                debug_cell_lines=args.debug_cell_lines,
                debug_drugs=args.debug_drugs,
                random_state=args.global_seed,
            )
            loader = test_loader  # Use test split for generation
        else:
            # This would require a different approach for single loader
            raise ValueError("Single loader not supported for leak-free dataloaders. Use --use-test-split.")

    print(f"Created dataloader with {len(loader)} batches")

    # Initialize data storage
    training_data = {
        'smiles': [],
        'biological_features': [],
        'compound_names': [],
        'rna_signatures': [],
        'image_features': []
    }

    # Initialize results storage
    results = []
    
    start_time = time.time()
    sample_counter = 0

    # Main generation loop
    print(f"Starting {args.inference_mode} generation...")
    
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Generating molecules ({args.inference_mode})")):
        if batch_idx >= args.max_batches and args.max_batches > 0:
            break
            
        # Extract batch data
        control_imgs = batch['control_images']
        treatment_imgs = batch['treatment_images']
        control_rna = batch['control_transcriptomics']
        treatment_rna = batch['treatment_transcriptomics']
        target_smiles = batch['target_smiles']
        compound_names = batch['compound_name']

        # Update training data storage
        training_data['smiles'].extend(target_smiles)
        training_data['compound_names'].extend(compound_names)

        # Encode biological features
        y_features, pad_mask = dual_rna_image_encoder(
            control_imgs, treatment_imgs, control_rna, treatment_rna,
            image_encoder, rna_encoder, device
        )
        
        # Store features for retrieval
        training_data['biological_features'].append(y_features.cpu())
        
        # Extract RNA and image features separately for analysis later
        rna_features = y_features[:, :, 128:].mean(dim=1).cpu()
        image_features = y_features[:, :, :128].mean(dim=1).cpu()
        training_data['rna_signatures'].extend([feat.cpu() for feat in rna_features])
        training_data['image_features'].extend([feat.cpu() for feat in image_features])
        
        batch_size = y_features.shape[0]
        shape = (batch_size, in_channels, latent_size, 1)
        
        # Generation based on mode
        if args.inference_mode == 'retrieval':
            if len(training_data['biological_features']) > 1:
                # Perform retrieval
                retrieved_smiles, similarity_scores = exact_retrieval(
                    y_features, training_data, top_k=args.retrieval_top_k, 
                    similarity_metric=args.similarity_metric
                )
                
                for i in range(batch_size):
                    sample_counter += 1
                    
                    if retrieved_smiles[i]:
                        best_candidate = retrieved_smiles[i][0]
                        confidence = similarity_scores[i][0]
                        all_candidates = retrieved_smiles[i]
                    else:
                        best_candidate = ""
                        confidence = 0.0
                        all_candidates = []
                    
                    # Basic validity check
                    is_valid, canonical_smiles = basic_validity_check(best_candidate)
                    
                    result = {
                        'sample_id': sample_counter,
                        'batch_idx': batch_idx,
                        'method': 'retrieval',
                        'target_smiles': target_smiles[i],
                        'compound_name': compound_names[i],
                        'generated_smiles': canonical_smiles if is_valid else best_candidate,
                        'is_valid': is_valid,
                        'confidence': float(confidence),
                        'generation_confidence': 0.0,
                        'retrieval_similarity': float(confidence),
                        'all_candidates': all_candidates,
                        'biological_features': y_features[i].cpu().numpy(),
                        'timestamp': time.time()
                    }
                    results.append(result)
            else:
                # Not enough training data for retrieval
                for i in range(batch_size):
                    sample_counter += 1
                    result = {
                        'sample_id': sample_counter,
                        'batch_idx': batch_idx,
                        'method': 'retrieval',
                        'target_smiles': target_smiles[i],
                        'compound_name': compound_names[i],
                        'generated_smiles': '',
                        'is_valid': False,
                        'confidence': 0.0,
                        'generation_confidence': 0.0,
                        'retrieval_similarity': 0.0,
                        'all_candidates': [],
                        'biological_features': y_features[i].cpu().numpy(),
                        'timestamp': time.time()
                    }
                    results.append(result)

        elif args.inference_mode == 'generation':
            # Pure generation mode
            all_generated = []
            all_confidences = []
            
            for sample_idx in range(args.num_samples_per_condition):
                samples = sample_with_cfg(
                    model=model, flow=flow, shape=shape, y_full=y_features, 
                    pad_mask=pad_mask, cfg_scale_rna=2.0, cfg_scale_image=2.0,
                    num_steps=args.num_sampling_steps, device=device
                )
                
                # Estimate confidence
                generation_confidence = estimate_generation_confidence(
                    model, flow, samples, y_features, pad_mask, device
                )
                
                samples = samples.squeeze(-1).permute((0, 2, 1))
                generated_smiles = AE_SMILES_decoder(samples, ae_model, stochastic=False, k=1)
                all_generated.append(generated_smiles)
                all_confidences.append(generation_confidence)
            
            # Process generated results
            for i in range(batch_size):
                sample_counter += 1
                
                # Collect all candidates for this sample
                candidates_with_conf = []
                for gen_idx in range(len(all_generated)):
                    candidate = all_generated[gen_idx][i]
                    confidence = all_confidences[gen_idx][i]
                    candidates_with_conf.append((candidate, float(confidence)))
                
                # Validate candidates and select best
                valid_candidates = []
                all_candidates = []
                
                for cand, conf in candidates_with_conf:
                    all_candidates.append(cand)
                    is_valid, canonical = basic_validity_check(cand)
                    if is_valid:
                        valid_candidates.append((canonical, conf))
                
                if valid_candidates:
                    # Select candidate with highest confidence
                    best_candidate, best_confidence = max(valid_candidates, key=lambda x: x[1])
                    is_valid = True
                else:
                    best_candidate = all_candidates[0] if all_candidates else ""
                    best_confidence = candidates_with_conf[0][1] if candidates_with_conf else 0.0
                    is_valid = False
                
                result = {
                    'sample_id': sample_counter,
                    'batch_idx': batch_idx,
                    'method': 'generation',
                    'target_smiles': target_smiles[i],
                    'compound_name': compound_names[i],
                    'generated_smiles': best_candidate,
                    'is_valid': is_valid,
                    'confidence': float(best_confidence),
                    'generation_confidence': float(best_confidence),
                    'retrieval_similarity': 0.0,
                    'all_candidates': all_candidates,
                    'biological_features': y_features[i].cpu().numpy(),
                    'timestamp': time.time()
                }
                results.append(result)
        
        elif args.inference_mode == 'adaptive':
            # Generate first
            samples = sample_with_cfg(
                model, flow, shape, y_features, pad_mask, 
                cfg_scale_rna=2.0, cfg_scale_image=2.0,
                num_steps=args.num_sampling_steps, device=device
            )
            
            generation_confidence = estimate_generation_confidence(
                model, flow, samples, y_features, pad_mask, device
            )
            
            samples = samples.squeeze(-1).permute((0, 2, 1))
            generated_smiles = AE_SMILES_decoder(samples, ae_model, stochastic=False, k=1)
            
            # Get retrieval backup if we have enough training data
            retrieved_smiles, retrieval_scores = [], []
            if len(training_data['biological_features']) > 1:
                retrieved_smiles, retrieval_scores = exact_retrieval(
                    y_features, training_data, top_k=1, similarity_metric=args.similarity_metric
                )
            
            # Process each sample
            for i in range(batch_size):
                sample_counter += 1
                
                gen_conf = generation_confidence[i]
                generated_candidate = generated_smiles[i]
                
                # Check generation validity
                gen_valid, gen_canonical = basic_validity_check(generated_candidate)
                
                # Decision logic for adaptive mode
                use_generation = (gen_conf >= args.confidence_threshold and gen_valid)
                
                if use_generation:
                    # Use generation
                    best_candidate = gen_canonical
                    method = 'generation'
                    confidence = float(gen_conf)
                    retrieval_sim = 0.0
                    is_valid = True
                else:
                    # Fall back to retrieval
                    if retrieved_smiles and i < len(retrieved_smiles) and retrieved_smiles[i]:
                        best_candidate = retrieved_smiles[i][0]
                        method = 'retrieval_fallback'
                        confidence = float(retrieval_scores[i][0]) if retrieval_scores[i] else 0.0
                        retrieval_sim = confidence
                        is_valid, best_candidate = basic_validity_check(best_candidate)
                    else:
                        best_candidate = gen_canonical if gen_valid else generated_candidate
                        method = 'retrieval_fallback'
                        confidence = 0.0
                        retrieval_sim = 0.0
                        is_valid = gen_valid
                
                result = {
                    'sample_id': sample_counter,
                    'batch_idx': batch_idx,
                    'method': method,
                    'target_smiles': target_smiles[i],
                    'compound_name': compound_names[i],
                    'generated_smiles': best_candidate,
                    'is_valid': is_valid,
                    'confidence': confidence,
                    'generation_confidence': float(gen_conf),
                    'retrieval_similarity': retrieval_sim,
                    'all_candidates': [generated_candidate],
                    'biological_features': y_features[i].cpu().numpy(),
                    'timestamp': time.time()
                }
                results.append(result)

    # Calculate basic statistics
    total_time = time.time() - start_time
    valid_count = sum(1 for r in results if r['is_valid'])
    
    print(f"\nGeneration Complete!")
    print(f"Total samples: {len(results)}")
    print(f"Valid molecules: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time/len(results):.2f}s/sample)")
    
    # Method distribution
    method_counts = {}
    for r in results:
        method = r['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"Method distribution:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} ({count/len(results)*100:.1f}%)")

    # Save results
    output_base = f"{os.path.dirname(os.path.dirname(args.ckpt))}/generated_molecules_{args.inference_mode}_{args.global_seed}"
    print(f"Saving results to {output_base}.json and .tsv")

    # Save detailed results as JSON
    results_json = {
        'metadata': {
            'inference_mode': args.inference_mode,
            'dataset': args.dataset,
            'use_test_split': args.use_test_split,
            'confidence_threshold': args.confidence_threshold,
            'retrieval_top_k': args.retrieval_top_k,
            'similarity_metric': args.similarity_metric,
            'num_samples_per_condition': args.num_samples_per_condition,
            'num_sampling_steps': args.num_sampling_steps,
            'global_seed': args.global_seed,
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method_distribution': method_counts,
            'total_samples': len(results),
            'valid_molecules': valid_count
        },
        'training_data': {
            'smiles': training_data['smiles'],
            'compound_names': training_data['compound_names']
        },
        'results': results
    }
    
    with open(f"{output_base}.json", 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    # Save lightweight TSV for quick inspection
    with open(f"{output_base}.tsv", 'w') as f:
        f.write("sample_id\tmethod\ttarget_smiles\tcompound_name\tgenerated_smiles\t"
               "is_valid\tconfidence\tgen_confidence\tretrieval_similarity\n")
        
        for r in results:
            f.write(f"{r['sample_id']}\t{r['method']}\t{r['target_smiles']}\t"
                   f"{r['compound_name']}\t{r['generated_smiles']}\t{r['is_valid']}\t"
                   f"{r['confidence']:.3f}\t{r['generation_confidence']:.3f}\t"
                   f"{r['retrieval_similarity']:.3f}\n")
    
    print(f"\nResults saved:")
    print(f"  Detailed: {output_base}.json")
    print(f"  Summary:  {output_base}.tsv")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(ReT_models.keys()) + list(ReT_SRA_models.keys()), default="pert2mol")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vae", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/dataloaders/checkpoint_autoencoder.ckpt")
    
    # Dataset arguments (matching train_pert2mol.py)
    parser.add_argument("--dataset", type=str, choices=["gdp", "lincs_rna", "cpgjump", "tahoe", "other"], default="gdp")
    parser.add_argument("--image-json-path", type=str, default=None)
    parser.add_argument("--drug-data-path", type=str, default=None)
    parser.add_argument("--raw-drug-csv-path", type=str, default=None)
    parser.add_argument("--metadata-control-path", type=str, default=None)
    parser.add_argument("--metadata-drug-path", type=str, default=None)
    parser.add_argument("--gene-count-matrix-path", type=str, default=None)
    parser.add_argument("--use-highly-variable-genes", action="store_true", default=True)
    parser.add_argument("--compound-name-label", type=str, default="compound")
    parser.add_argument("--cell-type-label", type=str, default="cell_line")
    parser.add_argument("--stratify-by", type=str, nargs='+', default=None, help="Columns for stratification in train/test split")
    parser.add_argument("--transpose-gene-count-matrix", action="store_true", default=False)
    parser.add_argument("--paired-rna-encoder", action="store_true", help="Use paired RNA encoder for control and treatment")
    
    # Data filtering arguments
    parser.add_argument("--exclude-cell-lines", type=str, nargs='+', default=None, help="Cell lines to exclude")
    parser.add_argument("--exclude-drugs", type=str, nargs='+', default=None, help="Drugs to exclude")
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--debug-samples", type=int, default=100, help="Number of samples in debug mode")
    parser.add_argument("--debug-cell-lines", type=str, nargs='+', default=None, help="Cell lines for debug mode")
    parser.add_argument("--debug-drugs", type=str, nargs='+', default=None, help="Drugs for debug mode")
    
    # Generation parameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples-per-condition", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--use-test-split", action="store_true", help="Use test split of the data (recommended)")
    
    # Inference mode arguments
    parser.add_argument("--inference-mode", type=str, choices=['retrieval', 'generation', 'adaptive'], 
                       default='retrieval', help='Inference mode: retrieval, generation, or adaptive')
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                       help='Confidence threshold for adaptive mode (0-1)')
    parser.add_argument("--retrieval-top-k", type=int, default=3,
                       help='Number of similar conditions to consider for retrieval')
    parser.add_argument("--similarity-metric", type=str, choices=['cosine', 'euclidean'], default='cosine',
                       help='Similarity metric for retrieval mode')
    
    args = parser.parse_args()
    
    print(args)
    main(args)
