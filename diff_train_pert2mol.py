# Inspired by LDMOL https://arxiv.org/pdf/2405.17829

"""
A minimal training script for ReT using PyTorch DDP.
"""
import argparse
import logging
import os
import pandas as pd
import sys
import os
import json
import signal
import atexit
from pathlib import Path
import torch
import hashlib
import random
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True # the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from pathlib import Path

from models import ReT_models
from models_sra import ReT_SRA_models, EMAManager, compute_sra_loss
from train_autoencoder import pert2mol_autoencoder
from utils import (
    AE_SMILES_encoder, regexTokenizer, get_hash,
    setup_signal_handlers, save_emergency_checkpoint, 
    find_latest_checkpoint, load_checkpoint_and_resume,
    EarlyStopping
    )
from encoders import ImageEncoder, RNAEncoder, PairedRNAEncoder
from encoders import dual_rna_image_encoder_separate as dual_rna_image_encoder
from dataloaders.dataloader import create_raw_drug_dataloader, create_leak_free_dataloaders
from dataloaders.dataset_gdp import create_gdp_dataloaders, create_gdp_rna_dataloaders, create_gdp_image_dataloaders
from dataloaders.dataset_lincs_rna import create_lincs_rna_dataloaders
from dataloaders.dataset_cpgjump import create_cpgjump_dataloaders
from dataloaders.dataset_tahoe import create_tahoe_dataloaders
from dataloaders.download import download_model, find_model
from diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType

GRACEFUL_SHUTDOWN = False
CHECKPOINT_DIR = None
CURRENT_STEP = 0
EXPERIMENT_DIR = None
AUTO_REQUEUE = False


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger




def create_dirs(args):
    os.makedirs(args.results_dir, exist_ok=True)
    # experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{get_hash(args)}-{model_string_name}"
    if args.prefix is not None:
        experiment_dir = f"{args.results_dir}/{args.prefix}-{model_string_name}"
    else:
        experiment_dir = f"{args.results_dir}/{get_hash(args)}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return experiment_dir, checkpoint_dir


def calculate_resume_position(resume_step, dataloader_length):
    """Calculate which epoch and batch to resume from."""
    if resume_step == 0:
        return 0, 0
    
    resume_epoch = resume_step // dataloader_length
    resume_batch_idx = resume_step % dataloader_length
    return resume_epoch, resume_batch_idx


def log_info(message, use_ddp, rank, logger):
    """Helper to avoid repeating rank checks"""
    if not use_ddp or rank == 0:
        logger.info(message)


def log_warning(message, use_ddp, rank, logger):
    """Helper for warnings"""
    if not use_ddp or rank == 0:
        logger.warning(message)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def sample_with_cfg(model, flow, shape, y_full, pad_mask, 
                   cfg_scale_rna=1.0, cfg_scale_image=1.0, 
                   num_steps=50, device=None):
    """
    Sample with classifier-free guidance for RNA and/or image conditioning.
    
    Args:
        y_full: Full conditioning [B, 2, 192] 
        cfg_scale_rna: CFG scale for RNA (1.0 = no guidance)
        cfg_scale_image: CFG scale for image (1.0 = no guidance)
    """
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]
    
    # Create conditioning variants
    y_rna_only = y_full.clone()
    y_rna_only[:, :, :128] = 0  # Zero out image features
    
    y_img_only = y_full.clone() 
    y_img_only[:, :, 128:] = 0  # Zero out RNA features
    
    # Stack all variants for batch processing
    if cfg_scale_rna != 1.0 and cfg_scale_image != 1.0:
        # Both RNA and image guidance
        y_batch = torch.cat([y_full, y_rna_only, y_img_only], dim=0)
        pad_mask_batch = pad_mask.repeat(3, 1)
        x_batch_shape = (batch_size * 3,) + shape[1:]
    elif cfg_scale_rna != 1.0:
        # RNA guidance only
        y_batch = torch.cat([y_full, y_rna_only], dim=0)
        pad_mask_batch = pad_mask.repeat(2, 1)
        x_batch_shape = (batch_size * 2,) + shape[1:]
    elif cfg_scale_image != 1.0:
        # Image guidance only  
        y_batch = torch.cat([y_full, y_img_only], dim=0)
        pad_mask_batch = pad_mask.repeat(2, 1)
        x_batch_shape = (batch_size * 2,) + shape[1:]
    else:
        # No guidance
        return flow.sample_dopri5(model, shape, num_steps=20,atol=1e-5, rtol=1e-3,model_kwargs=your_conditioning)
    
    # Sample with batched conditioning
    def cfg_model_fn(x, t, **kwargs):
        # Get predictions for all variants
        out = model(x, t, y=y_batch, pad_mask=pad_mask_batch)
        
        if cfg_scale_rna != 1.0 and cfg_scale_image != 1.0:
            # Split predictions
            out_cond, out_rna, out_img = torch.chunk(out, 3, dim=0)
            # Apply CFG relative to image-only baseline
            out_final = out_img + cfg_scale_image * (out_cond - out_img) + cfg_scale_rna * (out_rna - out_img)
        elif cfg_scale_rna != 1.0:
            out_cond, out_rna = torch.chunk(out, 2, dim=0)
            out_final = out_rna + cfg_scale_rna * (out_cond - out_rna)
        elif cfg_scale_image != 1.0:
            out_cond, out_img = torch.chunk(out, 2, dim=0)
            out_final = out_img + cfg_scale_image * (out_cond - out_img) 
        return out_final
    
    # # Custom sampling with CFG model
    # x = torch.randn(*shape, device=device)
    # dt = 1.0 / num_steps
    
    # for i in range(num_steps):
    #     t = torch.full((batch_size,), i * dt, device=device)
    #     t_discrete = (t * 999).long()
        
    #     with torch.no_grad():
    #         velocity = cfg_model_fn(x, t_discrete)
    #     x = x + dt * velocity
    
    # return x
    
    # Use diffusion sampling
    return flow.p_sample_loop(
        cfg_model_fn, 
        shape, 
        device=device,
        progress=False
    )


def run_validation(model, ema, test_loader, flow, ae_model, image_encoder, rna_encoder, device, use_ddp, rank, args, sra_teacher_manager=None, max_batches=100):
    """Run validation and return average losses"""
    # Store original training states
    model_training = model.training
    ema_training = ema.training
    
    model.eval()
    ema.eval()
    
    total_flow_loss = 0
    total_sra_loss = 0
    total_samples = 0
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= max_batches:  # Configurable limit
                    break
                    
                try:
                    # Encode target molecules
                    target_smiles = batch['target_smiles']
                    x = AE_SMILES_encoder(target_smiles, ae_model).permute((0, 2, 1)).unsqueeze(-1)
                    
                    # Prepare conditioning
                    control_imgs = batch['control_images']
                    treatment_imgs = batch['treatment_images']
                    control_rna = batch['control_transcriptomics']
                    treatment_rna = batch['treatment_transcriptomics']

                    y, pad_mask = dual_rna_image_encoder(
                        control_imgs, treatment_imgs, control_rna, treatment_rna,
                        image_encoder, rna_encoder, device
                    )

                    model_kwargs = dict(y=y.type(torch.float32), pad_mask=pad_mask.bool())
                    
                    # Validation flow loss using EMA model
                    def ema_model_wrapper(x_input, t_input, **kwargs):
                        output = ema(x_input, t_input, **kwargs)
                        if isinstance(output, tuple):
                            return output[0]  # Return only the main output, ignore SRA representation
                        return output
                        
                    loss_dict = flow.training_losses(ema_model_wrapper, x, model_kwargs=model_kwargs)
                    flow_loss = loss_dict["loss"].mean()
                    
                    # Validation SRA loss
                    sra_loss = torch.tensor(0.0, device=device)
                    if args.use_sra and sra_teacher_manager is not None:
                        try:
                            # Sample a timestep for SRA computation
                            current_t = torch.randint(0, flow.num_timesteps, (x.shape[0],), device=device)
                            
                            # Get student representation from current model
                            student_model = model.module if use_ddp else model
                            student_output = student_model.forward(
                                x, current_t, y=y.type(torch.float32), pad_mask=pad_mask.bool()
                            )
                            
                            # Handle tuple return from SRA model
                            if isinstance(student_output, tuple):
                                _, student_repr = student_output
                            else:
                                student_repr = None
                            
                            # Get teacher representation from SRA teacher
                            if student_repr is not None:
                                sra_teacher = sra_teacher_manager.get_teacher()
                                sra_teacher.eval()
                                
                                teacher_output = sra_teacher.forward(
                                    x, current_t, y=y.type(torch.float32), pad_mask=pad_mask.bool(),
                                    teacher_mode=True, teacher_timestep_offset=args.sra_timestep_offset_max // 2
                                )
                                
                                # Handle tuple return from teacher
                                if isinstance(teacher_output, tuple):
                                    _, teacher_repr = teacher_output
                                else:
                                    teacher_repr = None
                                
                                # Compute SRA loss if both representations are available
                                if teacher_repr is not None:
                                    # Get projection head from student model
                                    if hasattr(student_model, 'sra_projection_head'):
                                        projection_head = student_model.sra_projection_head
                                    else:
                                        projection_head = student_model.module.sra_projection_head
                                    
                                    sra_loss = compute_sra_loss(
                                        student_repr, teacher_repr, projection_head, 
                                        distance_type=args.sra_distance_type
                                    )
                        except Exception as e:
                            if not use_ddp or rank == 0:
                                logger.warning(f"SRA validation failed for batch {batch_idx}: {e}")
                            sra_loss = torch.tensor(0.0, device=device)
                    
                    # Accumulate losses
                    total_flow_loss += flow_loss.item() * x.shape[0]
                    total_sra_loss += sra_loss.item() * x.shape[0]
                    total_samples += x.shape[0]
                    
                except Exception as e:
                    if not use_ddp or rank == 0:
                        print(f"Warning: Validation failed for batch {batch_idx}: {e}")
                    continue
    
    finally:
        # Always restore original training states
        model.train(model_training)
        ema.train(ema_training)
    
    # Aggregate across all processes if using distributed training
    if use_ddp:
        total_flow_loss_tensor = torch.tensor(total_flow_loss, device=device)
        total_sra_loss_tensor = torch.tensor(total_sra_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        
        dist.all_reduce(total_flow_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_sra_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        
        avg_flow_loss = total_flow_loss_tensor.item() / max(total_samples_tensor.item(), 1)
        avg_sra_loss = total_sra_loss_tensor.item() / max(total_samples_tensor.item(), 1)
    else:
        avg_flow_loss = total_flow_loss / max(total_samples, 1)
        avg_sra_loss = total_sra_loss / max(total_samples, 1)
    
    return avg_flow_loss, avg_sra_loss


def main(args):
    """
    Trains a new ReT model with SRA enhancement and flexible single/multi-GPU support.
    Enhanced with signal handling, auto-resume capability, and early stopping.
    """
    global create_data_loader, gene_count_matrix, metadata_control, metadata_drug

    # Store references for cleanup
    train_loader = None
    test_loader = None
    
    try:
        assert torch.cuda.is_available(), "Training currently requires at least one GPU."

        # Setup distributed or single GPU training based on flag
        if args.use_distributed:
            # Multi-GPU distributed training
            dist.init_process_group("nccl")

            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
            seed = args.global_seed * dist.get_world_size() + rank
            torch.manual_seed(seed)
            torch.cuda.set_device(device)
            
            # Check CUDA compatibility (only print from rank 0)
            if rank == 0:
                print(f"create_data_loader: {create_data_loader}")
                try:
                    device_capability = torch.cuda.get_device_capability()
                    print(f"GPU capability: {device_capability}")
                    test_tensor = torch.randn(2, 2, device='cuda')
                    test_result = test_tensor * 2.0
                    print("CUDA compatibility test passed")
                except RuntimeError as e:
                    print(f"CUDA compatibility error: {e}")
                    print("Solution: Reinstall PyTorch with proper CUDA version for your GPU")
                    raise e
            
            print(f"Starting distributed training: rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
            
            # Setup experiment folder (only on rank 0)
            if rank == 0:
                experiment_dir, checkpoint_dir = create_dirs(args)
                logger = create_logger(experiment_dir)
                logger.info(f"Experiment directory created at {experiment_dir}")
            else:
                logger = create_logger(None)
                experiment_dir = checkpoint_dir = None
                
            use_ddp = True
            batch_size = args.batch_size
            
        else:
            # Single GPU training
            device = torch.device("cuda:0")
            rank = 0
            seed = args.global_seed
            torch.manual_seed(seed)
            torch.cuda.set_device(0)
            
            print(f"create_data_loader: {create_data_loader}")
            # Check CUDA compatibility
            try:
                device_capability = torch.cuda.get_device_capability()
                print(f"GPU capability: {device_capability}")
                test_tensor = torch.randn(2, 2, device='cuda')
                test_result = test_tensor * 2.0
                print("CUDA compatibility test passed")
            except RuntimeError as e:
                print(f"CUDA compatibility error: {e}")
                print("Solution: Reinstall PyTorch with proper CUDA version for your GPU")
                raise e
                
            print(f"Starting single GPU training on device={device}, seed={seed}.")
            
            # Setup experiment folder
            experiment_dir, checkpoint_dir = create_dirs(args)
            
            # Create simple logger for single GPU
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Experiment directory created at {experiment_dir}")
            
            use_ddp = False
            batch_size = args.batch_size

        # Setup signal handlers (only on rank 0 to avoid duplicate requeue attempts)
        if not use_ddp or rank == 0:
            setup_signal_handlers(checkpoint_dir, experiment_dir, args.auto_resume)
        
        # Initialize early stopping
        early_stopping = None
        if args.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                restore_best_weights=args.restore_best_weights,
                mode='min',  # Minimize validation loss
                verbose=not use_ddp or rank == 0
            )
            if not use_ddp or rank == 0:
                logger.info(f"Early stopping enabled: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
        
        latent_size = 127
        in_channels = 64
        cross_attn = 192
        conditioning_dim = 192

        model = ReT_SRA_models[args.model](
            input_size=latent_size,
            in_channels=in_channels,
            cross_attn=cross_attn,
            condition_dim=conditioning_dim,
            use_sra=args.use_sra,
            sra_layer_student=args.sra_layer_student,
            sra_layer_teacher=args.sra_layer_teacher,
            sra_projection_dim=args.sra_projection_dim
        )

        if args.ckpt:
            ckpt_path = args.ckpt
            state_dict = find_model(ckpt_path)
            msg = model.load_state_dict(state_dict, strict=False)  # strict=False for SRA compatibility
            if not use_ddp or rank == 0:
                logger.info('load ReT from ', ckpt_path, msg)

        # Setup model for distributed or single GPU
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        
        if use_ddp:
            model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
        else:
            model = model.to(device)

        # Initialize SRA teacher manager AFTER DDP wrapping
        sra_teacher_manager = None
        if args.use_sra:
            sra_teacher_manager = EMAManager(model.module if use_ddp else model, ema_decay=args.sra_ema_decay)
            if not use_ddp or rank == 0:
                logger.info(f"SRA enabled: student_layer={args.sra_layer_student}, teacher_layer={args.sra_layer_teacher}, lambda={args.sra_lambda}")
            
        # Create diffusion instance
        betas = get_named_beta_schedule("linear", num_diffusion_timesteps=1000)
        flow = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,  # Model predicts noise
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )

        # Load autoencoder (unchanged)
        ae_config = {
            'bert_config_decoder': './config_decoder.json',
            'bert_config_encoder': './config_encoder.json',
            'embed_dim': 256,
        }
        tokenizer = regexTokenizer(vocab_path='/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/dataloaders/vocab_bpe_300_sc.txt', max_len=127)
        ae_model = pert2mol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer, use_linear=True)
        
        if args.vae:
            if not use_ddp or rank == 0:
                logger.info(f'LOADING PRETRAINED MODEL.. {args.vae}')
            checkpoint = torch.load(args.vae, map_location='cpu')
            try:
                state_dict = checkpoint['model']
            except:
                state_dict = checkpoint['state_dict']
            msg = ae_model.load_state_dict(state_dict, strict=False)
            if not use_ddp or rank == 0:
                logger.info(f'autoencoder {msg}')
        
        for param in ae_model.parameters():
            param.requires_grad = False
        del ae_model.text_encoder
        ae_model = ae_model.to(device)
        ae_model.eval()
        if not use_ddp or rank == 0:
            logger.info(f'AE #parameters: {sum(p.numel() for p in ae_model.parameters())}, #trainable: {sum(p.numel() for p in ae_model.parameters() if p.requires_grad)}')

        logger.info(f"ReT Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Setup image encoder
        image_encoder = ImageEncoder(img_channels=4, output_dim=128).to(device)
        if use_ddp:
            image_encoder = DDP(image_encoder, device_ids=[rank], find_unused_parameters=True)
        
        for param in image_encoder.parameters():
            param.requires_grad = True
        image_encoder.train()
        if not use_ddp or rank == 0:
            logger.info(f'ImageEncoder #parameters: {sum(p.numel() for p in image_encoder.parameters())}, #trainable: {sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)}')

        # Setup rna encoder
        if args.paired_rna_encoder:
            rna_encoder = PairedRNAEncoder(
                input_dim=gene_count_matrix.shape[0], 
                output_dim=128, 
                dropout=0.1, 
                num_heads=4, 
                gene_embed_dim=512, 
                num_self_attention_layers=1, 
                num_cross_attention_layers=2,
                use_bidirectional_cross_attn=True
            ).to(device)
        else:
            rna_encoder = RNAEncoder(input_dim=gene_count_matrix.shape[0], output_dim=64, dropout=0.1).to(device)

        if use_ddp:
            rna_encoder = DDP(rna_encoder, device_ids=[rank], find_unused_parameters=True)
        for param in rna_encoder.parameters():
            param.requires_grad = True
        rna_encoder.train()
        if not use_ddp or rank == 0:
            logger.info(f'RNAEncoder #parameters: {sum(p.numel() for p in rna_encoder.parameters())}, #trainable: {sum(p.numel() for p in rna_encoder.parameters() if p.requires_grad)}')

        # Create optimizer
        if use_ddp:
            all_params = list(model.parameters()) + list(image_encoder.parameters()) + list(rna_encoder.parameters())
        else:
            all_params = list(model.parameters()) + list(image_encoder.parameters()) + list(rna_encoder.parameters())

        if not use_ddp or rank == 0:
            logger.info(f"Total parameters: {sum(p.numel() for p in all_params):,}, trainable: {sum(p.numel() for p in all_params if p.requires_grad):,}")
        opt = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0)

        # Check for existing checkpoints and resume if found
        resume_step = 0
        if not use_ddp or rank == 0:
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint and not args.ckpt and args.auto_resume:  # Don't auto-resume if explicit checkpoint specified
                resume_step = load_checkpoint_and_resume(
                    latest_checkpoint, model, image_encoder, rna_encoder, ema, opt,
                    sra_teacher_manager, use_ddp, rank, logger
                )
                
                # Also load early stopping state if available
                if early_stopping is not None:
                    try:
                        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                        if "early_stopping" in checkpoint:
                            early_stopping.load_state_dict(checkpoint["early_stopping"])
                            if not use_ddp or rank == 0:
                               logger.info(f"[RESUME] Loaded early stopping state: best_score={early_stopping.best_score:.6f}")
                    except Exception as e:
                        if not use_ddp or rank == 0:
                            logger.warning(f"[RESUME] Failed to load early stopping state: {e}")

        # Broadcast resume_step to all processes
        if use_ddp:
            resume_step_tensor = torch.tensor(resume_step, device=device)
            dist.broadcast(resume_step_tensor, src=0)
            resume_step = resume_step_tensor.item()

        # Setup data (unchanged)
        if use_ddp:
            train_dataset, test_dataset = create_data_loader(
                metadata_control=metadata_control,
                metadata_drug=metadata_drug, 
                gene_count_matrix=gene_count_matrix,
                image_json_path=args.image_json_path,
                drug_data_path=args.drug_data_path,
                raw_drug_csv_path=args.raw_drug_csv_path,
                use_highly_variable_genes=args.use_highly_variable_genes,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                compound_name_label=args.compound_name_label,
                exclude_cell_lines=args.exclude_cell_lines,
                exclude_drugs=args.exclude_drugs,
                include_cell_lines=args.include_cell_lines,
                include_drugs=args.include_drugs,
                debug_mode=args.debug_mode,
                debug_samples=args.debug_samples,
                random_state=args.random_state,
                split_train_test=True,
                test_size=args.test_size,
                return_datasets=True,
            )
            
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=True
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=False
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True  # Add this to keep workers alive
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True  # Add this to keep workers alive
            )
        else:
            train_loader, test_loader = create_data_loader(
                metadata_control=metadata_control,
                metadata_drug=metadata_drug, 
                gene_count_matrix=gene_count_matrix,
                image_json_path=args.image_json_path,
                drug_data_path=args.drug_data_path,
                raw_drug_csv_path=args.raw_drug_csv_path,
                use_highly_variable_genes=args.use_highly_variable_genes,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                stratify_by=args.stratify_by, 
                compound_name_label=args.compound_name_label,
                cell_type_label=args.cell_type_label,
                exclude_cell_lines=args.exclude_cell_lines,
                exclude_drugs=args.exclude_drugs,
                include_cell_lines=args.include_cell_lines,
                include_drugs=args.include_drugs,
                debug_mode=args.debug_mode,
                debug_samples=args.debug_samples,
                random_state=args.random_state,
                split_train_test=True,
                test_size=args.test_size,
            )

        # Prepare models for training
        if use_ddp:
            update_ema(ema, model.module, decay=0)
        else:
            update_ema(ema, model, decay=0)
        model.train()
        ema.eval()

        # Training loop with SRA, signal handling, and early stopping
        train_steps = resume_step
        log_steps = 0
        running_loss = 0
        running_flow_loss = 0
        running_sra_loss = 0
        start_time = time()
        validation_frequency = args.ckpt_every // 5  # Run validation 5 times per checkpoint cycle

        logger.info(f"Training for {args.epochs} epochs, starting from step {resume_step}...")
        
        training_completed = False
        
        for epoch in range(args.epochs):
            if GRACEFUL_SHUTDOWN:
                if not use_ddp or rank == 0:
                    logger.info("[SIGNAL] Graceful shutdown detected at start of epoch, exiting...")
                break

            if use_ddp:
                # Adjust epoch for sampler to account for resumed training
                effective_epoch = epoch + (resume_step // len(train_loader))
                train_sampler.set_epoch(effective_epoch)
            
            resume_epoch, resume_batch_idx = calculate_resume_position(resume_step, len(train_loader))
            if epoch < resume_epoch:
                continue

            if not use_ddp or rank == 0:
                logger.info(f"Beginning epoch {epoch}...")
                
            for batch_idx, batch in enumerate(train_loader):
                if epoch == resume_epoch and batch_idx < resume_batch_idx:
                    continue

                if GRACEFUL_SHUTDOWN:
                    if not use_ddp or rank == 0:
                        logger.info("[SIGNAL] Graceful shutdown initiated, saving checkpoint...")
                        save_emergency_checkpoint(
                            model, image_encoder, rna_encoder, ema, opt, args,
                            sra_teacher_manager, train_steps, use_ddp, rank
                        )
                        logger.info("[SIGNAL] Emergency checkpoint saved, exiting...")
                    break
                
                with torch.no_grad():
                    target_smiles = batch['target_smiles']
                    x = AE_SMILES_encoder(target_smiles, ae_model).permute((0, 2, 1)).unsqueeze(-1)
                    
                control_imgs = batch['control_images']
                treatment_imgs = batch['treatment_images']
                control_rna = batch['control_transcriptomics']
                treatment_rna = batch['treatment_transcriptomics']

                y, pad_mask = dual_rna_image_encoder(
                    control_imgs, treatment_imgs, control_rna, treatment_rna,
                    image_encoder, rna_encoder, device
                )

                # CFG training with three conditioning variants
                dropout_choice = torch.rand(y.shape[0], device=device)
                rna_only_mask = (dropout_choice < 0.1).float().view(-1, 1, 1)
                image_only_mask = ((dropout_choice >= 0.1) & (dropout_choice < 0.2)).float().view(-1, 1, 1)

                y[:, :, :128] = y[:, :, :128] * (1 - rna_only_mask)
                y[:, :, 128:] = y[:, :, 128:] * (1 - image_only_mask)

                student_model = model.module if use_ddp else model

                # Sample consistent timestep for both student and teacher
                current_t = flow.sample_time(x.shape[0], device)
                current_noise = torch.randn_like(x)

                # Use diffusion training losses
                current_t = torch.randint(0, flow.num_timesteps, (x.shape[0],), device=device)

                # Forward pass for student (captures both output and SRA representation)
                student_output, student_repr = student_model.forward(
                    x_t, (current_t * (flow.num_timesteps - 1)).long(), 
                    y=y.type(torch.float32), pad_mask=pad_mask.bool()
                )

                # Handle learn_sigma case
                if student_output.shape[1] == 2 * x.shape[1]:
                    predicted_velocity, _ = torch.split(student_output, x.shape[1], dim=1)
                else:
                    predicted_velocity = student_output

                # Flow loss using diffusion
                loss_dict = flow.training_losses(
                    lambda x_input, t_input, **kwargs: student_model.forward(x_input, t_input, **kwargs)[0],
                    x, t=current_t, model_kwargs=dict(y=y.type(torch.float32), pad_mask=pad_mask.bool())
                )
                flow_loss = loss_dict["loss"].mean()

                # SRA loss computation using same timestep
                sra_loss = torch.tensor(0.0, device=device)
                if args.use_sra and sra_teacher_manager is not None and student_repr is not None:
                    # Sample timestep offset for teacher (lower noise)
                    teacher_timestep_offset = torch.randint(1, args.sra_timestep_offset_max + 1, (1,)).item()
                    
                    # Get teacher representation (with lower noise)
                    teacher_model = sra_teacher_manager.get_teacher()
                    teacher_model.eval()
                    with torch.no_grad():
                        _, teacher_repr = teacher_model.forward(
                            x_t, (current_t * (flow.num_timesteps - 1)).long(),
                            y=y.type(torch.float32), pad_mask=pad_mask.bool(),
                            teacher_mode=True, teacher_timestep_offset=teacher_timestep_offset
                        )
                    
                    # Compute SRA alignment loss
                    if teacher_repr is not None:
                        projection_head = student_model.sra_projection_head if hasattr(student_model, 'sra_projection_head') else student_model.module.sra_projection_head
                        sra_loss = compute_sra_loss(
                            student_repr, teacher_repr, projection_head,
                            distance_type=args.sra_distance_type
                        )

                # Total loss
                total_loss = flow_loss + args.sra_lambda * sra_loss
                
                opt.zero_grad()
                total_loss.backward()
                opt.step()
                
                # Update EMA networks
                if use_ddp:
                    update_ema(ema, model.module)
                else:
                    update_ema(ema, model)
                    
                # Update SRA teacher
                if args.use_sra and sra_teacher_manager is not None:
                    if use_ddp:
                        sra_teacher_manager.update_teacher(model.module)
                    else:
                        sra_teacher_manager.update_teacher(model)

                # Logging
                running_loss += total_loss.item()
                running_flow_loss += flow_loss.item()
                running_sra_loss += sra_loss.item()
                log_steps += 1
                train_steps += 1
                
                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    
                    if use_ddp:
                        # Reduce losses across all processes
                        avg_total_loss = torch.tensor(running_loss / log_steps, device=device)
                        avg_flow_loss = torch.tensor(running_flow_loss / log_steps, device=device)
                        avg_sra_loss = torch.tensor(running_sra_loss / log_steps, device=device)
                        
                        dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_flow_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(avg_sra_loss, op=dist.ReduceOp.SUM)
                        
                        avg_total_loss = avg_total_loss.item() / dist.get_world_size()
                        avg_flow_loss = avg_flow_loss.item() / dist.get_world_size()
                        avg_sra_loss = avg_sra_loss.item() / dist.get_world_size()
                    else:
                        avg_total_loss = running_loss / log_steps
                        avg_flow_loss = running_flow_loss / log_steps
                        avg_sra_loss = running_sra_loss / log_steps
                    
                    if not use_ddp or rank == 0:
                        log_msg = f"(step={train_steps:07d}) Total Loss: {avg_total_loss:.4f}, Flow Loss: {avg_flow_loss:.4f}"
                        if args.use_sra:
                            log_msg += f", SRA Loss: {avg_sra_loss:.4f}"
                        log_msg += f", Train Steps/Sec: {steps_per_sec:.2f}"
                        logger.info(log_msg)
                        
                    running_loss = 0
                    running_flow_loss = 0
                    running_sra_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save checkpoint
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if not use_ddp or rank == 0:
                        if use_ddp:
                            checkpoint = {
                                "model": model.module.state_dict(),
                                "image_encoder": image_encoder.module.state_dict(),
                                "rna_encoder": rna_encoder.module.state_dict(),
                                "ema": ema.state_dict(),
                                "opt": opt.state_dict(),
                                "args": args,
                                "train_steps": train_steps
                            }
                            if args.use_sra and sra_teacher_manager is not None:
                                checkpoint["sra_teacher"] = sra_teacher_manager.get_teacher().state_dict()
                        else:
                            checkpoint = {
                                "model": model.state_dict(),
                                "image_encoder": image_encoder.state_dict(),
                                "rna_encoder": rna_encoder.state_dict(),
                                "ema": ema.state_dict(),
                                "opt": opt.state_dict(),
                                "args": args,
                                "train_steps": train_steps
                            }
                            if args.use_sra and sra_teacher_manager is not None:
                                checkpoint["sra_teacher"] = sra_teacher_manager.get_teacher().state_dict()
                        
                        # Add early stopping state
                        if early_stopping is not None:
                            checkpoint["early_stopping"] = early_stopping.state_dict()
                        
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                        # Update latest.pt symlink
                        latest_path = f"{checkpoint_dir}/latest.pt"
                        if os.path.exists(latest_path):
                            os.remove(latest_path)
                        os.symlink(os.path.basename(checkpoint_path), latest_path)
                    
                    if use_ddp:
                        dist.barrier()

                # Run validation periodically for early stopping
                if train_steps % validation_frequency == 0 and train_steps > 0:
                    if not use_ddp or rank == 0:
                        logger.info("Running validation...")
                    
                    val_flow_loss, val_sra_loss = run_validation(
                        model, ema, test_loader, flow, ae_model, 
                        image_encoder, rna_encoder, device, use_ddp, rank, args, sra_teacher_manager
                    )
                    
                    if not use_ddp or rank == 0:
                        val_total_loss = val_flow_loss + args.sra_lambda * val_sra_loss
                        val_msg = f"(step={train_steps:07d}) VALIDATION - Total Loss: {val_total_loss:.4f}, Flow Loss: {val_flow_loss:.4f}"
                        if args.use_sra:
                            val_msg += f", SRA Loss: {val_sra_loss:.4f}"
                        logger.info(val_msg)
                        
                        # Save validation metrics
                        val_metrics = {
                            'step': train_steps,
                            'val_total_loss': val_total_loss,
                            'val_flow_loss': val_flow_loss,
                            'val_sra_loss': val_sra_loss
                        }
                        
                        val_log_path = f"{experiment_dir}/validation_log.jsonl"
                        with open(val_log_path, 'a') as f:
                            f.write(json.dumps(val_metrics) + '\n')
                        
                        # Check early stopping
                        if early_stopping is not None:
                            should_stop = early_stopping(val_total_loss, model, train_steps)
                            if should_stop:
                                logger.info(f"[EARLY_STOPPING] Training stopped early at step {train_steps}")
                                
                                # Save final checkpoint with best weights
                                if use_ddp:
                                    final_checkpoint = {
                                        "model": model.module.state_dict(),
                                        "image_encoder": image_encoder.module.state_dict(),
                                        "rna_encoder": rna_encoder.module.state_dict(),
                                        "ema": ema.state_dict(),
                                        "opt": opt.state_dict(),
                                        "args": args,
                                        "train_steps": train_steps,
                                        "early_stopping": early_stopping.state_dict(),
                                        "early_stopped": True
                                    }
                                    if args.use_sra and sra_teacher_manager is not None:
                                        final_checkpoint["sra_teacher"] = sra_teacher_manager.get_teacher().state_dict()
                                else:
                                    final_checkpoint = {
                                        "model": model.state_dict(),
                                        "image_encoder": image_encoder.state_dict(),
                                        "rna_encoder": rna_encoder.state_dict(),
                                        "ema": ema.state_dict(),
                                        "opt": opt.state_dict(),
                                        "args": args,
                                        "train_steps": train_steps,
                                        "early_stopping": early_stopping.state_dict(),
                                        "early_stopped": True
                                    }
                                    if args.use_sra and sra_teacher_manager is not None:
                                        final_checkpoint["sra_teacher"] = sra_teacher_manager.get_teacher().state_dict()
                                
                                final_path = f"{checkpoint_dir}/final_early_stopped_{train_steps:07d}.pt"
                                torch.save(final_checkpoint, final_path)
                                logger.info(f"Saved final early-stopped checkpoint to {final_path}")
                                
                                # Update latest.pt symlink to point to final checkpoint
                                latest_path = f"{checkpoint_dir}/latest.pt"
                                if os.path.exists(latest_path):
                                    os.remove(latest_path)
                                os.symlink(os.path.basename(final_path), latest_path)
                                
                                break  # Exit training loop
                    
                    if use_ddp:
                        dist.barrier()
                        
                        # Broadcast early stopping decision to all processes
                        should_stop_tensor = torch.tensor(early_stopping.should_stop if early_stopping else False, device=device)
                        dist.broadcast(should_stop_tensor, src=0)
                        
                        if should_stop_tensor.item():
                            if rank != 0:
                                logger.info(f"[EARLY_STOPPING] Received stop signal from rank 0")
                            break  # Exit training loop on all processes
            
            # Check if we need to break from the outer epoch loop
            if GRACEFUL_SHUTDOWN:
                if not use_ddp or rank == 0:
                    logger.info("[SIGNAL] Graceful shutdown completed, exiting training loop...")
                break

            # Check for graceful shutdown or early stopping between epochs
            if early_stopping and early_stopping.should_stop:
                break
        else:
            # This else clause executes only if the for loop completed normally (not broken)
            training_completed = True
            if not use_ddp or rank == 0:
                logger.info("Training completed all epochs successfully!")

        model.eval()
        
        # Clean up and ensure proper termination
        if not use_ddp or rank == 0:
            if training_completed:
                logger.info("Training completed successfully. Cleaning up...")
            else:
                logger.info("Training stopped early or interrupted. Cleaning up...")
    
    except Exception as e:
        if not use_ddp or rank == 0:
            logger.error(f"Training failed with exception: {e}")
        raise e
    finally:
        # CRITICAL: Proper cleanup to prevent hanging
        try:
            # Force cleanup of DataLoader workers
            if train_loader is not None:
                del train_loader
            if test_loader is not None:
                del test_loader
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Cleanup distributed training
            if use_ddp:
                if not use_ddp or rank == 0:
                    logger.info("Cleaning up distributed training...")
                dist.barrier()  # Ensure all processes reach this point
                cleanup()
            
            # Final log message
            if not use_ddp or rank == 0:
                logger.info("Done! All resources cleaned up.")
                
            # Force exit to prevent hanging
            import sys
            sys.exit(0)
            
        except Exception as cleanup_error:
            if not use_ddp or rank == 0:
                print(f"Error during cleanup: {cleanup_error}")
            import sys
            sys.exit(1)

            
if __name__ == "__main__":
    # Default args here will train ReT with the hyperparameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--model", type=str, choices=list(ReT_SRA_models.keys()), default="pert2molSRA")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for data loading replicability")
    parser.add_argument("--global-seed", type=int, default=42, help="Global random seed for DDP reproducibility")
    parser.add_argument("--vae", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/dataloaders/checkpoint_autoencoder.ckpt") 
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--use-distributed", action="store_true", help="Enable distributed training across multiple GPUs")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["gdp", "gdp_rna", "gdp_image", "lincs_rna", "cpgjump", "tahoe", "other"], default="gdp")
    parser.add_argument("--image-json-path", type=str, default=None)
    parser.add_argument("--drug-data-path", type=str, default=None)
    parser.add_argument("--raw-drug-csv-path", type=str, default=None)
    parser.add_argument("--metadata-control-path", type=str, default=None)
    parser.add_argument("--metadata-drug-path", type=str, default=None)
    parser.add_argument("--gene-count-matrix-path", type=str, default=None)
    parser.add_argument("--use-highly-variable-genes", action="store_true", default=True)
    parser.add_argument("--disable-highly-variable-genes", action="store_false", dest="use_highly_variable_genes", help="Disable highly variable genes")
    parser.add_argument("--compound-name-label", type=str, default="compound")
    parser.add_argument("--cell-type-label", type=str, default="cell_line")
    parser.add_argument("--stratify-by", type=str, default=None, help="Column name in metadata to stratify train/test split")
    parser.add_argument("--transpose-gene-count-matrix", action='store_true', default=False)
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of dataset to use as test set when creating new split")
    
    parser.add_argument("--paired-rna-encoder", action="store_true", help="Use paired RNA encoder for control and treatment")
    
    parser.add_argument("--exclude-cell-lines", type=str, nargs='+', default=None, help="Cell lines to exclude from training")
    parser.add_argument("--exclude-drugs", type=str, nargs='+', default=None, help="Drugs to exclude from training")
    parser.add_argument("--include-cell-lines", type=str, nargs='+', default=None, help="Only use these cell lines (overrides exclude)")
    parser.add_argument("--include-drugs", type=str, nargs='+', default=None, help="Only use these drugs (overrides exclude)")

    parser.add_argument("--auto-resume", action="store_true", default=False, help="Automatically resume from latest checkpoint")
    parser.add_argument("--early-stopping-patience", type=int, default=0, 
                    help="Number of validation checks without improvement before stopping (0 = disabled)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001,
                    help="Minimum change to qualify as improvement for early stopping")
    parser.add_argument("--restore-best-weights", action="store_true", default=False,
                    help="Restore best model weights when early stopping triggers")

    parser.add_argument("--debug-mode", action='store_true', default=False)
    parser.add_argument("--debug-samples", type=int, default=2000, help="When in debug mode, use this many samples")

    parser.add_argument("--use-sra", action="store_true", default=True, help="Enable Self-Representation Alignment")
    parser.add_argument("--disable-sra", action="store_false", dest="use_sra", help="Disable Self-Representation Alignment")
    parser.add_argument("--sra-lambda", type=float, default=0.1, help="Weight for SRA loss")
    parser.add_argument("--sra-layer-student", type=int, default=4, help="Student layer for SRA extraction")
    parser.add_argument("--sra-layer-teacher", type=int, default=8, help="Teacher layer for SRA extraction") 
    parser.add_argument("--sra-projection-dim", type=int, default=None, help="SRA projection head output dim")
    parser.add_argument("--sra-ema-decay", type=float, default=0.9999, help="EMA decay for SRA teacher")
    parser.add_argument("--sra-distance-type", type=str, default="mse", choices=["mse", "cosine"], help="Distance function for SRA loss")
    parser.add_argument("--sra-timestep-offset-max", type=int, default=50, help="Maximum timestep offset for teacher (lower noise)")

    args = parser.parse_args()
    print(args)
    
    assert not (args.include_cell_lines and args.exclude_cell_lines), "Both --include-cell-lines and --exclude-cell-lines specified. Please choose one."

    if args.gene_count_matrix_path is not None:
        gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)

    if args.dataset == "gdp":
        create_data_loader = create_gdp_dataloaders
    elif args.dataset == "gdp_rna":
        create_data_loader = create_gdp_rna_dataloaders
    elif args.dataset == "lincs_rna":
        create_data_loader = create_lincs_rna_dataloaders
        gene_count_matrix = gene_count_matrix.T
    elif args.dataset == "cpgjump":
        create_data_loader = create_cpgjump_dataloaders
    elif args.dataset == "tahoe":
        create_data_loader = create_tahoe_dataloaders
        gene_count_matrix = gene_count_matrix.T
    elif args.dataset == "other":
        for i in ["image_json_path", "gene_count_matrix_path"]:
            try:
                if args.__dict__[i] is not None:
                    assert os.path.exists(args.__dict__[i]), f"Cannot find {args.__dict__[i]}."
            except Exception as e:
                logger.error(e, f"; Reset {i} to None")
                args.__dict__[i] = None
        assert not (args.image_json_path is None and args.gene_count_matrix_path is None), "Both image_json_path and gene_count_matrix_path are None. At least one must be provided."

        if args.transpose_gene_count_matrix:
            print("Transposing gene count matrix as per --transpose-gene-count-matrix flag.")
            gene_count_matrix = gene_count_matrix.T
        
        create_data_loader = create_leak_free_dataloaders

        if args.stratify_by is None:
            args.stratify_by = [args.compound_name_label]
            if args.cell_type_label is not None:
                args.stratify_by.append(args.cell_type_label)

        print(f"Stratifying train/test split by {args.stratify_by}")
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    print(f"create_data_loader: {create_data_loader}")

    metadata_control = metadata_drug = None
    if args.metadata_control_path is not None:
        metadata_control = pd.read_csv(args.metadata_control_path)
    
    if args.metadata_drug_path is not None:
        metadata_drug = pd.read_csv(args.metadata_drug_path)

    main(args)