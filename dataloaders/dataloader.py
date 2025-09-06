import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import tifffile
import pickle
import h5py
import json
import logging
from typing import Dict, List, Optional, Callable, Tuple
from torchvision import transforms
import scanpy as sc
import anndata as ad
import networkx as nx
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, RDKFingerprint
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import warnings
os.environ['RDKit_SILENCE_WARNINGS'] = '1'
import rdkit
rdkit.rdBase.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .dataset_pert2mol import DatasetWithDrugs, RawDrugDataset
from .utils import convert_to_aromatic_smiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_array(data, name, allow_negative=True, max_abs_value=1e6):
    """Comprehensive data validation helper."""
    if data is None:
        raise ValueError(f"{name}: Data is None")
    
    # Check for NaNs
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        raise ValueError(f"{name}: Contains {nan_count} NaN values")
    
    # Check for infinite values
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        raise ValueError(f"{name}: Contains {inf_count} infinite values")
    
    # Check for all zeros (might indicate loading issues)
    if np.all(data == 0):
        raise ValueError(f"{name}: All values are zero")
    
    return True


def create_vocab_mappings(dataset):
    """Create vocabulary mappings for categorical variables."""
    compounds = list(set(dataset.metadata_control['compound'].unique()).union(set(dataset.metadata_drug['compound'].unique())))
    cell_lines = list(set(dataset.metadata_control['cell_line'].unique()).union(set(dataset.metadata_drug['cell_line'].unique())))
    
    compound_to_idx = {comp: idx for idx, comp in enumerate(sorted(compounds))}
    cell_line_to_idx = {cl: idx for idx, cl in enumerate(sorted(cell_lines))}
    
    return compound_to_idx, cell_line_to_idx


def image_transform(images):
    """
    Transform for 16-bit multi-channel microscopy images.
    Args:
        images: numpy array of shape (channels, height, width)
    
    Returns:
        Normalized and contrast-enhanced images
    """
    # Normalize 16-bit to 0-1 range (CORRECTED from /255 to /65535)
    images_norm = (images / 32767.5) - 1.0
    # Apply per-channel contrast enhancement
    enhanced_images = np.zeros_like(images_norm)
    for i in range(images_norm.shape[0]):
        channel = images_norm[i]
        p1, p99 = np.percentile(channel, [1, 99])
        if p99 > p1:
            enhanced_images[i] = np.clip((channel - p1) / (p99 - p1) * 2 - 1, -1, 1)
        else:
            enhanced_images[i] = channel
    return enhanced_images.astype(np.float32)


def create_dataloader(
    metadata_control: pd.DataFrame,
    metadata_drug: pd.DataFrame,
    gene_count_matrix: pd.DataFrame,
    image_json_path: str,
    drug_data_path: str,

    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_size: Optional[int] = 256,

    use_highly_variable_genes: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,

    drug_encoder: Optional[torch.nn.Module] = None,
    debug_mode: bool = False,
    debug_samples: int = None,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    exclude_drugs: Optional[List[str]] = None,

    fallback_smiles_dict: Optional[Dict[str, str]] = None,
    enable_smiles_fallback: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with drug conditioning and debug options.
    
    Args:
        ... (same as before)
        debug_mode: If True, only load a subset of data for debugging
        debug_samples: Number of samples to load in debug mode
        debug_cell_lines: Specific cell lines to use for debugging
    """
    # Load image paths from JSON file
    with open(image_json_path, 'r') as f:
        image_json_dict = json.load(f)
    
    # Create dataset with debug options
    dataset = DatasetWithDrugs(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_dict=image_json_dict,
        drug_data_path=drug_data_path,

        fallback_smiles_dict=fallback_smiles_dict,
        enable_smiles_fallback=enable_smiles_fallback,

        transform=transform,
        target_size=target_size,
        
        drug_encoder=drug_encoder,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        debug_cell_lines=debug_cell_lines,
        debug_drugs=debug_drugs,
        exclude_drugs=exclude_drugs
    )
    
    if use_highly_variable_genes:
        adata = ad.AnnData(X=dataset.gene_count_matrix.T.values, 
                           obs=dataset.gene_count_matrix.T.index.to_frame(),
                           var=dataset.gene_count_matrix.T.columns.to_frame())
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,)
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        if zscore:
            sc.pp.scale(adata)
        if not normalize:
            adata.X = adata.layers['counts']
        dataset.gene_count_matrix = adata[:, hvg_genes].to_df().T

    # Enhanced collate function (same as before)
    def enhanced_collate_fn(batch):
        """Custom collate function for batch with drug data."""
        collated = {
            'control_transcriptomics': torch.stack([item['control_transcriptomics'] for item in batch]),
            'treatment_transcriptomics': torch.stack([item['treatment_transcriptomics'] for item in batch]),
            'control_images': torch.stack([item['control_images'] for item in batch]),
            'treatment_images': torch.stack([item['treatment_images'] for item in batch]),
            'compound_name': [item['compound_name'] for item in batch],
            'conditioning_info': [item['conditioning_info'] for item in batch]
        }

        # Handle drug_condition with proper error checking
        drug_conditions = []
        for item in batch:
            drug_cond = item['drug_condition']
            if drug_cond.numel() == 0:  # Empty tensor
                # Create a zero tensor with consistent shape
                drug_cond = torch.zeros(1047, dtype=torch.float32)  # Adjust size as needed
            drug_conditions.append(drug_cond)
        
        # Check if all drug conditions have the same shape
        if len(set(dc.shape for dc in drug_conditions)) == 1:
            collated['drug_condition'] = torch.stack(drug_conditions)
        else:
            # Pad to maximum length
            max_len = max(dc.shape[0] for dc in drug_conditions)
            padded_conditions = []
            for dc in drug_conditions:
                if dc.shape[0] < max_len:
                    padded = torch.zeros(max_len, dtype=torch.float32)
                    padded[:dc.shape[0]] = dc
                    padded_conditions.append(padded)
                else:
                    padded_conditions.append(dc)
            collated['drug_condition'] = torch.stack(padded_conditions)

        # Handle drug embeddings safely
        if 'drug_embeddings' in batch[0]:
            drug_keys = batch[0]['drug_embeddings'].keys()
            collated['drug_embeddings'] = {}
            for key in drug_keys:
                if isinstance(batch[0]['drug_embeddings'][key], torch.Tensor):
                    embeddings = [item['drug_embeddings'][key] for item in batch]
                    # Check if all have same shape
                    if len(set(emb.shape for emb in embeddings)) == 1:
                        collated['drug_embeddings'][key] = torch.stack(embeddings)
                    else:
                        # Handle variable shapes - pad or truncate as needed
                        collated['drug_embeddings'][key] = embeddings  # Keep as list
                else:
                    collated['drug_embeddings'][key] = [
                        item['drug_embeddings'][key] for item in batch
                    ]

        return collated
    
    # Adjust batch size and num_workers for debug mode
    if debug_mode:
        batch_size = min(batch_size, 4)
        num_workers = 0
        shuffle = False
        print(f"DEBUG MODE: Adjusted batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=enhanced_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, create_vocab_mappings(dataset)


def create_raw_drug_dataloader(
    metadata_control: pd.DataFrame,
    metadata_drug: pd.DataFrame,
    drug_data_path: str,
    raw_drug_csv_path: str,
    image_json_path: str = None,
    gene_count_matrix: pd.DataFrame = None,
    compound_name_label='compound',
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    transform=None,
    target_size: int = 256,
    use_highly_variable_genes: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,
    debug_mode: bool = False,
    debug_samples: int = 50,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    smiles_cache: Optional[Dict] = None,
    split_train_test: bool = True,
    test_size: float = 0.2,
    **kwargs
):
    """Create DataLoader for raw drug CSV data with biological conditioning"""
    # Load image paths
    if image_json_path is not None and os.path.exists(image_json_path):
        with open(image_json_path, 'r') as f:
            image_json_dict = json.load(f)
    else:
        image_json_dict = {}
        logger.info("No image JSON path provided or file does not exist - proceeding without images")

    # Create dataset
    dataset = RawDrugDataset(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_dict=image_json_dict,
        drug_data_path=drug_data_path,
        raw_drug_csv_path=raw_drug_csv_path,
        compound_name_label=compound_name_label,
        transform=transform or image_transform,
        target_size=target_size,
        smiles_cache=smiles_cache,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        debug_cell_lines=debug_cell_lines,
        debug_drugs=debug_drugs,
        smiles_only=True,
        **kwargs
    )

    # Apply highly variable gene selection
    if use_highly_variable_genes and gene_count_matrix is not None:
        adata = ad.AnnData(
            X=dataset.gene_count_matrix.T.values,
            obs=dataset.gene_count_matrix.T.index.to_frame(),
            var=dataset.gene_count_matrix.T.columns.to_frame()
        )
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        
        if zscore:
            sc.pp.scale(adata)
        if not normalize:
            adata.X = adata.layers['counts']
            
        dataset.gene_count_matrix = adata[:, hvg_genes].to_df().T
    
    # Collate function
    def conditional_collate_fn(batch):
        collated = {
            'control_transcriptomics': torch.stack([item['control_transcriptomics'] for item in batch]),
            'treatment_transcriptomics': torch.stack([item['treatment_transcriptomics'] for item in batch]),
            'control_images': torch.stack([item['control_images'] for item in batch]),
            'treatment_images': torch.stack([item['treatment_images'] for item in batch]),
            'compound_name': [item['compound_name'] for item in batch],
            'conditioning_info': [item['conditioning_info'] for item in batch],
            'target_smiles': [item['target_smiles'] for item in batch],
            'target_drug_info': [item['target_drug_info'] for item in batch]
        }
        
        # Handle drug conditions
        drug_conditions = []
        for item in batch:
            drug_cond = item['drug_condition']
            if drug_cond.numel() == 0:
                drug_cond = torch.zeros(1047, dtype=torch.float32)
            drug_conditions.append(drug_cond)
        
        if len(set(dc.shape for dc in drug_conditions)) == 1:
            collated['drug_condition'] = torch.stack(drug_conditions)
        else:
            max_len = max(dc.shape[0] for dc in drug_conditions)
            padded_conditions = []
            for dc in drug_conditions:
                if dc.shape[0] < max_len:
                    padded = torch.zeros(max_len, dtype=torch.float32)
                    padded[:dc.shape[0]] = dc
                    padded_conditions.append(padded)
                else:
                    padded_conditions.append(dc)
            collated['drug_condition'] = torch.stack(padded_conditions)
        
        return collated
    
    # Adjust for debug mode
    if debug_mode:
        batch_size = min(batch_size, 4)
        num_workers = 0
        shuffle = False
    
    if split_train_test:
        train_size = int(len(dataset) * (1-test_size))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=conditional_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=conditional_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, test_loader
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=conditional_collate_fn,
            pin_memory=torch.cuda.is_available()
        )


def _prepare_metadata_for_dataset(combined_df, compound_name_label='compound'):
    """Convert combined metadata to format expected by RawDrugDataset"""
    if combined_df.empty:
        return pd.DataFrame()
    
    metadata_drug = pd.DataFrame()
    
    # Essential columns for the dataset
    essential_cols = [
        compound_name_label, 'cell_line', 'compound_concentration_in_uM', 'timepoint'
    ]
    
    for col in essential_cols:
        if col in combined_df.columns:
            metadata_drug[col] = combined_df[col]
    
    # Use RNA sample IDs for transcriptomics data
    if 'sample_id:rna' in combined_df.columns:
        metadata_drug['sample_id'] = combined_df['sample_id:rna'].astype(str)
    
    # Use imaging json_key for image loading
    if 'json_key:hist' in combined_df.columns:
        metadata_drug['json_key'] = combined_df['json_key:hist'].astype(str)
    
    # Ensure data types match expectations
    for col in ['timepoint', 'compound_concentration_in_uM']:
        if col in metadata_drug.columns:
            metadata_drug[col] = pd.to_numeric(metadata_drug[col], errors='coerce')
    
    for col in [compound_name_label, 'cell_line', 'sample_id', 'json_key']:
        if col in metadata_drug.columns:
            metadata_drug[col] = metadata_drug[col].astype(str)
    
    return metadata_drug
 

def _create_modality_combinations(rna_df, imaging_df, compound_name_label='compound', suffix="train", 
                                  shared_cell_lines=None):
    """Create paired combinations within a split"""
    combinations = []    

    def _create_combined_row(rna_row, imaging_row, compound_name_label='compound'):
        """Create a combined row with proper suffixes"""
        combined_row = {}
        
        # Add RNA data with :rna suffix
        for col in rna_row.index:
            combined_row[f"{col}:rna"] = rna_row[col]
        
        # Add imaging data with :hist suffix
        for col in imaging_row.index:
            combined_row[f"{col}:hist"] = imaging_row[col]
        
        # Add shared condition columns without suffix (for dataloader compatibility)
        combined_row[compound_name_label] = rna_row[compound_name_label]
        combined_row['cell_line'] = rna_row['cell_line']
        combined_row['compound_concentration_in_uM'] = rna_row['compound_concentration_in_uM']
        combined_row['timepoint'] = rna_row['timepoint']
        
        return combined_row

    # Handle A549 separately (as in your original logic)
    if 'A549' in shared_cell_lines:
        # For A549, match on compound, cell_line, timepoint only (not concentration)
        a549_rna = rna_df[rna_df['cell_line'] == 'A549']
        a549_imaging = imaging_df[imaging_df['cell_line'] == 'A549']
        
        if not a549_rna.empty and not a549_imaging.empty:
            for _, rna_row in a549_rna.iterrows():
                matching_imaging = a549_imaging[
                    (a549_imaging[compound_name_label] == rna_row[compound_name_label]) &
                    (a549_imaging['timepoint'] == rna_row['timepoint'])
                ]
                
                for _, imaging_row in matching_imaging.iterrows():
                    combination = _create_combined_row(rna_row, imaging_row)
                    combinations.append(combination)
    
    # For other cell lines, use exact matching (including concentration)
    other_cell_lines = [cl for cl in shared_cell_lines if cl != 'A549']
    other_rna = rna_df[rna_df['cell_line'].isin(other_cell_lines)]
    other_imaging = imaging_df[imaging_df['cell_line'].isin(other_cell_lines)]
    
    if not other_rna.empty and not other_imaging.empty:
        # Group by exact matching conditions
        rna_grouped = other_rna.groupby([
            compound_name_label, 'cell_line', 'compound_concentration_in_uM', 'timepoint'
        ])
        
        for condition, rna_group in rna_grouped:
            compound, cell_line, conc, time = condition
            
            # Find exactly matching imaging samples
            matching_imaging = other_imaging[
                (other_imaging[compound_name_label] == compound) &
                (other_imaging['cell_line'] == cell_line) &
                (other_imaging['compound_concentration_in_uM'] == conc) &
                (other_imaging['timepoint'] == time)
            ]
            
            if not matching_imaging.empty:
                # Create all combinations for this condition
                for _, rna_row in rna_group.iterrows():
                    for _, imaging_row in matching_imaging.iterrows():
                        combination = _create_combined_row(rna_row, imaging_row)
                        combinations.append(combination)
    
    if combinations:
        combined_df = pd.DataFrame(combinations)
        logger.info(f"Created {len(combined_df)} {suffix} combinations")
        return combined_df
    else:
        logger.warning(f"No {suffix} combinations could be created")
        return pd.DataFrame()


def create_leak_free_dataloaders(
    metadata_control: pd.DataFrame,
    metadata_rna: pd.DataFrame,  # RNA-seq treatment data (GDPx1x2)
    metadata_imaging: pd.DataFrame,  # Imaging treatment data (GDPx3)
    shared_drugs: List[str],
    shared_cell_lines: List[str],
    gene_count_matrix: pd.DataFrame,
    image_json_path: str,
    drug_data_path: str,
    raw_drug_csv_path: str,
    use_highly_variable_genes: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,
    final_train_test_ratio: float = 4.0,  # Desired train:test ratio for final combinations
    test_size: Optional[float] = None,  # Deprecated - computed automatically
    random_state: int = 42,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    stratify_by: Optional[List[str]] = None,
    compound_name_label: str = 'compound',
    create_modality_combinations=_create_modality_combinations,
    prepare_metadata_for_dataset=_prepare_metadata_for_dataset,
    debug_mode: bool = False,
    debug_samples: int = 50,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    **dataloader_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create leak-free train/test dataloaders with balanced combination ratios.
    
    Args:
        final_train_test_ratio: Desired ratio of train:test combinations (default 4.0 for 80:20)
        test_size: Deprecated - ratio is computed automatically for balanced splits
    """
    
    # Calculate optimal per-modality split ratios
    R = final_train_test_ratio
    optimal_train_fraction = np.sqrt(R) / (1 + np.sqrt(R))
    computed_test_size = 1 - optimal_train_fraction
    
    logger.info(f"Target train:test combination ratio: {R}:1")
    logger.info(f"Computed per-modality train fraction: {optimal_train_fraction:.3f}")
    logger.info(f"Computed per-modality test fraction: {computed_test_size:.3f}")
    
    if test_size is not None:
        logger.warning(f"Ignoring provided test_size={test_size}. Using computed test_size={computed_test_size:.3f} for balanced splits.")
    
    # Filter to shared conditions
    rna_filtered = metadata_rna[
        (metadata_rna[compound_name_label].isin(shared_drugs)) &
        (metadata_rna['cell_line'].isin(shared_cell_lines))
    ].reset_index(drop=True)
    
    imaging_filtered = metadata_imaging[
        (metadata_imaging[compound_name_label].isin(shared_drugs)) &
        (metadata_imaging['cell_line'].isin(shared_cell_lines))
    ].reset_index(drop=True)
    
    logger.info(f"Filtered RNA samples: {len(rna_filtered)}")
    logger.info(f"Filtered imaging samples: {len(imaging_filtered)}")
    
    # Create stratification groups if specified
    def create_strata_labels(df, stratify_columns):
        if stratify_columns is None:
            return None
        strata = df[stratify_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        return strata
    
    # Split RNA-seq treatment data using computed ratios
    unique_rna_samples = rna_filtered['sample_id'].unique()
    logger.info(f"Found {len(unique_rna_samples)} unique RNA sample IDs")
    
    if len(unique_rna_samples) > 1:
        try:
            if stratify_by is not None:
                rna_sample_info = rna_filtered.groupby('sample_id').first()
                rna_strata = create_strata_labels(rna_sample_info.loc[unique_rna_samples], stratify_by)
            else:
                rna_strata = None
            
            rna_train_ids, rna_test_ids = train_test_split(
                unique_rna_samples,
                test_size=computed_test_size,
                random_state=random_state,
                stratify=rna_strata
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed for RNA samples: {e}. Using random split.")
            rna_train_ids, rna_test_ids = train_test_split(
                unique_rna_samples,
                test_size=computed_test_size,
                random_state=random_state
            )
        
        rna_train = rna_filtered[rna_filtered['sample_id'].isin(rna_train_ids)].reset_index(drop=True)
        rna_test = rna_filtered[rna_filtered['sample_id'].isin(rna_test_ids)].reset_index(drop=True)
    else:
        rna_train = rna_filtered
        rna_test = pd.DataFrame(columns=rna_filtered.columns)
    
    # Split imaging treatment data using same computed ratios
    unique_imaging_samples = imaging_filtered['json_key'].unique()
    logger.info(f"Found {len(unique_imaging_samples)} unique imaging sample IDs")
    
    if len(unique_imaging_samples) > 1:
        try:
            if stratify_by is not None:
                imaging_sample_info = imaging_filtered.groupby('json_key').first()
                imaging_strata = create_strata_labels(imaging_sample_info.loc[unique_imaging_samples], stratify_by)
            else:
                imaging_strata = None
            
            imaging_train_ids, imaging_test_ids = train_test_split(
                unique_imaging_samples,
                test_size=computed_test_size,
                random_state=random_state,
                stratify=imaging_strata
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed for imaging samples: {e}. Using random split.")
            imaging_train_ids, imaging_test_ids = train_test_split(
                unique_imaging_samples,
                test_size=computed_test_size,
                random_state=random_state
            )
        
        imaging_train = imaging_filtered[imaging_filtered['json_key'].isin(imaging_train_ids)].reset_index(drop=True)
        imaging_test = imaging_filtered[imaging_filtered['json_key'].isin(imaging_test_ids)].reset_index(drop=True)
    else:
        imaging_train = imaging_filtered
        imaging_test = pd.DataFrame(columns=imaging_filtered.columns)
    
    # Log split results with predicted combination counts
    rna_train_count = len(rna_train['sample_id'].unique())
    rna_test_count = len(rna_test['sample_id'].unique())
    imaging_train_count = len(imaging_train['json_key'].unique())
    imaging_test_count = len(imaging_test['json_key'].unique())
    
    logger.info(f"RNA train: {len(rna_train)} rows, {rna_train_count} unique samples")
    logger.info(f"RNA test: {len(rna_test)} rows, {rna_test_count} unique samples")
    logger.info(f"Imaging train: {len(imaging_train)} rows, {imaging_train_count} unique samples")
    logger.info(f"Imaging test: {len(imaging_test)} rows, {imaging_test_count} unique samples")
    
    # Estimate combination counts (rough approximation)
    est_train_combinations = len(rna_train) * len(imaging_train) / 100  # Rough estimate
    est_test_combinations = len(rna_test) * len(imaging_test) / 100    # Rough estimate
    if est_test_combinations > 0:
        est_ratio = est_train_combinations / est_test_combinations
        logger.info(f"Estimated train combinations: ~{est_train_combinations:.0f}")
        logger.info(f"Estimated test combinations: ~{est_test_combinations:.0f}")
        logger.info(f"Estimated train:test ratio: ~{est_ratio:.1f}:1 (target: {R}:1)")
    
    # Verify no overlap in unique sample IDs
    rna_train_unique = set(rna_train['sample_id'].unique())
    rna_test_unique = set(rna_test['sample_id'].unique())
    imaging_train_unique = set(imaging_train['json_key'].unique())
    imaging_test_unique = set(imaging_test['json_key'].unique())
    
    rna_split_overlap = rna_train_unique.intersection(rna_test_unique)
    imaging_split_overlap = imaging_train_unique.intersection(imaging_test_unique)
    
    if rna_split_overlap:
        logger.error(f"❌ RNA split overlap: {len(rna_split_overlap)} samples in both splits")
        raise ValueError("RNA sample leakage detected!")
    
    if imaging_split_overlap:
        logger.error(f"❌ Imaging split overlap: {len(imaging_split_overlap)} samples in both splits")
        raise ValueError("Imaging sample leakage detected!")
    
    logger.info("✓ No overlap in individual modality splits")
    
    # Create train and test combinations
    train_combinations = create_modality_combinations(rna_train, imaging_train, shared_cell_lines, "train")
    test_combinations = create_modality_combinations(rna_test, imaging_test, shared_cell_lines, "test")
    logger.info(f"Final train combinations: {len(train_combinations)}")
    logger.info(f"Final test combinations: {len(test_combinations)}")

    # Verify final train:test ratio
    if not train_combinations.empty and not test_combinations.empty:
        actual_ratio = len(train_combinations) / len(test_combinations)
        logger.info(f"Actual train:test combination ratio: {actual_ratio:.2f}:1 (target: {R}:1)")
        
        # Check for sample leakage in final combinations
        train_rna_ids = set(train_combinations['sample_id:rna'].astype(str))
        test_rna_ids = set(test_combinations['sample_id:rna'].astype(str))
        train_imaging_ids = set(train_combinations['json_key:hist'].astype(str))
        test_imaging_ids = set(test_combinations['json_key:hist'].astype(str))
        
        rna_overlap = train_rna_ids.intersection(test_rna_ids)
        imaging_overlap = train_imaging_ids.intersection(test_imaging_ids)
        
        if rna_overlap or imaging_overlap:
            logger.error(f"Final combination leakage - RNA: {len(rna_overlap)}, Imaging: {len(imaging_overlap)}")
        else:
            logger.info("✓ No sample leakage in final combinations")
    
    # Process gene matrix if needed
    if use_highly_variable_genes:
        logger.info("Selecting highly variable genes before creating dataloaders")
        logger.info(f"Original gene_count_matrix shape: {gene_count_matrix.shape}")
        adata = ad.AnnData(
            X=gene_count_matrix.T.values,
            obs=gene_count_matrix.T.index.to_frame(),
            var=gene_count_matrix.T.columns.to_frame()
        )
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        gene_count_matrix = gene_count_matrix.loc[hvg_genes]
        logger.info(f"Selected {len(hvg_genes)} highly variable genes")
        logger.info(f"Filtered gene_count_matrix shape: {gene_count_matrix.shape}")
    
    train_metadata_drug = prepare_metadata_for_dataset(train_combinations)
    test_metadata_drug = prepare_metadata_for_dataset(test_combinations)
    logger.info(f"Final train metadata samples: {len(train_metadata_drug)}")
    logger.info(f"train_metadata_drug.head():\n{train_metadata_drug.head()}")
    logger.info(f"Final test metadata samples: {len(test_metadata_drug)}")

    # Create dataloaders (you'll need to import create_raw_drug_dataloader)
    train_loader = None
    test_loader = None
    
    if not train_metadata_drug.empty:
        logger.info(f"Creating train dataloader with {len(train_metadata_drug)} samples")
        train_loader = create_raw_drug_dataloader(
            metadata_control=metadata_control,
            metadata_drug=train_metadata_drug,
            gene_count_matrix=gene_count_matrix,
            image_json_path=image_json_path,
            drug_data_path=drug_data_path,
            raw_drug_csv_path=raw_drug_csv_path,
            use_highly_variable_genes=False,  # Already processed
            compound_name_label=compound_name_label,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            split_train_test=False,
            debug_mode=debug_mode,
            debug_samples=debug_samples,
            debug_cell_lines=debug_cell_lines,
            debug_drugs=debug_drugs,
            **dataloader_kwargs
        )
    else:
        raise ValueError("No training combinations available - train dataloader not created")
    
    if not test_metadata_drug.empty:
        logger.info(f"Creating test dataloader with {len(test_metadata_drug)} samples")
        test_loader = create_raw_drug_dataloader(
            metadata_control=metadata_control,
            metadata_drug=test_metadata_drug,
            gene_count_matrix=gene_count_matrix,
            image_json_path=image_json_path,
            drug_data_path=drug_data_path,
            raw_drug_csv_path=raw_drug_csv_path,
            use_highly_variable_genes=False,  # Already processed
            compound_name_label=compound_name_label,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            split_train_test=False,
            debug_mode=debug_mode,
            debug_samples=debug_samples,
            debug_cell_lines=debug_cell_lines,
            debug_drugs=debug_drugs,
            **dataloader_kwargs
        )
    else:
        raise ValueError("No testing combinations available - test dataloader not created")
    
    logger.info(f"Final train set: {len(train_loader.dataset) if train_loader else 0} combinations")
    logger.info(f"Final test set: {len(test_loader.dataset) if test_loader else 0} combinations")
    
    return train_loader, test_loader

