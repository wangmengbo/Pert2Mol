import os
import sys
import logging
import json
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloaders.dataloader import (
	DatasetWithDrugs, image_transform, create_raw_drug_dataloader, 
	create_leak_free_dataloaders
)
from dataloaders.utils import validate_dosage_integration

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def prepare_metadata_for_dataset(df, compound_name_label='compound'):
    return df


def create_tahoe_dataloaders(
    metadata_control: pd.DataFrame=None,
    metadata_control_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/Tahoe-100M/metadata_control_sub.csv",
    metadata_drug: pd.DataFrame=None,
    metadata_drug_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/Tahoe-100M/metadata_drug_sub.csv",
    drug_data_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/Tahoe-100M/preprocessed_drug_data.h5",
    raw_drug_csv_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/Tahoe-100M/drug_metadata.csv",
    image_json_path: str = None,
    gene_count_matrix: pd.DataFrame = "/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/Tahoe-100M/gene_count_matrix.sub.parquet",
    compound_name_label: str = 'compound',
    smiles_label: str = 'canonical_smiles',
    batch_size: int = 4,
    shuffle: bool = True,
    shuffle_test: bool = False,
    num_workers: int = 0,
    transform=None,
    target_size: int = 256,
    use_highly_variable_genes: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,
    exclude_cell_lines: Optional[List[str]] = None,
    exclude_drugs: Optional[List[str]] = None,
    debug_mode: bool = False,
    debug_samples: int = 256,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    smiles_cache: Optional[Dict] = None,
    split_train_test: bool = True,
    test_size: float = 0.2,
    return_datasets: bool = False,  # Add this parameter
    random_state: int = 42,
    **kwargs
    ):

    if metadata_control is None:
        metadata_control = pd.read_csv(metadata_control_path)

    if metadata_drug is None:
        metadata_drug = pd.read_csv(metadata_drug_path)

    if gene_count_matrix is None:
        gene_count_matrix = pd.read_parquet(gene_count_matrix_path).T

    # Create leak-free train/test dataloaders
    train_loader, test_loader = create_leak_free_dataloaders(
        metadata_control=metadata_control,
        drug_data_path=drug_data_path,
        raw_drug_csv_path=raw_drug_csv_path,
        metadata_rna=metadata_drug,
        metadata_imaging=None,
        image_json_path=None,
        final_train_test_ratio = (1-test_size)/test_size,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_test=shuffle_test,
        stratify_by=['cell_name', 'compound'],
        cell_type_label='cell_name',
        extra_labels_to_match=['plate'],
        compound_name_label='compound',
        smiles_label='canonical_smiles',
        use_highly_variable_genes=False,
        prepare_metadata_for_dataset=prepare_metadata_for_dataset,
        exclude_cell_lines=exclude_cell_lines,
        exclude_drugs=exclude_drugs,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        debug_cell_lines=debug_cell_lines,
        debug_drugs=debug_drugs,
        random_state=random_state,
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    if return_datasets:
        return train_loader.dataset, test_loader.dataset
    else:
        return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_tahoe_dataloaders()
    # Test the first batch
    for batch in train_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)} - {len(v)} items")
        break

    validate_dosage_integration(train_loader)

