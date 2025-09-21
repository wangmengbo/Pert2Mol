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


def create_cpgjump_dataloaders(
    metadata_control: pd.DataFrame=None,
    metadata_control_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/cpg/cpg0000-jump-pilot/metadata_control.csv",
    metadata_drug: pd.DataFrame=None,
    metadata_drug_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/cpg/cpg0000-jump-pilot/metadata_drug.csv",
    drug_data_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/cpg/cpg0000-jump-pilot/preprocessed_drug_data.h5",
    raw_drug_csv_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/cpg/cpg0000-jump-pilot/compound_metadata.csv",
    image_json_path: str="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/dataset/cpg/cpg0000-jump-pilot/image_paths.4_channels.json",
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
    debug_samples: int = 256,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    smiles_cache: Optional[Dict] = None,
    split_train_test: bool = True,
    test_size: float = 0.1,
    return_datasets: bool = False,  # Add this parameter
    random_state: int = 42,
    **kwargs
    ):
    if metadata_control is None:
        metadata_control = pd.read_csv(metadata_control_path)

    if metadata_drug is None:
        metadata_drug = pd.read_csv(metadata_drug_path)

    # Create leak-free train/test dataloaders
    train_loader, test_loader = create_leak_free_dataloaders(
        metadata_control=metadata_control,
        drug_data_path=drug_data_path,
        raw_drug_csv_path=raw_drug_csv_path,
        metadata_rna=None,
        metadata_imaging=metadata_drug,
        image_json_path=image_json_path,
        final_train_test_ratio = (1-test_size)/test_size,
        batch_size=batch_size,
        shuffle=True,
        stratify_by=['Cell_type', 'compound'],
        cell_type_label='Cell_type',
        compound_name_label='compound',
        smiles_label='smiles',
        prepare_metadata_for_dataset=prepare_metadata_for_dataset,
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
    train_loader, test_loader = create_cpgjump_dataloaders()
    # Test the first batch
    for batch in train_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)} - {len(v)} items")
        break

    validate_dosage_integration(train_loader)

