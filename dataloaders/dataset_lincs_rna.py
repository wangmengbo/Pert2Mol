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


def create_lincs_rna_dataloaders(metadata_control: pd.DataFrame=None,
    metadata_control_path: str="/depot/natallah/data/Mengbo/dataset/chemCPA_data/metadata_control.csv",
    metadata_drug: pd.DataFrame=None,
    metadata_drug_path: str="/depot/natallah/data/Mengbo/dataset/chemCPA_data/metadata_treatment.csv",
    drug_data_path: str="/depot/natallah/data/Mengbo/dataset/chemCPA_data/preprocessed_drug_data.h5",
    raw_drug_csv_path: str="/depot/natallah/data/Mengbo/dataset/chemCPA_data/chemcpa_lincs_drug.csv",
    image_json_path: str = None,
    gene_count_matrix: pd.DataFrame=None,
    gene_count_matrix_path: str="/depot/natallah/data/Mengbo/dataset/chemCPA_data/gene_count_matrix.harmony.parquet",
    compound_name_label: str = 'compound',
    cell_type_label: str = 'cell_line',
    smiles_label: str = 'canonical_smiles',
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    transform=None,
    target_size: int = 256,
    use_highly_variable_genes: bool = False,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,
    exclude_cell_lines: Optional[List[str]] = None,
    exclude_drugs: Optional[List[str]] = None,
    debug_mode: bool = False,
    debug_samples: int = 50,
    include_cell_lines: Optional[List[str]] = None,
    include_drugs: Optional[List[str]] = None,
    smiles_cache: Optional[Dict] = None,
    split_train_test: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    return_datasets: bool = False,
    **kwargs):
    logger.info(f"{locals()}")

    metadata_control = pd.read_csv(metadata_control_path)
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
        shuffle_test=shuffle,
        stratify_by=['cell_line', 'compound'],
        cell_type_label=cell_type_label,
        compound_name_label=compound_name_label,
        smiles_label=smiles_label,
        use_highly_variable_genes=use_highly_variable_genes,
        prepare_metadata_for_dataset=prepare_metadata_for_dataset,
        exclude_cell_lines=exclude_cell_lines,
        exclude_drugs=exclude_drugs,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        include_cell_lines=include_cell_lines,
        include_drugs=include_drugs,
        random_state=random_state,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    if return_datasets:
        return train_loader.dataset, test_loader.dataset
    else:
        return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_lincs_rna_dataloaders()
    # Test the first batch
    for batch in train_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)} - {len(v)} items")
        break

    # validate_dosage_integration(train_loader)
