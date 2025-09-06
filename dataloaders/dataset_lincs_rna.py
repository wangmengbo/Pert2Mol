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
from .dataloader import DatasetWithDrugs, image_transform, create_raw_drug_dataloader

logger = logging.getLogger(__name__)


def create_lincs_rna_dataloaders(metadata_control: pd.DataFrame=None,
    metadata_drug: pd.DataFrame=None,
    drug_data_path: str=None,
    raw_drug_csv_path: str=None,
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
    split_train_test: bool = False,
    test_size: float = 0.2,
    **kwargs):
    # Your file paths
    IMAGE_JSON_PATH="N/A"
    DRUG_DATA_PATH="/depot/natallah/data/Mengbo/dataset/chemCPA_data/preprocessed_drug_data.h5"
    RAW_DRUG_CSV_PATH="/depot/natallah/data/Mengbo/dataset/chemCPA_data/chemcpa_lincs_drug.csv"
    METADATA_CONTROL_PATH="/depot/natallah/data/Mengbo/dataset/chemCPA_data/metadata_control.csv"
    METADATA_DRUG_PATH="/depot/natallah/data/Mengbo/dataset/chemCPA_data/metadata_treatment.csv"
    GENE_COUNT_MATRIX_PATH="/depot/natallah/data/Mengbo/dataset/chemCPA_data/gene_counts.parquet"
    COMPOUND_NAME_LABEL="pert_iname" 

    metadata_control = pd.read_csv(METADATA_CONTROL_PATH)
    metadata_drug = pd.read_csv(METADATA_DRUG_PATH)
    gene_count_matrix = pd.read_parquet(GENE_COUNT_MATRIX_PATH).T

    train_loader, test_loader = create_raw_drug_dataloader(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug, 
        gene_count_matrix=gene_count_matrix,
        image_json_path=IMAGE_JSON_PATH,
        drug_data_path=DRUG_DATA_PATH,
        raw_drug_csv_path=RAW_DRUG_CSV_PATH,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        compound_name_label=COMPOUND_NAME_LABEL,
        debug_mode=debug_mode,
        split_train_test=True,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
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
