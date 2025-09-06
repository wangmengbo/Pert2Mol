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
from .dataloader import (
	DatasetWithDrugs, image_transform, create_raw_drug_dataloader, 
	create_leak_free_dataloaders
)
from .utils import validate_dosage_integration

logger = logging.getLogger(__name__)


# Create combinations within each split
def create_modality_combinations(rna_df, imaging_df, shared_cell_lines, suffix="train", compound_name_label='compound'):
    """Create paired combinations within a split"""
    # logger.info(f"rna_df shape: {rna_df.shape}, imaging_df shape: {imaging_df.shape}")
    # logger.info(f"rna_df columns: {rna_df.columns.tolist()}")
    combinations = []

    def create_combined_row(rna_row, imaging_row, compound_name_label='compound'):
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
        # logger.info(f"A549 RNA samples: {len(a549_rna)}, Imaging samples: {len(a549_imaging)}")
        
        if not a549_rna.empty and not a549_imaging.empty:
            for _, rna_row in a549_rna.iterrows():
                matching_imaging = a549_imaging[
                    (a549_imaging[compound_name_label] == rna_row[compound_name_label]) &
                    (a549_imaging['timepoint'] == rna_row['timepoint'])
                ]
                
                for _, imaging_row in matching_imaging.iterrows():
                    combination = create_combined_row(rna_row, imaging_row)
                    combinations.append(combination)
        else:
            logger.warning("No A549 samples found in either RNA or imaging data for this split")
    # logger.info(f"A549 combinations created: {len(combinations)}")

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
                        combination = create_combined_row(rna_row, imaging_row)
                        combinations.append(combination)
    # logger.info(f"Total combinations created (including A549): {len(combinations)}")

    if combinations:
        combined_df = pd.DataFrame(combinations)
        logger.info(f"Created {len(combined_df)} {suffix} combinations")
        logger.info(f"combined_df.head():\n{combined_df.head()}")
        return combined_df
    else:
        logger.warning(f"No {suffix} combinations could be created")
        return pd.DataFrame()


def prepare_metadata_for_dataset(combined_df, compound_name_label='compound'):
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


def create_gdp_dataloaders(metadata_control: pd.DataFrame=None,
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
    return_datasets: bool = False,  # Add this parameter
    **kwargs
    ):
    # Your file paths
    IMAGE_JSON_PATH = "/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/image_paths.json"
    DRUG_DATA_PATH = "/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/PertRF/drug/PubChem/GDP_compatible/preprocessed_drugs.synonymous.pkl"
    RAW_DRUG_CSV_PATH = "/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/PertRF/drug/PubChem/GDP_compatible/complete_drug_data.csv"
    METADATA_CONTROL_PATH = "/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/metadata_control.csv"
    GENE_COUNT_MATRIX_PATH = "/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/GDPx1x2_gene_counts.parquet"

    # Load your original metadata
    gdpx1x2_metadata = pd.read_csv("/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/data/GDP_data/processed_data/GDPx1x2_metadata.csv")
    gdpx3_metadata = pd.read_csv("/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/data/GDP_data/GDPx3/metadata_updated.csv")
    if gene_count_matrix is None:
        gene_count_matrix = pd.read_parquet(GENE_COUNT_MATRIX_PATH)
    
    # Find shared conditions
    shared_drugs = list(set(gdpx3_metadata['compound'].unique()).intersection(
        gdpx1x2_metadata['compound'].unique()
    ))
    shared_drugs = [drug for drug in shared_drugs if drug != 'DMSO']
    logger.info(f"Found {len(shared_drugs)} shared drugs (excluding DMSO)")

    shared_cell_lines = list(set(gdpx3_metadata['cell_line'].unique()).intersection(
        gdpx1x2_metadata['cell_line'].unique()
    ))
    logger.info(f"Found {len(shared_cell_lines)} shared cell lines: {shared_cell_lines}")
    
    # Create your control metadata (keep this as is - it's fine to share controls)
    metadata_control = []
    
    # A549 control (special handling for dosage mismatch)
    metadata_control.append(pd.merge(
        gdpx3_metadata[(gdpx3_metadata['compound'] == 'DMSO') & (gdpx3_metadata['cell_line'] == 'A549')],
        gdpx1x2_metadata[(gdpx1x2_metadata['compound'] == 'DMSO') & (gdpx1x2_metadata['cell_line'] == 'A549')],
        left_on=['compound', 'cell_line', 'compound_concentration_in_uM', 'timepoint'],
        right_on=['compound', 'cell_line', 'compound_concentration_in_uM', 'timepoint'],
        suffixes=(':hist', ':rna'),
    ))
    
    # Other cell lines control (exact matching)
    metadata_control.append(pd.merge(
        gdpx3_metadata[(gdpx3_metadata['compound'] == 'DMSO') & (gdpx3_metadata['cell_line'] != 'A549')],
        gdpx1x2_metadata[(gdpx1x2_metadata['compound'] == 'DMSO') & (gdpx1x2_metadata['cell_line'] != 'A549')],
        left_on=['compound', 'cell_line', 'timepoint'],
        right_on=['compound', 'cell_line', 'timepoint'],
        suffixes=(':hist', ':rna'),
    ))
    
    metadata_control = pd.concat(metadata_control, ignore_index=True).drop_duplicates()
    
    # Fix data types
    for col in metadata_control.columns:
        if col.endswith('id'):
            metadata_control[col] = metadata_control[col].astype(str)

    # Create leak-free train/test dataloaders
    train_loader, test_loader = create_leak_free_dataloaders(
        metadata_control=metadata_control,
        metadata_rna=gdpx1x2_metadata[gdpx1x2_metadata['compound'] != 'DMSO'],
        metadata_imaging=gdpx3_metadata[gdpx3_metadata['compound'] != 'DMSO'],
        shared_drugs=shared_drugs,
        shared_cell_lines=shared_cell_lines,
        gene_count_matrix=gene_count_matrix,
        image_json_path=IMAGE_JSON_PATH,
        drug_data_path=DRUG_DATA_PATH,
        raw_drug_csv_path=RAW_DRUG_CSV_PATH,
        # test_size=test_size,
        final_train_test_ratio = (1-test_size)/test_size,
        batch_size=batch_size,
        shuffle=True,
        stratify_by=['cell_line', 'compound'],
        use_highly_variable_genes=True,
        n_top_genes=n_top_genes,
        prepare_metadata_for_dataset=prepare_metadata_for_dataset,
        create_modality_combinations=create_modality_combinations,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        debug_cell_lines=debug_cell_lines,
        debug_drugs=debug_drugs
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    if return_datasets:
        return train_loader.dataset, test_loader.dataset
    else:
        return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_gdp_dataloaders()
    # Test the first batch
    for batch in train_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)} - {len(v)} items")
        break

    validate_dosage_integration(train_loader)

