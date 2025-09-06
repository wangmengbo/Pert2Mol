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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_drug_data_hdf5(file_path: str) -> dict:
    """
    Load drug data from HDF5 with exact same structure as pickle version.
    Drop-in replacement for load_preprocessed_drug_data.
    """
    loaded_data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Load metadata from attributes
        for key in f.attrs.keys():
            attr_value = f.attrs[key]
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8')
            
            # Try to parse as JSON for complex structures
            if key in ['modality_dims', 'dataset_stats', 'preprocessing_params']:
                try:
                    loaded_data[key] = json.loads(attr_value)
                except (json.JSONDecodeError, TypeError):
                    loaded_data[key] = attr_value
            else:
                loaded_data[key] = attr_value
        
        # Load drug embeddings
        drug_embeddings = {}
        if 'drug_embeddings' in f:
            for compound_name in f['drug_embeddings'].keys():
                compound_group = f['drug_embeddings'][compound_name]
                embeddings = {}
                
                # Load datasets (numpy arrays) - check type first
                for dataset_name in compound_group.keys():
                    item = compound_group[dataset_name]
                    if isinstance(item, h5py.Dataset):
                        embeddings[dataset_name] = item[:]
                    # Skip groups - they should not be here for embeddings
                
                # Load attributes (scalars, strings)
                for attr_name in compound_group.attrs.keys():
                    attr_value = compound_group.attrs[attr_name]
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8')
                    embeddings[attr_name] = attr_value
                
                drug_embeddings[compound_name] = embeddings
        
        loaded_data['drug_embeddings'] = drug_embeddings
        
        # Load compound metadata
        compound_metadata = {}
        if 'compound_metadata' in f:
            for compound_name in f['compound_metadata'].keys():
                metadata_group = f['compound_metadata'][compound_name]
                metadata = {}
                
                for attr_name in metadata_group.attrs.keys():
                    attr_value = metadata_group.attrs[attr_name]
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8')
                    
                    # Try to convert back to appropriate type
                    if attr_name in ['cid']:
                        try:
                            metadata[attr_name] = int(attr_value)
                        except (ValueError, TypeError):
                            metadata[attr_name] = attr_value
                    elif attr_name in ['molecular_weight']:
                        try:
                            metadata[attr_name] = float(attr_value)
                        except (ValueError, TypeError):
                            metadata[attr_name] = attr_value
                    else:
                        metadata[attr_name] = attr_value
                
                compound_metadata[compound_name] = metadata
        
        loaded_data['compound_metadata'] = compound_metadata
    
    return loaded_data


def scale_down_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """
    Scale down multi-channel images to a given square pixel size using bilinear interpolation.
    
    Args:
        images: numpy array of shape (channels, height, width)
        target_size: int, target size for height and width (square)
        
    Returns:
        numpy array of scaled images with shape (channels, target_size, target_size)
    """
    # Convert numpy array to torch tensor
    images_tensor = torch.tensor(images)
    
    # Add batch dimension and ensure float32
    images_tensor = images_tensor.unsqueeze(0).float()  # (1, C, H, W)
    
    # Resize using bilinear interpolation
    images_resized = F.interpolate(
        images_tensor, 
        size=(target_size, target_size), 
        mode='bilinear',
        align_corners=False
    )
    
    # Remove batch dimension
    images_resized = images_resized.squeeze(0)
    
    # Convert back to numpy
    images_resized_np = images_resized.numpy()
    
    return images_resized_np


def convert_to_aromatic_smiles(smiles):
    """Convert SMILES to aromatic notation with lowercase aromatic atoms."""
    if not smiles:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            aromatic_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
            return aromatic_smiles
        else:
            return smiles
    except:
        return smiles


def validate_dosage_integration(dataloader, num_batches=3):
    """Validate that dosage information is properly included"""
    
    dosages_seen = []
    compounds_seen = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        print(f"\n--- Batch {i+1} ---")
        print(f"Batch size: {len(batch['conditioning_info'])}")
        
        for j, cond_info in enumerate(batch['conditioning_info']):
            dosage = cond_info['compound_concentration_in_uM']
            compound = cond_info['treatment']
            cell_line = cond_info['cell_line']
            timepoint = cond_info['timepoint']
            
            dosages_seen.append(dosage)
            compounds_seen.append(compound)
            
            print(f"  Sample {j}: {compound} at {dosage}µM in {cell_line} (t={timepoint}h)")
    
    print(f"\nSummary:")
    print(f"Unique dosages: {sorted(set(dosages_seen))}")
    print(f"Unique compounds: {set(compounds_seen)}")
    print(f"Dosage range: {min(dosages_seen):.3f} - {max(dosages_seen):.3f} µM")

