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

from .utils import convert_to_aromatic_smiles, scale_down_images, load_drug_data_hdf5

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class HistologyTranscriptomicsDataset(Dataset):
    """
    Custom Dataset for paired histology images and transcriptomics data
    with drug treatment conditioning.
    """
    
    def __init__(self, 
                 metadata_control: pd.DataFrame,
                 metadata_drug: pd.DataFrame, 
                 gene_count_matrix: pd.DataFrame = None,
                 image_json_dict: Dict[str, List[str]] = None,
                 transform: Optional[Callable] = None,
                 target_size: Optional[int] = None,
                 cell_type_label: str = 'cell_line',
                 ):
        """
        Args:
            metadata_control: Control dataset metadata with columns ['cell_line', 'sample_id', 'json_key']
            metadata_drug: Treatment dataset metadata with columns ['cell_line', 'compound', 'timepoint', 
                          'compound_concentration_in_uM', 'sample_id', 'json_key']
            gene_count_matrix: Transcriptomics data with sample_id as columns and genes as rows
            image_json_dict: Dictionary mapping json_key to list of image paths
            transform: Optional transform to be applied on images
            target_size: Optional target size for image resizing
        """
        self.metadata_control = metadata_control
        self.metadata_drug = metadata_drug
        self.gene_count_matrix = gene_count_matrix
        self.image_json_dict = image_json_dict
        self.transform = transform
        self.target_size = target_size
        self.cell_type_label = cell_type_label

        logger.debug(f"self.metadata_drug.columns={self.metadata_drug.columns.tolist()}")

        # Convert relevant columns to appropriate types
        for k in ['sample_id', self.cell_type_label, 'json_key', 'compound']:
            if k in self.metadata_control.columns:
                self.metadata_control[k] = self.metadata_control[k].astype(str)
            if k in self.metadata_drug.columns:
                self.metadata_drug[k] = self.metadata_drug[k].astype(str)
        
        for k in ['timepoint', 'compound_concentration_in_uM']:
            if k in self.metadata_control.columns:
                self.metadata_control[k] = self.metadata_control[k].astype(float)
            if k in self.metadata_drug.columns:
                self.metadata_drug[k] = self.metadata_drug[k].astype(float)

        # Group control metadata by cell_line for efficient sampling
        self.control_grouped = self.metadata_control.groupby(self.cell_type_label)
        
        # Get available cell lines in both datasets
        control_cell_lines = set(self.metadata_control[self.cell_type_label].unique())
        drug_cell_lines = set(self.metadata_drug[self.cell_type_label].unique())
        self.common_cell_lines = control_cell_lines.intersection(drug_cell_lines)
        logger.info(f"Found {len(self.common_cell_lines)} common cell lines between control and drug datasets")
        logger.info(f"#Control cell lines={len(control_cell_lines)} ({control_cell_lines}); #Drug cell lines={len(drug_cell_lines)} ({drug_cell_lines})")
        
        # Filter drug metadata to only include samples with matching control cell lines
        self.filtered_drug_metadata = self.metadata_drug[
            self.metadata_drug[cell_type_label].isin(self.common_cell_lines)
        ].reset_index(drop=True)
        
        # Pre-sample control samples for each treatment sample to ensure reproducibility
        self._create_control_treatment_mapping()
        
        logger.info(f"Dataset initialized with {len(self.filtered_drug_metadata)} treatment samples")
        logger.info(f"#Common cell lines: {len(self.common_cell_lines)}")
        logger.info(f"Pre-sampled {len(self.control_sample_mapping)} control-treatment pairs")
    
    def _create_control_treatment_mapping(self):
        """
        Pre-sample control samples for each treatment sample to ensure proper randomization
        and reproducibility. Uses vectorized sampling to avoid sequential sampling bias.
        """
        self.control_sample_mapping = {}
        
        # Group treatment samples by cell line
        treatment_grouped = self.filtered_drug_metadata.groupby(self.cell_type_label)
        
        for cell_line in self.common_cell_lines:
            if cell_line not in treatment_grouped.groups:
                continue
                
            # Get all treatment samples for this cell line
            treatment_indices = treatment_grouped.get_group(cell_line).index.tolist()
            n_treatment_samples = len(treatment_indices)
            
            # Get all available control samples for this cell line
            control_samples = self.control_grouped.get_group(cell_line)
            n_control_samples = len(control_samples)
            
            if n_control_samples == 0:
                logger.warning(f"No control samples found for cell line: {cell_line}")
                continue
            
            # Sample with replacement if needed (when treatment > control samples)
            replace_needed = n_treatment_samples > n_control_samples
            if replace_needed:
                logger.info(f"Cell line {cell_line}: Sampling {n_treatment_samples} controls from {n_control_samples} available (with replacement)")
            
            # Vectorized sampling: sample all needed controls at once for this cell line
            sampled_controls = control_samples.sample(
                n=n_treatment_samples, 
                replace=replace_needed
            )
            
            # Map each treatment index to its corresponding sampled control
            for i, treatment_idx in enumerate(treatment_indices):
                self.control_sample_mapping[treatment_idx] = sampled_controls.iloc[i]
        
        logger.info(f"Pre-sampled control samples for {len(self.control_sample_mapping)} treatment samples using vectorized approach")
    
    def __len__(self):
        return len(self.filtered_drug_metadata)
    
    def load_multi_channel_images(self, json_key: str) -> np.ndarray:
        """
        Load all TIFF images for a sample and concatenate as 3D array.
        
        Args:
            json_key: Key to locate image paths in the JSON dictionary
            
        Returns:
            3D numpy array of shape (channels, height, width)
        """
        image_paths = self.image_json_dict.get(json_key, [])
        if not image_paths:
            logger.debug(f"No image found for json_key: \"{json_key}\"")
            return np.zeros((4, self.target_size or 512, self.target_size or 512), dtype=np.float32)
        
        # Sort paths to ensure consistent channel order (w1, w2, w3, w4)
        image_paths = sorted(image_paths)
        
        images = []
        for i, path in enumerate(image_paths):
            try:
                img = tifffile.imread(path)
                # Ensure 2D image (H, W)
                if img.ndim > 2:
                    img = img.squeeze()

                images.append(img)
                
                # Log channel information for debugging
                channel_info = "w1=Blue" if "w1" in path else \
                              "w2=Green" if "w2" in path else \
                              "w3=Red" if "w3" in path else \
                              "w4=DeepRed" if "w4" in path else "Unknown"
                logger.debug(f"Loaded channel {i}: {channel_info} from {path}")
                
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                raise
        
        # Stack images along the channel dimension to form (C, H, W)
        images = np.stack(images, axis=0).astype(np.float32)
        
        # Scale images if target_size is provided
        if self.target_size is not None:
            images = scale_down_images(images, self.target_size)
        
        # Apply transform if provided
        if self.transform:
            images = self.transform(images)
        
        return images
    
    def get_transcriptomics_data(self, sample_id: str, adhoc_normalize: bool = False, 
                                 ran_default_size: int=128) -> np.ndarray:
        """
        Extract transcriptomics data for a given sample_id.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            1D numpy array of gene expression values
        """
        if self.gene_count_matrix is not None and sample_id not in self.gene_count_matrix.columns:
            logger.error(f"self.gene_count_matrix.columns[:10]={self.gene_count_matrix.columns[:10].tolist()}")
            logger.error(f"\"{sample_id}\" in self.gene_count_matrix.columns={sample_id in self.gene_count_matrix.columns}")
            logger.error(f"gene_count_matrix.shape={self.gene_count_matrix.shape}")
            logger.warning(f"Sample ID {sample_id} not found in gene count matrix")
            return np.zeros((ran_default_size,), dtype=np.float32)
        
        if not adhoc_normalize:
            return self.gene_count_matrix[sample_id].values.astype(np.float32)
        
        logger.warning("[DEPRECATED] Normalization in get_transcriptomics_data is deprecated. Use external preprocessing.")
        raw_data = self.gene_count_matrix[sample_id].values.astype(np.float32)
        
        # Apply the SAME transformation as the encoder
        log_data = np.log1p(raw_data)  # Log transform

        # Normalize (store global stats for consistency)
        if not hasattr(self, 'global_mean'):
            # Compute global statistics once
            all_log_data = np.log1p(self.gene_count_matrix.values)
            self.global_mean = np.mean(all_log_data)
            self.global_std = np.std(all_log_data)
        
        normalized_data = (log_data - self.global_mean) / (self.global_std + 1e-8)
        
        return normalized_data.astype(np.float32)  # Ensure float32 for consistency
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a paired sample (control + treatment) with conditioning information.
        
        Args:
            idx: Index of the treatment sample
            
        Returns:
            Dictionary containing paired data and conditioning information
        """
        # Get treatment sample metadata
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        cell_line = treatment_sample[self.cell_type_label]
        logger.debug(f"treatment_sample={treatment_sample.to_dict()}")
        
        # Use pre-sampled control sample (fixed pairing for reproducibility)
        control_sample = self.control_sample_mapping[idx]
        logger.debug(f"control_sample={control_sample.to_dict()}")
        
        # Load transcriptomics data
        treatment_transcriptomics = self.get_transcriptomics_data(treatment_sample['sample_id'])
        control_transcriptomics = self.get_transcriptomics_data(control_sample['sample_id'])
        
        # Load multi-channel images
        treatment_images = self.load_multi_channel_images(treatment_sample.get('json_key', ''))
        control_images = self.load_multi_channel_images(control_sample.get('json_key',''))
        
        # Prepare conditioning information
        conditioning_info = {
            'treatment': treatment_sample['compound'],
            'cell_line': cell_line,
            'timepoint': treatment_sample.get('timepoint', 24.0),
            'compound_concentration_in_uM': treatment_sample.get('compound_concentration_in_uM', 1.)
        }

        if np.isnan(conditioning_info.get('timepoint',24.)) or np.isnan(conditioning_info.get('compound_concentration_in_uM', 1.)):
            raise ValueError(f"NaN in conditioning info: {conditioning_info}")

        # Return paired data as tensors (CORRECTED - fixed image assignment)
        return {
            'control_transcriptomics': torch.tensor(control_transcriptomics),
            'treatment_transcriptomics': torch.tensor(treatment_transcriptomics),
            'control_images': torch.tensor(control_images),
            'treatment_images': torch.tensor(treatment_images),
            'conditioning_info': conditioning_info
        }


class DatasetWithDrugs(HistologyTranscriptomicsDataset):
    """
    Dataset that includes drug conditioning information.
    """
    def __init__(self,
                metadata_control: pd.DataFrame,
                metadata_drug: pd.DataFrame,
                drug_data_path: str,
                gene_count_matrix: pd.DataFrame = None,
                image_json_dict: Dict[str, List[str]] = None,
                transform: Optional[Callable] = None,
                target_size: Optional[int] = None,
                drug_encoder: Optional[torch.nn.Module] = None,
                debug_mode: bool = False,
                debug_samples: int = 50,
                debug_cell_lines: Optional[List[str]] = None,
                debug_drugs: Optional[List[str]] = None,
                exclude_drugs: Optional[List[str]] = None,
                fallback_smiles_dict=None,
                enable_smiles_fallback=False,
                smiles_cache: Optional[Dict] = None,
                smiles_only: bool = False,
                cell_type_label: str = 'cell_line',
                compound_name_label: str = 'compound',
                **kwargs
                ):
        """
        Args:
            smiles_only: If True, skip loading drug embeddings and only provide SMILES
        """
        # Store debug parameters
        self.debug_mode = debug_mode
        self.debug_samples = debug_samples
        self.debug_cell_lines = debug_cell_lines
        self.warned_compounds = set()
        self.smiles_only = smiles_only
        
        # Apply debug filtering to metadata BEFORE parent initialization
        if debug_mode:
            original_drug_size = len(metadata_drug)
            original_control_size = len(metadata_control)
            
            # Filter by specific cell lines if provided
            if debug_cell_lines:
                metadata_drug = metadata_drug[metadata_drug[cell_type_label].isin(debug_cell_lines)]
                metadata_control = metadata_control[metadata_control[cell_type_label].isin(debug_cell_lines)]
                print(f"DEBUG MODE: Filtered to cell lines: {debug_cell_lines}")
            
            if debug_drugs:
                metadata_drug = metadata_drug[metadata_drug[compound_name_label].isin(debug_drugs)]
                print(f"DEBUG MODE: Filtered to drugs: {debug_drugs}")
            
            if exclude_drugs:
                metadata_drug = metadata_drug[~metadata_drug[compound_name_label].isin(exclude_drugs)]
                print(f"DEBUG MODE: Excluded drugs: {exclude_drugs}")
            
            # Take only first N samples for debugging
            if debug_samples is not None and debug_samples > 0 and len(metadata_drug) > debug_samples:
                metadata_drug = metadata_drug.head(debug_samples).reset_index(drop=True)
            else:
                logger.warning("DEBUG MODE FALLBACK: debug_samples is None or larger than dataset size; not limiting samples.")
            
            # Ensure indices are reset after filtering
            metadata_drug = metadata_drug.reset_index(drop=True)

            print(f"DEBUG MODE: Reduced dataset size:")
            print(f"Drug metadata: {original_drug_size} → {len(metadata_drug)} samples")
            print(f"Control metadata: {original_control_size} → {len(metadata_control)} samples")
        
        # Initialize parent class with potentially filtered data
        super().__init__(
            metadata_control=metadata_control,
            metadata_drug=metadata_drug,
            gene_count_matrix=gene_count_matrix,
            image_json_dict=image_json_dict,
            transform=transform,
            target_size=target_size,
            cell_type_label=cell_type_label,
        )
        
        # Skip drug embedding loading if only SMILES needed
        if not smiles_only:
            # Load preprocessed drug data
            self.drug_data = self._load_drug_data(drug_data_path)
            self.drug_encoder = drug_encoder

            if self.drug_data is not None:
                self.compound_lookup = self.drug_data['drug_embeddings']
            else:
                self.compound_lookup = {}
                logger.warning("Drug data is None; compound lookup will be empty.")
            
            # Create compound mapping for quick lookup
            self._create_compound_mapping()
            
            if self.drug_data is not None:
                logger.info(f"Loaded drug data for {len(self.drug_data['drug_embeddings'])} compounds")
        else:
            # Skip all drug embedding logic for SMILES-only mode
            self.drug_data = None
            self.compound_lookup = {}
            self.compound_to_embeddings = {}
            self.available_compounds = set()
            logger.info("SMILES-only mode: Skipped drug embedding loading")

        if debug_mode:
            logger.info(f"DEBUG MODE: Final dataset size: {len(self)} samples")

        self.fallback_smiles_dict = fallback_smiles_dict or {}
        self.enable_smiles_fallback = enable_smiles_fallback

        if enable_smiles_fallback and not smiles_only:
            self._init_smiles_processor()

        self.smiles_cache = smiles_cache if smiles_cache is not None else {}
        self.warned_compounds = set()

    def _init_smiles_processor(self):
        """Initialize components for on-demand SMILES processing."""
        try:
            self.rdkit_available = True
            
            # Store processing parameters from drug_data for consistency
            self.fingerprint_size = self.drug_data.get('preprocessing_params', {}).get('fingerprint_size', 1024)
            self.normalize_descriptors = self.drug_data.get('preprocessing_params', {}).get('normalize_descriptors', True)
            
            # Get normalization parameters if they exist
            if 'modality_dims' in self.drug_data:
                sample_drug = next(iter(self.drug_data['drug_embeddings'].values()))
                if 'descriptors_2d' in sample_drug and hasattr(self, 'normalization_params'):
                    self.desc_mean = self.normalization_params.get('mean')
                    self.desc_std = self.normalization_params.get('std')
        except ImportError:
            logger.warning("RDKit not available - SMILES fallback disabled")
            self.rdkit_available = False

    def _compute_smiles_embeddings(self, smiles: str) -> Dict[str, torch.Tensor]:
        """Convert SMILES to drug embeddings on-demand."""
        if not self.rdkit_available:
            logger.warning(f"Cannot process SMILES {smiles} - RDKit not available")
            return self._get_zero_embeddings()
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, RDKFingerprint
            
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return self._get_zero_embeddings()
            
            # Compute Morgan fingerprint
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=self.fingerprint_size
            )
            fp_morgan = np.array(fp_morgan, dtype=np.float32)
            
            # Compute RDKit fingerprint
            fp_rdkit = RDKFingerprint(mol, fpSize=self.fingerprint_size)
            fp_rdkit = np.array(fp_rdkit, dtype=np.float32)
            
            # Compute 2D descriptors (same as drug_process.py)
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcExactMolWt(mol)
            ]
            descriptors_2d = np.array(descriptors, dtype=np.float32)
            
            # Apply normalization if available
            if self.normalize_descriptors and hasattr(self, 'desc_mean') and self.desc_mean is not None:
                descriptors_2d = (descriptors_2d - self.desc_mean) / (self.desc_std + 1e-8)
            
            # Convert to tensors
            result = {
                'fingerprint_morgan': torch.from_numpy(fp_morgan).float(),
                'fingerprint_rdkit': torch.from_numpy(fp_rdkit).float(),
                'descriptors_2d': torch.from_numpy(descriptors_2d).float(),
                'descriptors_3d': torch.zeros(5, dtype=torch.float32),  # No 3D info from SMILES
                'has_3d_structure': False,
                'has_2d_structure': False,  # Computed from SMILES, not actual 2D file
                'structure_source': 'SMILES_ON_DEMAND',
                'smiles': smiles
            }
            
            logger.info(f"Generated embeddings for new drug from SMILES: {smiles[:2]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return self._get_zero_embeddings()

    def _load_drug_data(self, drug_data_path: str) -> Dict:
        """Load preprocessed drug data."""
        try:
            if drug_data_path.endswith('.pickle'):
                with open(drug_data_path, 'rb') as f:
                    drug_data = pickle.load(f)
            elif drug_data_path.endswith('.h5') or drug_data_path.endswith('.hdf5'):
                drug_data = load_drug_data_hdf5(drug_data_path)
            else:
                raise ValueError("Unsupported drug data file format. Use .pickle or .h5/.hdf5")
            logger.info(f"Loaded preprocessed drug data from {drug_data_path}")
            return drug_data
        except Exception as e:
            logger.error(f"Failed to load drug data from {drug_data_path}: {e}")
            # raise
            return None
    
    def _create_compound_mapping(self):
        """Create mapping from compound names to drug embeddings."""
        if self.drug_data is None or 'drug_embeddings' not in self.drug_data:
            logger.warning("Drug data is None or missing 'drug_embeddings'; compound mapping will be empty.")
            self.compound_to_embeddings = {}
            self.available_compounds = set()
        else:
            self.compound_to_embeddings = self.drug_data['drug_embeddings']
            self.available_compounds = set(self.compound_to_embeddings.keys())
        
        # Check coverage
        required_compounds = set(self.metadata_drug['compound'].unique())
        missing_compounds = required_compounds - self.available_compounds
        
        if missing_compounds:
            logger.warning(f"Missing drug data for compounds: {missing_compounds}")
    
    def get_drug_embeddings(self, compound_name: str) -> Dict[str, torch.Tensor]:
        """Enhanced lookup with SMILES fallback."""
        # Try preprocessed lookup first
        if compound_name in self.smiles_cache:
            return self.smiles_cache[compound_name]

        if compound_name in self.compound_lookup:
            embeddings = self.compound_lookup[compound_name]
            result = {}
            for key, value in embeddings.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value).float()
                    result[key] = tensor.contiguous().clone()
                else:
                    result[key] = value
            self.smiles_cache[compound_name] = result
            return result
        # Try SMILES fallback if enabled
        elif self.enable_smiles_fallback:
            # Check if we have SMILES for this compound
            smiles = self.fallback_smiles_dict.get(compound_name)
            if smiles:
                if compound_name not in self.warned_compounds:
                    logger.info(f"Using SMILES fallback for unknown compound: {compound_name}")
                    self.warned_compounds.add(compound_name)
                
                computed_embeddings = self._compute_smiles_embeddings(smiles)
                # Store the new embeddings in the shared cache
                self.smiles_cache[compound_name] = computed_embeddings
                return computed_embeddings
            else:
                # Only warn once per compound
                if compound_name not in self.warned_compounds:
                    logger.warning(f"No SMILES available for unknown compound: {compound_name}")
                    self.warned_compounds.add(compound_name)
                zero_embeddings = self._get_zero_embeddings()
                # Also cache the "not found" result
                self.smiles_cache[compound_name] = zero_embeddings
                return zero_embeddings
        else:
            zero_embeddings = self._get_zero_embeddings()
            self.smiles_cache[compound_name] = zero_embeddings
            return zero_embeddings

    def _get_zero_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get zero embeddings for missing compounds."""
        # Handle case where drug_data is None
        if self.drug_data is None:
            modality_dims = {
                'fingerprint_morgan': 1024,
                'fingerprint_rdkit': 1024,
                'descriptors_2d': 18
            }
        else:
            modality_dims = self.drug_data.get('modality_dims', {
                'fingerprint_morgan': 1024,
                'fingerprint_rdkit': 1024,
                'descriptors_2d': 18
            })
        
        zero_embeddings = {
            'fingerprint_morgan': torch.zeros(
                modality_dims.get('fingerprint_morgan', 1024), 
                dtype=torch.float32
            ).contiguous(),
            'fingerprint_rdkit': torch.zeros(
                modality_dims.get('fingerprint_rdkit', 1024), 
                dtype=torch.float32
            ).contiguous(),
            'descriptors_2d': torch.zeros(
                modality_dims.get('descriptors_2d', 18), 
                dtype=torch.float32
            ).contiguous(),
            'descriptors_3d': torch.zeros(5, dtype=torch.float32).contiguous(),
            'has_3d_structure': False,
            'has_2d_structure': False,
            'structure_source': 'NONE',
            'smiles': ''
        }
        
        return zero_embeddings
    
    def encode_drug_condition(self, compound_name: str) -> torch.Tensor:
        """
        Encode drug into condition embedding.
        
        Args:
            compound_name: Name of the compound
            
        Returns:
            Drug condition tensor
        """
        drug_embeddings = self.get_drug_embeddings(compound_name)
        
        if self.drug_encoder is not None:
            # Use trained drug encoder
            # Create a mini-batch with single item
            batch_dict = {key: value.unsqueeze(0) for key, value in drug_embeddings.items() 
                         if isinstance(value, torch.Tensor)}
            batch_dict.update({key: [value] for key, value in drug_embeddings.items() 
                              if not isinstance(value, torch.Tensor)})
            
            with torch.no_grad():
                drug_condition = self.drug_encoder(batch_dict).squeeze(0)
            return drug_condition
        else:
            # Use raw embeddings (concatenate main modalities)
            main_embeddings = [
                drug_embeddings['fingerprint_morgan'],
                drug_embeddings['descriptors_2d']
            ]
            # Add 3D descriptors if available
            if drug_embeddings['descriptors_3d'].numel() > 0:
                main_embeddings.append(drug_embeddings['descriptors_3d'])
                
            return torch.cat(main_embeddings, dim=0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Enhanced getitem that includes drug conditioning or just SMILES.
        
        Returns:
            Dictionary with original data plus drug conditioning or just SMILES
        """
        # Get original data
        sample = super().__getitem__(idx)
        
        # Get drug information
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        compound_name = treatment_sample['compound']
        
        # Add compound name
        sample['compound_name'] = compound_name
        
        if self.smiles_only:
            # Only add SMILES, skip all drug embedding logic
            smiles = self.fallback_smiles_dict.get(compound_name, '')
            sample.update({
                'target_smiles': smiles,
                'drug_condition': torch.zeros(1047, dtype=torch.float32)  # Empty drug condition
            })
        else:
            # Add drug embeddings and condition (original behavior)
            drug_embeddings = self.get_drug_embeddings(compound_name)
            drug_condition = self.encode_drug_condition(compound_name)
            
            sample.update({
                'drug_embeddings': drug_embeddings,
                'drug_condition': drug_condition,
            })
        
        return sample


class RawDrugDataset(DatasetWithDrugs):
    """Dataset that handles raw PubChem CSV drug data and biological conditioning"""
    
    def __init__(self,
                metadata_control: pd.DataFrame,
                metadata_drug: pd.DataFrame,
                drug_data_path: str = None,
                raw_drug_csv_path: str = None,
                gene_count_matrix: pd.DataFrame = None,
                image_json_dict: Dict[str, List[str]] = None,
                transform=None,
                target_size=256,
                debug_mode=False,
                compound_name_label='compound',
                cell_type_label='cell_line',
                smiles_label='canonical_smiles',
                smiles_cache: Optional[Dict] = None,
                smiles_only: bool = False,
                **kwargs):
        
        # Load raw drug CSV data
        self.compound_name_label = compound_name_label
        self.raw_drug_df = pd.read_csv(raw_drug_csv_path)
        logger.info(f"Loaded {len(self.raw_drug_df)} raw drug entries from {raw_drug_csv_path}")
        
        # Create drug name to SMILES mapping
        self.drug_name_to_smiles = {}
        for _, row in self.raw_drug_df.iterrows():
            if pd.notna(row[smiles_label]) and pd.notna(row[self.compound_name_label]):
                self.drug_name_to_smiles[row[self.compound_name_label]] = convert_to_aromatic_smiles(row[smiles_label])
        
        logger.info(f"Created SMILES mapping for {len(self.drug_name_to_smiles)} drugs")
        
        if smiles_cache is None:
            self.smiles_cache = {}
        else:
            self.smiles_cache = smiles_cache
            
        # Initialize parent with fallback SMILES dictionary
        super().__init__(
            metadata_control=metadata_control,
            metadata_drug=metadata_drug,
            gene_count_matrix=gene_count_matrix,
            image_json_dict=image_json_dict,
            drug_data_path=drug_data_path,
            fallback_smiles_dict=self.drug_name_to_smiles,
            enable_smiles_fallback=False,
            transform=transform,
            target_size=target_size,
            cell_type_label=cell_type_label,
            debug_mode=debug_mode,
            smiles_cache=smiles_cache,
            smiles_only=smiles_only,
            **kwargs
        )
    
    def get_target_drug_info(self, idx):
        """Get target drug information for conditional generation"""
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        compound_name = treatment_sample['compound']
        
        drug_info = {
            'compound_name': compound_name,
            'smiles': self.drug_name_to_smiles.get(compound_name, ''),
        }
        
        # Add additional drug properties if available in raw CSV
        if compound_name in self.drug_name_to_smiles:
            drug_row = self.raw_drug_df[
                self.raw_drug_df[self.compound_name_label] == compound_name
            ]
            if not drug_row.empty:
                row = drug_row.iloc[0]
                drug_info.update({
                    'molecular_weight': row.get('molecular_weight', 0),
                    'xlogp': row.get('xlogp', 0),
                    'tpsa': row.get('tpsa', 0),
                })
        
        return drug_info
    
    def __getitem__(self, idx):
        # Get base sample with biological data
        sample = super().__getitem__(idx)
        
        # Add target drug information for conditional training
        target_drug_info = self.get_target_drug_info(idx)
        sample.update({
            'target_drug_info': target_drug_info,
            'target_smiles': target_drug_info['smiles']
        })
        
        return sample

    def _init_smiles_processor(self):
        """Initialize components for on-demand SMILES processing."""
        try:
            self.rdkit_available = True
            
            if self.drug_data is not None:
                self.fingerprint_size = self.drug_data.get('preprocessing_params', {}).get('fingerprint_size', 1024)
                self.normalize_descriptors = self.drug_data.get('preprocessing_params', {}).get('normalize_descriptors', True)
            else:
                self.fingerprint_size = 1024
                self.normalize_descriptors = True
            
            if self.drug_data is not None and ('modality_dims' in self.drug_data and self.drug_data['drug_embeddings']):
                sample_drug = next(iter(self.drug_data['drug_embeddings'].values()))
                if 'descriptors_2d' in sample_drug and hasattr(self, 'normalization_params'):
                    self.desc_mean = self.normalization_params.get('mean')
                    self.desc_std = self.normalization_params.get('std')
            else:
                self.desc_mean = None
                self.desc_std = None
                
        except ImportError:
            logger.warning("RDKit not available - SMILES fallback disabled")
            self.rdkit_available = False