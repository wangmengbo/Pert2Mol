# Inspired by LDMOL https://arxiv.org/pdf/2405.17829

from absl import logging
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from rdkit import RDLogger
import re
import multiprocessing as mp
from functools import partial
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import warnings
import json
import hashlib
import os
import signal
from glob import glob
import random
import argparse
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


@torch.no_grad()
def AE_SMILES_encoder(sm, ae_model):
    if sm[0][:5] == "[CLS]":    sm = [s[5:] for s in sm]
    text_input = ae_model.tokenizer(sm).to(ae_model.device)
    text_input_ids = text_input
    text_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(text_input.device)
    if hasattr(ae_model.text_encoder2, 'bert'):
        output = ae_model.text_encoder2.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
    else:
        output = ae_model.text_encoder2(text_input_ids, attention_mask=text_attention_mask, return_dict=True).last_hidden_state

    if hasattr(ae_model, 'encode_prefix'):
        output = ae_model.encode_prefix(output)
        if ae_model.output_dim*2 == output.size(-1):
            mean, logvar = torch.chunk(output, 2, dim=-1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            output = mean + std * torch.randn_like(mean)
    return output


@torch.no_grad()
def generate(model, image_embeds, text, stochastic=True, prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    token_output = model.text_encoder(text,
                                      attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=prop_att_mask,
                                      return_dict=True,
                                      is_decoder=True,
                                      return_logits=True,
                                      )[:, -1, :]  # batch*300
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1


@torch.no_grad()
def AE_SMILES_decoder(pv, model, stochastic=False, k=2, max_length=150):
    if hasattr(model, 'decode_prefix'):
        pv = model.decode_prefix(pv)

    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError('Tokenizer is not defined')
    # test
    model.eval()
    candidate = []
    if k == 1:
        text_input = torch.tensor([tokenizer.cls_token_id]).expand(pv.size(0), 1).to(model.device)  # batch*1
        for _ in range(max_length):
            output = generate(model, pv, text_input, stochastic=False)
            if output.sum() == 0:
                break
            
            text_input = torch.cat([text_input, output], dim=-1)
        for i in range(text_input.size(0)):
            sentence = text_input[i]
            cdd = tokenizer.decode(sentence)[0]#newtkn
            candidate.append(cdd)
    else:
        for prop_embeds in pv:
            prop_embeds = prop_embeds.unsqueeze(0)
            product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(model.device)
            values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
            product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(model.device), indices.squeeze(0).unsqueeze(-1)], dim=-1)
            current_p = values.squeeze(0)
            final_output = []
            for _ in range(max_length):
                values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
                k2_p = current_p[:, None] + values
                product_input_k2 = torch.cat([product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1)
                if tokenizer.sep_token_id in indices:
                    ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                    for e in ends:
                        p = k2_p[e[0], e[1]].cpu().item()
                        final_output.append((p, product_input_k2[e[0], e[1]]))
                        k2_p[e[0], e[1]] = -1e5
                    if len(final_output) >= k ** 1:
                        break
                current_p, i = torch.topk(k2_p.flatten(), k)
                next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
                product_input = torch.stack([product_input_k2[i[0], i[1]] for i in next_indices], dim=0)

            candidate_k = []
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
            for p, sentence in final_output:
                cdd = tokenizer.decode(sentence[:-1])[0]#newtkn
                candidate_k.append(cdd)
            if candidate_k == []:
                candidate.append("")
            else:
                candidate.append(candidate_k[0])
            # candidate.append(random.choice(candidate_k))
    return candidate


def get_validity(smiles):
    from rdkit import Chem
    v = []
    for l in smiles:
        try:
            if l == "":
                continue
            s = Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False)
            v.append(s)
        except:
            continue
    u = list(set(v))
    if len(u) == 0:
        return 0., 0.
    return len(v) / len(smiles)


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


class regexTokenizer():
    def __init__(self,vocab_path='/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/dataloaders/vocab_bpe_300_sc.txt',max_len=127):
        with open(vocab_path,'r') as f:
            x = f.readlines()
            x = [xx.replace('##', '') for xx in x]
            x2 = x.copy()
        x2.sort(key=len, reverse=True)
        pattern = "("+"|".join(re.escape(token).strip()[:-1] for token in x2)+")"
        self.rg = re.compile(pattern)

        self.idtotok  = { cnt:i.strip() for cnt,i in enumerate(x)}
        self.vocab_size = len(self.idtotok) #SOS, EOS, pad
        self.toktoid = { v:k for k,v in self.idtotok.items()}
        self.max_len = max_len
        self.cls_token_id = self.toktoid['[CLS]']
        self.sep_token_id = self.toktoid['[SEP]']
        self.pad_token_id = self.toktoid['[PAD]']

    def decode_one(self, iter):
        if self.sep_token_id in iter:   iter = iter[:(iter == self.sep_token_id).nonzero(as_tuple=True)[0][0].item()]
        # return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')
        return "".join([self.idtotok[i.item()] for i in iter[1:]])

    def decode(self,ids:torch.tensor):
        if len(ids.shape)==1:
            return [self.decode_one(ids)]
        else:
            smiles  = []
            for i in ids:
                smiles.append(self.decode_one(i))
            return smiles
    def __len__(self):
        return self.vocab_size

    def __call__(self,smis:list, truncation='max_len'):
        tensors = []
        lengths = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            length, tensor = self.encode_one(i)
            tensors.append(tensor)
            lengths.append(length)
        output = torch.concat(tensors,dim=0)
        if truncation == 'max_len':
            return output
        elif truncation == 'longest':
            return output[:, :max(lengths)]
        else:
            raise ValueError('truncation should be either max_len or longest')

    def encode_one(self, smi):
        smi = '[CLS]' + smi + '[SEP]'
        res = [self.toktoid[i] for i in self.rg.findall(smi)]
        token_length = len(res)
        if token_length < self.max_len:
            res += [self.pad_token_id]*(self.max_len-len(res))
        else:
            res = res[:self.max_len]
            # res[-1] = self.sep_token_id
        return token_length, torch.LongTensor([res])


def standardize_smiles_single(smiles_data):
    """
    Standardize a single SMILES or a tuple of (index, smiles).
    
    Args:
        smiles_data: Either a SMILES string or tuple of (index, smiles)
    
    Returns:
        Standardized SMILES string or tuple of (index, standardized_smiles)
    """
    # Handle both single SMILES and (index, smiles) tuples
    if isinstance(smiles_data, tuple):
        index, smiles = smiles_data
        return_tuple = True
    else:
        smiles = smiles_data
        return_tuple = False
    
    # Handle None, NaN, or empty strings
    if not smiles or pd.isna(smiles):
        result = None
    else:
        try:
            # Clean and parse SMILES
            clean_smiles = str(smiles).strip()
            mol = Chem.MolFromSmiles(clean_smiles)
            
            if mol is None:
                result = None
            else:
                # Standardize with consistent parameters
                result = Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True,  # Preserve stereochemistry
                    kekuleSmiles=False
                )
        except Exception as e:
            result = None
            raise e
    
    return (index, result) if return_tuple else result


def standardize_smiles_multiprocess(smiles_list, 
                                  n_processes=None, 
                                  chunk_size=None, 
                                  show_progress=True,
                                  preserve_order=True):
    """
    Standardize a list of SMILES using multiprocessing.
    
    Args:
        smiles_list (list): List of SMILES strings to standardize
        n_processes (int, optional): Number of processes to use. Defaults to CPU count.
        chunk_size (int, optional): Chunk size for multiprocessing. Auto-calculated if None.
        show_progress (bool): Whether to show progress bar
        preserve_order (bool): Whether to preserve the original order of SMILES
    
    Returns:
        list: List of standardized SMILES (None for invalid SMILES)
    """
    if not smiles_list:
        return []
    
    # Set default parameters
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(smiles_list))
    
    if chunk_size is None:
        # Auto-calculate chunk size based on list length and process count
        chunk_size = max(1, len(smiles_list) // (n_processes * 4))
    
    # For small lists, use single processing to avoid overhead
    if len(smiles_list) < 100:
        if show_progress:
            return [standardize_smiles_single(smi) for smi in tqdm(smiles_list, desc="Standardizing SMILES")]
        else:
            return [standardize_smiles_single(smi) for smi in smiles_list]
    
    # Prepare data for multiprocessing
    if preserve_order:
        # Include indices to preserve order
        indexed_data = [(i, smi) for i, smi in enumerate(smiles_list)]
        worker_func = standardize_smiles_single
    else:
        indexed_data = smiles_list
        worker_func = standardize_smiles_single
    
    # Use multiprocessing
    try:
        with mp.Pool(processes=n_processes) as pool:
            if show_progress:
                # Use imap for progress tracking
                results = list(tqdm(
                    pool.imap(worker_func, indexed_data, chunksize=chunk_size),
                    total=len(indexed_data),
                    desc=f"Standardizing SMILES ({n_processes} processes)"
                ))
            else:
                results = pool.map(worker_func, indexed_data, chunksize=chunk_size)
    
    except Exception as e:
        print(f"Multiprocessing failed: {e}. Falling back to single processing.")
        if show_progress:
            return [standardize_smiles_single(smi) for smi in tqdm(smiles_list, desc="Standardizing SMILES (fallback)")]
        else:
            return [standardize_smiles_single(smi) for smi in smiles_list]
    
    # Process results
    if preserve_order:
        # Sort by original index and extract standardized SMILES
        results.sort(key=lambda x: x[0])
        standardized_smiles = [result[1] for result in results]
    else:
        standardized_smiles = results
    
    return standardized_smiles


def standardize_smiles_batch_with_stats(smiles_list, 
                                      n_processes=None, 
                                      chunk_size=None,
                                      show_progress=True,
                                      return_mapping=False):
    """
    Standardize SMILES with detailed statistics and optional mapping.
    
    Args:
        smiles_list (list): List of SMILES strings
        n_processes (int, optional): Number of processes
        chunk_size (int, optional): Chunk size for multiprocessing
        show_progress (bool): Show progress bar
        return_mapping (bool): Return mapping of original -> standardized SMILES
    
    Returns:
        dict: Contains 'standardized_smiles', 'stats', and optionally 'mapping'
    """
    original_count = len(smiles_list)
    
    # Standardize SMILES
    standardized = standardize_smiles_multiprocess(
        smiles_list, 
        n_processes=n_processes,
        chunk_size=chunk_size,
        show_progress=show_progress
    )
    
    # Calculate statistics
    valid_count = sum(1 for smi in standardized if smi is not None)
    invalid_count = original_count - valid_count
    
    # Remove duplicates while preserving order
    unique_standardized = []
    seen = set()
    for smi in standardized:
        if smi is not None and smi not in seen:
            unique_standardized.append(smi)
            seen.add(smi)
        elif smi is None:
            unique_standardized.append(None)
    
    unique_valid_count = len(seen)
    duplicate_count = valid_count - unique_valid_count
    
    stats = {
        'original_count': original_count,
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'validity_rate': valid_count / original_count if original_count > 0 else 0,
        'unique_valid_count': unique_valid_count,
        'duplicate_count': duplicate_count,
        'uniqueness_rate': unique_valid_count / valid_count if valid_count > 0 else 0
    }
    
    result = {
        'standardized_smiles': standardized,
        'unique_standardized_smiles': [smi for smi in unique_standardized if smi is not None],
        'stats': stats
    }
    
    # Optionally return mapping
    if return_mapping:
        mapping = {}
        for orig, std in zip(smiles_list, standardized):
            if std is not None:
                mapping[orig] = std
        result['mapping'] = mapping
    
    return result


def get_hash(namespace):
    """
    Generate an 8-character hash from argparse Namespace object.
    
    Args:
        namespace: argparse.Namespace object from parse_args()
        
    Returns:
        str: 8-character hexadecimal hash
    """
    # Convert namespace to dict and create hash
    data = json.dumps(vars(namespace), sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:8]


def setup_signal_handlers(checkpoint_dir, experiment_dir, auto_requeue=False):
    """Setup signal handlers for graceful shutdown and checkpointing"""
    global CHECKPOINT_DIR, EXPERIMENT_DIR, AUTO_REQUEUE
    CHECKPOINT_DIR = checkpoint_dir
    EXPERIMENT_DIR = experiment_dir
    AUTO_REQUEUE = auto_requeue
    
    def signal_handler(signum, frame):
        # Import and modify the global variable in the main module
        import train_pert2mol
        print(f"\n[SIGNAL] Received signal {signum}. Initiating graceful shutdown...")
        train_pert2mol.GRACEFUL_SHUTDOWN = True
        
        # Try to requeue the job if running under SLURM and auto_requeue is enabled
        if AUTO_REQUEUE:
            try:
                import os
                job_id = os.environ.get('SLURM_JOB_ID')
                if job_id:
                    print(f"[SLURM] Attempting to requeue job {job_id} (auto-resume enabled)")
                    os.system(f'scontrol requeue {job_id}')
                else:
                    print("[SIGNAL] Auto-resume enabled but not running under SLURM")
            except Exception as e:
                print(f"[SLURM] Failed to requeue: {e}")
        else:
            print("[SIGNAL] Auto-resume disabled, will not requeue job")
    
    # Register handlers for catchable signals
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    signal.signal(signal.SIGINT, signal_handler)   # Interrupt signal (Ctrl+C)
    signal.signal(signal.SIGUSR1, signal_handler)  # User signal 1
    
    requeue_status = "enabled" if auto_requeue else "disabled"
    print(f"[SIGNAL] Signal handlers registered for graceful shutdown (auto-requeue: {requeue_status})")


def save_emergency_checkpoint(model, image_encoder, rna_encoder, ema, opt, args, 
                            sra_teacher_manager, train_steps, use_ddp, rank):
    """Save checkpoint during emergency shutdown"""
    global CHECKPOINT_DIR, CURRENT_STEP
    
    if not CHECKPOINT_DIR or (use_ddp and rank != 0):
        return
        
    try:
        print(f"[EMERGENCY] Saving checkpoint at step {train_steps}")
        
        checkpoint = {
            "model": model.module.state_dict() if use_ddp else model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps,
            "emergency_save": True
        }
        
        if hasattr(args, 'scheduler') and args.scheduler != "none":
            # You'll need to pass the scheduler to this function
            # checkpoint["scheduler"] = scheduler.state_dict()
            pass  # For now, skip scheduler in emergency saves

        # Only save encoder states if they exist
        if image_encoder is not None:
            checkpoint["image_encoder"] = image_encoder.module.state_dict() if use_ddp else image_encoder.state_dict()
        if rna_encoder is not None:
            checkpoint["rna_encoder"] = rna_encoder.module.state_dict() if use_ddp else rna_encoder.state_dict()
            
        if args.use_sra and sra_teacher_manager is not None:
            checkpoint["sra_teacher"] = sra_teacher_manager.get_teacher().state_dict()
        
        emergency_path = f"{CHECKPOINT_DIR}/emergency_{train_steps:07d}.pt"
        torch.save(checkpoint, emergency_path)
        print(f"[EMERGENCY] Checkpoint saved to {emergency_path}")
        
        # Also create a latest.pt symlink
        latest_path = f"{CHECKPOINT_DIR}/latest.pt"
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(os.path.basename(emergency_path), latest_path)
        
    except Exception as e:
        print(f"[EMERGENCY] Failed to save checkpoint: {e}")


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for latest.pt symlink first
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path) and os.path.islink(latest_path):
        target = os.path.join(checkpoint_dir, os.readlink(latest_path))
        if os.path.exists(target):
            return target
    
    # Fallback: find highest numbered checkpoint
    checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find maximum
    max_step = -1
    latest_file = None
    
    for file in checkpoint_files:
        basename = os.path.basename(file)
        if basename.startswith(('emergency_', '')):
            # Extract step number
            try:
                if basename.startswith('emergency_'):
                    step_str = basename[10:17]  # emergency_0123456.pt
                else:
                    step_str = basename[:7]     # 0123456.pt
                step = int(step_str)
                if step > max_step:
                    max_step = step
                    latest_file = file
            except (ValueError, IndexError):
                continue
    
    return latest_file


def load_checkpoint_and_resume(checkpoint_path, model, image_encoder, rna_encoder, ema, opt, 
                              sra_teacher_manager, use_ddp, rank, logger, scheduler=None):
    """Load checkpoint and return resume step"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0
    
    try:
        if not use_ddp or rank == 0:
            logger.info(f"[RESUME] Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model states
        if use_ddp:
            model.module.load_state_dict(checkpoint["model"])
            # Only load encoder states if they exist in both checkpoint and current setup
            if image_encoder is not None and "image_encoder" in checkpoint:
                image_encoder.module.load_state_dict(checkpoint["image_encoder"])
            if rna_encoder is not None and "rna_encoder" in checkpoint:
                rna_encoder.module.load_state_dict(checkpoint["rna_encoder"])
        else:
            model.load_state_dict(checkpoint["model"])
            if image_encoder is not None and "image_encoder" in checkpoint:
                image_encoder.load_state_dict(checkpoint["image_encoder"])
            if rna_encoder is not None and "rna_encoder" in checkpoint:
                rna_encoder.load_state_dict(checkpoint["rna_encoder"])
        
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        
        # Load SRA teacher if available
        if "sra_teacher" in checkpoint and sra_teacher_manager is not None:
            sra_teacher_manager.get_teacher().load_state_dict(checkpoint["sra_teacher"])
        
        resume_step = checkpoint.get("train_steps", 0)
        emergency_save = checkpoint.get("emergency_save", False)
        
        if not use_ddp or rank == 0:
            save_type = "emergency" if emergency_save else "regular"
            logger.info(f"[RESUME] Loaded {save_type} checkpoint, resuming from step {resume_step}")
        
        if scheduler is not None and "scheduler" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
                if not use_ddp or rank == 0:
                    logger.info(f"[RESUME] Loaded scheduler state")
            except Exception as e:
                if not use_ddp or rank == 0:
                    logger.warning(f"[RESUME] Failed to load scheduler state: {e}")
        
        return resume_step
        
    except Exception as e:
        if not use_ddp or rank == 0:
            logger.error(f"[RESUME] Failed to load checkpoint {checkpoint_path}: {e}")
        return 0
    

def load_checkpoint_and_resume_specified(checkpoint_path, model, image_encoder, rna_encoder, ema, opt, 
                                       sra_teacher_manager, use_ddp, rank, logger, scheduler=None):
    """Load checkpoint from specified path and return resume step"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0
    
    try:
        if not use_ddp or rank == 0:
            logger.info(f"[RESUME] Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model states
        if use_ddp:
            model.module.load_state_dict(checkpoint["model"])
            if image_encoder is not None and "image_encoder" in checkpoint:
                image_encoder.module.load_state_dict(checkpoint["image_encoder"])
            if rna_encoder is not None and "rna_encoder" in checkpoint:
                rna_encoder.module.load_state_dict(checkpoint["rna_encoder"])
        else:
            model.load_state_dict(checkpoint["model"])
            if image_encoder is not None and "image_encoder" in checkpoint:
                image_encoder.load_state_dict(checkpoint["image_encoder"])
            if rna_encoder is not None and "rna_encoder" in checkpoint:
                rna_encoder.load_state_dict(checkpoint["rna_encoder"])
        
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        
        # Load SRA teacher if available
        if "sra_teacher" in checkpoint and sra_teacher_manager is not None:
            sra_teacher_manager.get_teacher().load_state_dict(checkpoint["sra_teacher"])
        
        # Load scheduler if available
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        
        resume_step = checkpoint.get("train_steps", 0)
        
        if not use_ddp or rank == 0:
            logger.info(f"[RESUME] Successfully resumed from step {resume_step}")
        
        return resume_step
        
    except Exception as e:
        if not use_ddp or rank == 0:
            logger.error(f"[RESUME] Failed to load checkpoint {checkpoint_path}: {e}")
        return 0
    
    
class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True, mode='min', verbose=True):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
            mode (str): 'min' for minimization, 'max' for maximization
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.should_stop = False
        
        # Set comparison functions based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < (best - min_delta)
        elif mode == 'max':
            self.is_better = lambda current, best: current > (best + min_delta)
        else:
            raise ValueError(f"Mode {mode} is unknown, expected 'min' or 'max'")
    
    def __call__(self, current_score, model=None, step=None):
        """
        Check if training should stop early.
        
        Args:
            current_score (float): Current validation score
            model (torch.nn.Module, optional): Model to save best weights from
            step (int, optional): Current training step
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            if model is not None and self.restore_best_weights:
                self.best_weights = self._get_model_state(model)
            if self.verbose:
                print(f"Early stopping: Initial score set to {current_score:.6f}")
        elif self.is_better(current_score, self.best_score):
            # Improvement found - FIXED: Save old score before updating
            old_best = self.best_score
            self.best_score = current_score
            self.wait = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = self._get_model_state(model)
            if self.verbose:
                print(f"Early stopping: New best score {current_score:.6f} (improvement of {abs(current_score - old_best):.6f})")
        else:
            # No improvement
            self.wait += 1
            if self.verbose:
                print(f"Early stopping: No improvement for {self.wait}/{self.patience} checks. Best: {self.best_score:.6f}, Current: {current_score:.6f}")
            
            if self.wait >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping: Stopping training after {self.patience} checks without improvement")
                
                # Restore best weights if requested
                if model is not None and self.restore_best_weights and self.best_weights is not None:
                    self._restore_model_state(model, self.best_weights)
                    if self.verbose:
                        print("Early stopping: Restored best model weights")
                
                return True
        
        return False
    
    def _get_model_state(self, model):
        """Get model state dict, handling DDP wrapper."""
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()
    
    def _restore_model_state(self, model, state_dict):
        """Restore model state dict, handling DDP wrapper."""
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    def state_dict(self):
        """Return state dictionary for checkpoint saving."""
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'restore_best_weights': self.restore_best_weights,
            'mode': self.mode,
            'verbose': self.verbose,
            'wait': self.wait,
            'best_score': self.best_score,
            'best_weights': self.best_weights,
            'should_stop': self.should_stop
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.restore_best_weights = state_dict['restore_best_weights']
        self.mode = state_dict['mode']
        self.verbose = state_dict['verbose']
        self.wait = state_dict['wait']
        self.best_score = state_dict['best_score']
        self.best_weights = state_dict['best_weights']
        self.should_stop = state_dict['should_stop']


def collect_all_drugs_from_csv(raw_drug_csv_path, compound_name_label='compound', smiles_label='canonical_smiles'):
    """Collect all drugs from the raw drug CSV file for comprehensive retrieval."""
    
    print(f"Loading all drugs from raw CSV: {raw_drug_csv_path}")
    
    try:
        # Load the raw drug CSV
        drug_df = pd.read_csv(raw_drug_csv_path)
        print(f"Loaded {len(drug_df)} total drug entries from CSV")
        
        # Filter out entries without valid SMILES or compound names
        valid_entries = drug_df[
            drug_df[compound_name_label].notna() & 
            drug_df[smiles_label].notna()
        ].copy()
        
        print(f"Found {len(valid_entries)} entries with valid compound names and SMILES")
        
        # Get unique compounds and their SMILES
        compound_to_smiles = {}
        smiles_to_compound = {}
        
        for _, row in valid_entries.iterrows():
            compound = row[compound_name_label]
            smiles = row[smiles_label]
            
            # Convert to aromatic SMILES for consistency
            try:
                from dataloaders.utils import convert_to_aromatic_smiles
                canonical_smiles = convert_to_aromatic_smiles(smiles)
                if canonical_smiles:
                    compound_to_smiles[compound] = canonical_smiles
                    smiles_to_compound[canonical_smiles] = compound
            except:
                # Fallback to original SMILES if conversion fails
                compound_to_smiles[compound] = smiles
                smiles_to_compound[smiles] = compound
        
        # Create the training_drugs structure
        training_drugs = {
            'compound_names': list(compound_to_smiles.keys()),
            'smiles': list(smiles_to_compound.keys()),
            'compound_to_smiles': compound_to_smiles,
            'smiles_to_compound': smiles_to_compound
        }
        
        print(f"Collected {len(training_drugs['compound_names'])} unique compounds")
        print(f"Collected {len(training_drugs['smiles'])} unique SMILES")
        
        return training_drugs
        
    except Exception as e:
        print(f"Error loading drugs from CSV: {e}")
        # Fallback to empty structure
        return {
            'compound_names': [],
            'smiles': [],
            'compound_to_smiles': {},
            'smiles_to_compound': {}
        }


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} not a floating-point literal")
    
    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range (0.0, 1.0]")
    return x
