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


def dual_rna_image_encoder(control_images, treatment_images, control_rna, treatment_rna, 
                          image_encoder, rna_encoder, device):
    """
    Encode paired control and treatment images + RNA with selective dropout for CFG.
    """
    # Encode images
    control_img_features = image_encoder(control_images.to(device))
    treatment_img_features = image_encoder(treatment_images.to(device))
    
    # Encode RNA
    control_rna_features = rna_encoder(control_rna.to(device))
    treatment_rna_features = rna_encoder(treatment_rna.to(device))
    
    # Concatenate features
    control_features = torch.cat([control_img_features, control_rna_features], dim=-1)
    treatment_features = torch.cat([treatment_img_features, treatment_rna_features], dim=-1)
    
    combined_features = torch.stack([control_features, treatment_features], dim=1)
    attention_mask = torch.ones(combined_features.size(0), 2, dtype=torch.bool, device=device)
    
    return combined_features, attention_mask


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

