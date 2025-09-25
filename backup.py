import faiss

def compute_cosine_similarity_batch(query_embeddings, reference_embeddings, device='cuda', chunk_size=1000):
    """
    Efficiently compute cosine similarity between query and reference embeddings.
    
    Args:
        query_embeddings: [batch_size, embedding_dim] 
        reference_embeddings: [N_ref, embedding_dim]
        device: computation device
        chunk_size: process references in chunks to manage memory
    
    Returns:
        similarities: [batch_size, N_ref] similarity matrix
    """
    query_embeddings = query_embeddings.to(device)
    
    # Normalize embeddings for cosine similarity
    query_norm = F.normalize(query_embeddings, p=2, dim=1)  # [batch_size, dim]
    
    similarities = []
    
    # Process reference embeddings in chunks to manage memory
    for i in range(0, len(reference_embeddings), chunk_size):
        ref_chunk = reference_embeddings[i:i+chunk_size].to(device)
        ref_norm = F.normalize(ref_chunk, p=2, dim=1)  # [chunk_size, dim]
        
        # Compute cosine similarity: normalized dot product
        sim_chunk = torch.mm(query_norm, ref_norm.t())  # [batch_size, chunk_size]
        similarities.append(sim_chunk.cpu())
    
    return torch.cat(similarities, dim=1)  # [batch_size, N_ref]


def find_top_k_similar_vectorized(similarities, k):
    """
    Efficiently find top-k similar items using vectorized operations.
    
    Args:
        similarities: [batch_size, N_ref] similarity matrix
        k: number of top items to return
    
    Returns:
        top_k_indices: [batch_size, k] indices of top-k items
        top_k_similarities: [batch_size, k] similarity scores
    """
    # Use torch.topk for efficient top-k selection
    top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=1, largest=True)
    return top_k_indices, top_k_similarities


def optimized_conventional_retrieval(test_batch, training_data, y_features, args, device='cuda'):
    """Optimized conventional retrieval using vectorized operations."""
    
    retrieved_results = []
    target_smiles = test_batch['target_smiles']
    compound_names = test_batch['compound_name']
    batch_size = len(target_smiles)
    
    if not training_data.get('biological_features'):
        return [{'top_k_drugs': [], 'top_k_similarities': [], 'smiles_hit_rank': None, 
                'compound_hit_rank': None, 'smiles_in_top_k': False, 'compound_in_top_k': False}] * batch_size
    
    # Concatenate all training biological features
    all_training_features = torch.cat(training_data['biological_features'], dim=0)
    
    # Flatten embeddings
    query_flat = y_features.flatten(1)  # [batch_size, features]
    training_flat = all_training_features.flatten(1)  # [N, features]
    
    # Compute similarities efficiently
    similarities = compute_cosine_similarity_batch(query_flat, training_flat, device)
    
    # Find top-k for all queries at once
    top_k_indices, top_k_similarities = find_top_k_similar_vectorized(similarities, args.retrieval_top_k)
    
    # Process results
    for i in range(batch_size):
        target_smile = target_smiles[i]
        target_compound = compound_names[i]
        
        indices = top_k_indices[i].numpy()
        sims = top_k_similarities[i].numpy()
        
        top_k_smiles = [training_data['smiles'][idx] for idx in indices]
        top_k_compounds = [training_data['compound_names'][idx] for idx in indices]
        
        # Check if target is in top-k
        smiles_hit_rank = None
        compound_hit_rank = None
        
        for rank, (retrieved_smiles, retrieved_compound) in enumerate(zip(top_k_smiles, top_k_compounds)):
            # Check for SMILES match
            try:
                target_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(target_smile), canonical=True)
                retrieved_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(retrieved_smiles), canonical=True)
                if target_canonical == retrieved_canonical and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            except:
                if target_smile == retrieved_smiles and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            
            # Check for compound name match
            if target_compound == retrieved_compound and compound_hit_rank is None:
                compound_hit_rank = rank + 1
            
            if smiles_hit_rank is not None and compound_hit_rank is not None:
                break
        
        retrieved_results.append({
            'top_k_drugs': top_k_smiles,
            'top_k_similarities': sims.tolist(),
            'smiles_hit_rank': smiles_hit_rank,
            'compound_hit_rank': compound_hit_rank,
            'smiles_in_top_k': smiles_hit_rank is not None,
            'compound_in_top_k': compound_hit_rank is not None,
        })
    
    return retrieved_results


def optimized_retrieval_by_generation(test_batch, training_data, model, flow, ae_model, 
                                     y_features, pad_mask, device, args):
    """Optimized retrieval by generation using vectorized operations."""
    
    retrieved_results = []
    target_smiles = test_batch['target_smiles']
    compound_names = test_batch['compound_name']
    batch_size = len(target_smiles)
    
    # Step 1: Generate drug embeddings (unchanged)
    with torch.no_grad():
        shape = (batch_size, 64, 127, 1)
        x = torch.randn(*shape, device=device)
        dt = 1.0 / args.generation_steps
        
        for i in range(args.generation_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            t_discrete = (t * 999).long()
            
            velocity = model(x, t_discrete, y=y_features, pad_mask=pad_mask)
            if isinstance(velocity, tuple):
                velocity = velocity[0]
            
            if velocity.shape[1] == 2 * x.shape[1]:
                velocity, _ = torch.split(velocity, x.shape[1], dim=1)
            
            x = x + dt * velocity
        
        generated_drug_latents = x.squeeze(-1).permute((0, 2, 1))
    
    # Step 2: Get dataset drug embeddings efficiently
    dataset_smiles = training_data['smiles']
    dataset_compound_names = training_data['compound_names']
    
    dataset_latents = []
    batch_size_train = 32
    
    with torch.no_grad():
        for i in range(0, len(dataset_smiles), batch_size_train):
            batch_smiles = dataset_smiles[i:i+batch_size_train]
            train_latents_batch = AE_SMILES_encoder(batch_smiles, ae_model).permute((0, 2, 1))
            dataset_latents.append(train_latents_batch)
        
        dataset_latents = torch.cat(dataset_latents, dim=0)  # [N_drugs, seq_len, dim]
    
    # Step 3: Vectorized similarity computation
    # Flatten embeddings for similarity computation
    gen_flat = generated_drug_latents.flatten(1)  # [batch_size, seq_len*dim]
    dataset_flat = dataset_latents.flatten(1)     # [N_drugs, seq_len*dim]
    
    # Compute all similarities at once
    similarities = compute_cosine_similarity_batch(gen_flat, dataset_flat, device)
    
    # Find top-k for all queries
    top_k_indices, top_k_similarities = find_top_k_similar_vectorized(
        similarities, args.retrieval_by_generation_top_k
    )
    
    # Step 4: Process results
    for i in range(batch_size):
        target_smile = target_smiles[i]
        target_compound = compound_names[i]
        
        indices = top_k_indices[i].numpy()
        sims = top_k_similarities[i].numpy()
        
        top_k_smiles = [dataset_smiles[idx] for idx in indices]
        top_k_compounds = [dataset_compound_names[idx] for idx in indices]
        
        # Check if target is in top-k
        smiles_hit_rank = None
        compound_hit_rank = None
        
        for rank, (retrieved_smiles, retrieved_compound) in enumerate(zip(top_k_smiles, top_k_compounds)):
            # Check for SMILES match
            try:
                target_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(target_smile), canonical=True)
                retrieved_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(retrieved_smiles), canonical=True)
                if target_canonical == retrieved_canonical and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            except:
                if target_smile == retrieved_smiles and smiles_hit_rank is None:
                    smiles_hit_rank = rank + 1
            
            # Check for compound name match
            if target_compound == retrieved_compound and compound_hit_rank is None:
                compound_hit_rank = rank + 1
            
            if smiles_hit_rank is not None and compound_hit_rank is not None:
                break
        
        retrieved_results.append({
            'top_k_drugs': top_k_smiles,
            'top_k_similarities': sims.tolist(),
            'smiles_hit_rank': smiles_hit_rank,
            'compound_hit_rank': compound_hit_rank,
            'smiles_in_top_k': smiles_hit_rank is not None,
            'compound_in_top_k': compound_hit_rank is not None,
        })
    
    return retrieved_results, generated_drug_latents.cpu().numpy(), dataset_latents.cpu().numpy()


# For very large datasets (>100k drugs), consider approximate nearest neighbor search
def setup_faiss_index(embeddings, use_gpu=True):
    """Setup FAISS index for approximate nearest neighbor search."""
    d = embeddings.shape[1]  # dimension
    
    if use_gpu and torch.cuda.is_available():
        # GPU index for faster search
        index = faiss.IndexFlatIP(d)  # Inner product (for normalized vectors = cosine similarity)
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    else:
        # CPU index
        index = faiss.IndexFlatIP(d)
    
    # Normalize embeddings for cosine similarity
    embeddings_np = F.normalize(embeddings, p=2, dim=1).cpu().numpy().astype('float32')
    index.add(embeddings_np)
    
    return index


def faiss_search_top_k(index, query_embeddings, k):
    """Search top-k using FAISS index."""
    query_np = F.normalize(query_embeddings, p=2, dim=1).cpu().numpy().astype('float32')
    similarities, indices = index.search(query_np, k)
    return torch.from_numpy(indices), torch.from_numpy(similarities)

