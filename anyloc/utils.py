import numpy as np
import torch
from typing import List, Tuple
from torch.nn import functional as F
import faiss

def get_top_k_recall(top_k: List[int], db: torch.Tensor, 
        qu: torch.Tensor, gt_pos: np.ndarray, method: str="cosine", 
        norm_descs: bool=True, use_gpu: bool=False, 
        use_percentage: bool=True, sub_sample_db: int=1, 
        sub_sample_qu: int=1) -> Tuple[np.ndarray, np.ndarray, dict]:
    
    """
        Given a database and query (or queries), get the top 'k'
        retrievals (closest in database for each query) as indices (in
        database), distances, and recalls.
        
        Parameters:
        - top_k:    List of 'k' values for recall calculation. Eg:
                    `list(range(1, 11))`.
        - db:       Database descriptors of shape [n_db, d_dim].
        - qu:       Query descriptors of shape [n_qu, d_dim]. If only
                    one query (n_qu = 1), then shape [d_dim].
        - gt_pos:   Ground truth for retrievals. Should be object type
                    with gt_pos[i] having true database items 
                    (indices) for the query 'i' (in `qu`).
        - method:   Method for faiss search. In {'cosine', 'l2'}.
        - norm_descs:   If True, the descriptors are normalized in 
                        function.
        - use_gpu:  True if indexing (search) should be on GPU.
        - use_percentage:   If True, the recalls are returned as a
                            percentage (of queries resolved).
        - sub_sample_db:    Sub-sample database samples from the
                            ground truth 'gt_pos'
        - sub_sample_qu:    Sub-sample query samples from the ground
                            truth 'gt_pos'
        
        Returns:
        - distances:    The distances of queries to retrievals. The
                        shape is [n_qu, max(top_k)]. It is the 
                        distance (as specified in `method`) with the
                        database item retrieved (index in `indices`).
                        Sorted by distance.
        - indices:      Indices of the database items retrieved. The
                        shape is [n_qu, max(top_k)]. Sorted by 
                        distance.
        - recalls:      A dictionary with keys as top_k integers, and
                        values are the recall (number or percentage)
                        of correct retrievals for queries.
    """
    if len(qu.shape) == 1:
        qu = qu.unsqueeze(0)
    if norm_descs:
        db = F.normalize(db)
        qu = F.normalize(qu)
    D = db.shape[1]
    if method == "cosine":
        index = faiss.IndexFlatIP(D)
    elif method == "l2":
        index = faiss.IndexFlatL2(D)
    else:
        raise NotImplementedError(f"Method: {method}")
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0 , index)
    # Get the max(top-k) retrieval, then traverse list
    index.add(db)
    distances, indices = index.search(qu, max(top_k))
    recalls = dict(zip(top_k, [0]*len(top_k)))
    # print(qu.shape,indices.shape)
    for i_qu, qu_retr in enumerate(indices):
        for i_rec in top_k:
            # Correct database images (for the retrieval)
            """
                i_qu * sub_sample_qu
                    Sub-sampled queries (step)
                qu_retr[:i_rec] * sub_sample_db
                    Sub-sampled database (step in retrievals)
            """
            correct_retr = gt_pos[i_qu * sub_sample_qu]
            if np.any(np.isin(qu_retr[:i_rec] * sub_sample_db, 
                        correct_retr)):
                recalls[i_rec] += 1
    if use_percentage:
        for k in recalls:
            recalls[k] /= len(indices)
    return distances, indices, recalls