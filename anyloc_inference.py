import torch
from typing import List, Tuple
from torch.nn import functional as F

from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import argparse
import yaml
import importlib
import numpy as np
import faiss


def parser():
    parser = argparse.ArgumentParser(description='anyloc')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    # choose gpu to use
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')
    parser.add_argument('--retrieval', action='store_true')
    parser.add_argument('--viz', action='store_true') # visualize the retrieval results
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    
    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    return args, config

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
        print("Normalizing descriptors...")
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
    index.add(db.numpy())
    distances, indices = index.search(qu.numpy(), max(top_k))
    recalls = dict(zip(top_k, [0]*len(top_k)))
    print("Starting retrieval...")
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

def extract_gdesc(cfg):
    # basic setup
    data = importlib.import_module(cfg.data.type)
    loader = data.get_dataloader(cfg.data)
    model = importlib.import_module(cfg.model.type)
    model = model.get_model(cfg.model)
    save_dir = Path(cfg.data.output_path, 'gdesc')
    with torch.no_grad():
        for idx, data in tqdm(enumerate(loader), total=len(loader)):
            img = data['img']
            img_dirs = data['img_dir']
            # check if the image descriptors are already saved
            gd_dirs_list = [f'{Path(Path(img_dir).parent.name, Path(img_dir).stem)}.npy' for idx, img_dir in enumerate(img_dirs)]
            save_dir_list = [save_dir/gd_dir for gd_dir in gd_dirs_list]
            if all([Path(gd_dir).exists() for gd_dir in save_dir_list]):
                continue
            # if not, save the descriptors
            gd_batch = model(img) # shape: [batch_size, agg_dim], numpy.array()
            # Save the global descriptor to the save dir of images
            for idx, gd_dir in enumerate(save_dir_list):
                if not Path(gd_dir).exists():
                    gd_dir.parent.mkdir(parents=True, exist_ok=True)
                np.save(gd_dir, gd_batch[idx])
                # img_dir = Path(img_dir)
                # np.save(f'{img_dir.parent/img_dir.stem}.npy', gd_batch[idx])

def retrieve(cfg):
    # load dataset for gt
    dataset = importlib.import_module(cfg.data.type).get_dataset(cfg.data)
    # build database and query tensors
    database_tensors = dataset.get_database_descs_tensor()
    query_tensors = dataset.get_queries_descs_tensor()
    # load positive database indices for each query
    soft_positives_per_query = dataset.get_positives()
    # get top k retrieval
    dists, indices, recalls = get_top_k_recall(cfg.topk, database_tensors, query_tensors, 
                                               soft_positives_per_query, use_gpu=False, norm_descs=True)
    # save_path
    save_path = dataset.get_output_path()
    if not save_path.exists():
        save_path.mkdir(parents=True)
    # save dists (similarity), indices (retrieved indices), recalls (recall)
    topk = max(cfg.topk)
    np.save(save_path/f'dists_{topk}.npy', dists)
    np.save(save_path/f'indices_{topk}.npy', indices)
    np.save(save_path/f'recalls_{topk}.npy', recalls)
    print(f"Recalls: {recalls}")

# def viz(cfg):
#     retrieval_path = Path(cfg.pairs_from_retrieval.retrieval_results)
#     retrievals = np.load(retrieval_path, allow_pickle=True)
#     print(retrievals.shape)
    # for retrieval in tqdm(retrievals):
        # print(f"Retrieval: {retrieval}")
    #     for idx, (img_dir, sim) in enumerate(retrievals[retrieval][:20]):
    #         print(f"Image {idx}: {img_dir}, Similarity: {sim}")
    
    
if __name__ == "__main__":
    args, cfg = parser()
    if args.retrieval:
        retrieve(cfg)
    # if args.viz:
    #     viz(cfg)
    else:
        extract_gdesc(cfg)
    print('Finished, Exiting program')
    exit(0)