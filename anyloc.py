import torch
from torch import nn
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from torch.nn import functional as F
import fast_pytorch_kmeans as fpk
import einops as ein

from PIL import Image
import numpy as np
import tyro
import os
import sys
import glob
import natsort
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Literal, Union, List
from pathlib import Path
import logging


# --------------------      Logger      --------------------- #
def create_logger(name: str, level: str, log: str):
    levels = ['INFO', 'DEBUG', 'WARNING', 'ERROR']
    
    # logger format
    formatter = logging.Formatter(
        fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    
    # Stream handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    
    # Create file handler if provided
    if log is not None:
        # File handler
        f_handler = logging.FileHandler(log)
        f_handler.setFormatter(formatter)
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)
        logger.info(f'FileHandler created and saved to {log}')
    
    assert level in levels, f"'{level}' not  found in {levels}"
    logger.setLevel(level)
    
    return logger

# -------------------- Dino-v2 utilities -------------------- #

# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
                        "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int, 
                facet: _DINO_FACETS="token", use_cls=False, 
                norm_descs=True, device: str = "cpu") -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        self.vit_type: str = dino_model
        self.dino_model: nn.Module = torch.hub.load(
                'facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2*d_len]
                else:
                    res = res[:, :, 2*d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        return res
    
    def __del__(self):
        self.fh_handle.remove()

# -------------------- VLAD utilities -------------------- #

# VLAD global descriptor implementation
class VLAD:
    """
        An implementation of VLAD algorithm given database and query
        descriptors.
        
        Constructor arguments:
        - num_clusters:     Number of cluster centers for VLAD
        - desc_dim:         Descriptor dimension. If None, then it is
                            inferred when running `fit` method.
        - intra_norm:       If True, intra normalization is applied
                            when constructing VLAD
        - norm_descs:       If True, the given descriptors are 
                            normalized before training and predicting 
                            VLAD descriptors. Different from the
                            `intra_norm` argument.
        - dist_mode:        Distance mode for KMeans clustering for 
                            vocabulary (not residuals). Must be in 
                            {'euclidean', 'cosine'}.
        - vlad_mode:        Mode for descriptor assignment (to cluster
                            centers) in VLAD generation. Must be in
                            {'soft', 'hard'}
        - soft_temp:        Temperature for softmax (if 'vald_mode' is
                            'soft') for assignment
        - cache_dir:        Directory to cache the VLAD vectors. If
                            None, then no caching is done. If a str,
                            then it is assumed as the folder path. Use
                            absolute paths.
        
        Notes:
        - Arandjelovic, Relja, and Andrew Zisserman. "All about VLAD."
            Proceedings of the IEEE conference on Computer Vision and 
            Pattern Recognition. 2013.
    """
    def __init__(self, num_clusters: int, 
                desc_dim: Union[int, None]=None, 
                intra_norm: bool=True, norm_descs: bool=True, 
                dist_mode: str="cosine", vlad_mode: str="hard", 
                soft_temp: float=1.0, 
                cache_dir: Union[str,None]=None) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ['soft', 'hard']
        self.soft_temp = soft_temp
        # Set in the training phase
        self.c_centers = None
        self.kmeans = None
        # Set the caching
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(
                    self.cache_dir))
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            else:
                print("Warning: Cache directory already exists: " \
                        f"{self.cache_dir}")
        else:
            print("VLAD caching is disabled.")
    
    def can_use_cache_vlad(self):
        """
            Checks if the cache directory is a valid cache directory.
            For it to be valid, it must exist and should at least
            include the cluster centers file.
            
            Returns:
            - True if the cache directory is valid
            - False if 
                - the cache directory doesn't exist
                - exists but doesn't contain the cluster centers
                - no caching is set in constructor
        """
        if self.cache_dir is None:
            return False
        if not os.path.exists(self.cache_dir):
            return False
        if os.path.exists(f"{self.cache_dir}/c_centers.pt"):
            return True
        else:
            return False
    
    def can_use_cache_ids(self, 
                cache_ids: Union[List[str], str, None],
                only_residuals: bool=False) -> bool:
        """
            Checks if the given cache IDs exist in the cache directory
            and returns True if all of them exist.
            The cache is stored in the following files:
            - c_centers.pt:     Cluster centers
            - `cache_id`_r.pt:  Residuals for VLAD
            - `cache_id`_l.pt:  Labels for VLAD (hard assignment)
            - `cache_id`_s.pt:  Soft assignment for VLAD
            
            The function returns False if cache cannot be used or if
            any of the cache IDs are not found. If all cache IDs are
            found, then True is returned.
            
            This function is mainly for use outside the VLAD class.
        """
        if not self.can_use_cache_vlad():
            return False
        if cache_ids is None:
            return False
        if isinstance(cache_ids, str):
            cache_ids = [cache_ids]
        for cache_id in cache_ids:
            if not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_r.pt"):
                return False
            if self.vlad_mode == "hard" and not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_l.pt") and not \
                        only_residuals:
                return False
            if self.vlad_mode == "soft" and not os.path.exists(
                    f"{self.cache_dir}/{cache_id}_s.pt") and not \
                        only_residuals:
                return False
        return True
    
    # Generate cluster centers
    def fit(self, train_descs: Union[np.ndarray, torch.Tensor, None]):
        """
            Using the training descriptors, generate the cluster 
            centers (vocabulary). Function expects all descriptors in
            a single list (see `fit_and_generate` for a batch of 
            images).
            If the cache directory is valid, then retrieves cluster
            centers from there (the `train_descs` are ignored). 
            Otherwise, stores the cluster centers in the cache 
            directory (if using caching).
            
            Parameters:
            - train_descs:  Training descriptors of shape 
                            [num_train_desc, desc_dim]. If None, then
                            caching should be valid (else ValueError).
        """
        # Clustering to create vocabulary
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        # Check if cache exists
        if self.can_use_cache_vlad():
            print("Using cached cluster centers")
            self.c_centers = torch.load(
                    f"{self.cache_dir}/c_centers.pt")
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
                print(f"Desc dim set to {self.desc_dim}")
        else:
            if train_descs is None:
                raise ValueError("No training descriptors given")
            if type(train_descs) == np.ndarray:
                train_descs = torch.from_numpy(train_descs).\
                    to(torch.float32)
            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
            if self.norm_descs:
                train_descs = F.normalize(train_descs)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
            if self.cache_dir is not None:
                print("Caching cluster centers")
                torch.save(self.c_centers, 
                        f"{self.cache_dir}/c_centers.pt")
    
    def fit_and_generate(self, 
                train_descs: Union[np.ndarray, torch.Tensor]) \
                -> torch.Tensor:
        """
            Given a batch of descriptors over images, `fit` the VLAD
            and generate the global descriptors for the training
            images. Use only when there are a fixed number of 
            descriptors in each image.
            
            Parameters:
            - train_descs:  Training image descriptors of shape
                            [num_imgs, num_descs, desc_dim]. There are
                            'num_imgs' images, each image has 
                            'num_descs' descriptors and each 
                            descriptor is 'desc_dim' dimensional.
            
            Returns:
            - train_vlads:  The VLAD vectors of all training images.
                            Shape: [num_imgs, num_clusters*desc_dim]
        """
        # Generate vocabulary
        all_descs = ein.rearrange(train_descs, "n k d -> (n k) d")
        self.fit(all_descs)
        # For each image, stack VLAD
        return torch.stack([self.generate(tr) for tr in train_descs])
    
    def generate(self, query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None) -> torch.Tensor:
        """
            Given the query descriptors, generate a VLAD vector. Call
            `fit` before using this method. Use this for only single
            images and with descriptors stacked. Use function
            `generate_multi` for multiple images.
            
            Parameters:
            - query_descs:  Query descriptors of shape [n_q, desc_dim]
                            where 'n_q' is number of 'desc_dim' 
                            dimensional descriptors in a query image.
            - cache_id:     If not None, then the VLAD vector is
                            constructed using the residual and labels
                            from this file.
            
            Returns:
            - n_vlas:   Normalized VLAD: [num_clusters*desc_dim]
        """
        residuals = self.generate_res_vec(query_descs, cache_id)
        # Un-normalized VLAD vector: [c*d,]
        un_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == 'hard':
            # Get labels for assignment of descriptors
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(
                        f"{self.cache_dir}/{cache_id}_l.pt"):
                labels = torch.load(
                        f"{self.cache_dir}/{cache_id}_l.pt")
            else:
                labels = self.kmeans.predict(query_descs)   # [q]
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(labels, 
                            f"{self.cache_dir}/{cache_id}_l.pt")
            # Create VLAD from residuals and labels
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                # Sum of residuals for the descriptors in the cluster
                #  Shape:[q, c, d]  ->  [q', d] -> [d]
                cd_sum = residuals[labels==k,k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        else:       # Soft cluster assignment
            # Cosine similarity: 1 = close, -1 = away
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(
                        f"{self.cache_dir}/{cache_id}_s.pt"):
                soft_assign = torch.load(
                        f"{self.cache_dir}/{cache_id}_s.pt")
            else:
                cos_sims = F.cosine_similarity( # [q, c]
                        ein.rearrange(query_descs, "q d -> q 1 d"), 
                        ein.rearrange(self.c_centers, "c d -> 1 c d"), 
                        dim=2)
                soft_assign = F.softmax(self.soft_temp*cos_sims, 
                        dim=1)
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(soft_assign, 
                            f"{self.cache_dir}/{cache_id}_s.pt")
            # Soft assignment scores (as probabilities): [q, c]
            for k in range(0, self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                # Sum of residuals for all descriptors (for cluster k)
                cd_sum = ein.rearrange(w * residuals, 
                            "q c d -> (q c) d").sum(dim=0)  # [d]
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        # Normalize the VLAD vector
        n_vlad = F.normalize(un_vlad, dim=0)
        return n_vlad
    
    def generate_multi(self, 
            multi_query: Union[np.ndarray, torch.Tensor, list],
            cache_ids: Union[List[str], None]=None) \
            -> Union[torch.Tensor, list]:
        """
            Given query descriptors from multiple images, generate
            the VLAD for them.
            
            Parameters:
            - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                            There are 'n_imgs' and each image has
                            'n_kpts' keypoints, with 'd' dimensional
                            descriptor each. If a List (can then have
                            different number of keypoints in each 
                            image), then the result is also a list.
            - cache_ids:    Cache IDs for the VLAD vectors. If None,
                            then no caching is done (stored or 
                            retrieved). If a list, then the length
                            should be 'n_imgs' (one per image).
            
            Returns:
            - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate(q, c) \
                for (q, c) in zip(multi_query, cache_ids)]
        try:    # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:    # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass    # Let it remain as a list
        return res
    
    def generate_res_vec(self, 
                query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None) -> torch.Tensor:
        """
            Given the query descriptors, generate a VLAD vector. Call
            `fit` before using this method. Use this for only single
            images and with descriptors stacked. Use function
            `generate_multi` for multiple images.
            
            Parameters:
            - query_descs:  Query descriptors of shape [n_q, desc_dim]
                            where 'n_q' is number of 'desc_dim' 
                            dimensional descriptors in a query image.
            - cache_id:     If not None, then the VLAD vector is
                            constructed using the residual and labels
                            from this file.
            
            Returns:
            - residuals:    Residual vector: shape [n_q, n_c, d]
        """
        assert self.kmeans is not None
        assert self.c_centers is not None
        # Compute residuals (all query to cluster): [q, c, d]
        if cache_id is not None and self.can_use_cache_vlad() and \
                os.path.isfile(f"{self.cache_dir}/{cache_id}_r.pt"):
            residuals = torch.load(
                    f"{self.cache_dir}/{cache_id}_r.pt")
        else:
            if type(query_descs) == np.ndarray:
                query_descs = torch.from_numpy(query_descs)\
                    .to(torch.float32)
            if self.norm_descs:
                query_descs = F.normalize(query_descs)
            residuals = ein.rearrange(query_descs, "q d -> q 1 d") \
                    - ein.rearrange(self.c_centers, "c d -> 1 c d")
            if cache_id is not None and self.can_use_cache_vlad():
                cid_dir = f"{self.cache_dir}/"\
                        f"{os.path.split(cache_id)[0]}"
                if not os.path.isdir(cid_dir):
                    os.makedirs(cid_dir)
                    print(f"Created directory: {cid_dir}")
                torch.save(residuals, 
                        f"{self.cache_dir}/{cache_id}_r.pt")
        # print("residuals",residuals.shape)
        return residuals

    def generate_multi_res_vec(self, 
            multi_query: Union[np.ndarray, torch.Tensor, list],
            cache_ids: Union[List[str], None]=None) \
            -> Union[torch.Tensor, list]:
        """
            Given query descriptors from multiple images, generate
            the VLAD for them.
            
            Parameters:
            - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                            There are 'n_imgs' and each image has
                            'n_kpts' keypoints, with 'd' dimensional
                            descriptor each. If a List (can then have
                            different number of keypoints in each 
                            image), then the result is also a list.
            - cache_ids:    Cache IDs for the VLAD vectors. If None,
                            then no caching is done (stored or 
                            retrieved). If a list, then the length
                            should be 'n_imgs' (one per image).
                            
            Returns:
            - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate_res_vec(q, c) \
                for (q, c) in zip(multi_query, cache_ids)]
        try:    # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:    # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass    # Let it remain as a list
        return res


@dataclass
class Args:
    # Input directory containing images
    in_dir: str = "data/Pittsburgh250k/database/001/"
    # Image file extension
    imgs_ext: str = "jpg"
    # Output directory where global descriptors will be stored
    gdesc_dir: str = "data/Pittsburgh250k/gdesc/001/"
    dino_dir: str = "data/Pittsburgh250k/dinov2/001/"
    # Maximum edge length (expected) across all images (GPU OOM)
    max_img_size: int = 1024
    # Domain to use for loading VLAD cluster centers
    domain: Literal["aerial", "indoor", "urban"] = "urban"
    # Number of clusters (cluster centers for VLAD) - read from cache
    num_c: int = 32
    first_n: int = 10
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    domain: str = 'urban'
    
    
def main(args):
    # logger
    # logger = create_logger(args.logger_name, args.logger_level, args.logger_output)
    
    # Dino_v2 properties (parameters)
    desc_layer = args.desc_layer
    desc_facet = args.desc_facet
    num_c = args.num_c
    domain = args.domain
    
    # GPU device
    device = torch.device("cuda")
    
    # save directory
    gdesc_dir = Path(args.gdesc_dir)
    if not gdesc_dir.exists():
        gdesc_dir.mkdir(parents=True)
    dino_dir = Path(args.dino_dir, f'dinov2_l{desc_layer}_{desc_facet}')
    if not dino_dir.exists():
        dino_dir.mkdir(parents=True)
    
    # Load the DINO extractor model
    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
        desc_facet, device=device)
    
    base_tf = tvf.Compose([ # Base image transformations
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    
    # VLAD object (load cache)
    cache_dir = 'cache/'
    c_centers_file = Path(cache_dir , 'vocabulary' , 'dinov2_vitg14' , f'l{desc_layer}_{desc_facet}_c{num_c}' , domain , 'c_centers.pt')
    assert os.path.isfile(c_centers_file), "Vocabulary not cached!"
    c_centers = torch.load(c_centers_file)
    assert c_centers.shape[0] == num_c, "Wrong number of clusters!"
    # Main VLAD object
    vlad = VLAD(num_c, desc_dim=None, 
        cache_dir=os.path.dirname(c_centers_file))
    vlad.fit(None)  # Load the vocabulary
    
    # Global descriptor generation
    imgs_dir = Path(args.in_dir)
    assert imgs_dir.is_dir(), "Input directory doesn't exist!"
    img_fnames = [img for img in imgs_dir.glob(f"*.{args.imgs_ext}")]
    # img_fnames = glob.glob(f"{imgs_dir}/*.jpg")
    max_img_size = args.max_img_size # prevent GPU OOM
    
    if args.first_n is not None:
        img_fnames = img_fnames[:args.first_n]

    for img_fname in tqdm(img_fnames):
        # DINO features
        with torch.no_grad():
            pil_img = Image.open(img_fname).convert('RGB')
            img_pt = base_tf(pil_img).to(device)
            if max(img_pt.shape[-2:]) > max_img_size:
                c, h, w = img_pt.shape
            
                # Maintain aspect ratio
                if h == max(img_pt.shape[-2:]):
                    w = int(w * max_img_size / h)
                    h = max_img_size
                else:
                    h = int(h * max_img_size / w)
                    w = max_img_size
                
                print(f"To {(h, w) =}")
                
                img_pt = T.resize(img_pt, (h, w), 
                            interpolation=T.InterpolationMode.BICUBIC)
                
                print(f"Resized {img_fname} to {img_pt.shape = }")
            
            # Make image patchable (14, 14 patches)
            c, h, w = img_pt.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
            # Extract descriptor
            ret = extractor(img_pt) # [1, num_patches, desc_dim]
        
        # VLAD global descriptor
        gd = vlad.generate(ret.cpu().squeeze()) # VLAD:  [agg_dim]
        gd_np = gd.numpy()[np.newaxis, ...] # shape: [1, agg_dim]
        
        # Save the global descriptor
        np.save(f"{gdesc_dir}/{Path(img_fname).stem}.npy",
                gd_np)
        # Save the DINOv2 features
        np.save(f"{dino_dir}/{Path(img_fname).stem}.npy", 
                ret.cpu().numpy())


if __name__ == "__main__":
    args = tyro.cli(Args, description=__doc__)
    main(args)
    print('Finished, Exiting program')
    exit(0)
    