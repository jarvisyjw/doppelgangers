import torch
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from torch.nn import functional as F
from PIL import Image
import numpy as np
import tyro
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Literal, Union, List
from pathlib import Path


'''

This script is a mini version of the anyloc_vlad_generate.py script.
This script can be excuted alone without the need for the other scripts.
The detailed usage is provided below.

Load an AnyLoc-VLAD-[backbone] model from torch.hub
        The default settings are for AnyLoc-VLAD-DINOv2; and the
        'indoor' domain is used. The domain would depend on the 
        deployment setting/use case (environment).
        
        Parameters:
        - backbone (str):   The backbone to use. Should be "DINOv2" or
                            "DINOv1".
        - vit_model (str):  The ViT model (architecture) to use. Must
                            be compatible with the backbone. "/" and
                            "-" are ignored.
        - vit_layer (int):  The layer to use for feature extraction.
        - vit_facet (str):  The ViT facet to use for extraction.
        - num_c (int):      Number of cluster centers to use (for
                            VLAD clustering).
        - domain (str):     Domain for cluster centers.
        - device (torch.device):    Device for model; "cpu" or "cuda"
        
        Notes:
        - All string arguments are converted to lower case.

'''

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts250k", split="train"):
        super().__init__()
        
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name)
        
        self.dataset_folder = join(self.dataset_folder, "images", split)
        
        database_folder_name, queries_folder_name = "database", "queries"
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        self.resize = args.resize
        self.test_method = args.test_method
        
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, database_folder_name)
        queries_folder = join(self.dataset_folder, queries_folder_name)
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.database_paths = natsorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = natsorted(glob(join(queries_folder, "**", "*.jpg"),  recursive=True))
        
        # vpr bench
        if self.vprbench:
            self.database_utms, self.queries_utms, \
                self.soft_positives_per_query = \
                BaseDataset.generate_positives_and_utms("vpr_bench", 
                    None, None, 
                    join(self.dataset_folder,'ground_truth_new.npy'))
        # vg bench
        else:
            self.database_utms, self.queries_utms, \
                self.soft_positives_per_query = \
                BaseDataset.generate_positives_and_utms("vg_bench", 
                    self.database_paths, self.queries_paths, 
                    dist_thresh=args.val_positive_dist_threshold)
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
    
    @staticmethod
    def generate_positives_and_utms(db_type: str, 
                db_paths: Union[List[str], None],
                query_paths: Union[List[str], None], 
                gt_path: Union[str, None] = None, 
                dist_thresh: float = 25) \
                -> Tuple[Union[np.ndarray, None],
                        Union[np.ndarray, None], np.ndarray]:
        """
            Given a database type, generate the ground truth and the
            UTM coordinates (if applicable).
            Currently, only vpr_bench and vg_bench are implemented.
            - "vpr_bench" cannot get UTM path (they're set to None).
                The query and database ground truth is read from the
                'gt_path' file. If 'query_paths' is not None, then the
                indices of this is filtered (to correspond to correct
                queries). If 'db_paths' is not None, then the values
                for database (retrievals for corresponding queries)
                are subsequently also filtered.
            - "vg_bench" uses the db_paths and query_paths, and 
                parses the string to extract UTM coordinates. It then
                applies kNN to get the positives. The format must be
                `path/to/file/@utm_easting@utm_northing@...@.[ext]`.
                The positives are from UTM easting and northing.
            
            Parameters:
            - db_type:  The database type. "vpr_bench" or "vg_bench"
            - db_paths: The list of paths for the database images.
            - query_paths:  The list of paths for the query images.
            - gt_path:  Path to '.npy' ground truth for "vpr_bench"
            - dist_thresh:  Distance threshold (for +ve retrieval)
            
            Returns:
            - db_utms:  Database UTM coordinates
            - qu_utms:  Query UTM coordinates
            - sp_per_query: The soft-positive retrievals per query
                            - Shape: [n_q, ]    Dtype: np.object_
                            - Index [i] has the retrievals from the
                                database for the query 'i'
        """
        sp_per_query = db_utms = qu_utms = None
        if db_type == "vpr_bench":
            # UTM not possible
            db_utms = qu_utms = None
            # Positives
            pos_per_qu = np.load(gt_path, 
                                    allow_pickle=True)[:, 1]
            if query_paths is not None: # Filter queries
                _qs = [int(os.path.basename(qp).split(".")[0]) \
                        for qp in query_paths]
                pos_per_qu = pos_per_qu[_qs]
            if db_paths is not None:    # Filter database (retrievals)
                _db = [int(os.path.basename(db).split(".")[0]) \
                        for db in db_paths]
                _db_map = dict(zip(_db, range(len(_db))))   # Indices
                _p2 = [np.array(q_db)[np.isin(q_db, _db)] \
                        for q_db in pos_per_qu]
                _p3 = np.array([np.array([_db_map[v] for v in x]) \
                                for x in _p2], dtype=np.object_)
                pos_per_qu = _p3
            sp_per_query = pos_per_qu
        elif db_type == "vg_bench":
            db_utms = np.array([(path.split("@")[1], 
                    path.split("@")[2]) for path in db_paths], float)
            qu_utms = np.array([(path.split("@")[1],
                    path.split("@")[2]) for path in query_paths], 
                    float)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(db_utms)
            sp_per_query = knn.radius_neighbors(qu_utms, 
                    radius=dist_thresh, return_distance=False)
        else:
            raise NotImplementedError(f"Type: {db_type}")
        return db_utms, qu_utms, sp_per_query

    def get_image_paths(self):
        return self.images_paths

    def get_image_relpaths(self, i: Union[int, List[int]]) \
            -> Union[List[str], str]:
        """
            Get the relative path of the image at index i. Multiple 
            indices can be passed as a list (or int-like array).
        """
        indices = i
        if type(i) == int:
            indices = [i]
        img_paths = self.get_image_paths()
        s = 2 if self.vprbench else 4
        rel_paths = ["/".join(img_paths[k].split("/")[-s:]) \
                        for k in indices]
        if type(i) == int:
            return rel_paths[0]
        return rel_paths

    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        
        if self.use_mixVPR:
            # print("yes")
            img = mixVPR_transform(img)
        elif self.use_SAM:
            img = cv2.imread(self.images_paths[index])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            img = base_transform(img)
            # With database images self.test_method should always be "hard_resize"
            if self.test_method == "hard_resize":
                # self.test_method=="hard_resize" is the default, resizes all images to the same size.
                img = T.functional.resize(img, self.resize)
            else:
                img = self._test_query_transform(img)
        return img, index
    
    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = T.functional.resize(img, min(self.resize))
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = T.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = T.functional.resize(img, shorter_side)
            processed_img = torch.stack(T.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query


@dataclass
class Args:
    # if localization mode
    localize: bool = False
    # Root directory containing images
    root_dir: str = 'data/pitts250k/images'
    # mode: train, val, test
    mode: Literal["train", "val", "test"] = "train"
    # Input directory containing images
    in_dir: str = f'{root_dir}/{mode}/'
    # Image file extension
    imgs_ext: str = "jpg"
    descs_ext: str = "npy"
    # Output directory where global descriptors will be stored
    # gdesc_dir: str = f"{root_dir}/{mode}/"
    # Maximum edge length (expected) across all images (GPU OOM)
    max_img_size: int = 1024
    # Domain to use for loading VLAD cluster centers
    domain: Literal["aerial", "indoor", "urban"] = "urban"
    # Number of clusters (cluster centers for VLAD) - read from cache
    num_c: int = 32
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    domain: str = 'urban'


def extract_gdesc(image_fnames, model, base_tf, max_img_size, gdesc_dir):
    
    for img_fname in tqdm(img_fnames):
        # DINO features
        with torch.no_grad():
            pil_img = Image.open(img_fname).convert('RGB')
            img_pt = base_tf(pil_img).to('cuda')
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
            res = model(img_pt) # [1, num_patches, desc_dim]
        
        gd = res.cpu().squeeze() # VLAD:  [agg_dim]
        gd_np = gd.numpy()[np.newaxis, ...] # shape: [1, agg_dim]
        
        # Save the global descriptor
        np.save(f"{gdesc_dir}/{Path(img_fname).stem}.npy",
                gd_np)


def extract_global_descriptors(args):
    # parameters
    domain = args.domain
    max_img_size = args.max_img_size # prevent GPU OOM
    num_c = args.num_c
    desc_layer = args.desc_layer
    desc_facet = args.desc_facet

    imgs_dir = Path(args.in_dir)
    assert imgs_dir.is_dir(), "Input root directory doesn't exist!"
    
    qimgs_dir = imgs_dir / 'queries'
    assert qimgs_dir.is_dir(), "Queries directory doesn't exist!"
    
    rimgs_dir = imgs_dir / 'database'
    assert rimgs_dir.is_dir(), "Database directory doesn't exist!"
    
    qimg_fnames = [img for img in qimgs_dir.glob(f"*.{args.imgs_ext}")]
    rimg_fnames = [img for img in rimgs_dir.glob(f"*.{args.imgs_ext}")]
    
    # Load the anyloc extractor model from torch hub
    model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", 
            backbone="DINOv2", domain = domain, num_c = num_c,
            vit_model = 'vitg14', vit_layer = desc_layer, vit_facet = desc_facet,  
            device="cuda")
    
    # save directory
    qGdesc_dir = qimgs_dir
    rGdesc_dir = rimgs_dir
    
    base_tf = tvf.Compose([ # Base image transformations
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    
    # Global descriptor generation
    # extract for queries
    print("Extracting global descriptors for queries......")
    
    
    

    


# from anyloc utilities
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




def localize(args):
    # load query and database global descriptors
    
    
    

def main(args):
    
    if args.localize:
        # localization mode
        localize(args)
    else:
        # extract global descriptors
        extract_global_descriptors(args)


if __name__ == "__main__":
    args = tyro.cli(Args, description=__doc__)
    main(args)
    print('Finished, Exiting program')
    exit(0)