import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

from natsort import natsorted
from pathlib import Path
from typing import Union, List
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from tqdm import tqdm


base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class BaseDataset(Dataset):
    """Dataset with images from database and queries, 
      used for inference (testing and building cache).
      Dataset format should follow the 
      deep-image-retrieval-benchmarking format.
    """
    
    def __init__(self, datasets_folder="data", 
                dataset_name="pitts250k",
                split="train", _ext="jpg",
                max_img_size=1024, 
                positive_dist_threshold= 25, 
                cached = False):
            super().__init__()
            self.cached = cached
            self.dataset_name = dataset_name
            self.dataset_folder = Path(datasets_folder, dataset_name)
            self.dataset_folder = Path(self.dataset_folder, "images", split)
            database_folder_name, queries_folder_name = "database", "queries"
            if not self.dataset_folder.exists():
                  raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
            self.max_img_size = max_img_size
                        
            #### Read paths and UTM coordinates for all images.
            self.database_folder = self.dataset_folder/ database_folder_name
            self.queries_folder = self.dataset_folder/ queries_folder_name
            if not self.database_folder.exists():
                  raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
            if not self.queries_folder.exists():
                  raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")
            self.database_paths = natsorted(self.database_folder.glob(f"*.{_ext}"))
            self.queries_paths = natsorted(self.queries_folder.glob(f"*.{_ext}"))
            # if cached, load descriptors
            if self.cached:
                self.database_descs = natsorted(self.database_folder.glob("*.npy"))
                self.queries_descs = natsorted(self.queries_folder.glob("*.npy"))
            
            self.database_utms, self.queries_utms, \
                  self.soft_positives_per_query = \
                  BaseDataset.generate_positives_and_utms( self.database_paths, 
                                                      self.queries_paths, 
                  dist_thresh=positive_dist_threshold)
            
            self.images_paths = list(self.database_paths) + list(self.queries_paths)
            self.database_num = len(self.database_paths)
            self.queries_num = len(self.queries_paths)
    
    @staticmethod
    def generate_positives_and_utms(db_paths: Union[List[str], None],
                query_paths: Union[List[str], None], 
                dist_thresh: float = 25):
        
        """
            Given a database type, generate the ground truth and the
            UTM coordinates uses the db_paths and query_paths, and 
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
        db_utms = np.array([(str(path).split("@")[1], 
                str(path).split("@")[2]) for path in db_paths], float)
        qu_utms = np.array([(str(path).split("@")[1],
                str(path).split("@")[2]) for path in query_paths], 
                float)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(db_utms)
        sp_per_query = knn.radius_neighbors(qu_utms, 
                radius=dist_thresh, return_distance=False)
        return db_utms, qu_utms, sp_per_query
    
    @staticmethod
    def build_tensor_from_descs(descs: List[str]):
        """
            Given a list of paths to '.npy' files, build a tensor
            of shape [n, d] where 'n' is the number of descriptors
            and 'd' is the descriptor dimension.
            
            Parameters:
            - descs:    List of paths to '.npy' files.
            
            Returns:
            - desc_tensor:  Tensor of shape [n, d]
        """
        desc_tensor = []
        print('Loading descriptors...')
        for desc in tqdm(descs, total=len(descs)):
            desc_tensor.append(np.load(desc, allow_pickle=True)[0])
        desc_tensor = torch.from_numpy(np.array(desc_tensor))
        print(f'Descriptors loaded with size: {desc_tensor.shape}')
        return desc_tensor

    def get_image_paths(self):
        return self.images_paths

    def __getitem__(self, index):
        if self.cached:
            # load descriptors
            query_desc = np.load(self.queries_descs[index], allow_pickle=True)
            postives = self.soft_positives_per_query[index]
            data = { 'query_desc': torch.from_numpy(query_desc),
                     'pos_idx': postives
                    }
            return data
        else:
            # extract descriptors
            img = Image.open(self.images_paths[index]).convert('RGB')
            img_dir = self.images_paths[index]
            img = base_transform(img)
            if max(img.shape[-2:]) > self.max_img_size:
                c, h, w = img.shape
                # Maintain aspect ratio
                if h == max(img_pt.shape[-2:]):
                    w = int(w * self.max_img_size / h)
                    h = self.max_img_size
                else:
                    h = int(h * self.max_img_size / w)
                    w = self.max_img_size            
                img_pt = T.resize(img_pt, (h, w), 
                            interpolation=T.InterpolationMode.BICUBIC)     
            # Make image patchable (14, 14 patches)
            c, h, w = img.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img = tvf.CenterCrop((h_new, w_new))(img)
            data = { 'img': img,
                    'img_dir': str(img_dir)
                    }
            return data
    
    def __len__(self):
        return len(self.images_paths)
    
    def get_positives(self):
        return self.soft_positives_per_query
    
    def get_queries_paths(self):
        return self.queries_paths

    def get_database_paths(self):
        return self.database_paths
    
    def get_queries_descs(self):
        return self.queries_descs
    
    def get_database_descs(self):
        return self.database_descs
    
    def get_database_descs_tensor(self):
        return BaseDataset.build_tensor_from_descs(self.database_descs)
    
    def get_queries_descs_tensor(self):
        return BaseDataset.build_tensor_from_descs(self.queries_descs)
    
    def get_queries_positives(self):
        print("Getting positives...")
        return self.soft_positives_per_query
            

def get_dataset(cfg):
    dataset = BaseDataset(datasets_folder=cfg.dataset_folder,
                        dataset_name=cfg.dataset_name,
                        split=cfg.split,
                        _ext=cfg.image_ext,
                        max_img_size=cfg.max_img_size,
                        positive_dist_threshold=cfg.positive_dist_threshold,
                        cached=cfg.cached)
    return dataset

def get_dataloader(cfg):
    dataset = get_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, 
                            num_workers=cfg.num_workers,
                            pin_memory=True, shuffle=cfg.shuffle)
    return dataloader