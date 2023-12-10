import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from ..utils.dataset import read_loftr_matches

class PairsWiseDataset(Dataset):
    def __init__(self,
                 image_dir,
                 loftr_match_dir,
                 pair_path,
                 img_size,
                 **kwargs):
        """
        Doppelgangers dataset: loading images and loftr matches for Doppelgangers model.
        
        Args:
            image_dir (str): root directory for images.
            loftr_match_dir (str): root directory for loftr matches.
            pair_path (str): pair_list.npy path. This contains image pair information.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
        """
        super().__init__()

        self.image_dir = image_dir    
        self.loftr_match_dir = loftr_match_dir
        self.pairs_info = np.load(pair_path, allow_pickle=True)
        self.pairs_info_length = len(self.pairs_info)
        
        print('loading images, #pairs: ', len(self.pairs_info))
        self.img_size = img_size

        
    def __len__(self):
        return len(self.pairs_info)
    

    def __getitem__(self, idx):
        name0, name1, label, num_matches = self.pairs_info[idx]

        pair_matches = np.load(osp.join(self.loftr_match_dir, '%d.npy'%idx), allow_pickle=True).item()
        keypoints0 = np.array(pair_matches['kpt0'])
        keypoints1 = np.array(pair_matches['kpt1'])
        conf = pair_matches['conf']


        if np.sum(conf>0.8) == 0:
            matches = None
        else:
            F, mask = cv2.findFundamentalMat(keypoints0[conf>0.8],keypoints1[conf>0.8],cv2.FM_RANSAC, 3, 0.99)
            if mask is None or F is None:
                matches = None
            else:
                matches = np.array(np.ones((keypoints0.shape[0], 2)) * np.arange(keypoints0.shape[0]).reshape(-1,1)).astype(int)[conf>0.8][mask.ravel()==1]


        data = {
            'name0': name0,
            'name1': name1,
            'gt': int(label),
            'matches': len(matches)
        }


        return data

def get_datasets(cfg):
    pairwise_dataset = PairsWiseDataset(
                cfg.train.image_dir,
                cfg.train.loftr_match_dir,
                cfg.train.pair_path,
                img_size=getattr(cfg.train, "img_size", 640),
                phase='Train')

    return matching_dataset


# def init_np_seed(worker_id):
#     seed = torch.initial_seed()
#     np.random.seed(seed % 4294967296)


def get_data_loaders(cfg):
    # tr_dataset, te_dataset = get_datasets(cfg)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=tr_dataset, batch_size=cfg.train.batch_size,
    #     shuffle=True, num_workers=cfg.num_workers, drop_last=True,
    #     worker_init_fn=init_np_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=cfg.test.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)
    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader
    }
    return loaders


if __name__ == "__main__":
    pass