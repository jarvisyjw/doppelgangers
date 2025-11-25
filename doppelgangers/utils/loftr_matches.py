import torch
import cv2
import numpy as np
import os.path as osp
import os
import tqdm
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


from ..third_party.loftr import LoFTR, default_cfg


class LoFTRMatchingDataset(Dataset):
    def __init__(self, data_path, pair_path, img_size=1024, df=8, padding=True):
        """
        Dataset for LoFTR matching
        
        Args:
            data_path: Path to the image directory
            pair_path: Path to the numpy file containing image pairs
            img_size: Size to resize images to
            df: Downscale factor for the mask
            padding: Whether to apply padding
        """
        self.data_path = data_path
        self.pairs_info = np.load(pair_path, allow_pickle=True)
        self.img_size = img_size
        self.df = df
        self.padding = padding
        self.batch = True  # for batch processing
    
    def __len__(self):
        return len(self.pairs_info)
    
    def __getitem__(self, idx):
        # Extract image names
        if len(self.pairs_info[idx]) == 4:
            name0, name1, _, _ = self.pairs_info[idx]
        elif len(self.pairs_info[idx]) == 3:
            name0, name1, _ = self.pairs_info[idx]
        else:
            raise ValueError(f"Unexpected format for pair at index {idx}")
        
        # Get image paths
        img0_pth = osp.join(self.data_path, name0)
        img1_pth = osp.join(self.data_path, name1)
        
        # Read and process images
        img0_raw, mask0 = read_image(img0_pth, self.img_size, self.df, self.padding, self.batch)
        img1_raw, mask1 = read_image(img1_pth, self.img_size, self.df, self.padding, self.batch)
        
        return {
            'image0': torch.from_numpy(img0_raw),
            'image1': torch.from_numpy(img1_raw),
            'mask0': torch.from_numpy(mask0),
            'mask1': torch.from_numpy(mask1),
            'idx': idx,
            'name0': name0,
            'name1': name1
        }

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new


def read_image(img_pth, img_size, df, padding, batch=False):
    if str(img_pth).endswith('gif'):
        
        pil_image = ImageOps.grayscale(Image.open(str(img_pth)))
        img_raw = np.array(pil_image)
    else:
        img_raw = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)

    w, h = img_raw.shape[1], img_raw.shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        if batch:
            mask = np.zeros((pad_to, pad_to), dtype=bool)
            mask[:h_new, :w_new] = True
            mask = mask[::8,::8]
        else:
            mask = np.zeros((1,pad_to, pad_to), dtype=bool)
            mask[:,:h_new,:w_new] = True
            mask = mask[:,::8,::8]
    
    image = cv2.resize(img_raw, (w_new, h_new))
    if batch:
        pad_image = np.zeros((1, pad_to, pad_to), dtype=np.float32)
        pad_image[0,:h_new,:w_new]=image/255.
    else:
        pad_image = np.zeros((1,1, pad_to, pad_to), dtype=np.float32)
        pad_image[0,0,:h_new,:w_new]=image/255.

    return pad_image, mask


# def read_image(img_pth, img_size, df, padding):
#     if str(img_pth).endswith('gif'):
        
#         pil_image = ImageOps.grayscale(Image.open(str(img_pth)))
#         img_raw = np.array(pil_image)
#     else:
#         img_raw = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)

#     w, h = img_raw.shape[1], img_raw.shape[0]
#     w_new, h_new = get_resized_wh(w, h, img_size)
#     w_new, h_new = get_divisible_wh(w_new, h_new, df)

#     if padding:  # padding
#         pad_to = max(h_new, w_new)    
#         mask = np.zeros((1,pad_to, pad_to), dtype=bool)
#         mask[:,:h_new,:w_new] = True
#         mask = mask[:,::8,::8]
    
#     image = cv2.resize(img_raw, (w_new, h_new))
#     pad_image = np.zeros((1,1, pad_to, pad_to), dtype=np.float32)
#     pad_image[0,0,:h_new,:w_new]=image/255.

#     return pad_image, mask


def save_loftr_matches_batch(data_path, pair_path, output_path, model_weight_path="weights/outdoor_ds.ckpt", batch_size=4, num_workers=4):
    """
    TODO: Still bugy, need to fix.
    BUG: I think the original loftr's network does not support the batch operation, which is also not possible.
    However, I found that, the bugs are in the result extraction of loftr's processed result, the original implementation
    of loftr seems not support the batch operations.
                mkpts0 = batch_to_process['mkpts0_f'][i].cpu().numpy()
                mkpts1 = batch_to_process['mkpts1_f'][i].cpu().numpy()
                mconf = batch_to_process['mconf'][i].cpu().numpy()
    The shape of mkpts0_f and mkpts1_f is [N,2] regardless of batched image pairs as input.
    
    Process image pairs in batches using LoFTR matcher
    
    Args:
        data_path: Path to the image directory
        pair_path: Path to the numpy file containing image pairs
        output_path: Path to save the matching results
        model_weight_path: Path to the LoFTR model weights
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
    """
    
    # Initialize the matcher
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_weight_path)['state_dict'])
    matcher = matcher.eval().cuda()
    
    # Create the dataset and dataloader
    dataset = LoFTRMatchingDataset(data_path, pair_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4
    )
    
    # Create output directory if it doesn't exist
    if not osp.exists(output_path):
        if not osp.exists(osp.dirname(output_path)):
            os.makedirs(osp.dirname(output_path))
        os.mkdir(output_path)
    
    # Process batches
    for batch_data in tqdm.tqdm(dataloader):
        batch_indices = batch_data['idx'].numpy()
        
        # Skip already processed pairs
        to_process = []
        for i, idx in enumerate(batch_indices):
            if not osp.exists(f"{output_path}/{idx}.npy"):
                to_process.append(i)
        
        if not to_process:
            print('All pairs in this batch are already processed. Skipping batch.')
            continue
            
        # Prepare batch for processing
        batch_to_process = {
            'image0': batch_data['image0'][to_process].cuda(),
            'image1': batch_data['image1'][to_process].cuda(),
            'mask0': batch_data['mask0'][to_process].cuda(),
            'mask1': batch_data['mask1'][to_process].cuda()
        }
        print(batch_data['image0'].shape)
        print(batch_data['image1'].shape)
        # Process with LoFTR
        with torch.no_grad():
            matcher(batch_to_process)
            
            # Save results for each pair in the batch
            for i, idx in enumerate([batch_indices[j]] for j in to_process):
                # print(batch_to_process['mkpts0_f'].shape)
                # print(batch_to_process['mkpts1_f'].shape)
                # print(batch_to_process['mconf'].shape)
                
                mkpts0 = batch_to_process['mkpts0_f'][i].cpu().numpy()
                mkpts1 = batch_to_process['mkpts1_f'][i].cpu().numpy()
                mconf = batch_to_process['mconf'][i].cpu().numpy()
                # print(batch_to_process)
                
                np.save(f"{output_path}/{idx[0]}.npy", {
                    "kpt0": mkpts0,
                    "kpt1": mkpts1,
                    "conf": mconf
                })
            break


def save_loftr_matches(data_path, pair_path, output_path, model_weight_path="weights/outdoor_ds.ckpt"):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_weight_path)['state_dict'])
    matcher = matcher.eval().cuda()

    pairs_info = np.load(pair_path, allow_pickle=True)
    # pairs_info = np.loadtxt(pair_path, dtype=str, delimiter=', ')
    img_size = 1024
    df = 8
    padding = True

    if not osp.exists(output_path):
        if not osp.exists(osp.dirname(output_path)):
            os.makedirs(osp.dirname(output_path))
        os.mkdir(output_path)
        
    for idx in tqdm.tqdm(range(pairs_info.shape[0])):
        if osp.exists(output_path+'/%d.npy'%idx):
            continue
        if len(pairs_info[idx]) == 4:
            name0, name1, _, _, = pairs_info[idx]
        if len(pairs_info[idx]) == 3:
            name0, name1, _, = pairs_info[idx]

        img0_pth = osp.join(data_path, name0)
        img1_pth = osp.join(data_path, name1)
        img0_raw, mask0 = read_image(img0_pth, img_size, df, padding)
        img1_raw, mask1 = read_image(img1_pth, img_size, df, padding)        
        img0 = torch.from_numpy(img0_raw).cuda()
        img1 = torch.from_numpy(img1_raw).cuda()
        mask0 = torch.from_numpy(mask0).cuda()
        mask1 = torch.from_numpy(mask1).cuda()
        batch = {'image0': img0, 'image1': img1, 'mask0': mask0, 'mask1':mask1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            np.save(output_path+'/%d.npy'%idx, {"kpt0": mkpts0, "kpt1": mkpts1, "conf": mconf})

