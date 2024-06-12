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


@dataclass
class Args:
    # Input directory containing images
    in_dir: str = "data/Pittsburgh250k/database/002/"
    # Image file extension
    imgs_ext: str = "jpg"
    # Output directory where global descriptors will be stored
    gdesc_dir: str = "data/Pittsburgh250k/gdesc/002/"
    # Maximum edge length (expected) across all images (GPU OOM)
    max_img_size: int = 1024
    # Domain to use for loading VLAD cluster centers
    domain: Literal["aerial", "indoor", "urban"] = "urban"
    # Number of clusters (cluster centers for VLAD) - read from cache
    num_c: int = 32
    first_n: int = None
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    domain: str = 'urban'
    
def main(args):
    # parameters
    domain = args.domain
    max_img_size = args.max_img_size # prevent GPU OOM
    num_c = args.num_c
    desc_layer = args.desc_layer
    desc_facet = args.desc_facet

    imgs_dir = Path(args.in_dir)
    assert imgs_dir.is_dir(), "Input directory doesn't exist!"
    
    img_fnames = [img for img in imgs_dir.glob(f"*.{args.imgs_ext}")]
    
    # Load the anyloc extractor model from torch hub
    model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", 
            backbone="DINOv2", domain = domain, num_c = num_c,
            vit_model = 'vitg14', vit_layer = desc_layer, vit_facet = desc_facet,  
            device="cuda")
    
    # save directory
    gdesc_dir = Path(args.gdesc_dir)
    if not gdesc_dir.exists():
        gdesc_dir.mkdir(parents=True)
    
    base_tf = tvf.Compose([ # Base image transformations
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])
    
    # Global descriptor generation
    
    # use first n images for testing
    if args.first_n is not None:
        img_fnames = img_fnames[:args.first_n]

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


if __name__ == "__main__":
    args = tyro.cli(Args, description=__doc__)
    main(args)
    print('Finished, Exiting program')
    exit(0)