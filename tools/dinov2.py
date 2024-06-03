from typing import Literal, Union, List
import torch
from torch import nn
from torchvision.transforms import functional as T
from torch.nn import functional as F
from torchvision import transforms as tvf
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
import os
import glob
import tyro
import numpy as np 
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

@dataclass
class LocalArgs:
      img_dir: str = "data/GV-Bench/images/day1/"
      imgs_ext: str = "jpg"
      first_n: Union[int, None] = None
      max_img_size: int = 1024
      out_dir: str = "data/GV-Bench/dino_features/day1"
      

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
        

def main(args):
      device = torch.device("cuda")
      # Dino_v2 properties (parameters)
      desc_layer: int = 31
      desc_facet: Literal["query", "key", "value", "token"] = "value"
      
      # Load the DINO extractor model
      extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
            desc_facet, device=device)
      
      base_tf = tvf.Compose([ # Base image transformations
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
      ])
      
      # Global descriptor generation
      imgs_dir = args.img_dir
      assert os.path.isdir(imgs_dir), "Input directory doesn't exist!"
      img_fnames = glob.glob(f"{imgs_dir}/*.jpg")
      max_img_size = args.max_img_size # prevent GPU OOM
      
      if args.first_n is not None:
            img_fnames = img_fnames[:args.first_n]

      if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

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
        

            np.save(f"{args.out_dir}/{Path(img_fname).stem}.npy", ret.cpu().numpy())


def viz_dino_features(dino_features: str, image_path: str, grid_size = (45 , 73)):
      dino_features = np.load(dino_features)
      pca = PCA(n_components=3)
      dino_features = pca.fit_transform(dino_features[0])
      dino_out = dino_features.reshape((*grid_size, -1))
      dino_image = (dino_out - np.min(dino_out)) / (np.max(dino_out) - np.min(dino_out))
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,20))
      image = Image.open(image_path)
      ax1.imshow(image)
      ax2.imshow(dino_image)
      fig.tight_layout()
      fig.savefig("dino_features.png")
    #   plt.imshow(dino_image)
    #   fig.imsave("dino_features.png", dino_image)

      


if __name__ == "__main__":
    args = tyro.cli(LocalArgs)
    main(args)
    print("Done!")
    exit(0)
    # dino_features = np.load("data/GV-Bench/dino_features/day/1418132452169723.npy")
    # dino_features = 'data/GV-Bench/dino_features/day/1418132452169723.npy'
    # image_path = 'data/GV-Bench/images/day0/1418132452169723.jpg'
    # viz_dino_features(dino_features, image_path)
    # print(dino_features.shape)