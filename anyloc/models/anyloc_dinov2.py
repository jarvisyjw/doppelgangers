import torch
import numpy as np

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

class AnyLocExtractor(torch.nn.Module):
      def __init__(self, domain, num_c, desc_layer, desc_facet):
            super().__init__()
            self.domain = domain
            self.num_c = num_c
            self.desc_layer = desc_layer
            self.desc_facet = desc_facet
            
            self.model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", 
                  backbone="DINOv2", domain = self.domain, num_c = self.num_c,
                  vit_model = 'vitg14', vit_layer = self.desc_layer, vit_facet = self.desc_facet,  
                  device="cuda")

      def forward(self, data):
            data = data.cuda()
            res = self.model(data) # VLAD:  [agg_dim]
            return res.cpu().squeeze().numpy() # shape: [1, agg_dim]
      

def get_model(cfg):
      return AnyLocExtractor(cfg.domain, cfg.num_c, cfg.desc_layer, cfg.desc_facet)
