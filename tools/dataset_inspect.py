import numpy as np
import importlib
import argparse
import yaml
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
from keras.applications.vgg16 import VGG16

sys.path.append('/home/jarvis/jw_ws/Verification/doppelgangers')
from anyloc.datasets.base_datasets import get_dataset
from doppelgangers.datasets.pairwise_disambiguation_dataset import get_datasets

# train_noflip_pairs = np.load('data/doppelgangers_dataset/doppelgangers/pairs_metadata/train_pairs_noflip.npy', allow_pickle=True)
# train_flip_pairs = np.load('data/doppelgangers_dataset/doppelgangers/pairs_metadata/train_pairs_flip.npy', allow_pickle=True)
# test_pairs = np.load('data/doppelgangers_dataset/doppelgangers/pairs_metadata/test_pairs.npy', allow_pickle=True)

# # print shape of each pair
# print('Train noflip pairs:', train_noflip_pairs.shape)
# print('Train flip pairs:', train_flip_pairs.shape)
# print('Test pairs:', test_pairs.shape)
# print(f'Train noflip/Test ratio: {train_noflip_pairs.shape[0]/test_pairs.shape[0]:.2f}')
# print(f'Train flip/Test ratio: {train_flip_pairs.shape[0]/test_pairs.shape[0]:.2f}')

def parser():
    parser = argparse.ArgumentParser(description='anyloc')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

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

def copy_desc(source_path: Path, dest_path: Path, desc_list = None):
    if desc_list is not None:
        for desc in desc_list:
            source = source_path / desc
            dest = dest_path / desc
            # copy file from source path to dest path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    else:
        raise NotImplementedError
        # desc_list = dest_path.glob('**/*.jpg')

def viz_tsne(dataset, idx):
    viz_sample = dataset[idx]
    for sample in tqdm(viz_sample, total=len(viz_sample)):
        data = sample['data']
        print(data.shape)
        

if __name__ == '__main__':
    args, config = parser()
    tr_dataset = get_datasets(config.data)[0]
    viz_tsne(tr_dataset, 0)
    # # copy desc
    # source_path = Path('data/pitts250k/retrieval_results/train/gdesc')
    # dest_path = Path('data/pitts30k/retrieval_results/train/gdesc')
    # dest_image_paths = Path('data/pitts30k/images/train')
    # dest_desc_list = [str(Path(path.parent.name , path.with_suffix('.npy').name)) for path in dest_image_paths.glob('**/*.jpg')]
    # copy_desc(source_path, dest_path, dest_desc_list)
    print('Done')
      
      