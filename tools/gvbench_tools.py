from pathlib import Path
from tqdm import tqdm
import numpy as np
from doppelgangers.utils.loftr_matches import save_loftr_matches, save_loftr_matches_batch
import argparse

# transfer from txt to npy
def txt2npy(txt_file, npy_file):
    
    if not isinstance(txt_file, str):
        raise TypeError('txt file path must be a str')
    if Path(npy_file).is_dir():
        npy_file = Path(npy_file, "pairs.npy")
    
    if not Path(npy_file).parent.exists():
        Path(npy_file).parent.mkdir(parents=True, exist_ok=True)

    f = open(txt_file, 'r')
    pairs = []
    for line in tqdm(f.readlines()):
      line_str = line.strip('\n').split(' ')
      image0, image1, label = line_str
      line_numpy = np.array([str(image0), str(image1), int(label), 0], dtype=object)
      pairs.append(line_numpy)
    
    out = np.array(pairs)
    np.save(npy_file, out)
    print('Done!')
    return out

def parser():
    parser = argparse.ArgumentParser(description='Convert txt file to npy file')
    parser.add_argument('--txt_file', type=str)
    parser.add_argument('--npy_file', type=str)
    parser.add_argument('--loftr_matches_path', type=str)
    parser.add_argument('--image_root_path')
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parser()
    pairs_info = txt2npy(args.txt_file, args.npy_file)
    if args.batch_size:
        save_loftr_matches_batch(args.image_root_path, args.npy_file, args.loftr_matches_path, model_weight_path='weights/outdoor_ds.ckpt', batch_size=args.batch_size)
    else:
        save_loftr_matches(args.image_root_path, args.npy_file, args.loftr_matches_path)