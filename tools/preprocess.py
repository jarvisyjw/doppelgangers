import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2
import sys
import yaml
import warnings
import multiprocessing

sys.path.append('../doppelgangers')
sys.path.append('../anyloc')

from anyloc.datasets.base_datasets import get_dataset
from doppelgangers.utils.loftr_matches import save_loftr_matches

def sift_matches(root_dir: str, image0: str, image1: str):
    '''SIFT extraction
    '''
    if root_dir is None:
        warnings.warn("root_dir is None, assuming image0 and image1 are full paths")
        image0 = cv2.imread(str(image0))
        image1 = cv2.imread(str(image1))
    else:
        # read
        image0 = cv2.imread(str(Path(root_dir, image0)))
        image1 = cv2.imread(str(Path(root_dir, image1)))
    
    if image0 is None or image1 is None:
        raise FileNotFoundError("Error: One of the images did not load.")

    # extract
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(image0, None)
    kp1, des1 = sift.detectAndCompute(image1, None)

    if not kp0 or not kp1:
        return 0, None
    
    # match, mutual test ratio 0.75
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0,des1,k=2)
    print(matches)
    if not matches:
        return 0, None
    
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    
    # fundamental matrix estimation
    point_map = np.array([
        [kp0[match.queryIdx].pt[0],
        kp0[match.queryIdx].pt[1],
        kp1[match.trainIdx].pt[0],
        kp1[match.trainIdx].pt[1]] for match in good_matches
    ], dtype=np.float32)
    
    if point_map.shape[0] < 8:
        return 0, None
    F, inliers_idx = cv2.findFundamentalMat(point_map[:, :2], point_map[:, 2:], cv2.FM_RANSAC, 3.0)
    if inliers_idx is None:
        return 0, None
    inliers = point_map[inliers_idx.ravel() == 1]
    return len(inliers), inliers


# transfer from txt to npy
def txt2npy(root_dir, txt_file, npy_file):
    
    if not isinstance(txt_file, str):
        raise TypeError('txt file path must be a str')
    
    if Path(npy_file).is_dir():
        npy_file = Path(npy_file, "pairs.npy")

    f = open(txt_file, 'r')
    pairs = []
    for line in tqdm(f.readlines()):
      line_str = line.strip('\n').split(' ')
      image0, image1, label = line_str
      num_matches, _ = sift_matches(root_dir, image0, image1)
      line_numpy = np.array([str(image0), str(image1), int(label), num_matches], dtype=object)
      pairs.append(line_numpy)
    
    out = np.array(pairs)
    np.save(npy_file, out)
    print('Done!')


def check_pair(root_dir, image0, image1):
      assert Path(root_dir, image0).is_file(), "image0 is not a file"
      assert Path(root_dir, image1).is_file(), "image1 is not a file"
      num_matches, _ = sift_matches(root_dir, image0, image1)
      return num_matches


def check_txt(root_dir, txt_file, npy_file):
      if not isinstance(txt_file, str):
           raise TypeError('txt file path must be a str')
    
      if Path(npy_file).is_dir():
            npy_file = Path(npy_file, "pairs.npy")

      pairs = []
      missing = []
      f = open(txt_file, 'r')
      for line in tqdm(f.readlines()):
            line_str = line.strip('\n').split(', ')
            image0, image1, label = line_str
            print(image0, image1, label)
            try:
                  num_matches = check_pair(root_dir, image0, image1)
            except:
                  missing.append([image0, image1])
                  print("Error: ", image0, image1)
                  continue
            line_numpy = np.array([str(image0), str(image1), int(label), num_matches], dtype=object)
            print(line_numpy)
            pairs.append(line_numpy)

      missing = np.array(missing)
      out = np.array(pairs)
      np.save(npy_file, out)
      if len(missing) > 0:
            print(missing.shape)
            np.save(npy_file + ".missing", missing)
      print('Done!')


def split_data(all_pairs: str, split_pairs: str, length):
    if len(length) > 1:
        all_list = np.load(all_pairs, allow_pickle=True)
        if length[1] == -1:
            split_list = all_list[length[0]:]
        split_list = all_list[length[0]:length[1]]
        np.save(split_pairs, split_list)
    else:
        all_list = np.load(all_pairs, allow_pickle=True)
        split_list = all_list[:length]
        np.save(split_pairs, split_list)


def parser():
    parser = argparse.ArgumentParser(description='preprocessing tools')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    parser.add_argument('--npy_path', default='data/train.npy', type=str, help='npy path')
    parser.add_argument('--txt_path', default='data/train.txt', type=str, help='txt path')
    parser.add_argument('--root_dir')
    parser.add_argument('--output')
    parser.add_argument('--weights')
    parser.add_argument('--loftr', action='store_true')
    parser.add_argument('--pairs', action='store_true')
    parser.add_argument('--txt', action='store_true')
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


def npy2txt(npy: str, txt: str):
    pairs = np.load(npy, allow_pickle=True)
    with open(txt, 'w') as f:
        for pair in tqdm(pairs, total=len(pairs)):
            image0, image1, label, num_matches = pair
            f.write(f'{image0} {image1} {label}\n')
    print('Done!')

def match_pairs(retrieval, query_path, database_path, positives_per_query, pos, neg):
    query = query_path
    query_name = str(Path(query.parent.name, query.name))
    db = database_path
    db_name = str(Path(db.parent.name, db.name))
    num_matches, _ = sift_matches(None, query, db)
    if retrieval in positives_per_query:
        ret = np.array([query_name, db_name, 1, num_matches], dtype=object)
        pos +=1
    else:
        ret = np.array([query_name, db_name, 0, num_matches], dtype=object)
        neg += 1
    return ret, pos, neg

def pairs_from_retrieval(cfg):
    dataset = get_dataset(cfg.data)
    # get positive per query
    retrieval_results = np.load(cfg.pairs_from_retrieval.retrieval_results, allow_pickle=True)
    positives_per_query = dataset.get_positives()
    queries_paths = dataset.get_queries_paths()
    database_paths = dataset.get_database_paths()
    pairs = []
    pos = 0
    neg = 0
    
    for i, retrievals in tqdm(enumerate(retrieval_results), total=len(retrieval_results)):
        for retrieval in retrievals[:20]:
            query = queries_paths[i]
            query_name = str(Path(query.parent.name, query.name))
            db = database_paths[retrieval]
            db_name = str(Path(db.parent.name, db.name))
            num_matches, _ = sift_matches(None, query, db)
            if retrieval in positives_per_query[i]:
                ret = np.array([query_name, db_name, 1, num_matches], dtype=object)
                pos +=1
            else:
                ret = np.array([query_name, db_name, 0, num_matches], dtype=object)
                neg += 1
        print(ret)
        pairs.append(ret)
    out = np.array(pairs)
    np.save(cfg.pairs_from_retrieval.output_path, out)
    print('Pairs Generation Done!')
    print(f'Positive Pairs: {pos}, Negative Pairs: {neg}')


if __name__ == "__main__":
    args, cfg = parser()
    # Step 1: Generate pairs meta info
    if args.pairs:
        if args.txt:
            if Path(args.npy_path).is_file():
                raise Warning('npy file already exists, skip generation.')
            else:
                txt2npy(args.root_dir, args.txt_path, args.npy_path)
        else:
            # Process pairs from retreival results.
            pairs_from_retrieval(cfg)

    # Step 2: Extract loftr matches
    if args.loftr:
        '''Ectract loftr matches from pairs metadata
        Args:
            --root_dir: str, root directory of the images
            --npy_path: str, path to the pairs metadata
            --output: str, output directory to save the loftr matches
            --weights: str, path to the loftr weights
        '''
        save_loftr_matches(args.root_dir, args.npy_path, args.output, args.weights)