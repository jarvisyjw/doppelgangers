import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2
import sys
import yaml
import warnings
import time
import keyboard
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt

sys.path.append('../doppelgangers')
sys.path.append('../anyloc')

from anyloc.datasets.base_datasets import get_dataset
from doppelgangers.utils.loftr_matches import save_loftr_matches
from viz import plot_images, read_image, plot_sequence, plot_retreivals

def quit():
    global exitProgram
    exitProgram=True

def sift_matches(root_dir: str, image0: str, image1: str):
    '''SIFT extraction
    '''
    if root_dir is None:
        warnings.warn("root_dir is None, assuming image0 and image1 are full paths")
        t0 = time.time()
        image0 = cv2.imread(str(image0))
        image1 = cv2.imread(str(image1))
        t1 = time.time()
    else:
        # read
        t0 = time.time()
        image0 = cv2.imread(str(Path(root_dir, image0)))
        image1 = cv2.imread(str(Path(root_dir, image1)))
        t1 = time.time()

    if image0 is None or image1 is None:
        raise FileNotFoundError("Error: One of the images did not load.")

    # extract
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT_create()
    t2 = time.time()
    kp0, des0 = sift.detectAndCompute(image0, None)
    kp1, des1 = sift.detectAndCompute(image1, None)
    t3 = time.time()
    if not kp0 or not kp1:
        return 0, None
    
    # match, mutual test ratio 0.75
    bf = cv2.BFMatcher()
    t4 = time.time()
    matches = bf.knnMatch(des0,des1,k=2)
    t5 = time.time()
    # handling when there is less than 2 matches
    good_matches = []
    for match in matches:
        if len(match) == 2 and match[0].distance < 0.75 * match[1].distance:
            good_matches.append(match[0])
        elif len(match) == 1:
            good_matches.append(match[0])
    
    # fundamental matrix estimation
    point_map = np.array([
        [kp0[match.queryIdx].pt[0],
        kp0[match.queryIdx].pt[1],
        kp1[match.trainIdx].pt[0],
        kp1[match.trainIdx].pt[1]] for match in good_matches
    ], dtype=np.float32)
    
    if point_map.shape[0] < 8:
        return 0, None
    t6 = time.time()
    F, inliers_idx = cv2.findFundamentalMat(point_map[:, :2], point_map[:, 2:], cv2.FM_RANSAC, 3.0)
    t7 = time.time()
    if inliers_idx is None:
        return 0, None
    inliers = point_map[inliers_idx.ravel() == 1]
    # print(f'Total time: {t7 - t0}, read: {t1 - t0}, extract: {t3 - t2}, match: {t5 - t4}, fundamental: {t7 - t6}')
    return len(inliers), inliers

def sift_matches_mp(pairs: np.array):
    for pair in pairs:
        image0, image1, label, num_matches = pair
        num_matches, _ = sift_matches(None, image0, image1)
    return None

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

def txt2npy_mp(root_dir, txt_file, npy_file):
    # TODO: implement multiprocessing
    
    if not isinstance(txt_file, str):
        raise TypeError('txt file path must be a str')
    
    if Path(npy_file).is_dir():
        npy_file = Path(npy_file, "pairs_info.npy")

    f = open(txt_file, 'r')
    lines = f.readlines()
    
    start = time.time()
    
    pairs = []
    missing = []
    f = open(txt_file, 'r')
    for line in tqdm(f.readlines()):
        line_str = line.strip('\n').split(' ')
        image0, image1, label = line_str
        try:
            num_matches = check_pair(root_dir, image0, image1)
        except:
            missing.append([image0, image1])
            print("Error: ", image0, image1)
            continue
        line_numpy = np.array([str(image0), str(image1), int(label), num_matches], dtype=object)
        pairs.append(line_numpy)

    missing = np.array(missing)
    out = np.array(pairs)
    np.save(npy_file, out)
    if len(missing) > 0:
        print(missing.shape)
        np.save(npy_file + ".missing", missing)
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
    
    parser.add_argument('--loftr', action='store_true')
    parser.add_argument('--pairs', action='store_true')
    parser.add_argument('--txt', action='store_true')
    parser.add_argument('--viz', action='store_true')
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
        t0 = time.time()
        t_match = 0
        for retrieval in retrievals[:20]:
            query = queries_paths[i]
            query_name = str(Path(query.parent.name, query.name))
            db = database_paths[retrieval]
            db_name = str(Path(db.parent.name, db.name))
            t1 = time.time()
            num_matches, _ = sift_matches(None, query, db)
            t2 = time.time()
            if retrieval in positives_per_query[i]:
                ret = np.array([query_name, db_name, 1, num_matches], dtype=object)
                pos +=1
                t3 = time.time()
            else:
                ret = np.array([query_name, db_name, 0, num_matches], dtype=object)
                neg += 1
                t3 = time.time()
            # print(ret)
            pairs.append(ret)
            t_match += t2 - t1
        t4 = time.time()
        print(f'Total {i} time for one sample: {t4 - t0}, match time: {t_match}')
    out = np.array(pairs)
    np.save(cfg.pairs_from_retrieval.pairs_info, out)
    print('Pairs Generation Done!')
    print(f'Positive Pairs: {pos}, Negative Pairs: {neg}')


def pairs_from_retrieval_mp(cfg, num_cores=20):
    # TODO: implement muti-Process/Thread acceleration version
    dataset = get_dataset(cfg.data)
    # get positive per query
    retrieval_results = np.load(cfg.pairs_from_retrieval.retrieval_results, allow_pickle=True)
    positives_per_query = dataset.get_positives()
    queries_paths = dataset.get_queries_paths()
    database_paths = dataset.get_database_paths()
    pairs = []
    pos = 0
    neg = 0
    t0 = time.time()
    for i, retrievals in tqdm(enumerate(retrieval_results), total=len(retrieval_results)):
        for retrieval in retrievals[:20]:
            query = queries_paths[i]
            query_name = str(Path(query.parent.name, query.name))
            db = database_paths[retrieval]
            db_name = str(Path(db.parent.name, db.name))
            # num_matches, _ = sift_matches(None, query, db)
            if retrieval in positives_per_query[i]:
                ret = np.array([query_name, db_name, 1, 0], dtype=object)
                pos +=1
            else:
                ret = np.array([query_name, db_name, 0, 0], dtype=object)
                neg += 1
            # print(ret)
            pairs.append(ret)
    pairs = np.array(pairs)
    t1 = time.time()
    print(f'Positive Pairs: {pos}, Negative Pairs: {neg}')
    print(f'Pairs Search Done!')
    print(f'Total time for Search: {t1 - t0}')
    
    print(f'Using {num_cores} cores for SIFT matching acceleration.')
    # pairs = pairs[:100]
    pairs_list = [pairs[i::num_cores] for i in range(num_cores)]
    print('Matching images using multi-thread...')
    # Matching image using multi-thread
    
    def match_image_pairs(pairs:np.array, root_dir = cfg.pairs_from_retrieval.image_root):
        image_pairs = []
        for pair in tqdm(pairs):
            num_matches, _ = sift_matches(root_dir, pair[0], pair[1])
            line = np.array([pair[0], pair[1], pair[2], num_matches], dtype=object)
            # image0, image1 = load_image_pair(pair, root_dir)
            image_pairs.append(line)
        return image_pairs
    
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        image_pairs = list(executor.map(match_image_pairs, pairs_list))
    end = time.time()
    
    print(f'Matching image time: {end - start}')
    
    out = np.concatenate(image_pairs)
    np.save(cfg.pairs_from_retrieval.pairs_info, out)
    print('Pairs Generation Done!')
    print(f'Positive Pairs: {pos}, Negative Pairs: {neg}')


# def pairs_from_retrieval(cfg):
#     dataset = get_dataset(cfg.data)
#     txt_out = open(cfg.pairs_from_retrieval.txt_output_path, 'w')
#     # get positive per query
#     retrieval_results = np.load(cfg.pairs_from_retrieval.retrieval_results, allow_pickle=True)
#     positives_per_query = dataset.get_positives()
#     queries_paths = dataset.get_queries_paths()
#     database_paths = dataset.get_database_paths()
#     pos = 0
#     neg = 0
#     for i, retrievals in tqdm(enumerate(retrieval_results), total=len(retrieval_results)):
#         for retrieval in retrievals[:20]:
#             query = queries_paths[i]
#             query_name = str(Path(query.parent.name, query.name))
#             db = database_paths[retrieval]
#             db_name = str(Path(db.parent.name, db.name))
#             # num_matches, _ = sift_matches(None, query, db)
#             if retrieval in positives_per_query[i]:
#                 txt_out.write(f'{query_name} {db_name} 1\n')
#                 # ret = np.array([query_name, db_name, 1, num_matches], dtype=object)
#                 pos +=1
#             else:
#                 txt_out.write(f'{query_name} {db_name} 0\n')
#                 # ret = np.array([query_name, db_name, 0, num_matches], dtype=object)
#                 neg += 1
#     txt_out.close()
#     print('Pairs Generation Done!')
#     print(f'Positive Pairs: {pos}, Negative Pairs: {neg}')

if __name__ == "__main__":
    args, cfg = parser()
    # Step 1: Generate pairs meta info
    if args.pairs:
        if args.txt:
            conf = cfg.pairs_from_retrieval
            if Path(conf.pairs_info).is_file():
                raise Warning('npy file already exists, skip generation.')
            else:
                conf = cfg.pairs_from_retrieval
                txt2npy(conf.image_root, conf.txt_path, conf.pairs_info)
        else:
            # Process pairs from retreival results.
            # main_worker(cfg)
            # pairs_from_retrieval(cfg)
            pairs_from_retrieval_mp(cfg)

    # Step 2: Extract loftr matches
    if args.loftr:
        '''Ectract loftr matches from pairs metadata
        Args:
            --root_dir: str, root directory of the images
            --npy_path: str, path to the pairs metadata
            --output: str, output directory to save the loftr matches
            --weights: str, path to the loftr weights
        '''
        conf = cfg.pairs_from_retrieval
        save_loftr_matches(conf.image_root, conf.pairs_info, conf.loftr_output, conf.loftr_weights)
    
    # Step 3: Visualize retreival results
    if args.viz:
        # exitProgram = False
        # keyboard.add_hotkey('q', lambda: quit())
        # while not exitProgram:
            
        ### visualize pairs_from_retrieval
        # retrievals = np.load(cfg.pairs_from_retrieval.retrieval_results, allow_pickle=True)
        # dataset = get_dataset(cfg.data)
        # # get positive per query
        # positives_per_query = dataset.get_positives()
        # queries_paths = dataset.get_queries_paths()
        # database_paths = dataset.get_database_paths()
        
        # for idx, retrieval in tqdm(enumerate(retrievals), total=len(retrievals)):
        #     query = queries_paths[idx]
        #     topk = database_paths[retrieval[100]]
        #     query_image = read_image(query)
        #     topk_image = read_image(topk)
        #     plot_images([query_image, topk_image], titles=['Query', 'Retrieval'], figsize=4.5)
        #     plt.show()
        #     # topk_images = [read_image(database_paths[db]) for db in topk]
        #     # positives = np.zeros(len(topk))
        #     # positives = [1 if db in positives_per_query[idx] else 0 for db in topk]
        #     # positives = [1 for db in topk if db in positives_per_query[idx]]
        #     # plot_retreivals([query_image], topk_images, positives, figsize=(15, 10), dpi=100, pad=.5)
        #     # plt.show()
            
        ### visualize pairs_info
            pairs_info = cfg.pairs_from_retrieval.pairs_info
            pairs_info = np.load(pairs_info, allow_pickle=True)
            image_root = cfg.pairs_from_retrieval.image_root
            for pair in pairs_info[0::18]:
                image0, image1, label, num_matches = pair
                images = [read_image(Path(image_root, image0)), read_image(Path(image_root, image1))]
                plot_images(images, titles=[f'Image {i}' for i in range(2)], figsize=4.5)
                print(f'{image0} {image1} {label} {num_matches}')
                plt.show()
        # sys.exit()