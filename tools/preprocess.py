import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2
import sys
sys.path.append('../doppelgangers')
from doppelgangers.utils.loftr_matches import save_loftr_matches

def sift_matches(root_dir: str, image0: str, image1: str):
    # read
    image0 = cv2.imread(str(Path(root_dir, image0)))
    image1 = cv2.imread(str(Path(root_dir, image1)))

    # extract
    sift = cv2.xfeatures2d.SIFT_create()
#     sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(image0, None)
    kp1, des1 = sift.detectAndCompute(image1, None)

    # match, mutual test ratio 0.75
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0,des1,k=2)
    good_matches = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    
    # homography estimation
    point_map = np.array([
        [kp0[match.queryIdx].pt[0],
         kp0[match.queryIdx].pt[1],
         kp1[match.trainIdx].pt[0],
         kp1[match.trainIdx].pt[1]] for match in good_matches
    ], dtype=np.float32)

    F, inliers_idx = cv2.findFundamentalMat(point_map[:, :2], point_map[:, 2:], cv2.FM_RANSAC, 3.0)
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
      
      line_str = line.strip('\n').split(', ')
      image0, image1, label = line_str
      num_matches, _ = sift_matches(root_dir, image0, image1)
      line_numpy = np.array([str(image0), str(image1), int(label), num_matches], dtype=object)
      # print(line_numpy.shape)
      pairs.append(line_numpy)
    
    out = np.array(pairs)
#     print(out.shape)
#     print(out)
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
    parser = argparse.ArgumentParser(description='txt2npy')
    parser.add_argument('--npy_path', default='data/train.npy', type=str, help='npy path')
    parser.add_argument('--txt_path', default='data/train.txt', type=str, help='txt path')
    parser.add_argument('--root_dir')
    parser.add_argument('--output')
    parser.add_argument('--weights')
    return parser.parse_args()


def npy2txt(npy: str, txt: str):
    pairs = np.load(npy, allow_pickle=True)
    with open(txt, 'w') as f:
        for pair in tqdm(pairs, total=len(pairs)):
            image0, image1, label, num_matches = pair
            f.write(f'{image0} {image1} {label}\n')
    print('Done!')
    

if __name__ == "__main__":
      # args = parser()
    # all_pairs = 'data/robotcar_seasons/pairs_metadata/robotcar_qAutumn_dbNight.npy'
    # test_pairs = 'data/robotcar_seasons/pairs_metadata/robotcar_qAutumn_dbNight_test.npy'
    # #   split_data(all_pairs, split_pairs, [10001,-1]
    # train_pairs = 'data/robotcar_seasons/pairs_metadata/robotcar_qAutumn_dbNight_train.npy'

    # import sklearn.model_selection as model_selection
    # # model_selection.train_test_split()
    # all_list = np.load(all_pairs, allow_pickle=True)
    # train, test = model_selection.train_test_split(all_list, test_size=0.2, random_state=42)
    # print(f'####### DATA Length ####### \n train: {train.shape}, test: {test.shape}')
    # np.save(train_pairs, train)
    # np.save(test_pairs, test)

    npy = 'data/robotcar_seasons/pairs_metadata/robotcar_qAutumn_dbSunCloud_test.npy'
    txt = 'data/robotcar_seasons/pairs_metadata/robotcar_qAutumn_dbSunCloud_test.txt'
    npy2txt(npy, txt)
    #   root_dir = "data/robotcar_seasons/images"
    #   txt_path = "data/robotcar_seasons/pairs_metadata/robotcar_qAutumn_dbSunCloud_val_mini.txt"
    #   npy_path = "data/robotcar_seasons/pairs_metadata/robotcar_qAutumn_dbSunCloud.npy"
    #   # check_txt(root_dir, txt_path, npy_path)
    #   # txt2npy(root_dir, txt_path, npy_path)
    #   # txt2npy(args.root_dir, args.txt_path, args.npy_path)
    #   weights = 'weights/outdoor_ds.ckpt'
    #   # save_loftr_matches(args.root_dir, args.npy_path, args.output, args.weights)
    #   output = 'data/robotcar_seasons/loftr_matches/qAutumn_dbSunCloud'
    #   save_loftr_matches(root_dir, txt_path, output, weights)
      # load_data(args.npy_path)