import os.path as osp
import numpy as np
import torch
# import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import cv2


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


class PairWiseDataset(Dataset):
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
        # import pdb; pdb.set_trace()

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
        
        num_matches = keypoints0.shape[0]
        # if np.sum(conf>0.8) == 0:
        #     matches = None
        # else:
        #     F, mask = cv2.findFundamentalMat(keypoints0,keypoints1,cv2.FM_RANSAC, 3, 0.99)
        #     if mask is None or F is None:
        #         matches = None
        #         # print(f'{name0}: {name1} has no matches')
        #         # print(f'label is {label}')
        #         if label == 1:
        #             print(f'{name0}: {name1} has no matches')
        #             print('false negative')
        #             image0 = cv2.imread(osp.join(self.image_dir, name0))
        #             image1 = cv2.imread(osp.join(self.image_dir, name1))
        #             plot_images([image0, image1])
        #             plot_matches(keypoints0, keypoints1)
        #             plt.show()

        #     else:
                # import pdb; pdb.set_trace()
                # matches = len(keypoints0[mask.ravel()==1])
                # print(matches)
                # print(mask.ravel()==1)
                # num_matches = np.sum(np.array(mask.ravel()==1))
                # matches = np.array(np.ones((keypoints0.shape[0], 2)) * np.arange(keypoints0.shape[0]).reshape(-1,1)).astype(int)[:][mask.ravel()==1]
        # matches = matches.tolist()
        # if matches is not None:
        #     print('matches: ', matches)
        #     print('num_matches: ', len(matches))

        data = {
            'name0': name0,
            'name1': name1,
            'gt': int(label),
            'matches': num_matches
        }


        return data
        

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_dataset(cfg):
    pairwise_dataset = PairWiseDataset(
                cfg.image_dir,
                cfg.loftr_match_dir,
                cfg.pair_path,
                img_size=getattr(cfg, "img_size", 640),
                phase='Train')

    return pairwise_dataset


def get_data_loaders(cfg):
    pairwise_dataset = get_dataset(cfg)
    data_loader = torch.utils.data.DataLoader(
        dataset=pairwise_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)
    return data_loader


if __name__ == "__main__":
    pass