import os
import yaml
import time
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
from shutil import copy2
from pprint import pprint
import re
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs. '
                             'This is the fastest way to use PyTorch for '
                             'either single node or multi node data parallel '
                             'training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")
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


def max_recall(precision: np.ndarray, recall: np.ndarray):
    recall_idx = np.argmax(recall)
    idx = np.where(precision == 1.0)
    max_recall = np.max(recall[idx])
#     logger.debug(f'max recall{max_recall}')
    recall_idx = np.array(np.where(recall == max_recall)).reshape(-1)
    recall_idx = recall_idx[precision[recall_idx]==1.0]
#     logger.debug(f'recall_idx {recall_idx}')
    r_precision, r_recall = float(np.squeeze(precision[recall_idx])), float(np.squeeze(recall[recall_idx]))
#     logger.debug(f'precision: {r_precision}, recall: {r_recall}')
    return r_precision, r_recall


def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, average_precision, dataset = 'robotcar', exp_name = 'test'):
    # calculate max f2 score, and max recall
    f_score = 2 * (precision * recall) / (precision + recall)
    f_idx = np.argmax(f_score)
    f_precision, f_recall = np.squeeze(precision[f_idx]), np.squeeze(recall[f_idx])
    r_precision, r_recall = max_recall(precision, recall)
    plt.plot(recall, precision, label="{} (AP={:.3f})".format(dataset, average_precision))
    plt.vlines(r_recall, np.min(precision), r_precision, colors='green', linestyles='dashed')
    plt.vlines(f_recall, np.min(precision), f_precision, colors='red', linestyles='dashed')
    plt.hlines(f_precision, np.min(recall), f_recall, colors='red', linestyles='dashed')
    plt.hlines(r_precision, np.min(recall), r_recall, colors='green', linestyles='dashed')
    plt.xlim(0, None)
    plt.ylim(np.min(precision), None)
    plt.text(f_recall + 0.005, f_precision + 0.005, '[{:.3f}, {:.3f}]'.format(f_recall, f_precision))
    plt.text(r_recall + 0.005, r_precision + 0.005, '[{:.3f}, {:.3f}]'.format(r_recall, r_precision))
    plt.scatter(f_recall, f_precision, marker='o', color='red', label='Max F score')
    plt.scatter(r_recall, r_precision, marker='o', color='green', label='Max Recall')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curves for {}".format(exp_name))


def main_worker(gpu, ngpus_per_node, cfg, args):
    # basic setup
    cudnn.benchmark = True
    multi_gpu = False
    strict = True

    # initial dataset
    data_lib = importlib.import_module(cfg.data.type)
    loader = data_lib.get_data_loaders(cfg.data)
    

    # initial model
#     decoder_lib = importlib.import_module(cfg.models.decoder.type)
#     decoder = decoder_lib.decoder(cfg.models.decoder)
#     decoder = decoder.cuda()

    # load pretrained model
#     ckpt = torch.load(args.pretrained)
#     import copy
#     new_ckpt = copy.deepcopy(ckpt['dec'])
#     if not multi_gpu:
#         for key, value in ckpt['dec'].items():
#             if 'module.' in key:
#                 new_ckpt[key[len('module.'):]] = new_ckpt.pop(key)
#     elif multi_gpu:
#         for key, value in ckpt['dec'].items():                
#             if 'module.' not in key:
#                 new_ckpt['module.'+key] = new_ckpt.pop(key)
#     decoder.load_state_dict(new_ckpt, strict=strict)

    # evaluate on test set
#     decoder.eval()
#     acc = 0
#     sum = 0
    gt_list = list()
#     pred_list = list()
    prob_list = list()

    for bidx, data in tqdm.tqdm(enumerate(loader)):
      gt = data['gt']
      # import pdb; pdb.set_trace()
      # matches = data['matches']
      score = data['matches']
      print(score)
      # import pdb; pdb.set_trace()

      # score = data['matches'] / np.max(data['matches'])

      # data['image'] = data['image'].cuda()
      
      # score = decoder(data['image'])
      # pred = torch.argmax(score,dim=1).cuda()
      # acc += torch.sum(pred==gt).item()
      # sum += score.shape[0]                
      # for i in range(score.shape[0]):
      prob_list.append(score.cpu().numpy()[0])
            # pred_list.append(torch.argmax(score,dim=1)[i].cpu().numpy())
      gt_list.append(gt.cpu().numpy()[0])
    
    gt_list = np.array(gt_list).reshape(-1)
#     pred_list = np.array(pred_list).reshape(-1)
    prob_list = np.array(prob_list).reshape(-1)
    prob_list = prob_list / np.max(prob_list)

    if not os.path.exists(cfg.exp.save_dir):
            # if not os.path.exists(os.path.dirname(cfg.exp.save_dir)):
            #     os.makedirs(os.path.dirname(cfg.exp.save_dir))
            os.makedirs(cfg.exp.save_dir)


    np.save(os.path.join(cfg.exp.save_dir, "pair_probability_list.npy"), {'gt': gt_list, 'prob': prob_list})
    
    average_precision = average_precision_score(gt_list, prob_list)
    precision, recall, TH = precision_recall_curve(gt_list, prob_list)
    print(prob_list)
    print('Average precision: {0:0.2f}'.format(
          average_precision))
#     print('Max Recall: {0:0.2f}'.format(
#           np.max(recall)))
    plot_pr_curve(recall, precision, average_precision, cfg.data.name, cfg.exp.name)
    
      #   plt.savefig(os.path.join(cfg.exp.save_dir, "pr_curve.png"))
    # fig = plt.gcf()
    # fig.savefig(os.path.join(cfg.exp.save_dir, "pr_curve.png"))
#     plt.save(os.path.join(cfg.exp.save_dir, "pr_curve.png"))
    print("Test done.")


def main():
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, cfg, args)


if __name__ == '__main__':
    main()