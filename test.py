import os
import yaml
import time
import torch
import argparse
import importlib
from torch.backends import cudnn
import swanlab
from tools.eval_helper import plot_pr_curve
from scipy.special import softmax
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    
    # dryrun
    parser.add_argument('--dryrun', action='store_true',
                        help='If true, do a dry run to test the dataloaders.')

    # distributed training
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='Number of distributed nodes.')
    # parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist_backend', default='nccl', type=str,
    #                     help='distributed backend')
    # parser.add_argument('--distributed', action='store_true',
    #                     help='Use multi-processing distributed training to '
    #                          'launch N processes per node, which has N GPUs. '
    #                          'This is the fastest way to use PyTorch for '
    #                          'either single node or multi node data parallel '
    #                          'training')
    # parser.add_argument('--rank', default=0, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--gpu', default=None, type=int,
    #                     help='GPU id to use. None means using all '
    #                          'available GPUs.')

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

    # #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # Currently save dir and log_dir are the same
    # config.log_name = "val_logs/%s_val_%s" % (cfg_file_name, run_time)
    config.save_dir = "val_logs/%s_val_%s" % (cfg_file_name, run_time)
    # config.log_dir = "val_logs/%s_val_%s" % (cfg_file_name, run_time)
    # os.makedirs(config.log_dir + '/config')
    # copy2(args.config, config.log_dir + '/config')
    return args, config


def main_worker(cfg, args):
    # init swanlab logger
    logger = swanlab.init(
            project="doppelgangers",
            experiment_name=args.config.split('/')[-1],
            config={
                "config_file": args.config,
                "batch_size": cfg.data.test.batch_size,
                "pretrained": args.pretrained,
            }
        )
    
    # basic setup
    cudnn.benchmark = True
    # writer = SummaryWriter(log_dir=cfg.log_name)

    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, skip=-1)
    test_loader = loaders['test_loader']

    if args.dryrun:
        print("Dry run to test the dataloaders...")
        for i, sample in enumerate(test_loader):
            print("Sample %d:" % i)
            for key in sample:
                if isinstance(sample[key], torch.Tensor):
                    print("  %s: shape=%s, dtype=%s" %
                          (key, str(sample[key].shape), str(sample[key].dtype)))                        
                else:
                    print("  %s: %s" % (key, str(sample[key])))
        print("Dry run done.")
        return

    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)
    trainer.resume(args.pretrained, test=True)
    
    val_info = trainer.validate(test_loader, epoch=-1)
    trainer.log_val(val_info, logger=logger, step=-1)
    gt_list, prob_list = val_info['test/pr_curve']
    scores = softmax(prob_list, axis=1)[:, 1]
    precision, recall, TH = precision_recall_curve(gt_list, scores)
    average_precision = average_precision_score(gt_list, scores)
    AP, MR = plot_pr_curve(recall, precision, average_precision)
    print(f"Average Precision: {AP}, Max Recall@100 {MR}.")
    plt.savefig(Path(args.val_log,'pr_curve.pdf'))
    print("Test done")

def main():
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    ngpus_per_node = torch.cuda.device_count()
    # main_worker(args.gpu, ngpus_per_node, cfg, args)
    main_worker(cfg, args)



if __name__ == '__main__':
    main()