import os
import yaml
import time
import torch
import argparse
import importlib
from torch.backends import cudnn
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
import swanlab

# utils
from jw_utils.tools import set_seed, init_np_seed, init_distributed_mode, reduce_tensor


# def reduce_tensor(tensor):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= dist.get_world_size()
#     return rt


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Doppelgangers: Learning to Disambiguate Images of Similar Structures')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Total number of batches (None will read batch size from the [cfg]).')
    # parser.add_argument('--dist_url', default='env://', type=str,
    #                     help='url used to set up distributed training')


    # overfitting for debug:
    parser.add_argument('--overfit', default=False, action='store_true',
                        help='Overfit on a single batch for debugging purposes.')
    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained checkpoint")

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')
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
    
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    config.save_dir = f"logs/{cfg_file_name}_{run_time}"    
    return args, config


# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print


# def init_distributed_mode(args):
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"], 0)
#         args.world_size = int(os.environ['WORLD_SIZE'], 0)
#         args.gpu = int(os.environ['LOCAL_RANK'], 0 )
#         args.local_rank = int(os.environ['LOCAL_RANK'], 0)
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True
#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
#     torch.distributed.init_process_group(
#         backend=args.dist_backend, init_method=args.dist_url,
#         world_size=args.world_size, rank=args.rank)
#     setup_for_distributed(args.rank == 0)


@record
def main():
    # Basic setup
    cudnn.benchmark = True
    args, cfg = get_args()
    seed = getattr(cfg.trainer, "seed", 666)
    
    # Initialize distributed environment
    init_distributed_mode(args)
    
    # get local rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set random seeds for reproducibility
    set_seed(seed + local_rank, deterministic=False)
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # init Swanlab for logging
    if dist.get_rank() == 0:
        logger = swanlab.init(
            project="doppelgangers",
            experiment_name=args.config.split('/')[-1],
            config={
                "config_file": args.config,
                "batch_size": args.batch_size,
                "resume": args.resume,
                "pretrained": args.pretrained,
                "test_run": args.test_run
            }
        )
    
    # Create trainer
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)
    
    # Setup model for DDP
    trainer.decoder = trainer.decoder.to(device)
    trainer.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.decoder)
    
    trainer.decoder = DDP(
        trainer.decoder, 
        device_ids=[args.local_rank],
    )
    
    # Initialize datasets and loaders
    data_lib = importlib.import_module(cfg.data.type)
    if args.overfit:
        if args.batch_size is None:
            args.batch_size = cfg.data.train.batch_size
        print("Overfitting on a single batch for debugging purposes.")
        tr_dataset, _ = data_lib.get_datasets(cfg.data)
        tr_dataset = torch.utils.data.Subset(tr_dataset, list(range(0, args.batch_size)))
        te_dataset = tr_dataset
    else:
        tr_dataset, te_dataset = data_lib.get_datasets(cfg.data)

    
    # Adjust batch size based on number of GPUs if specified
    if args.batch_size is not None:
        cfg.data.train.batch_size = args.batch_size // dist.get_world_size()
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        tr_dataset, 
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        te_dataset, 
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )
    
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, 
        batch_size=cfg.data.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers, 
        pin_memory=True,
        drop_last=True, 
        worker_init_fn=init_np_seed
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, 
        batch_size=cfg.data.test.batch_size, 
        sampler = test_sampler,
        shuffle=False,
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        drop_last=False,
        worker_init_fn=init_np_seed
    )
    
    # Resume training if specified
    start_epoch = 0
    if args.resume:
        if args.pretrained is not None:
            start_epoch = trainer.resume(args.pretrained) + 1
        else:
            start_epoch = trainer.resume(cfg.resume.dir)
    
    # Test run if specified
    if args.test_run:
        if dist.get_rank() == 0:
            trainer.save(epoch=-1, step=-1)
        
        val_info = trainer.validate(test_loader, epoch=-1, multi_gpu=True)
        
        if dist.get_rank() == 0:
            trainer.log_val(val_info, logger=logger, epoch=-1)
            trainer.log_val(val_info, logger=logger, step=-1)
    
    # Main training loop
    print(f"Start epoch: {start_epoch} End epoch: {cfg.trainer.epochs}")
    step = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, cfg.trainer.epochs):
        train_sampler.set_epoch(epoch)
        # TODO: Set seed for each epoch if needed
        # break
        # Train for one epoch
        for bidx, data in enumerate(train_loader):
            step = bidx + len(train_loader) * epoch + 1
            logs_info = trainer.update(data)
            
            if step % int(cfg.viz.log_freq) == 0 and dist.get_rank() == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(f"Epoch {epoch} Batch [{bidx}/{len(train_loader)}] "
                      f"Time [{duration:.2f}s] Loss {logs_info['loss']:.5f}")
                trainer.log_train(logs_info, data, logger=logger, epoch=epoch, step=step)
            
            if int(cfg.viz.val_freq) > 0 and step % int(cfg.viz.val_freq) == 0:
                val_info = trainer.validate(test_loader, epoch=epoch)
                if dist.get_rank() == 0:
                    trainer.log_val(val_info, logger=logger, step=step)
        
        # Save checkpoint
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and dist.get_rank() == 0:
            trainer.save(epoch=epoch, step=step)
        
        # Validate at the end of specified epochs
        if (epoch + 1) % int(cfg.viz.save_freq) == 0:
            val_info = trainer.validate(test_loader, epoch=epoch)
            if dist.get_rank() == 0:
                trainer.log_val(val_info, logger=logger, epoch=epoch)
        
        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, logger=logger if dist.get_rank() == 0 else None)
    
    if dist.get_rank() == 0:
        logger.finish()
    
    # Clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    main()