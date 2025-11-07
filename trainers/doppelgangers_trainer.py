import os
import tqdm
import torch
import torch.distributed as dist
import importlib
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed, FocalLoss, plot_pr_curve, compute_ap
from scipy.special import softmax
import swanlab

from jw_utils.tools import reduce_tensor, all_gather_concat


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        # set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        decoder_lib = importlib.import_module(cfg.models.decoder.type)
        self.decoder = decoder_lib.decoder(cfg.models.decoder)
        self.decoder = self.decoder.cuda()

        print("decoder:")
        print(self.decoder)

        # The optimizer
        if not hasattr(self.cfg.trainer, "opt_dec"):
            self.cfg.trainer.opt_dec = self.cfg.trainer.opt

        self.opt_dec, self.scheduler_dec = get_opt(
            self.decoder.parameters(), self.cfg.trainer.opt_dec)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)

        self.loss = FocalLoss()

    # def multi_gpu_wrapper(self, wrapper):
    #     self.decoder = wrapper(self.decoder)


    def epoch_end(self, epoch, logger=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch=epoch)
            if logger is not None:
                logger.log({"train/opt_dec_lr": self.scheduler_dec.get_lr()[0]}, step =epoch)
                # writer.add_scalar(
                #     'train/opt_dec_lr', self.scheduler_dec.get_lr()[0], epoch)


    def update(self, data, *args, **kwargs):
        if 'no_update' in kwargs:
            no_update = kwargs['no_update']
        else:
            no_update = False
        if not no_update:
            self.decoder.train()
            self.opt_dec.zero_grad()

        data['image'] = data['image'].cuda()
        gt = data['gt'].cuda()
        score = self.decoder(data['image'])

        loss = self.loss(score, gt)
        if not no_update:
            loss.backward()
            self.opt_dec.step()

        return {
            'loss': loss.detach().cpu().item()
        }

    def log_train(self, train_info, data, logger=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if logger is None:
            return
        
        if train_info is None:
            return

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        
        # if train_log_info is not None:
        #     for k, v in train_log_info.items():
        #         if 'pr_curve' in k:
        #                 # v[0]: gt, v[1]: prob
        #                 plot_pr_curve(v[0], v[1], writer, step=step, epoch=epoch, name=k)
        #         else:
        #             if step is not None:
        #                 writer.add_scalar(k+'_step', v, step)
        #             else:
        #                 writer.add_scalar(k+'_epoch', v, epoch)
        
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            if step is not None:
                logger.log({"train/" + k: v}, step=step)
                # writer.add_scalar('train/' + k, v, step)
            else:
                assert epoch is not None
                logger.log({"train/" + k: v}, step=epoch)
                # writer.add_scalar('train/' + k, v, epoch)

        if visualize:
            with torch.no_grad():
                return

    def validate(self, test_loader, epoch, stage='test/', multi_gpu=False, *args, **kwargs):
        self.decoder.eval()
        device = next(self.decoder.parameters()).device
        
        total_samples = len(test_loader.dataset)
        
        # pre-allocate tensors:
        gt_tensor = torch.zeros(total_samples, dtype=torch.long, device=device)
        pred_tensor = torch.zeros(total_samples, dtype=torch.long, device=device)
        prob_tensor = torch.zeros((total_samples, 2), dtype=torch.float, device=device)
        
        # keep position
        current_idx = 0
        
        with torch.no_grad():
            for bidx, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                data['image'] = data['image'].cuda()
                gt = data['gt'].cuda()
                score = self.decoder(data['image'])
                pred = torch.argmax(score,dim=1)
                batch_size = gt.size(0)
                
                if dist.get_rank() == 0 :
                    print(f"batch_size:{batch_size}")
                    
                gt_tensor[current_idx:current_idx+batch_size] = gt
                pred_tensor[current_idx:current_idx+batch_size] = pred
                prob_tensor[current_idx:current_idx+batch_size] = score
                
                current_idx += batch_size
                # for i in range(score.shape[0]):
                #     prob_list.append(score[i].cpu().numpy())
                #     pred_list.append(torch.argmax(score,dim=1)[i].cpu().numpy())
                #     gt_list.append(gt[i].cpu().numpy())
        
        if current_idx < total_samples:
            gt_tensor = gt_tensor[:current_idx]
            pred_tensor = pred_tensor[:current_idx]
            prob_tensor = prob_tensor[:current_idx]
        
        if multi_gpu:
            gt_full = all_gather_concat(gt_tensor)
            pred_full = all_gather_concat(pred_tensor)
            prob_full = all_gather_concat(prob_tensor)
            
            if dist.get_rank() == 0:
                gt_list = gt_full.cpu().numpy().reshape(-1)
                pred_list = pred_full.cpu().numpy().reshape(-1)
                prob_list = prob_full.cpu().numpy().reshape(-1, 2)
                ap = compute_ap(gt_list, prob_list)
                print(stage + "average precision: ", ap)
                np.save(os.path.join(self.cfg.save_dir, stage[:-1]+"_doppelgangers_list.npy"), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
                np.save(os.path.join(self.cfg.save_dir, 'checkpoints', stage[:-1]+"_doppelgangers_list_%d.npy"%epoch), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
                all_res ={stage+'AP': ap, stage+'pr_curve': [gt_list, prob_list]}
                return all_res
            else:
                return None
            
        else:
            gt_list = gt_tensor.cpu().numpy().reshape(-1)
            pred_list = pred_tensor.cpu().numpy().reshape(-1)
            prob_list = prob_tensor.cpu().numpy().reshape(-1, 2)
            ap = compute_ap(gt_list, prob_list)
            print(stage + "average precision: ", ap)
            np.save(os.path.join(self.cfg.save_dir, stage[:-1]+"_doppelgangers_list.npy"), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
            np.save(os.path.join(self.cfg.save_dir, 'checkpoints', stage[:-1]+"_doppelgangers_list_%d.npy"%epoch), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
            all_res ={stage+'AP': ap, stage+'pr_curve': [gt_list, prob_list]}
            return all_res
                
        
            # print("Test on validation set:")
            # for bidx, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            #     data['image'] = data['image'].cuda()
            #     gt = data['gt'].cuda()
            #     score = self.decoder(data['image'])
            #     for i in range(score.shape[0]):
            #         prob_list.append(score[i].cpu().numpy())
            #         pred_list.append(torch.argmax(score,dim=1)[i].cpu().numpy())
            #         gt_list.append(gt[i].cpu().numpy())
        
        # gt_list = np.array(gt_list).reshape(-1)
        # pred_list = np.array(pred_list).reshape(-1)
        # prob_list = np.array(prob_list).reshape(-1, 2)
        # ap = compute_ap(gt_list, prob_list)

        # print(stage + "average precision: ", ap)
        
        # if not multi_gpu or (multi_gpu and dist.get_rank() == 0):
        #     np.save(os.path.join(self.cfg.save_dir, stage[:-1]+"_doppelgangers_list.npy"), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
        #     np.save(os.path.join(self.cfg.save_dir, 'checkpoints', stage[:-1]+"_doppelgangers_list_%d.npy"%epoch), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
        #     all_res ={stage+'AP': ap, stage+'pr_curve': [gt_list, prob_list]}
            
        # return all_res
    
    def validate_train(self, train_loader, epoch, stage='train/', *args, **kwargs):
        self.decoder.eval()
        gt_list = list()
        pred_list = list()
        prob_list = list()
        with torch.no_grad():
            print("Test on Training set:")
            for bidx, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                data['image'] = data['image'].cuda()
                gt = data['gt'].cuda()
                score = self.decoder(data['image'])
                for i in range(score.shape[0]):
                    prob_list.append(score[i].cpu().numpy())
                    pred_list.append(torch.argmax(score,dim=1)[i].cpu().numpy())
                    gt_list.append(gt[i].cpu().numpy())
        gt_list = np.array(gt_list).reshape(-1)
        pred_list = np.array(pred_list).reshape(-1)
        prob_list = np.array(prob_list).reshape(-1, 2)
        ap = compute_ap(gt_list, prob_list)
        
        print(stage + "average precision: ", ap)
        
        # np.save(os.path.join(self.cfg.save_dir, stage[:-1]+"_doppelgangers_list.npy"), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
        # np.save(os.path.join(self.cfg.save_dir, 'checkpoints', stage[:-1]+"_doppelgangers_list_%d.npy"%epoch), {'pred': pred_list, 'gt': gt_list, 'prob': prob_list})
 
        all_res ={stage+'AP': ap, stage+'pr_curve': [gt_list, prob_list]}
        return all_res
        

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_dec': self.opt_dec.state_dict(),
            'dec': self.decoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)
    
    def save_multigpu(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt_dec': self.opt_dec.state_dict(),
            'dec': self.decoder.module.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, multi_gpu=False, **kwargs):
        ckpt = torch.load(path)
        import copy
        new_ckpt = copy.deepcopy(ckpt['dec'])
        if not multi_gpu:
            for key, value in ckpt['dec'].items():
                if 'module.' in key:
                    new_ckpt[key[len('module.'):]] = new_ckpt.pop(key)
        elif multi_gpu:
            for key, value in ckpt['dec'].items():                
                if 'module.' not in key:
                    new_ckpt['module.'+key] = new_ckpt.pop(key)
        self.decoder.load_state_dict(new_ckpt, strict=strict)
        self.opt_dec.load_state_dict(ckpt['opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch
    

    def log_val(self, val_info, logger=None, step=None, epoch=None, **kwargs):
        if logger is not None:
            for k, v in val_info.items():
                if 'pr_curve' in k:
                    y_true = v[0]
                    y_scores = softmax(v[1], axis=1)
                    y_scores = y_scores[:, 1]
                    logger.log({"pr_curve": swanlab.pr_curve(y_true, y_scores, title=True)}, step=step if step is not None else epoch)
                    # v[0]: gt, v[1]: prob
                    # plot_pr_curve(v[0], v[1], writer, step=step, epoch=epoch, name=k)
                else:
                    if step is not None:
                        logger.log({k+'_step': v}, step = step)
                    else:
                        logger.log({k+'_epoch': v}, step = epoch)
                        
            # if train_info is not None:
            #     for k, v in train_info.items():
            #         if step is not None:
            #             writer.add_scalar('train/'+k+'_step', v, step)
            #         else:
            #             writer.add_scalar('train/'+k+'_epoch', v, epoch)
