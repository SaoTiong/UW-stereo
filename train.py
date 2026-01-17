from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim


import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from core.raft_stereo import RAFTStereo

from evaluate_stereo import *
import core.stereo_datasets as datasets

try:
    from torch.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = os.environ.get("DDP_BACKEND", "nccl")
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, is_main=True):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.is_main = is_main
        self.writer = SummaryWriter(log_dir='runs') if is_main else None

    def _print_training_status(self):
        if not self.is_main:
            return
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def train(args):

    distributed, local_rank, rank, world_size = ddp_setup()
    device = torch.device("cuda", local_rank)

    model = RAFTStereo(args).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    if distributed:
        dataset = train_loader.dataset
        per_gpu_batch = max(1, args.batch_size // world_size)
        if rank == 0 and args.batch_size % world_size != 0:
            logging.warning(
                "batch_size %d not divisible by world_size %d, using per-gpu batch %d",
                args.batch_size,
                world_size,
                per_gpu_batch,
            )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_gpu_batch,
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last,
        )

    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, is_main=(rank == 0))

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        if rank == 0:
            logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        if rank == 0:
            logging.info(f"Done loading checkpoint")

    model.train()
    base_model = model.module if distributed else model
    base_model.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000

    scaler = GradScaler('cuda', enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    epoch = 0
    while should_keep_training:
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader, disable=(rank != 0))):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(device, non_blocking=True) for x in data_blob]

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            assert model.training

            loss, metrics = sequence_loss(flow_predictions, flow, valid)
            if logger.writer is not None:
                logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
                logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                if rank == 0:
                    save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save(model.state_dict(), save_path)

                results = validate_uwstereo(
                    base_model,
                    iters=args.valid_iters,
                    max_samples=args.uwstereo_val_max_samples,
                    seed=args.uwstereo_val_seed,
                )

                logger.write_dict(results)

                model.train()
                base_model.freeze_bn()
                if distributed:
                    dist.barrier()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            if rank == 0:
                save_path = Path('checkpoints/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path}")
                torch.save(model.state_dict(), save_path)
        epoch += 1

    if rank == 0:
        print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    if rank == 0:
        torch.save(model.state_dict(), PATH)

    if distributed:
        dist.destroy_process_group()

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
   


    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--uwstereo_val_max_samples', type=int, default=0, help='max UW Stereo samples to evaluate (0 = all)')
    parser.add_argument('--uwstereo_val_seed', type=int, default=0, help='random seed for UW Stereo sampling')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)
