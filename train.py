import argparse
import json
import os
import math

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc

import horovod.torch as hvd
import kfac


def main(args):
    makedirs(args)
    snapshotargs(args)
    print("start training!")
    cuda_avail = torch.cuda.is_available()

    # horovod
    hvd.init()
    verbose = True if hvd.rank() == 0 else False
    torch.manual_seed(args.seed)
    if cuda_avail:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)
    
    # data loader config
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda_avail else {}

    loader_train, loader_valid, sampler_train, sampler_valid = horovod_data_loaders(args, kwargs)
    print("rank : {}, data loading finished".format(hvd.local_rank()))
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    if cuda_avail:
        unet.cuda()
    
    print("rank : {}, model initialize finished".format(hvd.local_rank()))

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    use_kfac = True if args.kfac_update_freq > 0 else False
    if use_kfac:
        # base lr in original distributed k-fac paper is 0.1
        optimizer = optim.SGD(unet.parameters(), lr=args.base_lr * hvd.size(), momentum=args.momentum,
                        weight_decay=args.weight_decay)
        preconditioner = kfac.KFAC(unet, lr=args.base_lr * hvd.size(), factor_decay=args.stat_decay, 
                               damping=args.damping, kl_clip=args.kl_clip, 
                               fac_update_freq=args.kfac_cov_update_freq, 
                               kfac_update_freq=args.kfac_update_freq,
                               diag_blocks=args.diag_blocks,
                               diag_warmup=args.diag_warmup,
                               distribute_layer_factors=args.distribute_layer_factors)
        kfac_param_scheduler = kfac.KFACParamScheduler(preconditioner,
                damping_alpha=args.damping_alpha,
                damping_schedule=args.damping_schedule,
                update_freq_alpha=args.kfac_update_freq_alpha,
                update_freq_schedule=args.kfac_update_freq_schedule)
    else:
        optimizer = optim.Adam(unet.parameters(), lr=args.lr * hvd.size())
    # horovod optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, 
                                         named_parameters=unet.named_parameters(),
                                         op=hvd.Average,
                                         backward_passes_per_step=args.batches_per_allreduce)

    # Broadcast parameters from rank 0 at weight initialization
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(unet.state_dict(), root_rank=0)
    if use_kfac:
        #without kfac, default to adam
        lrs = create_lr_schedule(hvd.size(), args.warmup_epochs, args.lr_decay)
        lr_scheduler = [LambdaLR(optimizer, lrs)]
        lr_scheduler.append(LambdaLR(preconditioner, lrs))

    # logging
    logger = Logger(args.logs) if verbose and hvd.local_rank() == 0 else None
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(args.epochs),
                      total=args.epochs):
        for phase in ["train", "valid"]:
            # skip evaluation if not process 0
            if phase == "valid" and hvd.local_rank() != 0:
                continue
            if phase == "train":
                unet.train()
                sampler_train.set_epoch(epoch)
            else:
                unet.eval()

            validation_pred = []
            validation_true = []
            train_pred = []
            train_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                if cuda_avail:
                    x, y_true = x.cuda(), y_true.cuda() # default to GPU.

                optimizer.zero_grad()

                # support batches_per_all_reduce
                with torch.set_grad_enabled(phase == "train"):
                    if phase == "valid":
                        y_pred = unet(x)
                        loss = dsc_loss(y_pred, y_true)
                    if phase == "valid" and logger:
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = "image/{}".format(i)
                                num_images = args.vis_images - i * args.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    # if phase == "train" and logger:
                    #     y_train_np = y_pred.detach().cpu().numpy()
                    #     train_pred.extend(
                    #         [y_train_np[s] for s in range(y_train_np.shape[0])]
                    #     )
                    #     y_true_np = y_true.detach().cpu().numpy()
                    #     train_true.extend(
                    #         [y_true_np[s] for s in range(y_true_np.shape[0])]
                    #     )

                    if phase == "train":
                        for idx in range(0, len(x), args.batch_size):
                            x_batch = x[idx:idx + args.batch_size]
                            y_true_batch = y_true[idx: idx + args.batch_size]
                            y_pred_batch = unet(x_batch)
                            loss = dsc_loss(y_pred_batch, y_true_batch)
                            loss.div_(math.ceil(float(len(data)) / args.batch_size))
                            loss.backward()
                        optimizer.synchronize()
                        if use_kfac:
                            preconditioner.step(epoch=epoch)
                        with optimizer.skip_synchronize():
                            optimizer.step()
                        if logger:
                            loss_train.append(loss.item())

                if phase == "train" and (step + 1) % 10 == 0 and logger:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []
                # if phase == "train" and (step + 1) % args.loss_print_frequency == 0:
                    # print("rank {}, train loss type {}, value {}".format(hvd.local_rank(), type(loss), loss.item()))
                    # print("data size is {}, batch size is {}".format(x.size(), args.batch_size))

            # if phase == "train" and logger:
            #     print("check training dsc per volume")
            #     mean_dsc = np.mean(
            #         dsc_per_volume(
            #             train_pred,
            #             train_true,
            #             loader_train.dataset.patient_slice_index,
            #         )
            #     )
            #     print("rank {}, type {}, train mean dsc value is {}".format(hvd.local_rank(), type(mean_dsc), mean_dsc))

            if phase == "valid" and logger:
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                #if (step + 1) % args.loss_print_frequency == 0:
                print("check validation dsc per volume")
                print("rank {}, type {}, val mean dsc value is {}".format(hvd.local_rank(), type(mean_dsc), mean_dsc))
                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []
            for scheduler in lr_scheduler:
                scheduler.step()
            if use_kfac:
                kfac_param_scheduler.step(epoch)

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

def horovod_data_loaders(args, kwargs):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    # init training set's dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=hvd.size(),
        rank=hvd.rank()
    )
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size * args.batches_per_allreduce,
        # shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init,
        sampler=train_sampler,
        **kwargs
    )

    # init validation set's dataloader

    # valid_sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset_valid,
    #     num_replicas=hvd.size(),
    #     rank=hvd.rank()
    # )
    # non distributed version of validation since we need the whole dataset to calculate related metric.
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    # valid_loader = DataLoader(
    #     dataset_valid,
    #     batch_size=args.batch_size,
    #     drop_last=False,
    #     num_workers=args.workers,
    #     worker_init_fn=worker_init,
    #     sampler=valid_sampler
    # )

    return train_loader, valid_loader, train_sampler, None

def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        # shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    # print("total example in val set is {}, total num of slices is {}".format(len(validation_pred), sum(num_slices)))
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        # print("y_pred statistics: min {}, max {}, avg {}".format(np.min(y_pred), np.max(y_pred), np.mean(y_pred)))
        # print("y_true statistics: min {}, max {}, avg {}".format(np.min(y_true), np.max(y_true), np.mean(y_true)))
        dsc_list.append(dsc(y_pred, y_true))
        # print("latest dsc result is ", dsc_list[-1])
        index += num_slices[p]
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)

def create_lr_schedule(workers, warmup_epochs, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        if epoch < warmup_epochs:
            lr_adj = 1. / workers * (epoch * (workers - 1) / warmup_epochs + 1)
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch >= e:
                    lr_adj *= alpha
        return lr_adj
    return lr_schedule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument(
        "--loss-print-frequency",
        type=int,
        default=10,
        help="how many step do we print the training loss",
    )
    parser.add_argument(
        '--warmup-epochs', 
        type=int, 
        default=10, 
        metavar='WE',
        help='number of warmup epochs (default: 5)'
    )
    parser.add_argument(
        '--batches-per-allreduce', 
        type=int, 
        default=1,
        help='number of batches processed locally before '
         'executing allreduce across workers; it multiplies '
         'total batch size.'
    )
    # SGD Parameters
    parser.add_argument('--base-lr', type=float, default=0.0001, metavar='LR',
                    help='base learning rate (default: 0.1)')
    # 0.0001 without lr decay appears to work better.
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[1000, 1500],
                    help='epoch intervals to decay lr')
    # used to be [100, 150]
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=500,
                        help='iters between kfac inv ops (0 for no kfac updates) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=50,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-schedule', nargs='+', type=int, default=None,
                        help='KFAC update freq schedule (default None)')
    parser.add_argument('--stat-decay', type=float, default=0.93,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    # 0.92 > 0.93 ~= 0.80 (but converges much faster) ~= 0.60  ~= 0.50 > 0.90 > 0.70 > 0.20
    parser.add_argument('--damping', type=float, default=0.003,
                        help='KFAC damping factor (defaultL 0.003)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-schedule', nargs='+', type=int, default=None,
                        help='KFAC damping decay schedule (default None)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--diag-blocks', type=int, default=1,
                        help='Number of blocks to approx layer factor with (default: 1)')
    parser.add_argument('--diag-warmup', type=int, default=5,
                        help='Epoch to start diag block approximation at (default: 5)')
    parser.add_argument('--distribute-layer-factors', action='store_true', default=False,
                        help='Compute A and G for a single layer on different workers')
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    
    args = parser.parse_args()
    main(args)
