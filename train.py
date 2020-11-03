import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc

import horovod.torch as hvd


def main(args):
    makedirs(args)
    snapshotargs(args)
    print("start training!")

    # horovod
    hvd.init()
    verbose = True if hvd.rank() == 0 else False
    torch.cuda.set_device(hvd.local_rank())

    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid, sampler_train, sampler_valid = horovod_data_loaders(args)
    print("rank : {}, data loading finished".format(hvd.local_rank()))
    # loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    # unet.to(device)
    # default to running on device with gpu
    unet.cuda()
    print("rank : {}, model initialize finished".format(hvd.local_rank()))

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr * hvd.size())
    # horovod optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, 
                                         named_parameters=unet.named_parameters(),
                                         op=hvd.Average,
                                         backward_passes_per_step=1)

    # Broadcast parameters from rank 0 at weight initialization
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(unet.state_dict(), root_rank=0)

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
                x, y_true = x.cuda(), y_true.cuda() # default to GPU.

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
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
                        if logger:
                            loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0 and logger:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []
                if phase == "train" and (step + 1) % args.loss_print_frequency == 0:
                    print("rank {}, train loss type {}, value {}".format(hvd.local_rank(), type(loss), loss.item()))
                    print("data size is ", x.size())

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
                print("check validation dsc per volume")
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                print("rank {}, type {}, val mean dsc value is {}".format(hvd.local_rank(), type(mean_dsc), mean_dsc))
                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

def horovod_data_loaders(args):
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
        batch_size=args.batch_size,
        # shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        sampler=train_sampler
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
    print("total example in val set is {}, total num of slices is {}".format(len(validation_pred), sum(num_slices)))
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        print("y_pred statistics: min {}, max {}, avg {}".format(np.min(y_pred), np.max(y_pred), np.mean(y_pred)))
        print("y_true statistics: min {}, max {}, avg {}".format(np.min(y_true), np.max(y_true), np.mean(y_true)))
        dsc_list.append(dsc(y_pred, y_true))
        print("latest dsc result is ", dsc_list[-1])
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
        default=100,
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
        default=1,
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
    args = parser.parse_args()
    main(args)
