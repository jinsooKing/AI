import os
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchmetrics import Accuracy
from timm.data.mixup import Mixup

from lib.datasets import CIFAR10_MEAN, CIFAR10_STD
from lib.convnet_v2 import ConvNet
from lib.engines import train_one_epoch, eval_one_epoch
from lib.utils import save_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='convnet_v2', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str)

    # data
    parser.add_argument('--data', default='data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)

    # model
    parser.add_argument('--blocks', nargs='+', default=[3, 3, 9, 3], type=int)
    parser.add_argument('--dims', nargs='+', default=[64, 128, 256, 512], type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--droppath', default=0.1, type=float)

    # train
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--mixup_alpha', default=0.8, type=float)
    parser.add_argument('--cutmix_alpha', default=1.0, type=float)    
    parser.add_argument('--weight_decays', default=0.05, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--random_erasing', default=0.25, type=float)
    parser.add_argument('--random_augment', nargs='+', default=(2, 9), type=int)
    
    args = parser.parse_args()
    return args


def main(args):
    # -------------------------------------------------------------------------
    # Set Logger & Checkpoint Dirs
    # -------------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{args.title}.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )
    
    # -------------------------------------------------------------------------
    # Data Processing Pipeline
    # -------------------------------------------------------------------------
    train_transform = T.Compose([
        T.RandomCrop((32, 32), padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(num_ops=args.random_augment[0], magnitude=args.random_augment[1]),
        T.ToTensor(),
        T.RandomErasing(p=args.random_erasing),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_data = CIFAR10(args.data, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_data = CIFAR10(args.data, train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, 
        prob=1.0, switch_prob=0.5, label_smoothing=args.label_smoothing, 
        num_classes=args.num_classes)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = ConvNet(args.blocks, args.dims, args.droppath, args.dropout, args.num_classes)
    model = model.to(args.device)


    # -------------------------------------------------------------------------
    # Performance Metic, Loss Function, Optimizer
    # -------------------------------------------------------------------------
    metric_fn = Accuracy(task='multiclass', num_classes=args.num_classes)
    metric_fn = metric_fn.to(args.device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decays)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    # -------------------------------------------------------------------------
    # Run Main Loop
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        # train one epoch
        train_summary = train_one_epoch(
            model, train_loader, metric_fn, loss_fn, mixup_fn, args.device,
            optimizer, scheduler)

        # evaluate one epoch
        val_summary = eval_one_epoch(
            model, val_loader, metric_fn, loss_fn, args.device
        )
        log = (f'epoch {epoch+1}, '
               + f'train_loss: {train_summary["loss"]:.4f}, '
               + f'val_loss: {val_summary["loss"]:.4f}, '
               + f'val_accuracy: {val_summary["accuracy"]:.4f}')
        print(log)
        logging.info(log)

        # save model
        checkpoint_path = f'{args.checkpoint_dir}/{args.title}_last.pt'
        save_model(checkpoint_path, model, optimizer, scheduler, epoch+1)


if __name__=="__main__":
    args = get_args()
    main(args)