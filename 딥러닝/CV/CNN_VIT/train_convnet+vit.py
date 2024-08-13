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
from lib.your_model import CONVIT
from lib.engines import train_one_epoch, eval_one_epoch
from lib.utils import save_model


class Args:
    title = 'your_model'
    device = 'cuda:0'
    checkpoint_dir = 'checkpoints'
    data = 'data'
    batch_size = 128
    blocks = [3, 3]
    dims = [64, 128]
    num_classes = 10
    dropout = 0.1
    droppath = 0.1
    epochs = 250
    lr = 1e-3
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    weight_decays = 0.05
    label_smoothing = 0.1
    random_erasing = 0.25
    random_augment = (2, 9)
    
    vit_num_heads = 4
    vit_blocks = 12
    vit_embed_dim = 384
    image_size = 15
    patch_size = 3
    vit_expansion_ratio = 2
    vit_droppath = 0.1
    vit_dropout = 0.1
    
    
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
    model = CONVIT(args.blocks, args.dims, args.droppath, 
                   args.vit_blocks, args.vit_embed_dim, args.num_classes, args.image_size, 
                   args.patch_size, args.vit_num_heads, args.vit_expansion_ratio, args.vit_droppath, args.vit_dropout)
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
    args = Args()
    main(args)





