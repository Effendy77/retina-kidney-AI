#!/usr/bin/env python3
"""
Training script for ESRD survival model.
Usage:
    python main_survival.py --train_csv data/folds/train.csv \
                            --val_csv data/folds/val.csv \
                            --image_root data/images
"""

import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.model.survival_model import SurvivalModel
from src.datasets.fundus_survival_dataset import FundusSurvivalDataset
from src.data.collate import survival_collate_fn
from src.train.train_survival import train_survival_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESRD survival model")

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = FundusSurvivalDataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        transform=transform,
    )
    val_dataset = FundusSurvivalDataset(
        csv_path=args.val_csv,
        image_root=args.image_root,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=survival_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=survival_collate_fn,
    )

    model = SurvivalModel(
        backbone_name=args.backbone,
        pretrained=True,
        dropout=0.1,
    )

    history = train_survival_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
    )

    print("\nTraining complete.")
    print("Final validation C-index:", history["val_cindex"][-1])


if __name__ == "__main__":
    main()
