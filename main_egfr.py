#!/usr/bin/env python3
"""
Training script for eGFR regression.

Usage:
    python main_egfr.py \
        --train_csv data/folds/train_reg.csv \
        --val_csv data/folds/val_reg.csv \
        --image_root data/images
"""

import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.datasets.egfr_regression_dataset import EGFRRegressionDataset
from src.model.retfound_regression import RETFoundRegression
from src.train.train_regression import train_regression_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train RETFound-based eGFR regression model")

    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV file")
    parser.add_argument("--val_csv", type=str, required=True, help="Validation CSV file")
    parser.add_argument("--image_root", type=str, required=True, help="Directory containing images")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocessing for eGFR images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = EGFRRegressionDataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        transform=transform,
    )
    val_dataset = EGFRRegressionDataset(
        csv_path=args.val_csv,
        image_root=args.image_root,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Model
    model = RETFoundRegression(
        backbone_name=args.backbone,
        pretrained=True,
        dropout=0.1,
    )

    # Training
    history = train_regression_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
    )

    print("\nTraining complete.")
    print("Final validation MAE:", history["val_mae"][-1])
    print("Final validation RMSE:", history["val_rmse"][-1])


if __name__ == "__main__":
    main()
