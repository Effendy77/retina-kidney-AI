import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MultimodalKidneyDatasetV2(Dataset):
    """
    Ablation-ready dataset:
      - Loads LEFT-eye RGB fundus image
      - Loads LEFT-eye vessel mask
      - Loads tabular features (10 cols)
      - Supports ablation flags:
            use_image, use_mask, use_tabular, use_retinal_features
      - Returns eGFR regression target
    """

    def __init__(
        self,
        csv_path,
        image_root,
        mask_root,
        use_image=True,
        use_mask=True,
        use_tabular=True,
        use_retinal_features=True,
        silence_qrisk=False, # New parameter to silence qrisk warning
    ):
        super().__init__()

        # ------------------------------
        # TABULAR FEATURES FIRST!
        # ------------------------------
        self.tabular_features = [
            "age",
            "sex",
            "diabetes",
            "hypertension",
            "qrisk3",
            "dm_htn_combined",
            "fractal_dim",
            "vessel_density",
            "eccentricity",
            "mean_width_px",
        ]

        # ------------------------------
        # Validate paths
        # ------------------------------
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Image root not a directory: {image_root}")
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"Mask root not a directory: {mask_root}")

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # ------------------------------
        # Validate required columns
        # ------------------------------
        required_cols = ["fundus_image", "vessel_mask", "egfr"]
        required_cols.extend(self.tabular_features)

        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Store paths
        self.image_root = image_root
        self.mask_root = mask_root

        # Ablation flags
        self.use_image = use_image
        self.use_mask = use_mask
        self.use_tabular = use_tabular
        self.use_retinal_features = use_retinal_features
        # new QRISK warning silencer
        self.silence_qrisk = silence_qrisk

        # ------------------------------
        # Image transforms
        # ------------------------------
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Mask transforms
        self.mask_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        print(f"[INFO] Loaded dataset: {len(self.df)} samples")
        print(f"[INFO] Ablation flags: image={use_image}, mask={use_mask}, "
              f"tabular={use_tabular}, retinal_features={use_retinal_features}, "
              f"silence_qrisk={silence_qrisk}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --------------------------
        # SAMPLE ID (optional)
        # --------------------------
        sample_id = row["eid"] if "eid" in row else idx

        # --------------------------
        # IMAGE
        # --------------------------
        img_path = row["fundus_image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"[WARN] Missing image → {img_path}, filling zeros.")
            img = Image.new("RGB", (224, 224), color=0)

        image_tensor = self.img_transform(img)
        if not self.use_image:
            image_tensor = torch.zeros_like(image_tensor)

        # --------------------------
        # MASK
        # --------------------------
        mask_path = row["vessel_mask"]
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(self.mask_root, mask_path)

        try:
            mask = Image.open(mask_path).convert("L")
        except Exception:
            print(f"[WARN] Missing mask → {mask_path}, filling zeros.")
            mask = Image.new("L", (224, 224), color=0)

        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        if not self.use_mask:
            mask_tensor = torch.zeros_like(mask_tensor)

        # --------------------------
        # TABULAR
        # --------------------------
        tab_series = row[self.tabular_features].astype(float).fillna(0.0)
        tab = torch.tensor(tab_series.values, dtype=torch.float32)
        
        # Silence QRISK3 for ablation (keep tensor shape unchanged)
        if self.silence_qrisk:
            qrisk_idx = self.tabular_features.index("qrisk3")
            tab[qrisk_idx] = 0.0 
            
        if not self.use_tabular:
            tab = torch.zeros_like(tab)

        if not self.use_retinal_features:
            retinal_cols = ["fractal_dim", "vessel_density", "eccentricity", "mean_width_px"]
            idxs = [self.tabular_features.index(f) for f in retinal_cols]
            tab[idxs] = 0.0

        # --------------------------
        # TARGET
        # --------------------------
        target = torch.tensor(row["egfr"], dtype=torch.float32)

        return {
            "sample_id": sample_id,
            "image": image_tensor,
            "mask": mask_tensor,
            "tabular": tab,
            "egfr": target,
        }
