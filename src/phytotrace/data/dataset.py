"""PyTorch Dataset implementations for PhytoTrace AI."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from phytotrace.data.validators import check_and_fix_exif_orientation
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.dataset")


class TomatoLeafClassificationDataset(Dataset):
    """Dataset for 3-class baseline classifier:
    0: healthy
    1: leaf_miner_damage
    2: other_damage
    """

    LABEL_MAP = {
        "none": 0,
        "healthy": 0,
        "leaf_miner_mine": 1,
        "leaf_miner_damage": 1,
        "chewing": 2,
        "blight": 2,
        "punctures": 2,
        "curling": 2,
        "other": 2,
        "other_damage": 2,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transforms: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = resolve_path(image_dir)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = str(row.get("image_id", ""))
        rel_path = str(row.get("image_path", ""))
        
        # Resolve file path
        img_path = resolve_path(rel_path)
        if not img_path.exists():
            img_path = self.image_dir / Path(rel_path).name
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {rel_path} or {img_path}")

        # Load image with EXIF orientation correction
        bgr = check_and_fix_exif_orientation(img_path)
        if bgr is None:
            raise ValueError(f"Failed to read image at {img_path}")
            
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if self.transforms:
            tensor_img = self.transforms(pil_img)
        else:
            tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        raw_label = str(row.get("visible_damage_type", "other")).lower().strip()
        label_idx = self.LABEL_MAP.get(raw_label, 2)  # Default to other_damage if unknown

        return tensor_img, label_idx, image_id


class TomatoLeafSegmentationDataset(Dataset):
    """Reviewed, aligned image/mask pairs in the U-Net checkpoint's RGB [0, 1] space."""

    def __init__(self, image_paths, mask_paths, target_size=(256, 256), augment=False):
        if not image_paths or len(image_paths) != len(mask_paths):
            raise ValueError("Each image needs a reviewed leaf mask.")
        self.image_paths = [resolve_path(p) for p in image_paths]
        self.mask_paths = [resolve_path(p) for p in mask_paths]
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx], self.mask_paths[idx]
        bgr = check_and_fix_exif_orientation(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None or mask.shape != bgr.shape[:2]:
            raise ValueError(f"Missing or misaligned leaf mask: {img_path.name}")
        values = set(np.unique(mask).tolist())
        if values not in ({0, 1}, {0, 255}):
            raise ValueError(f"Expected a binary mask with foreground and background: {mask_path.name}")

        rgb = cv2.cvtColor(cv2.resize(bgr, self.target_size), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        target = torch.from_numpy(mask > 0).float().unsqueeze(0)
        if self.augment:
            for axis in (-1, -2):
                if torch.rand(()) < 0.5:
                    image, target = image.flip(axis), target.flip(axis)
            image = (image * (0.85 + 0.30 * torch.rand(()))).clamp(0, 1)
        return image, target


class TomatoDamageEvidenceDataset(Dataset):
    """Dataset for multi-label pest evidence classification (Tuta / Liriomyza).
    
    Includes label mask so unverified species are NOT treated as negative training samples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        image_id = str(row.get("image_id", ""))
        rel_path = str(row.get("image_path", ""))
        img_path = resolve_path(rel_path)

        bgr = check_and_fix_exif_orientation(img_path)
        if bgr is None:
            raise ValueError(f"Failed to load image for evidence {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if self.transforms:
            tensor_img = self.transforms(pil_img)
        else:
            tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        confirmed = str(row.get("confirmed_pest", "unconfirmed")).lower().strip()
        status = str(row.get("label_verification_status", "unverified_field_guess")).lower().strip()

        # Multi-label targets: [tuta_evidence, liriomyza_evidence]
        # Label mask: [tuta_is_verified, liriomyza_is_verified]
        targets = torch.zeros(2, dtype=torch.float32)
        valid_mask = torch.zeros(2, dtype=torch.float32)

        if status in {"expert_reviewed", "lab_confirmed"}:
            if confirmed == "tuta_absoluta":
                targets[0] = 1.0
                targets[1] = 0.0
                valid_mask[0] = 1.0
                valid_mask[1] = 1.0
            elif confirmed == "liriomyza_spp":
                targets[0] = 0.0
                targets[1] = 1.0
                valid_mask[0] = 1.0
                valid_mask[1] = 1.0
            elif confirmed == "mixed":
                targets[0] = 1.0
                targets[1] = 1.0
                valid_mask[0] = 1.0
                valid_mask[1] = 1.0
            elif confirmed == "none":
                targets[0] = 0.0
                targets[1] = 0.0
                valid_mask[0] = 1.0
                valid_mask[1] = 1.0

        return tensor_img, targets, valid_mask, image_id

