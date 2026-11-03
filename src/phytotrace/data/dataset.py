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
    """Dataset for binary leaf segmentation (leaf vs background)."""

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: Optional[List[Path]] = None,
        target_size: Tuple[int, int] = (256, 256),
    ):
        self.image_paths = [resolve_path(p) for p in image_paths]
        self.mask_paths = [resolve_path(p) for p in mask_paths] if mask_paths else None
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        bgr = check_and_fix_exif_orientation(img_path)
        if bgr is None:
            raise ValueError(f"Failed to read image {img_path}")
            
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float() / 255.0

        if self.mask_paths and idx < len(self.mask_paths) and self.mask_paths[idx].exists():
            mask_arr = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(mask_arr, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask_tensor = (torch.from_numpy(resized_mask).float() / 255.0).clamp(0, 1).unsqueeze(0)
        else:
            # Fallback zero mask if no mask supplied
            mask_tensor = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)

        return img_tensor, mask_tensor


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

