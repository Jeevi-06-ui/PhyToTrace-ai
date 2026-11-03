"""Deduplication and duplicate cluster detection for PhytoTrace AI."""

import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
import cv2
import numpy as np
from PIL import Image

from phytotrace.utils.logging import get_logger

logger = get_logger("phytotrace.deduplicate")


def compute_sha256(file_path: Path) -> str:
    """Compute exact SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for perceptual near-duplicate detection.
    
    Args:
        image_path: Path to the image file.
        hash_size: Grid width (generates hash_size * hash_size bits).
        
    Returns:
        Hexadecimal string representation of the hash.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize to (hash_size + 1, hash_size)
            resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(resized)
            
            # Compare adjacent pixels horizontally
            diff = pixels[:, 1:] > pixels[:, :-1]
            
            # Pack boolean array into an integer and then to hex string
            decimal_val = 0
            for bit in diff.flatten():
                decimal_val = (decimal_val << 1) | int(bit)
            return hex(decimal_val)[2:].zfill((hash_size * hash_size) // 4)
    except Exception as e:
        logger.warning("Could not compute dHash for %s: %s", image_path, e)
        return ""


def hamming_distance(hex1: str, hex2: str) -> int:
    """Compute Hamming distance between two hex hashes of equal length."""
    if not hex1 or not hex2 or len(hex1) != len(hex2):
        return 9999
    val1 = int(hex1, 16)
    val2 = int(hex2, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")


def detect_duplicates(
    image_paths: List[Path],
    near_dup_threshold: int = 4,
) -> Tuple[Dict[str, List[Path]], List[Set[Path]]]:
    """Scan images to find exact duplicates and near-duplicate clusters.
    
    Args:
        image_paths: List of image paths.
        near_dup_threshold: Maximum Hamming distance to consider as near duplicate.
        
    Returns:
        (exact_duplicates_dict, near_duplicate_clusters)
    """
    sha_map: Dict[str, List[Path]] = {}
    dhashes: List[Tuple[Path, str]] = []
    
    for p in image_paths:
        try:
            sha = compute_sha256(p)
            sha_map.setdefault(sha, []).append(p)
            
            dh = compute_dhash(p)
            if dh:
                dhashes.append((p, dh))
        except Exception as e:
            logger.error("Error processing %s during deduplication: %s", p, e)
            
    exact_duplicates = {k: v for k, v in sha_map.items() if len(v) > 1}
    
    # Cluster near-duplicates
    near_clusters: List[Set[Path]] = []
    visited: Set[Path] = set()
    
    for i in range(len(dhashes)):
        p1, h1 = dhashes[i]
        if p1 in visited:
            continue
            
        cluster = {p1}
        for j in range(i + 1, len(dhashes)):
            p2, h2 = dhashes[j]
            if p2 in visited:
                continue
            dist = hamming_distance(h1, h2)
            if dist <= near_dup_threshold:
                cluster.add(p2)
                
        if len(cluster) > 1:
            near_clusters.append(cluster)
            visited.update(cluster)
            
    return exact_duplicates, near_clusters

