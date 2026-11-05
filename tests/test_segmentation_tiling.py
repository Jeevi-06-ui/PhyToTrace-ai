"""Unit tests for high-resolution overlapping tiling and mask stitching."""

import numpy as np
import pytest

from phytotrace.segmentation.tiling import (
    compare_tiling_vs_resizing,
    extract_tile_crops,
    generate_overlapping_tiles,
    stitch_tile_masks,
)


def test_generate_overlapping_tiles_coverage():
    """Verify that tile boxes completely cover the canvas without out-of-bound errors."""
    h, w = 1000, 1500
    tile_size = (400, 400)
    overlap = 100

    boxes = generate_overlapping_tiles((h, w), tile_size=tile_size, overlap=overlap)
    assert len(boxes) > 1

    # Check boundaries
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)

    assert min_x == 0
    assert min_y == 0
    assert max_x == w
    assert max_y == h

    for xmin, ymin, xmax, ymax in boxes:
        assert 0 <= xmin < xmax <= w
        assert 0 <= ymin < ymax <= h
        assert xmax - xmin == tile_size[1]
        assert ymax - ymin == tile_size[0]


def test_extract_tile_crops():
    """Verify image crops match expected tile shapes."""
    img = np.zeros((800, 1000, 3), dtype=np.uint8)
    boxes = generate_overlapping_tiles((800, 1000), tile_size=(500, 500), overlap=100)
    crops = extract_tile_crops(img, boxes)

    assert len(crops) == len(boxes)
    for c in crops:
        assert c.shape == (500, 500, 3)


def test_stitch_tile_masks_binary_union():
    """Verify that individual tile masks stitch back into the full canvas with union pooling."""
    canvas_h, canvas_w = 600, 800
    tile_size = (400, 400)
    overlap = 200

    boxes = generate_overlapping_tiles((canvas_h, canvas_w), tile_size=tile_size, overlap=overlap)

    # Place a synthetic mine spot in the first tile at local coord (100, 100)
    tile_masks = [np.zeros(tile_size, dtype=np.uint8) for _ in boxes]
    tile_masks[0][100:150, 100:150] = 255

    stitched = stitch_tile_masks((canvas_h, canvas_w), tile_masks, boxes, blend_mode="max")
    assert stitched.shape == (canvas_h, canvas_w)

    # First tile starts at (0, 0), so canvas coord is (100:150, 100:150)
    assert np.all(stitched[100:150, 100:150] == 255)
    assert stitched[0, 0] == 0
    assert stitched[500, 500] == 0


def test_compare_tiling_vs_resizing_analytics():
    """Verify that the comparison function produces meaningful spatial metrics."""
    res = compare_tiling_vs_resizing((3024, 4032), tile_size=(640, 640), overlap=128)

    assert "native_dimensions" in res
    assert res["full_resize_effective_resolution_pct"] < 25.0  # ~15-20%
    assert res["tiling_effective_resolution_pct"] == 100.0
    assert res["tiling_mode_tile_count"] > 1
    assert "SUB_PIXEL_RISK" in res["downsampled_feature_status"]

