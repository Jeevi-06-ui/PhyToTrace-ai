#!/usr/bin/env python3
"""PhytoTrace AI — CVAT Segmentation Annotation Importer & Mask Rasterizer."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from phytotrace.segmentation.io import (
    MatchResolutionStatus,
    MatchResult,
    find_coco_annotation_entry,
    parse_coco_segmentation_json,
    polygons_to_binary_mask,
    resolve_filename_match,
)
from phytotrace.segmentation.schemas import (
    AnnotationStatus,
    ImageSegmentationCatalogRecord,
    ReviewerDecision,
)
from phytotrace.segmentation.validators import (
    validate_mine_within_leaf_mask,
    validate_polygon_geometry,
)
from phytotrace.segmentation.visualization import save_segmentation_comparison_figure
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Import CVAT COCO polygon annotations and rasterize to binary masks.")
    parser.add_argument(
        "--coco_json", "--coco-path", "--coco-json",
        dest="coco_json",
        type=str,
        required=True,
        help="Path to exported CVAT COCO instances_default.json",
    )
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata.csv", help="Path to master metadata CSV")
    parser.add_argument("--image_dir", type=str, default="data/raw/original_photos", help="Directory containing raw photos")
    parser.add_argument("--masks_dir", type=str, default="data/segmentation/masks", help="Output directory for binary masks")
    parser.add_argument("--out_catalog", type=str, default="data/segmentation/catalog.csv", help="Segmentation tracking catalog CSV")
    parser.add_argument("--reviewer", type=str, default="human_annotator_01", help="Name/ID of annotation reviewer")
    parser.add_argument("--reviewer_decisions", type=str, default="data/segmentation/reviewer_decisions.csv", help="Path to human reviewer decisions CSV")
    parser.add_argument("--generate_overlays", action="store_true", help="Generate side-by-side verification overlays")
    args = parser.parse_args()

    coco_p = resolve_path(args.coco_json)
    meta_p = resolve_path(args.metadata)
    img_dir_p = resolve_path(args.image_dir)
    masks_p = resolve_path(args.masks_dir)
    cat_p = resolve_path(args.out_catalog)
    rev_dec_p = resolve_path(args.reviewer_decisions)

    print("=" * 70)
    print("🌿 PHYTOTRACE AI — PHASE 3 SEGMENTATION ANNOTATION IMPORTER")
    print("=" * 70)

    if not coco_p.is_file():
        print(f"❌ Error: COCO annotation file not found at {coco_p}")
        return

    meta_df = pd.read_csv(meta_p, dtype=str).fillna("")
    print(f"Loaded master metadata catalog ({len(meta_df)} images).")

    # Load recorded human reviewer decisions if available
    reviewer_decisions_map = {}
    if rev_dec_p.is_file():
        rev_df = pd.read_csv(rev_dec_p, dtype=str).fillna("")
        for _, r in rev_df.iterrows():
            rec = r.to_dict()
            if "image_id" in rec and rec["image_id"]:
                reviewer_decisions_map[rec["image_id"]] = rec
            if "original_filename" in rec and rec["original_filename"]:
                reviewer_decisions_map[rec["original_filename"]] = rec
        print(f"Loaded {len(rev_df)} human reviewer decisions from {rev_dec_p}.")

    # 1. Parse COCO JSON
    print(f"Parsing CVAT COCO JSON: {coco_p}...")
    parsed_annotations = parse_coco_segmentation_json(coco_p)
    print(f"Registered {len(parsed_annotations)} distinct image entries from COCO export.")

    # Upfront verification: Ensure every image in COCO resolves to exactly one metadata entry
    all_metadata_filenames = meta_df["original_filename"].tolist()
    coco_unresolved = []
    for coco_fname in parsed_annotations.keys():
        chk_res = resolve_filename_match(coco_fname, all_metadata_filenames)
        if chk_res.status not in (MatchResolutionStatus.EXACT, MatchResolutionStatus.NORMALIZED):
            coco_unresolved.append((coco_fname, chk_res.status, chk_res.message))
    if coco_unresolved:
        print(f"⚠️ Warning: {len(coco_unresolved)} COCO images could not be uniquely resolved to metadata:")
        for cf, st, msg in coco_unresolved:
            print(f"   - {cf} [{st.value}]: {msg}")

    leaf_mask_dir = masks_p / "leaf"
    mine_mask_dir = masks_p / "mine"
    vis_dir = masks_p / "visualizations"
    leaf_mask_dir.mkdir(parents=True, exist_ok=True)
    mine_mask_dir.mkdir(parents=True, exist_ok=True)
    if args.generate_overlays:
        vis_dir.mkdir(parents=True, exist_ok=True)

    catalog_records = []
    exact_matches = 0
    normalized_matches = 0
    ambiguous_matches = 0
    unmatched_images = 0
    total_leaf_polys = 0
    total_mine_polys = 0
    verified_negative_count = 0
    discrepancy_count = 0
    warnings_list = []

    # 2. Process each image in the master dataset
    for idx, row in meta_df.iterrows():
        img_id = str(row["image_id"])
        fname = Path(str(row["original_filename"])).name
        img_path = resolve_path(row.get("image_path", ""))

        if not img_path.is_file():
            img_path = img_dir_p / fname
        if not img_path.is_file() and img_dir_p.is_dir():
            # Fallback: case-insensitive search in image_dir
            for p in img_dir_p.iterdir():
                if p.name.lower() == fname.lower():
                    img_path = p
                    break

        img_bgr = None
        if img_path.is_file():
            img_bgr = cv2.imread(str(img_path))

        h, w = (img_bgr.shape[:2]) if img_bgr is not None else (int(row.get("height_px", 0)), int(row.get("width_px", 0)))

        # Safe hierarchical filename resolution against COCO entries
        ann_data, match_res = find_coco_annotation_entry(parsed_annotations, fname)
        in_coco = ann_data is not None and ann_data.get("present_in_coco", False)

        # Retrieve recorded human reviewer decisions if available
        rev_entry = reviewer_decisions_map.get(img_id) or reviewer_decisions_map.get(fname)
        rev_decision = rev_entry.get("decision", "pending_review").strip().lower() if rev_entry else "pending_review"
        rev_notes = rev_entry.get("rationale", "") if rev_entry else ""
        rev_reviewer = rev_entry.get("reviewer_id", args.reviewer) if rev_entry else args.reviewer

        notes_parts = []
        if match_res.status == MatchResolutionStatus.EXACT:
            exact_matches += 1
        elif match_res.status == MatchResolutionStatus.NORMALIZED:
            normalized_matches += 1
            notes_parts.append(f"Normalized match: CVAT '{match_res.matched_target}'")
        elif match_res.status == MatchResolutionStatus.AMBIGUOUS:
            ambiguous_matches += 1
            warn_msg = f"{img_id} ({fname}): Ambiguous collision with candidates {match_res.candidates}."
            warnings_list.append(warn_msg)
            notes_parts.append(f"AMBIGUOUS MATCH: {match_res.candidates}")
        else:
            unmatched_images += 1

        leaf_polys = ann_data.get("leaf", []) if ann_data else []
        mine_polys = ann_data.get("mine_damage", []) if ann_data else []

        total_leaf_polys += len(leaf_polys)
        total_mine_polys += len(mine_polys)

        leaf_status = AnnotationStatus.UNANNOTATED
        mine_status = AnnotationStatus.UNANNOTATED
        leaf_mask_rel = ""
        mine_mask_rel = ""
        is_verified_neg = False
        ready = False

        vis_damage = str(row.get("visible_damage_type", "unlabelled")).strip().lower()

        # CASE 1: Image not present in COCO or zero leaf polygons drawn
        if not in_coco or not leaf_polys:
            if in_coco and not leaf_polys:
                notes_parts.append("Image present in CVAT task but missing required 'leaf' polygon.")
                leaf_status = AnnotationStatus.IN_PROGRESS
            else:
                notes_parts.append("Awaiting CVAT polygon annotation.")

        # CASE 2: Leaf polygon is present and dimensions valid
        elif leaf_polys and h > 0 and w > 0:
            # Rasterize leaf boundary mask
            leaf_mask = polygons_to_binary_mask(leaf_polys, h, w)
            leaf_mask_fname = f"{img_id}_leaf_mask.png"
            cv2.imwrite(str(leaf_mask_dir / leaf_mask_fname), leaf_mask)
            leaf_mask_rel = f"data/segmentation/masks/leaf/{leaf_mask_fname}"
            leaf_status = AnnotationStatus.COMPLETED
            notes_parts.append(f"Leaf polygons: {len(leaf_polys)}")

            # Process mine annotations with human reviewer governance
            if rev_decision == "confirmed_negative":
                if mine_polys:
                    # DATA INTEGRITY SAFEGUARD: Reject CONFIRMED_NEGATIVE when COCO contains positive mine polygons!
                    # NEVER override positive polygons with empty masks based only on reviewer status.
                    mine_mask = polygons_to_binary_mask(mine_polys, h, w)
                    mine_mask_fname = f"{img_id}_mine_mask.png"
                    cv2.imwrite(str(mine_mask_dir / mine_mask_fname), mine_mask)
                    mine_mask_rel = f"data/segmentation/masks/mine/{mine_mask_fname}"
                    mine_status = AnnotationStatus.FLAGGED
                    ready = False
                    is_verified_neg = False
                    discrepancy_count += 1
                    notes_parts.append(
                        f"CONTRADICTION: Reviewer decision is CONFIRMED_NEGATIVE, but COCO contains "
                        f"{len(mine_polys)} mine_damage polygons! CVAT polygons must be deleted before "
                        "verified negative status can be granted."
                    )
                    warn_msg = (
                        f"{img_id} ({fname}): CONTRADICTION — reviewer status is CONFIRMED_NEGATIVE, but "
                        f"COCO contains {len(mine_polys)} mine_damage polygons! CVAT annotations and human decisions must agree."
                    )
                    warnings_list.append(warn_msg)
                else:
                    # Consistent: COCO contains exactly 0 mine polygons AND reviewer explicitly confirms negative
                    empty_mine_mask = np.zeros((h, w), dtype=np.uint8)
                    mine_mask_fname = f"{img_id}_mine_mask.png"
                    cv2.imwrite(str(mine_mask_dir / mine_mask_fname), empty_mine_mask)
                    mine_mask_rel = f"data/segmentation/masks/mine/{mine_mask_fname}"
                    mine_status = AnnotationStatus.COMPLETED
                    is_verified_neg = True
                    ready = True
                    verified_negative_count += 1
                    notes_parts.append(f"REVIEW [{rev_reviewer}]: CONFIRMED_NEGATIVE — Verified 0 mines in COCO: {rev_notes}")

            elif rev_decision == "needs_correction":
                # Expert human reviewer determined polygon annotations must be corrected in CVAT
                if mine_polys:
                    mine_mask = polygons_to_binary_mask(mine_polys, h, w)
                    mine_mask_fname = f"{img_id}_mine_mask.png"
                    cv2.imwrite(str(mine_mask_dir / mine_mask_fname), mine_mask)
                    mine_mask_rel = f"data/segmentation/masks/mine/{mine_mask_fname}"
                    notes_parts.append(f"Mine polygons: {len(mine_polys)}")
                mine_status = AnnotationStatus.FLAGGED
                ready = False
                discrepancy_count += 1
                notes_parts.append(f"REVIEW [{rev_reviewer}]: NEEDS_CORRECTION in CVAT — {rev_notes}")
                warn_msg = f"{img_id} ({fname}): NEEDS_CORRECTION in CVAT — {rev_notes}"
                warnings_list.append(warn_msg)

            elif rev_decision == "uncertain_exclude":
                # Ambiguous leaf; explicitly excluded from training
                mine_status = AnnotationStatus.FLAGGED
                ready = False
                notes_parts.append(f"REVIEW [{rev_reviewer}]: UNCERTAIN_EXCLUDE — {rev_notes}")

            elif rev_decision == "confirmed_mine" or (rev_decision == "pending_review" and mine_polys):
                # Mine damage polygons present
                if mine_polys:
                    mine_mask = polygons_to_binary_mask(mine_polys, h, w)
                    mine_mask_fname = f"{img_id}_mine_mask.png"
                    cv2.imwrite(str(mine_mask_dir / mine_mask_fname), mine_mask)
                    mine_mask_rel = f"data/segmentation/masks/mine/{mine_mask_fname}"
                    mine_status = AnnotationStatus.COMPLETED
                    notes_parts.append(f"Mine polygons: {len(mine_polys)}")

                    # Containment check: Mine damage must reside inside the leaf perimeter
                    valid_containment, out_ratio, cont_errs = validate_mine_within_leaf_mask(mine_mask, leaf_mask)
                    if not valid_containment and out_ratio > 0.05:
                        warn_msg = f"{img_id} ({fname}): Containment violation ({out_ratio*100:.1f}% mine area outside leaf boundary)."
                        warnings_list.append(warn_msg)
                        notes_parts.append(f"WARN: {out_ratio*100:.1f}% mine outside leaf")
                        mine_status = AnnotationStatus.FLAGGED
                        ready = False
                    else:
                        ready = True
                        if not valid_containment:
                            notes_parts.append(f"Notice: {out_ratio*100:.1f}% mine outside leaf (within 5% tolerance)")

                    # Biological conflict check: If image is labeled healthy in metadata, check reviewer decision
                    if vis_damage == "healthy":
                        if rev_decision == "confirmed_mine":
                            notes_parts.append(f"AUDIT TRAIL: Metadata misclassification ({vis_damage} -> confirmed_mine) verified by {rev_reviewer}: {rev_notes}")
                        else:
                            warn_msg = f"{img_id} ({fname}): BIOLOGICAL CONFLICT — Image is classified as 'healthy' in metadata, but {len(mine_polys)} mine_damage polygons were drawn in CVAT!"
                            warnings_list.append(warn_msg)
                            notes_parts.append(f"CONFLICT: Healthy leaf has {len(mine_polys)} mine polygons! Delete mines in CVAT to verify negative.")
                            mine_status = AnnotationStatus.FLAGGED
                            ready = False
                            discrepancy_count += 1
                    elif vis_damage == "other_damage":
                        if rev_decision == "confirmed_mine":
                            notes_parts.append(f"AUDIT TRAIL: Mixed damage leaf (confirmed mines co-occurring with other damage) verified by {rev_reviewer}: {rev_notes}")
                        else:
                            notes_parts.append(f"Check: {len(mine_polys)} mine polygons on other_damage leaf")
                else:
                    # In case confirmed_mine was set on an image with 0 mine polygons
                    mine_status = AnnotationStatus.FLAGGED
                    ready = False
                    discrepancy_count += 1
                    warn_msg = f"{img_id} ({fname}): DISCREPANCY — confirmed_mine but 0 mine polygons in CVAT export!"
                    warnings_list.append(warn_msg)
                    notes_parts.append("DISCREPANCY: Confirmed mine but 0 polygons in CVAT.")

            else:
                # Sub-case: Zero mine damage polygons in CVAT (unreviewed or pending review)
                if vis_damage in ("healthy", "other_damage"):
                    # Unreviewed negative candidate: requires explicit human reviewer decision
                    mine_status = AnnotationStatus.IN_PROGRESS
                    ready = False
                    notes_parts.append("Candidate negative leaf; requires human reviewer confirmation before training eligibility.")
                elif vis_damage == "leaf_miner_damage":
                    mine_status = AnnotationStatus.FLAGGED
                    ready = False
                    discrepancy_count += 1
                    warn_msg = f"{img_id} ({fname}): DISCREPANCY — marked as leaf_miner_damage in metadata, but 0 mine polygons in CVAT export!"
                    warnings_list.append(warn_msg)
                    notes_parts.append("DISCREPANCY: Labeled leaf_miner_damage, but 0 mines drawn. Requires human review.")
                else:
                    mine_status = AnnotationStatus.UNANNOTATED
                    ready = False
                    notes_parts.append("Unlabelled metadata category; cannot verify absence of mine damage.")

            # Optional verification overlay
            if args.generate_overlays and img_bgr is not None and leaf_status == AnnotationStatus.COMPLETED:
                vis_p = vis_dir / f"{img_id}_overlay.jpg"
                active_mine = cv2.imread(str(resolve_path(mine_mask_rel)), cv2.IMREAD_GRAYSCALE) if mine_mask_rel else None
                save_segmentation_comparison_figure(
                    img_bgr,
                    leaf_mask,
                    active_mine,
                    vis_p,
                    f"Image {img_id}: {fname} ({'Neg' if is_verified_neg else 'Mined'})",
                )

        record = ImageSegmentationCatalogRecord(
            image_id=img_id,
            original_filename=fname,
            plant_id=str(row.get("plant_id", "")),
            farm_id=str(row.get("farm_id", "")),
            visible_damage_type=vis_damage,
            leaf_mask_status=leaf_status,
            mine_annotation_status=mine_status,
            annotation_reviewer=rev_reviewer if (leaf_polys or is_verified_neg or rev_entry) else "none",
            reviewer_decision=rev_decision,
            reviewer_notes=rev_notes,
            num_damage_regions=len(mine_polys),
            image_quality=str(row.get("image_quality", "sharp")),
            ready_for_training=ready,
            is_verified_negative=is_verified_neg,
            leaf_mask_path=leaf_mask_rel,
            mine_mask_path=mine_mask_rel,
            notes=" | ".join(notes_parts),
        )
        catalog_records.append(record.to_dict())

    # 3. Save catalog CSV
    cat_df = pd.DataFrame(catalog_records)
    cat_p.parent.mkdir(parents=True, exist_ok=True)
    cat_df.to_csv(cat_p, index=False)

    print("\n✅ INGESTION AUDIT SUMMARY:")
    print(f"   • Total metadata records processed  : {len(cat_df)}")
    print(f"   • COCO images registered            : {len(parsed_annotations)}")
    print(f"   • Exact filename matches            : {exact_matches}")
    print(f"   • Normalized filename matches       : {normalized_matches}")
    print(f"   • Ambiguous filename collisions     : {ambiguous_matches}")
    print(f"   • Unmatched metadata records        : {unmatched_images}")
    print(f"   • Leaf polygons imported            : {total_leaf_polys}")
    print(f"   • Mine damage polygons imported     : {total_mine_polys}")
    print(f"   • Completed leaf masks              : {(cat_df['leaf_mask_status'] == 'completed').sum()}")
    print(f"   • Completed mine annotations        : {(cat_df['mine_annotation_status'] == 'completed').sum()}")
    print(f"   • Verified negative mine masks      : {verified_negative_count}")
    if "reviewer_decision" in cat_df.columns:
        print(f"   • Human Review - Confirmed Mine     : {(cat_df['reviewer_decision'] == 'confirmed_mine').sum()}")
        print(f"   • Human Review - Needs Correction   : {(cat_df['reviewer_decision'] == 'needs_correction').sum()}")
        print(f"   • Human Review - Confirmed Negative : {(cat_df['reviewer_decision'] == 'confirmed_negative').sum()}")
        print(f"   • Human Review - Uncertain / Exclude: {(cat_df['reviewer_decision'] == 'uncertain_exclude').sum()}")
    print(f"   • Flagged biological conflicts/disc : {discrepancy_count}")
    print(f"   • Ready for training                : {cat_df['ready_for_training'].sum()}/{len(cat_df)}")
    print(f"   • Saved catalog CSV to              : {cat_p}")

    if warnings_list:
        print(f"\n⚠️ Quality Warnings & Discrepancies ({len(warnings_list)}):")
        for w in warnings_list[:8]:
            print(f"   - {w}")
        if len(warnings_list) > 8:
            print(f"   ... and {len(warnings_list) - 8} more.")
    print("=" * 70)


if __name__ == "__main__":
    main()
