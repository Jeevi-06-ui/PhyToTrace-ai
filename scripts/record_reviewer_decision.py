#!/usr/bin/env python3
"""PhytoTrace AI — Record & Audit Human Expert Reviewer Decisions.

Provides an auditable CLI interface to record reviewer decisions for ambiguous
or conflicting annotations, maintaining an immutable decision history log.
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import pandas as pd

from phytotrace.segmentation.schemas import ReviewerDecision
from phytotrace.utils.paths import resolve_path

VALID_DECISIONS = [
    ReviewerDecision.CONFIRMED_NEGATIVE.value,
    ReviewerDecision.CONFIRMED_MINE.value,
    ReviewerDecision.NEEDS_CORRECTION.value,
    ReviewerDecision.UNCERTAIN_EXCLUDE.value,
]


def main():
    parser = argparse.ArgumentParser(description="Record human reviewer decision for PhytoTrace segmentation.")
    parser.add_argument("--image_id", type=str, required=True, help="Image ID (e.g. IMG_0010) or 'ALL_HEALTHY'")
    parser.add_argument("--decision", type=str, required=True, choices=VALID_DECISIONS, help="Reviewer decision")
    parser.add_argument("--reviewer", type=str, default="expert_reviewer", help="Name/ID of expert reviewer")
    parser.add_argument("--notes", type=str, required=True, help="Reviewer biological rationale")
    parser.add_argument("--ledger", type=str, default="data/segmentation/reviewer_decisions.csv", help="Review ledger path")
    args = parser.parse_args()

    ledger_p = resolve_path(args.ledger)
    meta_p = resolve_path("data/metadata/metadata.csv")
    manifest_p = resolve_path("data/segmentation/pilot_manifest.csv")
    history_dir = resolve_path("data/segmentation/backups/reviewer_decisions_history")

    if not ledger_p.is_file():
        print(f"❌ Ledger file not found at {ledger_p}")
        sys.exit(1)

    meta_df = pd.read_csv(meta_p, dtype=str).fillna("")
    pilot_df = pd.read_csv(manifest_p, dtype=str).fillna("") if manifest_p.is_file() else pd.DataFrame()
    split_map = dict(zip(pilot_df["image_id"], pilot_df["split_partition"])) if not pilot_df.empty else {}
    class_map = dict(zip(meta_df["image_id"], meta_df["visible_damage_type"]))
    file_map = dict(zip(meta_df["image_id"], meta_df["original_filename"]))

    df = pd.read_csv(ledger_p, dtype=str).fillna("")

    # Backup previous ledger before modification
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = history_dir / f"reviewer_decisions_{ts}.csv"
    shutil.copy2(ledger_p, backup_file)

    target_ids = []
    if args.image_id.upper() == "ALL_HEALTHY":
        target_ids = [iid for iid, cls in class_map.items() if cls == "healthy" and iid in set(pilot_df["image_id"])]
        print(f"Applying decision to ALL {len(target_ids)} healthy pilot images: {target_ids}")
    else:
        target_ids = [args.image_id.strip()]

    updated_count = 0
    now_iso = datetime.now().isoformat()

    for iid in target_ids:
        if iid not in class_map:
            print(f"⚠️ Warning: Image ID '{iid}' not found in metadata catalog. Skipping.")
            continue

        fname = file_map.get(iid, "")
        split = split_map.get(iid, "train")
        meta_cls = class_map.get(iid, "healthy")
        corr_cls = "healthy" if args.decision == ReviewerDecision.CONFIRMED_NEGATIVE.value else ("leaf_miner_damage" if args.decision == ReviewerDecision.CONFIRMED_MINE.value else meta_cls)

        mask = df["image_id"] == iid
        if mask.any():
            df.loc[mask, "decision"] = args.decision
            df.loc[mask, "reviewer_id"] = args.reviewer
            df.loc[mask, "corrected_metadata_class"] = corr_cls
            df.loc[mask, "rationale"] = args.notes
            df.loc[mask, "reviewed_timestamp"] = now_iso
        else:
            new_row = {
                "image_id": iid,
                "original_filename": fname,
                "split_partition": split,
                "visible_damage_type": meta_cls,
                "reviewer_id": args.reviewer,
                "decision": args.decision,
                "original_metadata_class": meta_cls,
                "corrected_metadata_class": corr_cls,
                "rationale": args.notes,
                "reviewed_timestamp": now_iso,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        updated_count += 1
        print(f"✅ Recorded {args.decision} for {iid} ({fname}): {args.notes}")

    df.to_csv(ledger_p, index=False)
    print(f"\n📁 Successfully updated {updated_count} records in {ledger_p}.")
    print(f"📦 Audit backup created at {backup_file}.")


if __name__ == "__main__":
    main()

