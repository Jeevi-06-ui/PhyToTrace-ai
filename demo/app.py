"""PhytoTrace AI — Phase 2 Dataset Inventory, Labelling Console & ML Dashboard."""

from datetime import datetime
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from phytotrace.data.augmentation import get_baseline_transforms
from phytotrace.data.validators import check_and_fix_exif_orientation
from phytotrace.device import get_device
from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.preprocessing.quality import assess_image_quality
from phytotrace.segmentation.tiling import compare_tiling_vs_resizing
from phytotrace.segmentation.visualization import render_segmentation_overlay
from phytotrace.training.train_baseline import run_baseline_training_smoke_test, train_baseline_pipeline
from phytotrace.utils.paths import resolve_path

# Streamlit Page Config
st.set_page_config(
    page_title="PhytoTrace AI — Dataset Labelling & ML Engine",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (PhytoTrace Forest Green & Mint theme)
st.markdown("""
<style>
    .main { background-color: #f7faf8; }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 5px solid #2e7d32;
        margin-bottom: 12px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-green { background-color: #e8f5e9; color: #2e7d32; }
    .badge-amber { background-color: #fff3e0; color: #e65100; }
    .badge-coral { background-color: #ffebee; color: #c62828; }
    .badge-gray  { background-color: #eceff1; color: #455a64; }
    .stButton>button {
        background-color: #1b5e20;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def load_metadata_df() -> pd.DataFrame:
    meta_p = resolve_path("data/metadata/metadata.csv")
    if meta_p.is_file():
        return pd.read_csv(meta_p, dtype=str).fillna("")
    return pd.DataFrame()


def save_metadata_df(df: pd.DataFrame) -> None:
    meta_p = resolve_path("data/metadata/metadata.csv")
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_p, index=False)


def filter_dataframe(df: pd.DataFrame, filter_option: str) -> pd.DataFrame:
    """Filter metadata dataframe based on category string."""
    if df.empty:
        return df.copy().reset_index(drop=True)
    if "Unlabelled" in filter_option:
        return df[df["visible_damage_type"] == "unlabelled"].reset_index(drop=False)
    elif "Healthy" in filter_option:
        return df[df["visible_damage_type"] == "healthy"].reset_index(drop=False)
    elif "Leaf Miner" in filter_option:
        return df[df["visible_damage_type"] == "leaf_miner_damage"].reset_index(drop=False)
    elif "Other Damage" in filter_option:
        return df[df["visible_damage_type"] == "other_damage"].reset_index(drop=False)
    return df.reset_index(drop=False)


def resolve_navigation_index(image_count: int, requested_idx: int = 0) -> int | None:
    """
    Return valid navigation index, or None if image_count == 0.
    Handles 0 images (None), 1 image (0), and multiple images (clamped).
    """
    if image_count <= 0:
        return None
    if image_count == 1:
        return 0
    return max(0, min(requested_idx, image_count - 1))


def main():
    # Sidebar
    st.sidebar.title("🌿 PhytoTrace AI")
    st.sidebar.markdown("**Phase 2: Dataset Labelling & ML Engine**")
    st.sidebar.caption("Tomato Leaf Infestation Diagnostic System")
    st.sidebar.divider()

    device = get_device("auto")
    mps_on = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    st.sidebar.markdown(f"**Compute Device:** `{device.type.upper()}`")
    st.sidebar.markdown(f"**Apple MPS Acceleration:** {'✅ Active' if mps_on else '❌ Inactive'}")

    ckpt_path = resolve_path("models/checkpoints/baseline/baseline_mobilenetv3_best.pt")
    smoke_ckpt = resolve_path("models/checkpoints/smoke_test/baseline_mobilenetv3_best.pt")

    st.sidebar.markdown("### Model Status")
    if ckpt_path.is_file():
        st.sidebar.success("✅ Real Baseline Checkpoint Available")
    else:
        st.sidebar.warning("⏳ Real Model Awaiting Labelling & Training")

    if smoke_ckpt.is_file():
        st.sidebar.info("ℹ️ MPS Smoke Test Checkpoint Available")

    st.sidebar.divider()
    st.sidebar.info(
        "🔬 Scientific Rule: We do NOT start training on unreviewed images. "
        "Review images in the 'Labelling & Review' tab to confirm labels before running model training."
    )

    # Main Tabs
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏷️ Image Labelling & Review",
        "📊 Dataset Inventory",
        "⚙️ Training Console",
        "📈 Evaluation & Metrics",
        "🔍 Single-Image Prediction",
        "✂️ Phase 3: Segmentation Readiness",
    ])

    # =========================================================================
    # TAB 0: INTERACTIVE IMAGE LABELLING & REVIEW
    # =========================================================================
    with tab0:
        st.header("Tomato Leaf Image Labelling & Review Interface")
        st.markdown(
            "Review each original photograph, inspect visual symptoms, and assign ground-truth labels "
            "(`healthy`, `leaf_miner_damage`, `other_damage`, or keep `unlabelled`)."
        )

        df = load_metadata_df()

        if df.empty:
            st.warning("No metadata catalog found. Run `python scripts/build_dataset_inventory.py` first.")
            if st.button("🔄 Scan data/raw/ and Generate Inventory"):
                with st.spinner("Scanning data/raw/..."):
                    from scripts.build_dataset_inventory import build_inventory_and_metadata
                    build_inventory_and_metadata()
                    st.rerun()
            return

        # Summary KPIs
        total_imgs = len(df)
        healthy_count = int(np.sum(df["visible_damage_type"] == "healthy"))
        miner_count = int(np.sum(df["visible_damage_type"] == "leaf_miner_damage"))
        other_count = int(np.sum(df["visible_damage_type"] == "other_damage"))
        unlabelled_count = int(np.sum(df["visible_damage_type"] == "unlabelled"))

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total Images", total_imgs)
        with k2:
            st.metric("Healthy", healthy_count)
        with k3:
            st.metric("Leaf Miner Damage", miner_count)
        with k4:
            st.metric("Other Damage", other_count)
        with k5:
            badge_color = "badge-coral" if unlabelled_count > 0 else "badge-green"
            st.markdown(f"""
            <div style="padding: 6px;">
                <small>Needs Review</small><br>
                <span class="status-badge {badge_color}">{unlabelled_count} UNLABELLED</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Filter View
        filter_options = [
            f"All Images ({total_imgs})",
            f"Unlabelled Only ({unlabelled_count})",
            f"Healthy ({healthy_count})",
            f"Leaf Miner Damage ({miner_count})",
            f"Other Damage ({other_count})",
        ]
        filter_option = st.selectbox(
            "Filter Images by Current Status:",
            filter_options,
        )

        filtered_df = filter_dataframe(df, filter_option)

        # Image navigation — handles 0, 1, or multiple images
        image_count = len(filtered_df)
        if image_count == 0:
            st.info("No images found for this filter. Try selecting 'All Images'.")
            sel_idx = None
        elif image_count == 1:
            sel_idx = 0
            st.caption("Showing the only image matching this filter.")
        else:
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            with col_nav2:
                sel_idx = st.slider(
                    "Reviewing Image",
                    min_value=0,
                    max_value=image_count - 1,
                    value=0,
                    format="%d",
                )

        # Only access the image when a valid selection exists
        if sel_idx is not None and not filtered_df.empty:
            target_row = filtered_df.iloc[sel_idx]
            orig_df_idx = target_row["index"]
            img_rel_path = target_row.get("image_path", "")
            img_full_path = resolve_path(img_rel_path)

            col_view, col_label = st.columns([1.2, 1])

            with col_view:
                st.subheader(f"Photograph #{sel_idx + 1}: {target_row.get('original_filename', '')}")
                if img_full_path.is_file():
                    try:
                        bgr = check_and_fix_exif_orientation(img_full_path)
                        if bgr is not None:
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            st.image(rgb, use_container_width=True)
                            h_orig, w_orig = bgr.shape[:2]
                            st.caption(f"Resolution: {w_orig}x{h_orig} px | File: `{img_rel_path}`")
                    except Exception as e:
                        st.error(f"Cannot decode image: {e}")
                else:
                    st.error(f"File not found on disk at {img_full_path}")

            with col_label:
                st.subheader("Assign Label & Metadata")
                current_label = target_row.get("visible_damage_type", "unlabelled")

                label_options = ["unlabelled", "healthy", "leaf_miner_damage", "other_damage"]
                try:
                    default_choice_idx = label_options.index(current_label)
                except ValueError:
                    default_choice_idx = 0

                row_key = f"{target_row.get('image_id', sel_idx)}_{sel_idx}"

                new_label = st.radio(
                    "Select Verified Category:",
                    label_options,
                    index=default_choice_idx,
                    key=f"label_{row_key}",
                    help="Categorize visible symptom. Mine galleries -> leaf_miner_damage; Spotting/blight/chewing -> other_damage; Clean leaflet -> healthy.",
                )

                st.markdown("#### Metadata Attributes")
                farm_val = st.text_input("Farm / Site ID:", value=target_row.get("farm_id", "BLOCK_3"), key=f"farm_{row_key}")
                plant_val = st.text_input("Plant ID:", value=target_row.get("plant_id", "PL001"), key=f"plant_{row_key}")
                leaf_val = st.text_input("Leaf ID:", value=target_row.get("leaf_id", "L01"), key=f"leaf_{row_key}")
                notes_val = st.text_area("Field Notes / Observations:", value=target_row.get("notes", ""), key=f"notes_{row_key}")

                verif_status = st.selectbox(
                    "Label Verification Status:",
                    ["unverified_field_guess", "expert_reviewed", "lab_confirmed"],
                    index=1 if new_label != "unlabelled" else 0,
                    key=f"verif_{row_key}",
                )

                if st.button("💾 Save Label & Update Metadata", key=f"save_{row_key}"):
                    df.at[orig_df_idx, "visible_damage_type"] = new_label
                    df.at[orig_df_idx, "farm_id"] = farm_val
                    df.at[orig_df_idx, "plant_id"] = plant_val
                    df.at[orig_df_idx, "leaf_id"] = leaf_val
                    df.at[orig_df_idx, "notes"] = notes_val
                    df.at[orig_df_idx, "label_verification_status"] = verif_status
                    save_metadata_df(df)
                    st.success(f"Saved #{target_row.get('image_id')} as '{new_label}'!")
                    st.rerun()

        st.divider()

        # Batch Operations Helper
        st.subheader("⚡ Quick Batch Labelling Assistant")
        st.markdown(
            "Your dataset has 46 `BLK_3_1002_PLxxx_NH.JPG` photographs. In field scouting, `_NH` designates non-healthy leaves. "
            "Inspect the contact sheet: if a batch consists predominantly of leaf-miner mines, you can label them quickly here."
        )

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("Assign all remaining unlabelled '_NH' to 'leaf_miner_damage' (Draft)"):
                count_updated = 0
                for idx, row in df.iterrows():
                    fname = str(row.get("original_filename", ""))
                    cur_lbl = str(row.get("visible_damage_type", ""))
                    if "_NH" in fname and cur_lbl == "unlabelled":
                        df.at[idx, "visible_damage_type"] = "leaf_miner_damage"
                        df.at[idx, "label_verification_status"] = "unverified_field_guess"
                        count_updated += 1
                save_metadata_df(df)
                st.success(f"Updated {count_updated} non-healthy images to 'leaf_miner_damage'!")
                st.rerun()

        with bcol2:
            if st.button("Reset all labels to initial filename suggestions"):
                from scripts.build_dataset_inventory import build_inventory_and_metadata
                build_inventory_and_metadata()
                st.success("Reset metadata to clean default inventory!")
                st.rerun()

    # =========================================================================
    # TAB 1: DATASET INVENTORY & CONTACT SHEET
    # =========================================================================
    with tab1:
        st.header("Dataset Inventory & Visual Contact Sheet")
        st.markdown(
            "Inventory of all 95 downloaded photographs in `data/raw/original_photos/`. "
            "Generated without modifying or deleting original files."
        )

        df = load_metadata_df()
        inv_csv_p = resolve_path("reports/dataset_inventory.csv")
        contact_sheet_p = resolve_path("reports/thumbnail_contact_sheet.jpg")

        # Contact sheet preview
        st.subheader("Visual Thumbnail Contact Sheet (All 95 Photos)")
        if contact_sheet_p.is_file():
            st.image(str(contact_sheet_p), use_container_width=True)
            with open(contact_sheet_p, "rb") as cs_f:
                st.download_button(
                    label="📥 Download High-Res Contact Sheet (1.4 MB JPG)",
                    data=cs_f.read(),
                    file_name="phytotrace_thumbnail_contact_sheet.jpg",
                    mime="image/jpeg",
                )
        else:
            st.info("Contact sheet not found. Run `python scripts/build_dataset_inventory.py` to generate.")

        st.divider()

        # Detailed Inventory Table
        st.subheader("Metadata & File Inventory Catalog")
        if inv_csv_p.is_file():
            df_inv = pd.read_csv(inv_csv_p)
            st.dataframe(df_inv, use_container_width=True)
            with open(inv_csv_p, "rb") as inv_f:
                st.download_button(
                    label="📥 Download Inventory CSV",
                    data=inv_f.read(),
                    file_name="dataset_inventory.csv",
                    mime="text/csv",
                )
        else:
            st.dataframe(df, use_container_width=True)

    # =========================================================================
    # TAB 2: TRAINING CONSOLE
    # =========================================================================
    with tab2:
        st.header("MobileNetV3-Small Training Console")
        st.markdown(
            "Execute model training once labels are reviewed and confirmed in Tab 0."
        )

        df = load_metadata_df()
        unlabelled_left = int(np.sum(df["visible_damage_type"] == "unlabelled")) if not df.empty else 999
        healthy_left = int(np.sum(df["visible_damage_type"] == "healthy")) if not df.empty else 0
        miner_left = int(np.sum(df["visible_damage_type"] == "leaf_miner_damage")) if not df.empty else 0
        other_left = int(np.sum(df["visible_damage_type"] == "other_damage")) if not df.empty else 0

        # Readiness banner
        if unlabelled_left > 0:
            st.warning(
                f"⚠️ Review Pending: {unlabelled_left} images are still marked as 'unlabelled'. "
                f"Current verified distribution: Healthy={healthy_left}, Leaf Miner={miner_left}, Other Damage={other_left}. "
                "Review labels before running final training."
            )
        else:
            st.success("✅ All 95 images have assigned labels! Ready for group-aware splitting and training.")

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("### Architecture Specifications")
            st.markdown("- **Backbone:** Pretrained MobileNetV3-Small (ImageNet)")
            st.markdown("- **Pooling:** Global Average Pooling (GAP)")
            st.markdown("- **Classification Head:** Linear(1024, 3) $\\rightarrow$ [healthy, leaf_miner_damage, other_damage]")

        with tc2:
            st.markdown("### Training Strategy")
            st.markdown("- **Stage A (Head Warmup):** 5 epochs with AdamW (LR: 1e-3)")
            st.markdown("- **Stage B (Fine-Tuning):** 20 epochs with Cosine Annealing (LR: 1e-4)")
            st.markdown("- **Batch Size:** 16 (optimized for Apple Silicon MPS)")

        st.divider()

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if st.button("🧪 Run Training Smoke Test on MPS"):
                with st.spinner("Executing forward/backward pass, Stage A, Stage B on Apple MPS..."):
                    smoke_res = run_baseline_training_smoke_test()
                    st.success("✅ Smoke Test Passed Successfully on Apple Silicon MPS!")
                    st.json({
                        "device": smoke_res["device"],
                        "status": smoke_res["status"],
                        "best_val_macro_f1": smoke_res["best_val_macro_f1"],
                        "checkpoint": smoke_res["best_checkpoint"],
                    })

        with t_col2:
            if st.button("🚀 Train Baseline on Reviewed Dataset"):
                if unlabelled_left > 0:
                    st.error(f"Cannot start training: {unlabelled_left} images are still unlabelled. Complete labelling in Tab 0.")
                else:
                    with st.spinner("Creating group-aware splits and training MobileNetV3-Small..."):
                        from phytotrace.data.split import make_group_splits, save_split_manifest
                        manifest = make_group_splits(df, group_col="farm_id", fallback_group_col="plant_id")
                        save_split_manifest(manifest, "data/splits/splits.json")
                        res = train_baseline_pipeline()
                        st.success("Training complete!")
                        st.json(res)

        # Training Curves Display
        st.subheader("Training Curves")
        exp_dir = resolve_path("experiments")
        curve_images = list(exp_dir.glob("**/loss_and_validation_curves.png"))
        if curve_images:
            latest_curve = sorted(curve_images, key=lambda p: p.stat().st_mtime)[-1]
            st.image(str(latest_curve), caption=f"Learning Curves: {latest_curve.parent.name}", use_container_width=True)

    # =========================================================================
    # TAB 3: EVALUATION & METRICS
    # =========================================================================
    with tab3:
        st.header("Scientific Evaluation & Held-Out Benchmark")
        eval_report_p = resolve_path("reports/evaluation_report.json")
        if eval_report_p.is_file():
            with open(eval_report_p, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            base_eval = eval_data.get("baseline_classifier", {})
            st.json(base_eval)
        else:
            st.info("Evaluation report not yet generated. Run `python scripts/evaluate.py`.")

    # =========================================================================
    # TAB 4: SINGLE-IMAGE PREDICTION
    # =========================================================================
    with tab4:
        st.header("Single-Image Prediction & Class Breakdown")
        avail_ckpts = []
        if ckpt_path.is_file():
            avail_ckpts.append(str(ckpt_path))
        if smoke_ckpt.is_file():
            avail_ckpts.append(str(smoke_ckpt))

        selected_ckpt = st.selectbox("Select Model Checkpoint:", avail_ckpts) if avail_ckpts else None

        uploaded = st.file_uploader("Upload Tomato Leaf Photograph:", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is not None and selected_ckpt:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Uploaded Photograph", width=360)

                qc = assess_image_quality(img_bgr)
                if not qc["is_acceptable"]:
                    st.warning(f"⚠️ Quality Warning: {', '.join(qc['reasons'])}")

                model = MobileNetV3Baseline(num_classes=3, pretrained=False).to(device)
                state = torch.load(selected_ckpt, map_location=device, weights_only=True)
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                else:
                    model.load_state_dict(state)
                model.eval()

                pil_img = Image.fromarray(img_rgb)
                transforms = get_baseline_transforms(is_training=False)
                tensor = transforms(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = model.predict_probabilities(tensor).squeeze(0).cpu().numpy()

                classes = MobileNetV3Baseline.CLASS_NAMES
                pred_idx = int(np.argmax(probs))
                pred_class = classes[pred_idx]
                pred_conf = float(probs[pred_idx])

                st.markdown(f"### Predicted: `{pred_class.upper()}` ({pred_conf*100:.1f}%)")
                for i, name in enumerate(classes):
                    score = float(probs[i])
                    st.write(f"**{name.replace('_', ' ').title()}**: {score*100:.1f}%")
                    st.progress(score)

    # =========================================================================
    # TAB 5: PHASE 3 SEGMENTATION READINESS & CVAT VIEWER
    # =========================================================================
    with tab5:
        st.header("✂️ Phase 3: Leaf & Mine Damage Segmentation Readiness")
        st.markdown(
            "Phase 3 transitions PhytoTrace from whole-image classification to pixel-level localization. "
            "Because early serpentine leaf mines occupy $<3\\%$ of the leaf surface, whole-image downsampling "
            "causes false negatives. High-precision instance segmentation of both **leaf boundaries** and "
            "**active mines** is required."
        )

        seg_catalog_p = resolve_path("data/segmentation/catalog.csv")
        seg_report_p = resolve_path("reports/segmentation_readiness_report.json")
        pilot_manifest_p = resolve_path("data/segmentation/pilot_manifest.csv")

        if seg_catalog_p.is_file():
            seg_df = pd.read_csv(seg_catalog_p, dtype=str).fillna("")
        else:
            seg_df = pd.DataFrame()

        if pilot_manifest_p.is_file():
            pilot_df = pd.read_csv(pilot_manifest_p, dtype=str).fillna("")
        else:
            pilot_df = pd.DataFrame()

        if seg_report_p.is_file():
            with open(seg_report_p) as f:
                seg_report = json.load(f)
        else:
            seg_report = {}

        # Readiness Metrics
        total_seg = len(seg_df)
        leaf_done = (seg_df["leaf_mask_status"] == "completed").sum() if not seg_df.empty else 0
        mine_done = (seg_df["mine_annotation_status"] == "completed").sum() if not seg_df.empty else 0
        
        is_neg_col = "is_verified_negative" in seg_df.columns
        neg_verified = (
            seg_df["is_verified_negative"].apply(lambda x: str(x).lower() in ["true", "1"]).sum()
            if (not seg_df.empty and is_neg_col) else 0
        )
        
        flagged_issues = (
            (seg_df["mine_annotation_status"] == "flagged").sum() +
            seg_df["notes"].str.contains("WARN|DISCREPANCY|violation", case=False).sum()
        ) if not seg_df.empty else 0

        fully_ready = seg_df["ready_for_training"].apply(lambda x: str(x).lower() in ["true", "1"]).sum() if not seg_df.empty else 0
        pct_ready = (fully_ready / max(1, total_seg) * 100.0) if total_seg > 0 else 0.0

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Cataloged Images", total_seg)
            st.metric("Verified Leaf Masks", f"{leaf_done} / {total_seg}")
        with col_m2:
            st.metric("Verified Mine Masks", f"{mine_done} / {total_seg}")
            st.metric("Verified Negative Empty Masks", f"{neg_verified}")
        with col_m3:
            st.metric("Quality Warnings & Discrepancies", f"{flagged_issues}")
            st.metric("Training Readiness", f"{fully_ready} ({pct_ready:.1f}%)")

        # Scientific Integrity Alert
        if fully_ready == 0:
            st.warning(
                "⚠️ **Segmentation Readiness Status: AWAITING CVAT ANNOTATIONS**\n\n"
                "PhytoTrace enforces strict scientific data integrity. **Zero synthetic or fabricated masks** are permitted. "
                "The training gate will unlock once genuine human polygon annotations from CVAT are imported."
            )
        else:
            st.success(f"✅ **{fully_ready} / {total_seg} images verified and ready for segmentation training!**")

        st.divider()

        # Split into two columns: Image & Overlay Inspector vs. Catalog Data
        st.subheader("🖼️ Interactive Segmentation & Mask Inspector")

        if not seg_df.empty:
            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                scope_choice = st.radio(
                    "Inspection Scope:",
                    ["All Cataloged Images (95)", "Pilot Set (18 Priority Images)"],
                    horizontal=True,
                    key="seg_scope_choice",
                )
            with col_f2:
                damage_filter = st.selectbox(
                    "Filter Catalog by Damage Class:",
                    ["All Categories", "Leaf Miner Damage Only", "Healthy Only", "Other Damage Only"],
                    key="seg_filter_damage",
                )

            sub_seg_df = seg_df.copy()
            if "Pilot" in scope_choice and not pilot_df.empty:
                pilot_ids = set(pilot_df["image_id"])
                sub_seg_df = sub_seg_df[sub_seg_df["image_id"].isin(pilot_ids)]

            if damage_filter == "Leaf Miner Damage Only":
                sub_seg_df = sub_seg_df[sub_seg_df["visible_damage_type"] == "leaf_miner_damage"]
            elif damage_filter == "Healthy Only":
                sub_seg_df = sub_seg_df[sub_seg_df["visible_damage_type"] == "healthy"]
            elif damage_filter == "Other Damage Only":
                sub_seg_df = sub_seg_df[sub_seg_df["visible_damage_type"] == "other_damage"]

            img_choices = sub_seg_df["original_filename"].tolist()
            if img_choices:
                selected_filename = st.selectbox("Select Photograph to Inspect:", img_choices, key="seg_img_choice")
                row = sub_seg_df[sub_seg_df["original_filename"] == selected_filename].iloc[0]

                # Check if pilot rationale exists
                p_match = pilot_df[pilot_df["original_filename"] == selected_filename] if not pilot_df.empty else None
                if p_match is not None and not p_match.empty:
                    p_info = p_match.iloc[0]
                    st.info(
                        f"🎯 **Pilot Sample #{p_info.get('pilot_order', '')} | Partition: `{p_info.get('split_partition', '')}`**\n\n"
                        f"**Rationale:** {p_info.get('selection_rationale', '')}"
                    )

                raw_matches = list(resolve_path("data/raw").glob(f"**/{selected_filename}"))
                img_path = raw_matches[0] if raw_matches else None

                col_img_left, col_img_right = st.columns([1, 1])
                img_cv = None
                img_rgb = None

                with col_img_left:
                    st.markdown("**Original Field Photograph**")
                    if img_path and img_path.is_file():
                        img_cv = cv2.imread(str(img_path))
                        if img_cv is not None:
                            img_cv = check_and_fix_exif_orientation(str(img_path), img_cv)
                            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True)
                            st.caption(f"File: `{selected_filename}` | Dimensions: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
                        else:
                            st.error(f"Failed to read image at {img_path}")
                    else:
                        st.error(f"Image file not found: {selected_filename}")

                with col_img_right:
                    st.markdown("**Segmentation Overlay (Leaf: Green | Mine: Red)**")
                    leaf_mask_p = resolve_path(row["leaf_mask_path"]) if row["leaf_mask_path"] else None
                    mine_mask_p = resolve_path(row["mine_mask_path"]) if row["mine_mask_path"] else None

                    has_leaf = leaf_mask_p and leaf_mask_p.is_file()
                    has_mine = mine_mask_p and mine_mask_p.is_file()

                    if (has_leaf or has_mine) and img_rgb is not None:
                        leaf_mask = cv2.imread(str(leaf_mask_p), cv2.IMREAD_GRAYSCALE) if has_leaf else None
                        mine_mask = cv2.imread(str(mine_mask_p), cv2.IMREAD_GRAYSCALE) if has_mine else None
                        overlay = render_segmentation_overlay(img_rgb, leaf_mask=leaf_mask, mine_mask=mine_mask)
                        st.image(overlay, use_container_width=True)
                        st.success(f"Verified Masks Loaded: Leaf={'Yes' if has_leaf else 'No'}, Mine={'Yes' if has_mine else 'No'}")
                    else:
                        st.info(
                            f"📌 **Status: {row['notes']}**\n\n"
                            f"- **Leaf Mask:** `{row['leaf_mask_status']}`\n"
                            f"- **Mine Mask:** `{row['mine_annotation_status']}`\n"
                            f"- **Damage Type:** `{row['visible_damage_type']}`\n"
                            f"- **Plant ID:** `{row['plant_id']}`\n\n"
                            "Once annotations are completed in CVAT and imported via `python scripts/import_segmentation_annotations.py`, "
                            "exact color-coded mask overlays will appear here."
                        )

                # Resolution Tiling Simulator Expander
                if img_rgb is not None:
                    with st.expander("🔬 Tiling vs. Full-Image Downsampling Resolution Simulator", expanded=False):
                        tile_stats = compare_tiling_vs_resizing(img_rgb.shape[:2], tile_size=(640, 640), overlap=128)
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.markdown("#### Mode A: Full-Image Resizing (640x640)")
                            st.write(f"• **Scale Factor:** `{tile_stats['full_resize_scale_factor']}` ({tile_stats['full_resize_effective_resolution_pct']}% of native)")
                            st.write(f"• **1mm Mine Width:** `{tile_stats['downsampled_mine_feature_width_px']} px`")
                            st.write(f"• **Feature Status:** `{tile_stats['downsampled_feature_status']}`")
                        with col_t2:
                            st.markdown("#### Mode B: Overlapping Tiling (640x640)")
                            st.write(f"• **Tiles Generated:** `{tile_stats['tiling_mode_tile_count']}` (Overlap: {tile_stats['tiling_mode_overlap_px']} px)")
                            st.write(f"• **1mm Mine Width:** `{tile_stats['tiled_mine_feature_width_px']} px` (100% native)")
                            st.write(f"• **Feature Status:** `{tile_stats['tiled_feature_status']}`")
                        st.caption(tile_stats["recommendation"])

                with st.expander("🔍 View Image Metadata Record"):
                    st.json(row.to_dict())

            else:
                st.info("No images match the selected filter.")

        st.divider()

        st.subheader("📋 Phase 3 Segmentation Catalog")
        if not seg_df.empty:
            st.dataframe(
                seg_df,
                use_container_width=True,
                height=300,
            )

        with st.expander("📖 CVAT Annotation Protocol & Import Instructions (Phase 3)", expanded=False):
            st.markdown("""
            ### CVAT Annotation Protocol Summary
            PhytoTrace Phase 3 requires standard CVAT polygon annotation:
            1. **Class 1: `leaf`** (Shape: Polygon) — Trace complete boundary of the tomato leaf or leaflet. Exclude background, stems, and neighboring foliage.
            2. **Class 2: `mine_damage`** (Shape: Polygon) — Trace serpentine or blotch mine boundaries on the leaf surface. **Rule: Must be biologically contained within the leaf polygon.**
            3. **Agnostic Principle:** Do **NOT** assign species labels (*Tuta* vs *Liriomyza*) at this stage. That is reserved for Phase 4 morphological evidence extraction.
            4. **Export Format:** Export annotations from CVAT as **COCO 1.0 (instances_default.json)**.
            5. **Import & Ingestion:** Run the automated rasterizer:
               ```bash
               .venv/bin/python scripts/import_segmentation_annotations.py --coco-path /path/to/instances_default.json
               ```
            6. **Readiness Audit:** Re-run the audit anytime:
               ```bash
               .venv/bin/python scripts/audit_segmentation_readiness.py
               ```
            """)


if __name__ == "__main__":
    main()
