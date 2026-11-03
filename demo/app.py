"""PhytoTrace AI — Phase 2 Training, Dataset & Diagnostic Dashboard."""

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
from phytotrace.training.train_baseline import run_baseline_training_smoke_test, train_baseline_pipeline
from phytotrace.utils.paths import resolve_path

# Page configuration
st.set_page_config(
    page_title="PhytoTrace AI — Phase 2 ML Engine Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for PhytoTrace forest green & mint theme
st.markdown("""
<style>
    .main { background-color: #f8faf8; }
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
    .stButton>button {
        background-color: #1b5e20;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def load_dataset_metadata():
    meta_p = resolve_path("data/metadata/metadata.csv")
    if meta_p.is_file():
        return pd.read_csv(meta_p, dtype=str).fillna("")
    return pd.DataFrame()


def main():
    # Sidebar
    st.sidebar.title("🌿 PhytoTrace AI")
    st.sidebar.markdown("**Phase 2: Baseline Model & Dataset Console**")
    st.sidebar.caption("Target: MobileNetV3-Small (Healthy / Leaf Miner / Other)")
    st.sidebar.divider()

    device = get_device("auto")
    mps_on = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    st.sidebar.markdown(f"**Compute Device:** `{device.type.upper()}`")
    st.sidebar.markdown(f"**Apple MPS Acceleration:** {'✅ Active' if mps_on else '❌ Inactive'}")

    ckpt_path = resolve_path("models/checkpoints/baseline/baseline_mobilenetv3_best.pt")
    smoke_ckpt = resolve_path("models/checkpoints/smoke_test/baseline_mobilenetv3_best.pt")
    
    st.sidebar.markdown("### Model Checkpoints")
    if ckpt_path.is_file():
        st.sidebar.success("✅ Real Baseline Checkpoint Available")
    else:
        st.sidebar.warning("⏳ Real Baseline Not Trained Yet")

    if smoke_ckpt.is_file():
        st.sidebar.info("ℹ️ Smoke Test Checkpoint Available")

    st.sidebar.divider()
    st.sidebar.info(
        "Scientific Rule: A classification model identifies whether damage resembles leaf mining, "
        "but does NOT prove insect species identity without stereomicroscopy or larvae."
    )

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset Explorer",
        "⚙️ Training Console",
        "📈 Evaluation & Metrics",
        "🔍 Single-Image Prediction",
    ])

    # ==========================================
    # TAB 1: DATASET EXPLORER
    # ==========================================
    with tab1:
        st.header("Tomato Leaf Dataset Explorer")
        st.markdown(
            "Inspect imported leaf photographs, verify 3-class distribution balance, and check group-aware splitting status."
        )

        df = load_dataset_metadata()
        raw_dir = resolve_path("data/raw")
        raw_images = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg"))
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <small>Total Ingested Images</small><br>
                <b style="font-size: 1.5rem;">{len(raw_images)}</b>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <small>Metadata Catalog Records</small><br>
                <b style="font-size: 1.5rem;">{len(df)}</b>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            farms = len(df["farm_id"].unique()) if not df.empty and "farm_id" in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <small>Independent Farms</small><br>
                <b style="font-size: 1.5rem;">{farms}</b>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            readiness = "READY FOR TRAINING" if (len(raw_images) >= 30 and len(df) >= 30) else "INSUFFICIENT DATA"
            badge_cls = "badge-green" if readiness == "READY FOR TRAINING" else "badge-coral"
            st.markdown(f"""
            <div class="metric-card">
                <small>Training Readiness</small><br>
                <span class="status-badge {badge_cls}">{readiness}</span>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Class Distribution Analysis")
        if not df.empty and "visible_damage_type" in df.columns:
            # Map into 3 classes
            mapped_classes = []
            for val in df["visible_damage_type"]:
                v = str(val).lower().strip()
                if v in {"none", "healthy"}:
                    mapped_classes.append("healthy")
                elif v in {"leaf_miner_mine", "leaf_miner_damage"}:
                    mapped_classes.append("leaf_miner_damage")
                else:
                    mapped_classes.append("other_damage")
            
            df_counts = pd.Series(mapped_classes).value_counts().reset_index()
            df_counts.columns = ["Target Class", "Count"]
            
            col_chart, col_tbl = st.columns([2, 1])
            with col_chart:
                st.bar_chart(df_counts.set_index("Target Class"))
            with col_tbl:
                st.dataframe(df_counts, use_container_width=True)
        else:
            st.warning("Metadata catalog is currently empty. Awaiting student field photograph ingestion.")

        st.divider()

        # Split Information
        st.subheader("Group-Aware Split Manifest Status")
        split_p = resolve_path("data/splits/splits.json")
        if split_p.is_file():
            with open(split_p, "r", encoding="utf-8") as f:
                s_manifest = json.load(f)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.info(f"**Training Set (70%):** {len(s_manifest.get('train', []))} images")
            with sc2:
                st.info(f"**Validation Set (15%):** {len(s_manifest.get('val', []))} images")
            with sc3:
                st.info(f"**Test Set (15%):** {len(s_manifest.get('test', []))} images")
        else:
            st.caption("No split manifest generated yet. Run `scripts/make_splits.py` once photographs are ingested.")

        # Ingestion Instructions Card
        st.subheader("How to Add Real Leaf Photographs")
        st.code("""
# 1. Place raw unedited camera photos in a staging folder (e.g., incoming_photos/)
# 2. Ingest with SHA-256 deduplication and EXIF auto-rotation:
python scripts/ingest_images.py --source_dir incoming_photos/ --collector_id COL_01 --farm_id FARM_01

# 3. Validate dataset integrity:
python scripts/validate_dataset.py

# 4. Generate group-aware train/val/test splits (zero leakage):
python scripts/make_splits.py --group_col farm_id --fallback_group_col plant_id
        """, language="bash")

    # ==========================================
    # TAB 2: TRAINING CONSOLE
    # ==========================================
    with tab2:
        st.header("MobileNetV3-Small Training Console")
        st.markdown(
            "Configure transfer learning hyperparameters, run forward/backward smoke tests, and initiate model training."
        )

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("### Architecture Specifications")
            st.markdown("- **Backbone:** Pretrained MobileNetV3-Small (ImageNet)")
            st.markdown("- **Pooling:** Global Average Pooling (GAP)")
            st.markdown("- **Regularization:** Hardswish non-linearity, Dropout = 0.2")
            st.markdown("- **Output Head:** Linear(1024, 3) $\\rightarrow$ [healthy, leaf_miner_damage, other_damage]")
            st.markdown("- **Loss Function:** PyTorch CrossEntropyLoss")

        with tc2:
            st.markdown("### Training Strategy & Schedules")
            st.markdown("- **Stage A (Head Warmup):** Freeze backbone; train classification head with AdamW (LR: 1e-3, 5 epochs)")
            st.markdown("- **Stage B (Fine-Tuning):** Unfreeze last 3 inverted residual blocks; train with Cosine Annealing (LR: 1e-4, 20 epochs)")
            st.markdown("- **Batch Size:** 16 (optimized for 16 GB Apple Silicon MPS)")
            st.markdown("- **Early Stopping:** Patience = 6 epochs based on validation Macro F1")

        st.divider()

        # Training Actions
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🧪 Run Training Smoke Test on MPS"):
                with st.spinner("Executing forward/backward pass, Stage A, Stage B, and checkpointing on Apple MPS..."):
                    smoke_res = run_baseline_training_smoke_test()
                    st.success("✅ Smoke Test Passed Successfully!")
                    st.json({
                        "device": smoke_res["device"],
                        "status": smoke_res["status"],
                        "best_val_macro_f1": smoke_res["best_val_macro_f1"],
                        "checkpoint": smoke_res["best_checkpoint"],
                    })

        with btn_col2:
            if st.button("🚀 Start Real Training Run"):
                meta_p = resolve_path("data/metadata/metadata.csv")
                split_p = resolve_path("data/splits/splits.json")
                if not meta_p.is_file() or not split_p.is_file():
                    st.error("❌ INSUFFICIENT_DATA: Real metadata or splits not found in data/raw. Please ingest field images first.")
                else:
                    with st.spinner("Training MobileNetV3-Small on real dataset..."):
                        res = train_baseline_pipeline()
                        st.success("Training complete!")
                        st.json(res)

        # Training Curves Display
        st.subheader("Training & Validation History")
        exp_dir = resolve_path("experiments")
        curve_images = list(exp_dir.glob("**/loss_and_validation_curves.png"))
        if curve_images:
            latest_curve = sorted(curve_images, key=lambda p: p.stat().st_mtime)[-1]
            st.image(str(latest_curve), caption=f"Learning Curves: {latest_curve.parent.name}", use_container_width=True)
        else:
            st.caption("No training history curves recorded yet.")

    # ==========================================
    # TAB 3: EVALUATION & BENCHMARKS
    # ==========================================
    with tab3:
        st.header("Scientific Evaluation & Held-Out Benchmark")
        st.markdown(
            "Independent evaluation on the frozen, held-out test split (15%). "
            "PhytoTrace maintains strict scientific honesty: un-evaluated benchmarks display `NOT EVALUATED`."
        )

        eval_report_p = resolve_path("reports/evaluation_report.json")
        if eval_report_p.is_file():
            with open(eval_report_p, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            
            base_eval = eval_data.get("baseline_classifier", {})
            eval_status = base_eval.get("status", "NOT_EVALUATED")

            if eval_status == "EVALUATED":
                st.success(f"✅ Baseline Classifier Evaluated on {base_eval.get('test_samples')} Test Images")
                ec1, ec2, ec3, ec4 = st.columns(4)
                with ec1:
                    st.metric("Test Accuracy", f"{base_eval.get('accuracy', 0.0)*100:.1f}%")
                with ec2:
                    st.metric("Macro F1", f"{base_eval.get('macro_f1', 0.0):.4f}")
                with ec3:
                    st.metric("Avg Latency", f"{base_eval.get('avg_inference_time_per_image_ms', 0)} ms")
                with ec4:
                    st.metric("Misclassifications", base_eval.get("misclassified_count", 0))

                st.subheader("Per-Class Performance Metrics")
                per_class = base_eval.get("per_class", {})
                if per_class:
                    df_pc = pd.DataFrame.from_dict(per_class, orient="index")
                    st.dataframe(df_pc, use_container_width=True)

                st.subheader("Confusion Matrix")
                cm = base_eval.get("confusion_matrix")
                if cm:
                    df_cm = pd.DataFrame(
                        cm,
                        index=["True Healthy", "True Leaf Miner", "True Other"],
                        columns=["Pred Healthy", "Pred Leaf Miner", "Pred Other"],
                    )
                    st.dataframe(df_cm, use_container_width=True)

                st.subheader("Misclassified Example Gallery")
                gallery = base_eval.get("misclassified_gallery", [])
                if gallery:
                    st.dataframe(pd.DataFrame(gallery), use_container_width=True)
                else:
                    st.caption("No misclassified samples in held-out test set.")
            else:
                st.warning(f"⚠️ Status: {eval_status}")
                st.info(f"Reason: {base_eval.get('reason')}")
        else:
            st.info("Evaluation report not yet generated. Run `python scripts/evaluate.py`.")

    # ==========================================
    # TAB 4: SINGLE-IMAGE PREDICTION
    # ==========================================
    with tab4:
        st.header("Single-Image Prediction & Class Breakdown")
        st.markdown(
            "Upload any tomato leaf photograph to inspect MobileNetV3-Small class probabilities, "
            "photometric quality checks, and calibrated uncertainty."
        )

        # Select which checkpoint to test
        avail_ckpts = []
        if ckpt_path.is_file():
            avail_ckpts.append(str(ckpt_path))
        if smoke_ckpt.is_file():
            avail_ckpts.append(str(smoke_ckpt))

        selected_ckpt = None
        if avail_ckpts:
            selected_ckpt = st.selectbox("Select Model Checkpoint:", avail_ckpts)
        else:
            st.warning("⚠️ No model checkpoints exist yet. Run a training smoke test or train with real data.")

        uploaded = st.file_uploader("Upload Tomato Leaf Photograph:", type=["jpg", "jpeg", "png", "webp"])

        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                pred_col1, pred_col2 = st.columns([1, 1])
                with pred_col1:
                    st.subheader("Uploaded Photograph")
                    st.image(img_rgb, use_container_width=True)

                with pred_col2:
                    st.subheader("Prediction Analysis")
                    if selected_ckpt and Path(selected_ckpt).is_file():
                        # Quality Check
                        qc = assess_image_quality(img_bgr)
                        if not qc["is_acceptable"]:
                            st.warning(f"⚠️ Quality Warning: {', '.join(qc['reasons'])}")

                        # Load model and predict
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

                        # Display Result Card
                        st.markdown(f"""
                        <div class="metric-card">
                            <small>Predicted Category</small><br>
                            <span class="status-badge badge-green" style="font-size: 1.1rem;">{pred_class.upper()}</span>
                            <br><small>Confidence: <b>{pred_conf*100:.1f}%</b></small>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display Class Probabilities
                        st.markdown("#### Class Score Breakdown:")
                        for i, name in enumerate(classes):
                            score = float(probs[i])
                            st.write(f"**{name.replace('_', ' ').title()}**: {score*100:.1f}%")
                            st.progress(score)

                        st.divider()
                        st.caption(
                            "🔬 Scientific Caveat: High confidence for 'leaf_miner_damage' confirms visible mining injury, "
                            "but does NOT determine whether Tuta or Liriomyza caused it. Subsequent segmentation and "
                            "stereomicroscopy are required for confirmed species diagnosis."
                        )
                    else:
                        st.error("Model weights not available. Train the model to see predictions.")


if __name__ == "__main__":
    main()
