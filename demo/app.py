"""PhytoTrace AI — Streamlit Demonstration Interface."""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from phytotrace.device import get_device
from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.inference.visualization import create_side_by_side_comparison

# Streamlit Page Config
st.set_page_config(
    page_title="PhytoTrace AI — Explainable Leaf-Miner Diagnostic Engine",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for PhytoTrace styling (Forest green, mint, crisp cards)
st.markdown("""
<style>
    .main {
        background-color: #f7faf8;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
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
    .badge-cyan  { background-color: #e0f7fa; color: #00838f; }
    .stButton>button {
        background-color: #1b5e20;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return InferencePipeline()


def main():
    st.sidebar.title("🌿 PhytoTrace AI")
    st.sidebar.markdown("**Explainable Mixed Leaf-Miner Diagnostic Engine**")
    st.sidebar.caption("Target: *Tuta absoluta* & *Liriomyza* spp. in Tomato")
    st.sidebar.divider()

    # System Status in Sidebar
    device = get_device("auto")
    st.sidebar.markdown(f"**Compute Device:** `{device.type.upper()}`")
    
    pipeline = load_pipeline()
    
    st.sidebar.markdown("### Model Component Status")
    b_stat = "✅ Trained" if pipeline.baseline_model is not None else "⏳ Awaiting Weights"
    l_stat = "✅ UNet Active" if pipeline.leaf_segmenter.is_trained else "ℹ️ Heuristic Mode"
    m_stat = "✅ YOLO Active" if pipeline.mine_segmenter.is_trained else "⏳ Awaiting Annotations"
    s_stat = "✅ Active" if pipeline.species_wrapper.is_trained else "🔒 DATA_NOT_READY (Needs Lab Labels)"

    st.sidebar.markdown(f"- **Baseline Classifier:** {b_stat}")
    st.sidebar.markdown(f"- **Leaf Segmenter:** {l_stat}")
    st.sidebar.markdown(f"- **Mine Localization:** {m_stat}")
    st.sidebar.markdown(f"- **Pest Evidence (Tuta/Liriomyza):** {s_stat}")

    st.sidebar.divider()
    st.sidebar.info(
        "Scientific Rule: Mine symptoms alone do NOT guarantee pest species identity. "
        "PhytoTrace marks species uncertain unless verified evidence exists."
    )

    # Main Area
    st.title("PhytoTrace — Field Diagnostic Inspector")
    st.markdown(
        "Upload a tomato leaf photograph to run automated quality control, canopy segmentation, "
        "mine damage localization, and uncertainty-calibrated diagnosis."
    )

    uploaded_file = st.file_uploader(
        "Select tomato leaf photograph (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Failed to decode uploaded image. Please check file format.")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Original Image Preview")
            st.image(img_rgb, use_container_width=True)

        with st.spinner("Executing multi-stage analysis pipeline..."):
            result = pipeline.predict(img_bgr, image_id=Path(uploaded_file.name).stem)

        with col2:
            st.subheader("Diagnostic Overlay")
            annotated_bgr = result.get("_annotated_image")
            if annotated_bgr is not None:
                st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.warning("Overlay unavailable due to quality rejection.")

        st.divider()

        # Results Dashboard
        st.subheader("Diagnostic Assessment Summary")
        m1, m2, m3, m4 = st.columns(4)

        decision = result["decision"]
        if decision in {"no_visible_mine", "healthy"}:
            badge_class = "badge-green"
        elif decision in {"possible_tuta", "possible_liriomyza", "possible_mixed"}:
            badge_class = "badge-amber"
        else:
            badge_class = "badge-coral"

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <small>Decision Outcome</small><br>
                <span class="status-badge {badge_class}">{decision.replace('_', ' ').upper()}</span>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <small>Uncertainty Status</small><br>
                <b>{result['uncertainty_status'].replace('_', ' ').title()}</b>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            dmg_val = f"{result['damage_percentage']}%" if result['damage_percentage'] is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <small>Visible Damage Percentage</small><br>
                <b style="font-size: 1.3rem;">{dmg_val}</b>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <small>Image Quality</small><br>
                <b>{result['quality_status'].replace('_', ' ').title()}</b>
            </div>
            """, unsafe_allow_html=True)

        # Scientific Explanation & Safety Notice
        st.markdown(f"**Explanation:** {result['explanation']}")
        st.warning(result["safety_notice"])

        if result.get("limitations"):
            st.caption(f"**Model Limitations / Warnings:** {', '.join(result['limitations'])}")

        # Download Report
        st.divider()
        report_export = {k: v for k, v in result.items() if k != "_annotated_image"}
        json_str = json.dumps(report_export, indent=2)
        st.download_button(
            label="📥 Download Diagnostic JSON Report",
            data=json_str,
            file_name=f"phytotrace_report_{result['image_id']}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()

