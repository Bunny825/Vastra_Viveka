import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from pathlib import Path
from PIL import Image
import os
import tempfile

# --------------- Page Config ---------------
st.set_page_config(
    page_title="Vastra Viveka | Smart Dress Code Analyzer",
    page_icon="logo.png" if os.path.exists("logo.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- Custom CSS ---------------
st.markdown("""
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }

    /* ---------- Header ---------- */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85) !important;
        margin: 0.3rem 0 0 0 !important;
        font-size: 1rem;
    }

    /* ---------- Cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card h3 {
        margin: 0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #302b63 !important;
    }
    .metric-card p {
        margin: 0.2rem 0 0 0 !important;
        color: #555 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ---------- Status badges ---------- */
    .badge-positive {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px;
    }
    .badge-negative {
        display: inline-block;
        background: #f8d7da;
        color: #721c24;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px;
    }

    /* ---------- Info box ---------- */
    .info-box {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .info-box p {
        margin: 0 !important;
        color: #333 !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
        color: #999;
        font-size: 0.8rem;
    }

    /* ---------- Button styling ---------- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.2s;
    }

    /* ---------- Slider ---------- */
    .stSlider > div > div {
        color: #667eea;
    }

    /* ---------- Upload area ---------- */
    .stFileUploader > div {
        border-radius: 12px;
    }

    /* ---------- Hide Streamlit branding ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --------------- Model Config ---------------
MODELS = {
    "Mask Detector": {
        "file": "mask_best.pt",
        "icon": "shield-check",
        "emoji": "😷",
        "subtitle": "Mask & No-Mask Classifier",
        "description": "Detects whether a person is wearing a face mask or not. Useful for health & safety compliance.",
        "positive_label": "mask",
    },
    "Formal vs Informal": {
        "file": "formal_best.pt",
        "icon": "briefcase",
        "emoji": "🧥",
        "subtitle": "Formal & Informal Classifier",
        "description": "Classifies attire as formal or informal. Great for workplace dress code monitoring.",
        "positive_label": "formal",
    },
    "Traditional vs Non-Traditional": {
        "file": "trad_best.pt",
        "icon": "palette",
        "emoji": "👕",
        "subtitle": "Traditional & Non-Traditional Classifier",
        "description": "Identifies traditional vs non-traditional clothing styles.",
        "positive_label": "traditional",
    },
    "Helmet Detector": {
        "file": "helmet_best.pt",
        "icon": "hard-hat",
        "emoji": "🪖",
        "subtitle": "Helmet & No-Helmet Classifier",
        "description": "Detects helmet usage for construction or road safety compliance.",
        "positive_label": "helmet",
    },
}


# --------------- Cache Model Loading ---------------
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model with caching to avoid reloading on every rerun."""
    base = Path(__file__).parent
    return YOLO(str(base / model_path))


# --------------- Sidebar ---------------
with st.sidebar:
    st.markdown("## Vastra Viveka")
    st.markdown("---")
    st.markdown("#### Select Detector")
    option = option_menu(
        menu_title=None,
        options=list(MODELS.keys()),
        icons=[m["icon"] for m in MODELS.values()],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0 !important", "background-color": "transparent"},
            "icon": {"color": "#a78bfa", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "10px 14px",
                "border-radius": "8px",
                "color": "#e0e0e0",
                "--hover-color": "rgba(255,255,255,0.08)",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "color": "white",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.markdown("#### Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Only show detections above this confidence level",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#888; font-size:0.75rem; text-align:center;'>"
        "Built with YOLOv11 + Streamlit</p>",
        unsafe_allow_html=True,
    )


# --------------- Main Content ---------------
cfg = MODELS[option]
model = load_model(cfg["file"])

# Header
st.markdown(
    f"""<div class="main-header">
        <h1>{cfg['emoji']} {cfg['subtitle']}</h1>
        <p>{cfg['description']}</p>
    </div>""",
    unsafe_allow_html=True,
)

# --------------- Detection helper ---------------
def run_detection(image_bgr, conf_thresh):
    """Run YOLO on an image and return annotated image + detection stats."""
    results = model(image_bgr, conf=conf_thresh, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label_name = model.names[cls]

            detections.append({"label": label_name, "conf": conf, "bbox": (x1, y1, x2, y2)})

            is_positive = cls == 0
            color = (0, 200, 80) if is_positive else (0, 40, 230)

            # Draw box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)

            # Label background
            text = f"{label_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_bgr, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(image_bgr, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, detections


def show_detection_summary(detections):
    """Display detection stats as styled metric cards."""
    if not detections:
        st.markdown(
            '<div class="info-box"><p>No detections found. Try lowering the confidence threshold.</p></div>',
            unsafe_allow_html=True,
        )
        return

    # Count per class
    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    cols = st.columns(len(counts) + 1)
    with cols[0]:
        st.markdown(
            f'<div class="metric-card"><h3>{len(detections)}</h3><p>Total Detections</p></div>',
            unsafe_allow_html=True,
        )
    for i, (label, count) in enumerate(counts.items(), 1):
        with cols[i]:
            st.markdown(
                f'<div class="metric-card"><h3>{count}</h3><p>{label}</p></div>',
                unsafe_allow_html=True,
            )

    # Badges
    st.markdown("")
    badge_html = ""
    for d in detections:
        cls_type = "positive" if d["label"].lower() in (cfg["positive_label"], "mask", "formal", "traditional", "helmet") else "negative"
        badge_html += f'<span class="badge-{cls_type}">{d["label"]} ({d["conf"]:.0%})</span> '
    st.markdown(badge_html, unsafe_allow_html=True)


# --------------- Input Mode Tabs ---------------
tab_upload, tab_webcam = st.tabs(["📁  Upload Image", "📷  Webcam (Live)"])

# --- Upload Tab ---
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supports JPG, PNG, BMP, WEBP formats",
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is not None:
            with st.spinner("Running detection..."):
                annotated_rgb, detections = run_detection(image_bgr, confidence_threshold)

            # Results layout
            col_img, col_stats = st.columns([3, 1])
            with col_img:
                st.image(annotated_rgb, caption="Detection Result", use_container_width=True)
            with col_stats:
                st.markdown("### Results")
                show_detection_summary(detections)
        else:
            st.error("Could not decode the uploaded image. Please try a different file.")

# --- Webcam Tab ---
with tab_webcam:
    st.markdown(
        '<div class="info-box"><p><strong>Tip:</strong> '
        "Make sure your browser has camera permissions enabled. "
        "The webcam feed runs YOLO inference in real-time on each frame.</p></div>",
        unsafe_allow_html=True,
    )

    # State
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    col_start, col_stop, _ = st.columns([1, 1, 4])
    with col_start:
        if st.button("▶  Start Camera", use_container_width=True, type="primary"):
            st.session_state.webcam_running = True
            st.rerun()
    with col_stop:
        if st.button("⏹  Stop Camera", use_container_width=True):
            st.session_state.webcam_running = False
            st.rerun()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam. Check your camera permissions and make sure no other app is using the camera.")
            st.session_state.webcam_running = False
        else:
            video_placeholder = st.empty()
            summary_placeholder = st.empty()

            while cap.isOpened() and st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from webcam.")
                    break

                annotated_rgb, detections = run_detection(frame, confidence_threshold)
                video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                # Live detection count overlay
                if detections:
                    counts = {}
                    for d in detections:
                        counts[d["label"]] = counts.get(d["label"], 0) + 1
                    summary_text = " | ".join(f"**{label}**: {count}" for label, count in counts.items())
                    summary_placeholder.markdown(f"Detected: {summary_text}")
                else:
                    summary_placeholder.markdown("*No detections in current frame*")

            cap.release()
            st.info("Camera stopped.")


# --------------- Footer ---------------
st.markdown(
    '<div class="footer">Vastra Viveka &mdash; Smart Dress Code Analyzer &nbsp;|&nbsp; '
    "Powered by YOLOv11 &amp; Streamlit</div>",
    unsafe_allow_html=True,
)
