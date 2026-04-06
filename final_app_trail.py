import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from pathlib import Path
import os

# --------------- Page Config ---------------
st.set_page_config(
    page_title="Vastra Viveka | Smart Dress Code Analyzer",
    page_icon="logo.png" if os.path.exists("logo.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- Session State Init ---------------
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

# --------------- Custom CSS ---------------
st.markdown("""
<style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* ---------- Global ---------- */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: #f8f9fc;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1040 40%, #2d1b69 70%, #1a1040 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    section[data-testid="stSidebar"] * {
        color: #c4b5fd !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e9e5ff !important;
    }
    .sidebar-brand {
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .sidebar-brand h2 {
        background: linear-gradient(135deg, #a78bfa, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin: 0 !important;
    }
    .sidebar-brand p {
        color: #7c72a0 !important;
        font-size: 0.72rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin: 0.1rem 0 0 0 !important;
    }
    .sidebar-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
        margin: 1rem 0;
    }

    /* ---------- Main Header ---------- */
    .hero-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a855f7 70%, #7c3aed 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25), 0 2px 8px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
    }
    .hero-header::after {
        content: '';
        position: absolute;
        bottom: -60%;
        left: -10%;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: rgba(255,255,255,0.03);
    }
    .hero-header h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero-header p {
        color: rgba(255,255,255,0.8) !important;
        margin: 0.4rem 0 0 0 !important;
        font-size: 0.95rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* ---------- Metric Cards ---------- */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.3rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
        border-color: #c7d2fe;
    }
    .metric-card h3 {
        margin: 0 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0 !important;
        color: #6b7280 !important;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    /* ---------- Badges ---------- */
    .badge-positive {
        display: inline-block;
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        margin: 3px;
        border: 1px solid #6ee7b7;
    }
    .badge-negative {
        display: inline-block;
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        margin: 3px;
        border: 1px solid #fca5a5;
    }

    /* ---------- Info / Tip Box ---------- */
    .tip-box {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-left: 4px solid #6366f1;
        padding: 1rem 1.4rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .tip-box p {
        margin: 0 !important;
        color: #312e81 !important;
        font-size: 0.9rem;
    }

    /* ---------- Webcam Placeholder ---------- */
    .cam-placeholder {
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .cam-placeholder .cam-icon {
        font-size: 3.5rem;
        margin-bottom: 0.8rem;
        opacity: 0.5;
    }
    .cam-placeholder h3 {
        color: #475569 !important;
        font-weight: 600 !important;
        margin: 0 0 0.3rem 0 !important;
    }
    .cam-placeholder p {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        margin: 0 !important;
    }

    /* ---------- Upload Placeholder ---------- */
    .upload-placeholder {
        background: linear-gradient(135deg, #faf5ff, #f3e8ff);
        border: 2px dashed #d8b4fe;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-placeholder .up-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
        opacity: 0.6;
    }
    .upload-placeholder h3 {
        color: #6b21a8 !important;
        font-weight: 600 !important;
        margin: 0 0 0.3rem 0 !important;
    }
    .upload-placeholder p {
        color: #a78bfa !important;
        font-size: 0.88rem;
        margin: 0 !important;
    }

    /* ---------- Section Label ---------- */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }

    /* ---------- Results Panel ---------- */
    .results-panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .results-panel h3 {
        color: #1f2937 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin: 0 0 1rem 0 !important;
        padding-bottom: 0.7rem;
        border-bottom: 2px solid #f3f4f6;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Plus Jakarta Sans', sans-serif;
        padding: 0.55rem 1.8rem;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 8px 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.2rem;
    }

    /* ---------- Slider ---------- */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #8b5cf6;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        padding: 1.8rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }
    .app-footer p {
        color: #9ca3af !important;
        font-size: 0.78rem;
        margin: 0 !important;
        letter-spacing: 0.3px;
    }

    /* ---------- Hide Streamlit chrome ---------- */
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
        "description": "Detects whether a person is wearing a face mask. Useful for health & safety compliance monitoring.",
        "positive_label": "mask",
    },
    "Formal vs Informal": {
        "file": "formal_best.pt",
        "icon": "briefcase",
        "emoji": "🧥",
        "subtitle": "Formal & Informal Classifier",
        "description": "Classifies attire as formal or informal. Great for workplace dress code enforcement.",
        "positive_label": "formal",
    },
    "Traditional vs Non-Traditional": {
        "file": "trad_best.pt",
        "icon": "palette",
        "emoji": "👕",
        "subtitle": "Traditional & Non-Traditional Classifier",
        "description": "Identifies traditional vs non-traditional clothing styles for cultural event monitoring.",
        "positive_label": "traditional",
    },
    "Helmet Detector": {
        "file": "helmet_best.pt",
        "icon": "hard-hat",
        "emoji": "🪖",
        "subtitle": "Helmet & No-Helmet Classifier",
        "description": "Detects helmet usage for construction site and road safety compliance.",
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
    st.markdown(
        '<div class="sidebar-brand">'
        "<h2>Vastra Viveka</h2>"
        "<p>Smart Dress Code AI</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<p class="section-label">Detector Mode</p>', unsafe_allow_html=True)
    option = option_menu(
        menu_title=None,
        options=list(MODELS.keys()),
        icons=[m["icon"] for m in MODELS.values()],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0 !important", "background-color": "transparent"},
            "icon": {"color": "#a78bfa", "font-size": "17px"},
            "nav-link": {
                "font-size": "0.85rem",
                "text-align": "left",
                "margin": "3px 0",
                "padding": "10px 14px",
                "border-radius": "10px",
                "color": "#c4b5fd",
                "--hover-color": "rgba(139, 92, 246, 0.12)",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                "color": "white",
                "font-weight": "600",
                "box-shadow": "0 4px 15px rgba(99, 102, 241, 0.35)",
            },
        },
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Settings</p>', unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "Display Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Show detections above this confidence. The model always runs at low conf internally for best accuracy.",
    )

    show_labels = st.toggle("Show Labels", value=True, help="Show/hide class labels on bounding boxes")
    box_thickness = st.select_slider("Box Thickness", options=[1, 2, 3, 4], value=2)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:0.7rem; color:#6b5b95 !important;'>"
        "YOLOv11 + Streamlit<br>Vastra Viveka v2.0</p>",
        unsafe_allow_html=True,
    )


# --------------- Main Content ---------------
cfg = MODELS[option]
model = load_model(cfg["file"])

# Hero Header
st.markdown(
    f"""<div class="hero-header">
        <h1>{cfg['emoji']}  {cfg['subtitle']}</h1>
        <p>{cfg['description']}</p>
    </div>""",
    unsafe_allow_html=True,
)


# --------------- Detection Engine ---------------
def run_detection(image_bgr, display_conf):
    """
    Run YOLO with a very low internal confidence (0.05) so the model returns
    all possible detections. Then filter by the user's display_conf threshold.
    This gives much better accuracy than passing a high conf directly to YOLO.
    """
    # Run inference at low conf to capture everything
    results = model(image_bgr, conf=0.05, iou=0.45, verbose=False)
    detections = []
    annotated = image_bgr.copy()

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            # Post-filter by user threshold
            if conf < display_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label_name = model.names[cls]

            detections.append({"label": label_name, "conf": conf, "cls": cls, "bbox": (x1, y1, x2, y2)})

            is_positive = cls == 0
            color = (34, 197, 94) if is_positive else (59, 130, 246)  # green / blue
            neg_color = (239, 68, 68)  # red for dangerous classes
            draw_color = color if is_positive else neg_color

            # BGR for OpenCV
            bgr = (draw_color[2], draw_color[1], draw_color[0])

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, box_thickness)

            # Corner accents for a modern look
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), bgr, box_thickness + 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), bgr, box_thickness + 2)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), bgr, box_thickness + 2)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), bgr, box_thickness + 2)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), bgr, box_thickness + 2)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), bgr, box_thickness + 2)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), bgr, box_thickness + 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), bgr, box_thickness + 2)

            # Label with background pill
            if show_labels:
                text = f"{label_name} {conf:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.55
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                pill_y = max(y1 - th - 14, 0)
                cv2.rectangle(annotated, (x1, pill_y), (x1 + tw + 12, pill_y + th + 12), bgr, -1)
                cv2.putText(annotated, text, (x1 + 6, pill_y + th + 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return image_rgb, detections


def show_detection_summary(detections):
    """Display detection stats as styled metric cards and badges."""
    if not detections:
        st.markdown(
            '<div class="tip-box"><p>No objects detected. Try <strong>lowering</strong> the confidence threshold in the sidebar.</p></div>',
            unsafe_allow_html=True,
        )
        return

    # Count per class
    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    # Metric cards
    card_cols = st.columns(min(len(counts) + 1, 4))
    with card_cols[0]:
        st.markdown(
            f'<div class="metric-card"><h3>{len(detections)}</h3><p>Total Found</p></div>',
            unsafe_allow_html=True,
        )
    for i, (label, count) in enumerate(counts.items()):
        if i + 1 < len(card_cols):
            with card_cols[i + 1]:
                st.markdown(
                    f'<div class="metric-card"><h3>{count}</h3><p>{label}</p></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")

    # Confidence badges
    badge_html = '<div style="margin-top: 0.5rem;">'
    positive_labels = {cfg["positive_label"], "mask", "formal", "traditional", "helmet"}
    for d in sorted(detections, key=lambda x: -x["conf"]):
        badge_type = "positive" if d["label"].lower() in positive_labels else "negative"
        badge_html += f'<span class="badge-{badge_type}">{d["label"]}  {d["conf"]:.0%}</span> '
    badge_html += "</div>"
    st.markdown(badge_html, unsafe_allow_html=True)

    # Average confidence
    avg_conf = sum(d["conf"] for d in detections) / len(detections)
    st.caption(f"Average confidence: {avg_conf:.1%}")


# --------------- Input Mode Tabs ---------------
tab_upload, tab_webcam = st.tabs(["📁  Upload Image", "📷  Webcam (Live)"])

# ==================== UPLOAD TAB ====================
with tab_upload:
    uploaded_file = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supports JPG, PNG, BMP, WEBP",
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is not None:
            with st.spinner("Analyzing image..."):
                annotated_rgb, detections = run_detection(image_bgr, confidence_threshold)

            col_img, col_stats = st.columns([2.5, 1])
            with col_img:
                st.image(annotated_rgb, caption=f"Detection Result  |  {len(detections)} object(s) found", use_container_width=True)
            with col_stats:
                st.markdown('<div class="results-panel"><h3>Detection Results</h3>', unsafe_allow_html=True)
                show_detection_summary(detections)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Could not decode the uploaded image. Please try a different file.")
    else:
        st.markdown(
            '<div class="upload-placeholder">'
            '<div class="up-icon">📤</div>'
            "<h3>Upload an Image</h3>"
            "<p>Drag and drop or click above to select a file</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ==================== WEBCAM TAB ====================
with tab_webcam:
    st.markdown(
        '<div class="tip-box"><p>'
        "<strong>How it works:</strong> Click Start Camera to begin live detection. "
        "YOLO runs on each frame in real-time. Adjust the confidence threshold in the sidebar to tune sensitivity."
        "</p></div>",
        unsafe_allow_html=True,
    )

    col_start, col_stop, col_space = st.columns([1, 1, 4])
    with col_start:
        start_clicked = st.button("▶  Start Camera", use_container_width=True, type="primary")
    with col_stop:
        stop_clicked = st.button("⏹  Stop Camera", use_container_width=True)

    if start_clicked:
        st.session_state.webcam_running = True
    if stop_clicked:
        st.session_state.webcam_running = False

    # --- Camera is OFF: show placeholder ---
    if not st.session_state.webcam_running:
        st.markdown(
            '<div class="cam-placeholder">'
            '<div class="cam-icon">🎥</div>'
            "<h3>Camera is Off</h3>"
            "<p>Press <strong>Start Camera</strong> to begin live detection</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # --- Camera is ON ---
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam. Check camera permissions or close other apps using it.")
            st.session_state.webcam_running = False
        else:
            video_placeholder = st.empty()
            summary_placeholder = st.empty()

            while cap.isOpened() and st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost webcam feed.")
                    break

                annotated_rgb, detections = run_detection(frame, confidence_threshold)
                video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                if detections:
                    counts = {}
                    for d in detections:
                        counts[d["label"]] = counts.get(d["label"], 0) + 1
                    parts = []
                    for label, count in counts.items():
                        parts.append(f"**{label}**: {count}")
                    summary_placeholder.markdown(f"🔍  {' &nbsp;|&nbsp; '.join(parts)}  &nbsp;&nbsp; _({len(detections)} total)_")
                else:
                    summary_placeholder.markdown("_Scanning... no detections in current frame_")

            cap.release()
            st.info("Camera stopped.")


# --------------- Footer ---------------
st.markdown(
    '<div class="app-footer">'
    "<p>Vastra Viveka &mdash; Smart Dress Code Analyzer &nbsp;&bull;&nbsp; "
    "Powered by YOLOv11 &amp; Streamlit &nbsp;&bull;&nbsp; v2.0</p>"
    "</div>",
    unsafe_allow_html=True,
)
