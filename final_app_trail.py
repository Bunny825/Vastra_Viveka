import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from pathlib import Path
from PIL import Image
from datetime import datetime
from io import BytesIO
import os
import json
import time
import base64
import struct
import wave

# ================================================================
#                         PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Vastra Viveka | Smart Dress Code Analyzer",
    page_icon="logo.png" if os.path.exists("logo.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
#                       SESSION STATE INIT
# ================================================================
defaults = {
    "webcam_running": False,
    "detection_history": [],       # list of {timestamp, model, detections}
    "history_chart_data": [],      # for the live chart
    "alert_active": False,
    "selected_models": ["Mask Detector"],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ================================================================
#                         CUSTOM CSS
# ================================================================
st.markdown("""
<style>
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
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #9ca3af !important;
        margin-bottom: 0.5rem;
    }

    /* ---------- Hero Header ---------- */
    .hero-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a855f7 70%, #7c3aed 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 300px; height: 300px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
    }
    .hero-header h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        position: relative; z-index: 1;
    }
    .hero-header p {
        color: rgba(255,255,255,0.8) !important;
        margin: 0.4rem 0 0 0 !important;
        font-size: 0.95rem;
        position: relative; z-index: 1;
    }

    /* ---------- Metric Cards ---------- */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.3rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
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

    /* ---------- Tip Box ---------- */
    .tip-box {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-left: 4px solid #6366f1;
        padding: 1rem 1.4rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .tip-box p { margin: 0 !important; color: #312e81 !important; font-size: 0.9rem; }

    /* ---------- Alert Banner ---------- */
    .alert-banner {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        animation: pulse-alert 1.5s ease-in-out infinite;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4);
    }
    .alert-banner h4 { color: white !important; margin: 0 !important; font-size: 1.1rem !important; }
    .alert-banner p { color: rgba(255,255,255,0.9) !important; margin: 0.3rem 0 0 0 !important; font-size: 0.9rem; }
    @keyframes pulse-alert {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    /* ---------- Placeholders ---------- */
    .cam-placeholder {
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 4rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .cam-placeholder .cam-icon { font-size: 3.5rem; margin-bottom: 0.8rem; opacity: 0.5; }
    .cam-placeholder h3 { color: #475569 !important; font-weight: 600 !important; margin: 0 0 0.3rem 0 !important; }
    .cam-placeholder p { color: #94a3b8 !important; font-size: 0.9rem; margin: 0 !important; }

    .upload-placeholder {
        background: linear-gradient(135deg, #faf5ff, #f3e8ff);
        border: 2px dashed #d8b4fe;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-placeholder .up-icon { font-size: 3rem; margin-bottom: 0.8rem; opacity: 0.6; }
    .upload-placeholder h3 { color: #6b21a8 !important; font-weight: 600 !important; margin: 0 0 0.3rem 0 !important; }
    .upload-placeholder p { color: #a78bfa !important; font-size: 0.88rem; margin: 0 !important; }

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

    /* ---------- History Card ---------- */
    .history-entry {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .history-entry .he-time { color: #9ca3af; font-size: 0.8rem; }
    .history-entry .he-model { color: #6366f1; font-weight: 600; font-size: 0.85rem; }
    .history-entry .he-count { color: #1f2937; font-weight: 700; font-size: 1.1rem; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Plus Jakarta Sans', sans-serif;
        padding: 0.55rem 1.8rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 12px;
        padding: 6px;
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        padding: 1.8rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }
    .app-footer p { color: #9ca3af !important; font-size: 0.78rem; margin: 0 !important; }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================================================================
#                       MODEL CONFIG
# ================================================================
MODELS = {
    "Mask Detector": {
        "file": "mask_best.pt",
        "icon": "shield-check",
        "emoji": "😷",
        "subtitle": "Mask & No-Mask Classifier",
        "description": "Detects whether a person is wearing a face mask. Health & safety compliance.",
        "positive_label": "mask",
        "violation_cls": 1,   # class index that represents a violation
        "violation_name": "No Mask",
    },
    "Formal vs Informal": {
        "file": "formal_best.pt",
        "icon": "briefcase",
        "emoji": "🧥",
        "subtitle": "Formal & Informal Classifier",
        "description": "Classifies attire as formal or informal. Workplace dress code enforcement.",
        "positive_label": "formal",
        "violation_cls": 1,
        "violation_name": "Informal Attire",
    },
    "Traditional vs Non-Traditional": {
        "file": "trad_best.pt",
        "icon": "palette",
        "emoji": "👕",
        "subtitle": "Traditional & Non-Traditional Classifier",
        "description": "Identifies traditional vs non-traditional clothing styles.",
        "positive_label": "traditional",
        "violation_cls": 1,
        "violation_name": "Non-Traditional",
    },
    "Helmet Detector": {
        "file": "helmet_best.pt",
        "icon": "cone-striped",
        "emoji": "⛑",
        "subtitle": "Helmet & No-Helmet Classifier",
        "description": "Detects helmet usage for construction and road safety compliance.",
        "positive_label": "helmet",
        "violation_cls": 1,
        "violation_name": "No Helmet",
    },
}

POSITIVE_LABELS = {"mask", "formal", "traditional", "helmet"}

# Color palette for multi-model - each model gets a distinct color (BGR)
MODEL_COLORS_BGR = [
    (234, 102, 99),   # indigo-ish
    (246, 92, 139),   # purple-ish
    (241, 168, 88),   # amber
    (94, 197, 34),    # green
]


# ================================================================
#                     CACHED MODEL LOADING
# ================================================================
@st.cache_resource
def load_model(model_path: str):
    base = Path(__file__).parent
    return YOLO(str(base / model_path))


# ================================================================
#                          SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        "<h2>Vastra Viveka</h2>"
        "<p>Smart Dress Code AI</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # --- Multi-Model Selection ---
    st.markdown('<p class="section-label">Detectors</p>', unsafe_allow_html=True)

    multi_mode = st.toggle("Multi-Model Mode", value=False, help="Run multiple detectors on the same image/frame")

    if multi_mode:
        selected_models = []
        for name in MODELS:
            if st.checkbox(f"{MODELS[name]['emoji']} {name}", value=(name == "Mask Detector"), key=f"chk_{name}"):
                selected_models.append(name)
        if not selected_models:
            selected_models = ["Mask Detector"]
        st.session_state.selected_models = selected_models
    else:
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
        st.session_state.selected_models = [option]

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Settings</p>', unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "Display Confidence",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Show detections above this confidence. Model runs at low conf internally for best accuracy.",
    )
    show_labels = st.toggle("Show Labels", value=True)
    box_thickness = st.select_slider("Box Thickness", options=[1, 2, 3, 4], value=2)
    enable_alerts = st.toggle("Violation Alerts", value=True, help="Show warning banner when violations detected")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:0.7rem; color:#6b5b95 !important;'>"
        "YOLOv11 + Streamlit<br>Vastra Viveka v3.0</p>",
        unsafe_allow_html=True,
    )


# ================================================================
#                      LOAD SELECTED MODELS
# ================================================================
active_models = {}
for model_name in st.session_state.selected_models:
    cfg = MODELS[model_name]
    active_models[model_name] = {
        "model": load_model(cfg["file"]),
        "config": cfg,
    }

# Hero Header - show active models
if len(active_models) == 1:
    name = list(active_models.keys())[0]
    cfg = MODELS[name]
    hero_title = f"{cfg['emoji']}  {cfg['subtitle']}"
    hero_desc = cfg["description"]
else:
    emojis = " ".join(MODELS[n]["emoji"] for n in active_models)
    hero_title = f"{emojis}  Multi-Model Detection"
    hero_desc = f"Running {len(active_models)} detectors simultaneously: {', '.join(active_models.keys())}"

st.markdown(
    f'<div class="hero-header"><h1>{hero_title}</h1><p>{hero_desc}</p></div>',
    unsafe_allow_html=True,
)


# ================================================================
#                     DETECTION ENGINE
# ================================================================
def run_detection(image_bgr, display_conf):
    """Run all active YOLO models on image with post-filtering."""
    all_detections = []
    annotated = image_bgr.copy()

    for idx, (model_name, mdata) in enumerate(active_models.items()):
        yolo_model = mdata["model"]
        mcfg = mdata["config"]
        results = yolo_model(image_bgr, conf=0.05, iou=0.45, verbose=False)

        # Pick color for this model
        base_color_bgr = MODEL_COLORS_BGR[idx % len(MODEL_COLORS_BGR)]

        for r in results:
            for box in r.boxes:
                conf = box.conf[0].item()
                if conf < display_conf:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                label_name = yolo_model.names[cls]
                is_positive = cls == 0
                is_violation = cls == mcfg.get("violation_cls", 1)

                all_detections.append({
                    "label": label_name,
                    "conf": conf,
                    "cls": cls,
                    "bbox": (x1, y1, x2, y2),
                    "model": model_name,
                    "is_violation": is_violation,
                })

                # Color: green for positive, red for violation
                if is_positive:
                    bgr = (80, 200, 34)
                else:
                    bgr = (68, 68, 239)

                # In multi-model mode, tint by model color
                if len(active_models) > 1:
                    bgr = base_color_bgr

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, box_thickness)

                # Corner accents
                cl = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
                t = box_thickness + 2
                cv2.line(annotated, (x1, y1), (x1 + cl, y1), bgr, t)
                cv2.line(annotated, (x1, y1), (x1, y1 + cl), bgr, t)
                cv2.line(annotated, (x2, y1), (x2 - cl, y1), bgr, t)
                cv2.line(annotated, (x2, y1), (x2, y1 + cl), bgr, t)
                cv2.line(annotated, (x1, y2), (x1 + cl, y2), bgr, t)
                cv2.line(annotated, (x1, y2), (x1, y2 - cl), bgr, t)
                cv2.line(annotated, (x2, y2), (x2 - cl, y2), bgr, t)
                cv2.line(annotated, (x2, y2), (x2, y2 - cl), bgr, t)

                # Label pill
                if show_labels:
                    prefix = f"[{model_name[:4]}] " if len(active_models) > 1 else ""
                    text = f"{prefix}{label_name} {conf:.0%}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.55
                    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
                    pill_y = max(y1 - th - 14, 0)
                    cv2.rectangle(annotated, (x1, pill_y), (x1 + tw + 12, pill_y + th + 12), bgr, -1)
                    cv2.putText(annotated, text, (x1 + 6, pill_y + th + 6), font, scale, (255, 255, 255), 2, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return image_rgb, all_detections


def generate_beep_wav(freq=700, duration_ms=300, volume=0.5):
    """Generate a short beep as WAV bytes for st.audio."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            val = int(volume * 32767 * np.sin(2 * np.pi * freq * t))
            wf.writeframes(struct.pack("<h", val))
    return buf.getvalue()


@st.cache_data
def get_alert_sound():
    """Cache the alert beep so it's only generated once."""
    return generate_beep_wav(freq=700, duration_ms=300, volume=0.5)


def check_violations(detections):
    """Check for violations and show alert banner + play sound."""
    violations = [d for d in detections if d.get("is_violation", False)]
    if violations and enable_alerts:
        v_names = set(f"{d['label']}" for d in violations)
        v_list = ", ".join(v_names)
        st.markdown(
            f'<div class="alert-banner">'
            f'<h4>Warning  -  Violation Detected</h4>'
            f'<p>{len(violations)} violation(s) found: {v_list}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Play alert sound
        st.audio(get_alert_sound(), format="audio/wav", autoplay=True)
    return violations


def add_to_history(detections, source="image"):
    """Record detection event into session history."""
    if not detections:
        return
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "datetime": datetime.now().isoformat(),
        "source": source,
        "models": list(set(d["model"] for d in detections)),
        "total": len(detections),
        "violations": sum(1 for d in detections if d.get("is_violation")),
        "details": [
            {"label": d["label"], "conf": round(d["conf"], 3), "model": d["model"]}
            for d in detections
        ],
    }
    st.session_state.detection_history.append(entry)
    # Keep chart data
    st.session_state.history_chart_data.append({
        "time": entry["timestamp"],
        "detections": entry["total"],
        "violations": entry["violations"],
    })


def show_detection_summary(detections):
    """Display detection stats as metric cards and badges."""
    if not detections:
        st.markdown(
            '<div class="tip-box"><p>No detections found. Try lowering the confidence threshold.</p></div>',
            unsafe_allow_html=True,
        )
        return

    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

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
    badge_html = '<div style="margin-top:0.5rem;">'
    for d in sorted(detections, key=lambda x: -x["conf"]):
        badge_type = "positive" if d["label"].lower() in POSITIVE_LABELS else "negative"
        badge_html += f'<span class="badge-{badge_type}">{d["label"]} {d["conf"]:.0%}</span> '
    badge_html += "</div>"
    st.markdown(badge_html, unsafe_allow_html=True)

    avg_conf = sum(d["conf"] for d in detections) / len(detections)
    st.caption(f"Average confidence: {avg_conf:.1%}")


def get_image_download(image_rgb):
    """Convert annotated image to downloadable PNG bytes."""
    img = Image.fromarray(image_rgb)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_report(detections, source_name="uploaded image"):
    """Generate a text report for download."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models_used = ", ".join(set(d["model"] for d in detections))

    lines = [
        "=" * 60,
        "VASTRA VIVEKA - DETECTION REPORT",
        "=" * 60,
        f"Generated: {now}",
        f"Source: {source_name}",
        f"Models Used: {models_used}",
        f"Confidence Threshold: {confidence_threshold:.0%}",
        f"Total Detections: {len(detections)}",
        f"Violations: {sum(1 for d in detections if d.get('is_violation'))}",
        "",
        "-" * 60,
        "DETECTION DETAILS",
        "-" * 60,
    ]

    for i, d in enumerate(detections, 1):
        lines.append(f"  [{i}] {d['label']} | Confidence: {d['conf']:.1%} | Model: {d['model']} | Violation: {'YES' if d.get('is_violation') else 'No'}")

    # Summary counts
    counts = {}
    for d in detections:
        key = f"{d['model']} > {d['label']}"
        counts[key] = counts.get(key, 0) + 1

    lines.append("")
    lines.append("-" * 60)
    lines.append("SUMMARY")
    lines.append("-" * 60)
    for key, count in sorted(counts.items()):
        lines.append(f"  {key}: {count}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("End of Report")
    lines.append("=" * 60)

    return "\n".join(lines)


# ================================================================
#                       MAIN TABS
# ================================================================
tab_upload, tab_video, tab_webcam, tab_history = st.tabs([
    "📁  Upload Image",
    "🎬  Upload Video",
    "📷  Webcam (Live)",
    "📊  History & Analytics",
])


# ==================== IMAGE UPLOAD TAB ====================
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

            # Alert check
            check_violations(detections)

            # Record to history
            add_to_history(detections, source="image")

            col_img, col_stats = st.columns([2.5, 1])
            with col_img:
                st.image(annotated_rgb, caption=f"Detection Result  |  {len(detections)} object(s) found", use_container_width=True)

                # Download buttons
                dl_col1, dl_col2, _ = st.columns([1, 1, 3])
                with dl_col1:
                    st.download_button(
                        "📥 Download Image",
                        data=get_image_download(annotated_rgb),
                        file_name=f"vastra_viveka_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                with dl_col2:
                    if detections:
                        st.download_button(
                            "📄 Download Report",
                            data=generate_report(detections, uploaded_file.name),
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

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
            "<p>Drag and drop or click above to select an image file</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ==================== VIDEO UPLOAD TAB ====================
with tab_video:
    video_file = st.file_uploader(
        "Upload a video for analysis",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supports MP4, AVI, MOV, MKV formats",
        key="video_uploader",
    )

    if video_file is not None:
        # Save to temp file for OpenCV
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        st.markdown(
            f'<div class="tip-box"><p><strong>Video Info:</strong> {total_frames} frames, '
            f'{fps} FPS, ~{total_frames // fps}s duration. '
            f'Processing every 5th frame for speed.</p></div>',
            unsafe_allow_html=True,
        )

        frame_skip = st.slider("Process every Nth frame", min_value=1, max_value=30, value=5,
                                help="Higher = faster processing, lower = more thorough")

        if st.button("▶  Analyze Video", type="primary"):
            progress_bar = st.progress(0, text="Processing video...")
            video_placeholder = st.empty()
            stats_placeholder = st.empty()

            all_video_detections = []
            frame_idx = 0
            processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    annotated_rgb, detections = run_detection(frame, confidence_threshold)
                    all_video_detections.extend(detections)
                    processed += 1

                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                    # Update progress
                    pct = min(frame_idx / max(total_frames, 1), 1.0)
                    progress_bar.progress(pct, text=f"Frame {frame_idx}/{total_frames} | {len(detections)} detections")

                frame_idx += 1

            cap.release()
            os.unlink(tfile.name)

            progress_bar.progress(1.0, text="Video analysis complete!")

            # Record to history
            add_to_history(all_video_detections, source="video")

            # Video summary
            check_violations(all_video_detections)

            st.markdown("### Video Analysis Summary")
            show_detection_summary(all_video_detections)

            # Downloadable report
            if all_video_detections:
                st.download_button(
                    "📄 Download Video Report",
                    data=generate_report(all_video_detections, video_file.name),
                    file_name=f"video_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
    else:
        st.markdown(
            '<div class="upload-placeholder">'
            '<div class="up-icon">🎬</div>'
            "<h3>Upload a Video</h3>"
            "<p>Drag and drop or click above to select a video file (MP4, AVI, MOV)</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ==================== WEBCAM TAB ====================
with tab_webcam:
    st.markdown(
        '<div class="tip-box"><p>'
        "<strong>How it works:</strong> Click Start Camera to begin live detection. "
        "YOLO runs on each frame in real-time. Adjust confidence in the sidebar."
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
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam. Check camera permissions or close other apps using it.")
            st.session_state.webcam_running = False
        else:
            video_placeholder = st.empty()
            alert_placeholder = st.empty()
            summary_placeholder = st.empty()
            frame_count = 0

            while cap.isOpened() and st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost webcam feed.")
                    break

                annotated_rgb, detections = run_detection(frame, confidence_threshold)
                video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                # Violation alert
                violations = [d for d in detections if d.get("is_violation")]
                if violations and enable_alerts:
                    v_names = set(d["label"] for d in violations)
                    alert_placeholder.markdown(
                        f'<div class="alert-banner"><h4>⚠ Violation Detected!</h4>'
                        f'<p>{", ".join(v_names)}</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    alert_placeholder.empty()

                # Summary
                if detections:
                    counts = {}
                    for d in detections:
                        counts[d["label"]] = counts.get(d["label"], 0) + 1
                    parts = [f"**{label}**: {count}" for label, count in counts.items()]
                    summary_placeholder.markdown(f"🔍  {' &nbsp;|&nbsp; '.join(parts)}  &nbsp;&nbsp; _({len(detections)} total)_")
                else:
                    summary_placeholder.markdown("_Scanning... no detections in current frame_")

                # Record every 30th frame to history
                frame_count += 1
                if frame_count % 30 == 0 and detections:
                    add_to_history(detections, source="webcam")

            cap.release()
            st.info("Camera stopped.")


# ==================== HISTORY & ANALYTICS TAB ====================
with tab_history:
    import pandas as pd
    import altair as alt

    history = st.session_state.detection_history

    if not history:
        st.markdown("")
        st.markdown(
            '<div class="cam-placeholder">'
            '<div class="cam-icon" style="font-size:2.5rem;">📊</div>'
            "<h3>No History Yet</h3>"
            "<p>Upload an image, video, or start the webcam to begin recording detections.<br>"
            "Session history resets when you close the browser tab.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # ---- Summary Metrics Row ----
        total_events = len(history)
        total_detections = sum(h["total"] for h in history)
        total_violations = sum(h["violations"] for h in history)
        compliant = total_detections - total_violations
        violation_rate = (total_violations / total_detections * 100) if total_detections > 0 else 0

        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><h3>{total_events}</h3><p>Scan Events</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h3>{total_detections}</h3><p>Total Detections</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h3>{total_violations}</h3><p>Violations</p></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><h3>{violation_rate:.1f}%</h3><p>Violation Rate</p></div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")

        # ---- Charts ----
        chart_data = st.session_state.history_chart_data
        if len(chart_data) >= 1:
            df_chart = pd.DataFrame(chart_data)
            df_chart["index"] = range(1, len(df_chart) + 1)

            # Common theme for all charts - white bg, clean look
            CHART_THEME = {
                "background": "white",
                "padding": {"left": 15, "right": 15, "top": 10, "bottom": 10},
            }

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown(
                    '<div class="results-panel"><h3>Detections Over Time</h3>',
                    unsafe_allow_html=True,
                )
                area_chart = (
                    alt.Chart(df_chart, **CHART_THEME)
                    .mark_area(
                        interpolate="monotone",
                        line={"color": "#6366f1", "strokeWidth": 2},
                        fill="#c7d2fe",
                        opacity=0.5,
                    )
                    .encode(
                        x=alt.X("index:O", title="Scan #", axis=alt.Axis(grid=False, labelAngle=0)),
                        y=alt.Y("detections:Q", title="Detections", scale=alt.Scale(domainMin=0), axis=alt.Axis(grid=True, gridColor="#f0f0f0", gridDash=[3, 3])),
                        tooltip=[
                            alt.Tooltip("time:N", title="Time"),
                            alt.Tooltip("detections:Q", title="Detections"),
                        ],
                    )
                    .properties(height=250, background="white")
                    .configure_view(strokeWidth=0, fill="white")
                    .configure_axis(
                        labelFontSize=11,
                        titleFontSize=12,
                        titleColor="#6b7280",
                        labelColor="#9ca3af",
                    )
                )
                st.altair_chart(area_chart, use_container_width=True, theme=None)
                st.markdown("</div>", unsafe_allow_html=True)

            with chart_col2:
                st.markdown(
                    '<div class="results-panel"><h3>Violations Over Time</h3>',
                    unsafe_allow_html=True,
                )
                bar_chart = (
                    alt.Chart(df_chart, **CHART_THEME)
                    .mark_bar(
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4,
                        color="#ef4444",
                        size=20,
                    )
                    .encode(
                        x=alt.X("index:O", title="Scan #", axis=alt.Axis(grid=False, labelAngle=0)),
                        y=alt.Y("violations:Q", title="Violations", scale=alt.Scale(domainMin=0), axis=alt.Axis(grid=True, gridColor="#f0f0f0", gridDash=[3, 3])),
                        tooltip=[
                            alt.Tooltip("time:N", title="Time"),
                            alt.Tooltip("violations:Q", title="Violations"),
                        ],
                    )
                    .properties(height=250, background="white")
                    .configure_view(strokeWidth=0, fill="white")
                    .configure_axis(
                        labelFontSize=11,
                        titleFontSize=12,
                        titleColor="#6b7280",
                        labelColor="#9ca3af",
                    )
                )
                st.altair_chart(bar_chart, use_container_width=True, theme=None)
                st.markdown("</div>", unsafe_allow_html=True)

            # ---- Compliance Donut + Breakdown ----
            if total_detections > 0:
                st.markdown("")
                donut_col, breakdown_col = st.columns([1, 2])

                with donut_col:
                    st.markdown(
                        '<div class="results-panel"><h3>Compliance Overview</h3>',
                        unsafe_allow_html=True,
                    )
                    donut_df = pd.DataFrame({
                        "Status": ["Compliant", "Violation"],
                        "Count": [compliant, total_violations],
                    })
                    donut = (
                        alt.Chart(donut_df, background="white", padding={"left": 10, "right": 10, "top": 10, "bottom": 10})
                        .mark_arc(innerRadius=55, outerRadius=90, cornerRadius=3)
                        .encode(
                            theta=alt.Theta("Count:Q"),
                            color=alt.Color(
                                "Status:N",
                                scale=alt.Scale(
                                    domain=["Compliant", "Violation"],
                                    range=["#22c55e", "#ef4444"],
                                ),
                                legend=alt.Legend(
                                    orient="bottom",
                                    titleFontSize=0,
                                    labelFontSize=12,
                                    symbolSize=80,
                                    labelColor="#6b7280",
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("Status:N"),
                                alt.Tooltip("Count:Q"),
                            ],
                        )
                        .properties(height=220, background="white")
                        .configure_view(strokeWidth=0, fill="white")
                    )
                    st.altair_chart(donut, use_container_width=True, theme=None)
                    st.markdown("</div>", unsafe_allow_html=True)

                with breakdown_col:
                    st.markdown(
                        '<div class="results-panel"><h3>Detection Breakdown</h3>',
                        unsafe_allow_html=True,
                    )
                    label_counts = {}
                    for h in history:
                        for d in h["details"]:
                            label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1

                    breakdown_df = pd.DataFrame(
                        [{"Label": k, "Count": v} for k, v in sorted(label_counts.items(), key=lambda x: -x[1])]
                    )
                    h_bar = (
                        alt.Chart(breakdown_df, background="white", padding={"left": 10, "right": 25, "top": 10, "bottom": 10})
                        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color="#6366f1")
                        .encode(
                            x=alt.X("Count:Q", title="Count", axis=alt.Axis(grid=True, gridColor="#f0f0f0", gridDash=[3, 3]), scale=alt.Scale(domainMin=0)),
                            y=alt.Y("Label:N", title="", sort="-x", axis=alt.Axis(labelFontSize=12, labelColor="#374151")),
                            tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Count:Q")],
                        )
                        .properties(height=220, background="white")
                        .configure_view(strokeWidth=0, fill="white")
                        .configure_axis(
                            labelFontSize=11,
                            titleFontSize=12,
                            titleColor="#6b7280",
                            labelColor="#6b7280",
                        )
                    )
                    st.altair_chart(h_bar, use_container_width=True, theme=None)
                    st.markdown("</div>", unsafe_allow_html=True)

        # ---- Detection Log Table ----
        st.markdown("")
        st.markdown(
            '<div class="results-panel"><h3>Detection Log</h3>',
            unsafe_allow_html=True,
        )
        log_rows = []
        for h in reversed(history[-50:]):
            for d in h["details"]:
                is_v = d["label"].lower() not in POSITIVE_LABELS
                log_rows.append({
                    "Time": h["timestamp"],
                    "Source": h["source"].capitalize(),
                    "Model": d["model"],
                    "Class": d["label"],
                    "Confidence": round(d["conf"] * 100, 1),
                    "Status": "Violation" if is_v else "Compliant",
                })
        if log_rows:
            log_df = pd.DataFrame(log_rows)
            st.dataframe(
                log_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="small"),
                    "Source": st.column_config.TextColumn("Source", width="small"),
                    "Model": st.column_config.TextColumn("Model", width="medium"),
                    "Class": st.column_config.TextColumn("Class", width="small"),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence %",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                        width="medium",
                    ),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                },
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Export & Clear ----
        st.markdown("")
        exp_col1, exp_col2, _ = st.columns([1, 1, 4])
        with exp_col1:
            history_json = json.dumps(history, indent=2)
            st.download_button(
                "Export History (JSON)",
                data=history_json,
                file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col2:
            if st.button("Clear History", use_container_width=True):
                st.session_state.detection_history = []
                st.session_state.history_chart_data = []
                st.rerun()


# ================================================================
#                          FOOTER
# ================================================================
st.markdown(
    '<div class="app-footer">'
    "<p>Vastra Viveka &mdash; Smart Dress Code Analyzer &nbsp;&bull;&nbsp; "
    "Powered by YOLOv11 &amp; Streamlit &nbsp;&bull;&nbsp; v3.0</p>"
    "</div>",
    unsafe_allow_html=True,
)
