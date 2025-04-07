import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Set Streamlit Page Config
st.set_page_config(page_title="Vastra Viveka", layout="wide")

# Sidebar Navigation using Option Menu
with st.sidebar:
    st.title("Situation Selection")
    option = option_menu(
        menu_title=None,
        options=["Mask Detector", "Formal vs Informal", "Traditional vs Non-Traditional", "Helmet Detector"],
        icons=["mask", "tshirt", "tshirt", "shield"],
        menu_icon="list",
        default_index=0,
        orientation="vertical"
    )

st.title("Vastra Viveka")

# Load Models
if option == "Mask Detector":
    model = YOLO("mask_best.pt")
    st.subheader("😷 Mask and No Mask Classifier")

elif option == "Formal vs Informal":
    model = YOLO("formal_best.pt")
    st.subheader("🧥 Formal and Informal Classifier")

elif option == "Traditional vs Non-Traditional":
    model = YOLO("trad_best.pt")
    st.subheader("👕 Traditional and Non-Traditional Classifier")

elif option == "Helmet Detector":
    model = YOLO("helmet_best.pt")
    st.subheader("🪖 Helmet and No Helmet Classifier")

# Streamlit-webrtc video processor
class AttireDetectionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = model(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                highest_conf_cls = max(zip(box.conf.tolist(), box.cls.tolist()), key=lambda x: x[0])
                highest_conf_idx = int(highest_conf_cls[1])
                highest_conf = highest_conf_cls[0]

                label = f"{model.names[highest_conf_idx]}: {highest_conf:.2f}"
                color = (0, 255, 0) if highest_conf_idx == 0 else (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start/Stop buttons
if "start" not in st.session_state:
    st.session_state.start = False
if "stop" not in st.session_state:
    st.session_state.stop = False

if st.button("Start"):
    st.session_state.start = True
    st.session_state.stop = False
if st.button("Stop"):
    st.session_state.stop = True
    st.session_state.start = False

# Activate streamlit-webrtc if "Start" is clicked
if st.session_state.start and not st.session_state.stop:
    webrtc_streamer(
        key=option.lower().replace(" ", "-"),
        video_processor_factory=AttireDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

