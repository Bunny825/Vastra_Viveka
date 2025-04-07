import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Vastra Viveka", layout="wide")

# Sidebar
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

# Load appropriate YOLO model
model_paths = {
    "Mask Detector": "mask_best.pt",
    "Formal vs Informal": "formal_best.pt",
    "Traditional vs Non-Traditional": "trad_best.pt",
    "Helmet Detector": "helmet_best.pt"
}
model = YOLO(model_paths[option])
st.subheader(f"🔍 {option} Classifier")

# Video processing class
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]}: {conf:.2f}"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream from browser webcam
webrtc_streamer(
    key="yolo-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
