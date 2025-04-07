import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu

# Set Streamlit Page Config
st.set_page_config(page_title="Smart Scope", layout="wide")

# Sidebar Navigation using Option Menu
with st.sidebar:
    st.title("Situation Selection")
    option = option_menu(
        menu_title=None,  # Removed duplicate title inside menu
        options=["Mask Detector", "Formal vs Informal", "Traditional vs Non-Traditional", "Helmet Detector"],
        menu_icon="list",
        default_index=0,
        orientation="vertical"
    )

st.title("Smart Scope")

# Load Models
if option == "Mask Detector":
    model = YOLO("/home/bunnys-weapon/Documents/mini_project_codes/mask_best.pt")
    st.subheader("😷 Mask and No Mask Classifier")
    
elif option == "Formal vs Informal":
    model = YOLO("/home/bunnys-weapon/Documents/mini_project_codes/formal_best.pt")
    st.subheader("🧥 Formal and Informal Classifier")

elif option == "Traditional vs Non-Traditional":
    model = YOLO("/home/bunnys-weapon/Documents/mini_project_codes/trad_best.pt")
    st.subheader("👕 Traditional and Non-Traditional Classifier")
    
elif option == "Helmet Detector":
    model = YOLO("/home/bunnys-weapon/Documents/mini_project_codes/helmet_best.pt")
    st.subheader("🪖 Helmet and No Helmet Classifier")

# Initialize stop state
if "stop" not in st.session_state:
    st.session_state.stop = False
if "start" not in st.session_state:
    st.session_state.start = False

# Start & Stop buttons
if st.button("Start"):
    st.session_state.start = True
    st.session_state.stop = False
if st.button("Stop"):
    st.session_state.stop = True
    st.session_state.start = False

# Open webcam only when 'Start' is clicked
if st.session_state.start:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open webcam. Check your camera permissions.")
    else:
        video_placeholder = st.empty()

        while cap.isOpened():
            if st.session_state.stop:
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            # Run YOLO inference
            results = model(frame)

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

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        st.write("Video stream stopped.")
