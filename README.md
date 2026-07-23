# Vastra Viveka — Real-Time Detection & Compliance Analyzer

A real-time computer-vision application that runs multiple custom-trained **YOLOv11** models to
analyze dress-code and safety compliance from images and a live webcam feed.

## Detectors

- **Attire** — traditional vs. formal/informal wear classification
- **Face mask** — mask / no-mask detection
- **Helmet / PPE** — helmet compliance detection

## Features

- Real-time detection on uploaded images and **live webcam**.
- Switch between multiple trained YOLOv11 models from the sidebar.
- **Detection history** and live analytics charts.
- Compliance **alerts** on detection.
- Clean Streamlit interface.

## Tech Stack

- **Ultralytics YOLOv11** — custom-trained models (attire, mask, helmet)
- **OpenCV**, **NumPy**, **Pillow**
- **Streamlit** + `streamlit-option-menu`

## Getting Started

```bash
pip install -r requirements.txt
streamlit run final_app_trail.py
```

Choose a detector, then upload an image or start the webcam to see real-time detections, history,
and analytics.
