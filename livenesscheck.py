import streamlit as st
import cv2
import torch
import mediapipe as mp
from collections import deque, Counter
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

# -----------------------------
# Constants
# -----------------------------
MODEL_ID = "nguyenkhoa/dinov2_Liveness_detection_v2.2.3"
LIVE_THRESHOLD = 0.75
SPOOF_THRESHOLD = 0.75
WINDOW_SIZE = 10            # frames for probability smoothing
LABEL_WINDOW_SIZE = 10      # frames for majority voting

# -----------------------------
# Load model (runs once)
# -----------------------------
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

processor, model = load_model()

# -----------------------------
# Initialize MediaPipe face detector
# -----------------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Labels
id2label = model.config.id2label

# -----------------------------
# Session state buffers
# -----------------------------
if "prob_buffer" not in st.session_state:
    st.session_state.prob_buffer = deque(maxlen=WINDOW_SIZE)
if "label_buffer" not in st.session_state:
    st.session_state.label_buffer = deque(maxlen=LABEL_WINDOW_SIZE)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Real-Time Webcam Liveness Detection")

frame_placeholder = st.empty()  # for video display

# -----------------------------
# Webcam stream
# -----------------------------
camera_frame = st.camera_input("Look at the camera")

if camera_frame is not None:
    # Convert frame to OpenCV
    frame = np.array(Image.open(camera_frame))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = rgb_frame.copy()

    # -----------------------------
    # Face detection
    # -----------------------------
    results = face_detector.process(rgb_frame)
    face_crop = None

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = rgb_frame.shape

        x1 = max(0, int(bbox.xmin * w - 0.2 * bbox.width * w))
        y1 = max(0, int(bbox.ymin * h - 0.2 * bbox.width * w))
        x2 = min(w, int((bbox.xmin + bbox.width) * w + 0.2 * bbox.width * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h + 0.2 * bbox.width * w))

        face_crop = rgb_frame[y1:y2, x1:x2]

        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    else:
        st.session_state.label_buffer.append("NO FACE")
        st.image(display_frame, caption="No face detected")
        st.stop()

    # -----------------------------
    # Liveness detection
    # -----------------------------
    if face_crop is not None and face_crop.size > 0:
        pil_image = Image.fromarray(face_crop)
        inputs = processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            st.session_state.prob_buffer.append(probs)

        avg_probs = torch.stack(list(st.session_state.prob_buffer)).mean(dim=0)
        pred_id = torch.argmax(avg_probs).item()
        confidence = avg_probs[pred_id].item()
        raw_label = id2label[pred_id]

        # Confidence thresholding
        if raw_label.lower() == "live" and confidence >= LIVE_THRESHOLD:
            label = "LIVE"
            color = (0, 255, 0)
        elif raw_label.lower() == "spoof" and confidence >= SPOOF_THRESHOLD:
            label = "SPOOF"
            color = (0, 0, 255)
        else:
            label = "UNCERTAIN"
            color = (0, 255, 255)

        # Multi-frame voting
        st.session_state.label_buffer.append(label)
        most_common_label = Counter(st.session_state.label_buffer).most_common(1)[0][0]

        # Draw result
        cv2.putText(
            display_frame,
            f"{most_common_label} ({confidence:.2f})",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    # Display frame
    frame_placeholder.image(display_frame, channels="RGB")

    # -----------------------------
    # Auto-refresh
    # -----------------------------
    st.experimental_rerun()  # Automatically fetch new frame
