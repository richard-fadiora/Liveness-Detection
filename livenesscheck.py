import cv2
import torch
import mediapipe as mp
from collections import deque
from collections import Counter
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "nguyenkhoa/dinov2_Liveness_detection_v2.2.3"
LIVE_THRESHOLD = 0.75
SPOOF_THRESHOLD = 0.75
LABEL_WINDOW_SIZE = 10  # Number of frames to vote over
label_buffer = deque(maxlen=LABEL_WINDOW_SIZE)

# Load model
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

# Face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# Labels
id2label = model.config.id2label

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press Q to quit")

WINDOW_SIZE = 10  # ~0.3 seconds
prob_buffer = deque(maxlen=WINDOW_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detector.process(rgb_frame)

    face_crop = None

    if results.detections:
        detection = results.detections[0]  # first face
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x2 = x1 + bw
        y2 = y1 + bh

        # Padding
        pad = int(0.2 * bw)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = frame[y1:y2, x1:x2]

        # Draw face box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # No face case
    if face_crop is None or face_crop.size == 0:
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        cv2.imshow("Webcam Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
        continue

    # Convert face to PIL
    image = Image.fromarray(
        cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    )

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        prob_buffer.append(probs)

    avg_probs = torch.stack(list(prob_buffer)).mean(dim=0)
    pred_id = torch.argmax(avg_probs).item()
    confidence = avg_probs[pred_id].item()
    raw_label = id2label[pred_id]

    if raw_label.lower() == "live" and confidence >= LIVE_THRESHOLD:
        label = "LIVE"
        color = (0, 255, 0)

    elif raw_label.lower() == "spoof" and confidence >= SPOOF_THRESHOLD:
        label = "SPOOF"
        color = (0, 0, 255)

    else:
        label = "UNCERTAIN"
        color = (0, 255, 255)

    label_buffer.append(label)

    if len(label_buffer) > 0:
        most_common_label = Counter(label_buffer).most_common(1)[0][0]
    else:
        most_common_label = label  # fallback

    # Draw liveness result
    text = f"{most_common_label.upper()} ({confidence:.2f})"

    cv2.putText(
        frame,
        text,
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Webcam Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
