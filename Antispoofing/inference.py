"""
Face Anti-Spoofing Inference Module
Supports: Image, Video, and Webcam inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import warnings
from huggingface_hub import hf_hub_download
from collections import deque
import time

warnings.filterwarnings('ignore')


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class WRN101_2_Early(nn.Module):
    """Wide ResNet-101-2 backbone (layer 1-2 only)"""

    def __init__(self, ckpt_path=None):
        super().__init__()
        base = models.wide_resnet101_2(weights=None)

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model" in checkpoint:
                full_state_dict = checkpoint["model"]
                backbone_state_dict = {}
                for key, value in full_state_dict.items():
                    if key.startswith("backbone."):
                        new_key = key.replace("backbone.", "")
                        backbone_state_dict[new_key] = value
                base.load_state_dict(backbone_state_dict, strict=False)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Transformer(nn.Module):
    """Temporal encoder"""

    def __init__(self, embed_dim=512, num_heads=8, depth=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, embed_dim))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.mean(dim=[3, 4])
        x = x + self.pos_embedding[:, :T, :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return x


class DeepFakeModel(nn.Module):
    """Complete anti-spoofing model"""

    def __init__(self, backbone_ckpt=None, freeze_backbone=True):
        super().__init__()
        self.backbone = WRN101_2_Early(ckpt_path=backbone_ckpt)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.temporal_encoder = Transformer(
            embed_dim=512,
            num_heads=8,
            depth=4,
            mlp_ratio=2.0,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, frames):
        B, T = frames.shape[:2]
        frames_flat = frames.view(B * T, *frames.shape[2:])

        with torch.set_grad_enabled(self.backbone.training):
            feats_flat = self.backbone(frames_flat)

        feats = feats_flat.view(B, T, *feats_flat.shape[1:])
        x = self.temporal_encoder(feats)
        logits = self.classifier(x)
        return logits


# ============================================================
# YOLO FACE DETECTOR
# ============================================================
class YOLOFaceDetector:
    """YOLO-based face detector"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.conf_threshold = conf_threshold
        self.input_size = 640

    def detect_faces(self, image: np.ndarray):
        """Detect faces in image"""
        img_height, img_width = image.shape[:2]

        # Preprocess
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: img_batch})
        predictions = outputs[0]

        # Post-process
        faces = []
        if len(predictions.shape) == 3:
            predictions = predictions[0].T

            for pred in predictions:
                conf = pred[4]
                if conf > self.conf_threshold:
                    x_center, y_center, w, h = pred[:4]

                    x_center = x_center * img_width / self.input_size
                    y_center = y_center * img_height / self.input_size
                    w = w * img_width / self.input_size
                    h = h * img_height / self.input_size

                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)

                    faces.append((x1, y1, x2, y2, float(conf)))

        return faces


# ============================================================
# MAIN INFERENCE CLASS
# ============================================================
class AntiSpoofingDetector:
    """
    Face Anti-Spoofing Detector

    Usage:
        # With full checkpoint (recommended)
        detector = AntiSpoofingDetector(
            model_path="antispoofing_full.pth"
        )

        # Or with separate files (legacy)
        detector = AntiSpoofingDetector(
            antispoofing_model_path="antispoofing_best.pth",
            face_recognition_ckpt="faceRecognition_arcface_ckpt.pth"
        )
    """

    def __init__(
            self,
            model_path=None,  # ← NEW: Full checkpoint path
            antispoofing_model_path=None,  # ← LEGACY: Separate checkpoint
            face_recognition_ckpt=None,  # ← LEGACY: Separate backbone
            yolo_model_path=None,
            device="cuda",
            num_frames=10,
            threshold=0.5
    ):
        """
        Initialize detector

        Args:
            model_path: Path to FULL model checkpoint (backbone + temporal + classifier)
            antispoofing_model_path: [LEGACY] Path to anti-spoofing checkpoint only
            face_recognition_ckpt: [LEGACY] Path to face recognition backbone
            yolo_model_path: Path to YOLO face detector (.onnx)
            device: 'cuda' or 'cpu'
            num_frames: Number of frames to analyze
            threshold: Classification threshold (0.5 = balanced)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.threshold = threshold

        print("=" * 60)
        print("FACE ANTI-SPOOFING DETECTOR")
        print("=" * 60)
        print(f"Device: {self.device}")

        # ============================================================
        # DETERMINE LOADING MODE
        # ============================================================
        if model_path is not None:
            # NEW MODE: Full checkpoint (1 file)
            print("\n🔧 Loading mode: FULL CHECKPOINT")
            print(f"   Model: {Path(model_path).name}")

            # Auto-download if not exists
            if not Path(model_path).exists():
                print("\n📥 Downloading full model...")
                model_path = self._download_full_model()

            use_full_checkpoint = True

        elif antispoofing_model_path is not None:
            # LEGACY MODE: Separate files
            print("\n🔧 Loading mode: LEGACY (Separate Files)")
            print(f"   Anti-spoofing: {Path(antispoofing_model_path).name}")

            if face_recognition_ckpt is None:
                print("\n📥 Downloading face recognition backbone...")
                face_recognition_ckpt = self._download_backbone()

            print(f"   Backbone: {Path(face_recognition_ckpt).name}")
            use_full_checkpoint = False

        else:
            # Auto-download full checkpoint
            print("\n📥 No model specified, downloading full checkpoint...")
            model_path = self._download_full_model()
            use_full_checkpoint = True

        # ============================================================
        # LOAD YOLO DETECTOR
        # ============================================================
        if yolo_model_path is None:
            print("\n📥 Downloading YOLO face detector...")
            yolo_model_path = self._download_yolo()

        print(f"\n🔧 Loading YOLO detector: {Path(yolo_model_path).name}")
        self.yolo = YOLOFaceDetector(yolo_model_path, conf_threshold=0.5)

        # ============================================================
        # LOAD ANTI-SPOOFING MODEL
        # ============================================================
        if use_full_checkpoint:
            # NEW: Load full checkpoint (all weights in one file)
            print(f"\n🔧 Loading FULL model: {Path(model_path).name}")

            # Create model WITHOUT loading backbone separately
            self.model = DeepFakeModel(
                backbone_ckpt=None,  # Don't load backbone separately
                freeze_backbone=False
            ).to(self.device)

            # Load full checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                auc = checkpoint.get('auc', 'N/A')
                info = checkpoint.get('info', '')
                print(f"   ✓ Full model loaded successfully")
                print(f"   ✓ Validation AUC: {auc}")
                if info:
                    print(f"   ℹ {info}")
            else:
                self.model.load_state_dict(checkpoint, strict=True)
                print(f"   ✓ Full model loaded successfully")

        else:
            # LEGACY: Load separate files
            print(f"\n🔧 Loading anti-spoofing model (LEGACY mode)")

            # Create model WITH backbone loading
            self.model = DeepFakeModel(
                backbone_ckpt=face_recognition_ckpt,
                freeze_backbone=True
            ).to(self.device)

            # Load anti-spoofing checkpoint
            checkpoint = torch.load(antispoofing_model_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                auc = checkpoint.get('auc', 'N/A')
                print(f"   ✓ Model loaded (Validation AUC: {auc})")
            else:
                self.model.load_state_dict(checkpoint, strict=False)
                print(f"   ✓ Model loaded")

        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("=" * 60)
        print("✓ Detector ready!\n")

    def _download_yolo(self):
        """Download YOLO model from Hugging Face"""
        try:
            path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection",
                filename="model.onnx"
            )
            return path
        except Exception as e:
            print(f"⚠ Warning: Could not download YOLO model: {e}")
            print("Please provide yolo_model_path manually")
            raise

    def _download_backbone(self):
        """Download face recognition backbone (LEGACY)"""
        try:
            path = hf_hub_download(
                repo_id="biometric-ai-lab/Face_Recognition",
                filename="faceRecognition_arcface_ckpt.pth"
            )
            return path
        except Exception as e:
            print(f"⚠ Warning: Could not download backbone: {e}")
            print("Please provide face_recognition_ckpt manually")
            raise

    def _download_full_model(self):
        """Use local model path instead of downloading"""
        local_path = Path("Antispoofing/antispoofing_full.pth")  # adjust to where your file is
        if not local_path.exists():
            raise FileNotFoundError(f"Full checkpoint not found at {local_path}")
        return str(local_path)


    def _detect_and_crop_face(self, frame):
        """Detect and crop face from frame"""
        faces = self.yolo.detect_faces(frame)

        if len(faces) == 0:
            return None, None

        # Get largest face
        x1, y1, x2, y2, conf = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))

        # Add padding
        w = x2 - x1
        h = y2 - y1
        pad = int(0.1 * max(w, h))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)

        face = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2, conf)

        return face, bbox

    def _extract_frames_from_video(self, video_path, max_frames=None):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if max_frames is None:
            max_frames = min(total_frames, self.num_frames * 3)

        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames, fps

    def predict_image(self, image_path):
        """
        Predict single image (requires video for temporal analysis)

        Args:
            image_path: Path to image file

        Returns:
            dict with keys: 'prediction', 'confidence', 'is_attack'
        """
        print(f"\n🔍 Analyzing image: {image_path}")

        # Load image
        if isinstance(image_path, str):
            frame = cv2.imread(image_path)
        else:
            frame = image_path

        if frame is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Detect face
        face, bbox = self._detect_and_crop_face(frame)

        if face is None:
            return {
                'prediction': 'NO_FACE',
                'confidence': 0.0,
                'is_attack': None,
                'bbox': None
            }

        print(f"   ✓ Face detected (confidence: {bbox[4]:.3f})")

        # Note: Single image cannot do temporal analysis
        # We'll duplicate the frame to simulate sequence
        print(f"   ⚠ Warning: Single image mode - duplicating frame for temporal analysis")

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil)

        # Create sequence by duplicating
        frames_tensor = face_tensor.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(frames_tensor)
            prob = torch.sigmoid(logits).item()

        is_attack = prob > self.threshold
        prediction = "SPOOFING_ATTACK" if is_attack else "GENUINE"

        print(f"   📊 Prediction: {prediction}")
        print(f"   📊 Confidence: {prob:.4f}")

        return {
            'prediction': prediction,
            'confidence': prob,
            'is_attack': is_attack,
            'bbox': bbox
        }

    def predict_video(self, video_path, sample_frames=30):
        """
        Predict video with temporal analysis

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample from video

        Returns:
            dict with keys: 'prediction', 'confidence', 'is_attack', 'frame_scores'
        """
        print(f"\n🎬 Analyzing video: {video_path}")

        # Extract frames
        frames, fps = self._extract_frames_from_video(video_path, max_frames=sample_frames)
        print(f"   ✓ Extracted {len(frames)} frames (FPS: {fps:.1f})")

        # Process frames in sliding windows
        frame_scores = []
        face_crops = []

        for i, frame in enumerate(frames):
            face, bbox = self._detect_and_crop_face(frame)

            if face is not None:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_tensor = self.transform(face_pil)
                face_crops.append(face_tensor)

        if len(face_crops) < self.num_frames:
            print(f"   ⚠ Warning: Only {len(face_crops)} faces detected (need {self.num_frames})")
            # Pad with duplicates
            while len(face_crops) < self.num_frames:
                face_crops.append(face_crops[-1] if face_crops else torch.zeros(3, 224, 224))

        # Process in sliding windows
        scores = []
        window_size = self.num_frames
        stride = max(1, window_size // 2)

        for start_idx in range(0, len(face_crops) - window_size + 1, stride):
            window = face_crops[start_idx:start_idx + window_size]
            frames_tensor = torch.stack(window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(frames_tensor)
                prob = torch.sigmoid(logits).item()
                scores.append(prob)

        # Aggregate scores
        if len(scores) == 0:
            avg_prob = 0.5
        else:
            avg_prob = np.mean(scores)

        is_attack = avg_prob > self.threshold
        prediction = "SPOOFING_ATTACK" if is_attack else "GENUINE"

        print(f"   📊 Prediction: {prediction}")
        print(f"   📊 Average confidence: {avg_prob:.4f}")
        if scores:
            print(f"   📊 Score range: [{min(scores):.4f}, {max(scores):.4f}]")

        return {
            'prediction': prediction,
            'confidence': avg_prob,
            'is_attack': is_attack,
            'frame_scores': scores,
            'num_windows': len(scores)
        }

    def run_webcam(self, camera_id=0, frame_skip=2):
        """
        Run real-time detection on webcam

        Args:
            camera_id: Camera device ID (usually 0)
            frame_skip: Process every N frames (for speed)
        """
        print(f"\n📷 Starting webcam detection (Camera {camera_id})")
        print(f"   Frame skip: {frame_skip} (process every {frame_skip + 1} frames)")
        print(f"   Press 'q' to quit, 's' to show stats\n")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")

        # Set camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0 or camera_fps > 120:
            camera_fps = 30

        # Buffers
        frame_buffer = deque(maxlen=self.num_frames)
        score_history = deque(maxlen=5)

        # State
        frame_counter = 0
        last_bbox = None
        last_label = "Initializing..."
        last_prob = None

        # Statistics
        total_frames = 0
        start_time = time.time()
        last_fps_update = start_time
        display_fps = 0

        stats = {
            'total_inferences': 0,
            'fake_detected': 0,
            'real_detected': 0
        }

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1
                frame_counter += 1

                # Detect face
                face, bbox = self._detect_and_crop_face(frame)

                if face is not None:
                    last_bbox = bbox

                    # Process every N frames
                    should_process = (frame_counter % (frame_skip + 1)) == 0

                    if should_process:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(face_rgb)
                        face_tensor = self.transform(face_pil)
                        frame_buffer.append(face_tensor)

                        # Run inference when buffer is full
                        if len(frame_buffer) >= self.num_frames:
                            frames_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                logits = self.model(frames_tensor)
                                prob = torch.sigmoid(logits).item()

                            score_history.append(prob)
                            smooth_prob = sum(score_history) / len(score_history)

                            last_label = "FAKE" if smooth_prob > self.threshold else "REAL"
                            last_prob = smooth_prob

                            stats['total_inferences'] += 1
                            if last_label == "FAKE":
                                stats['fake_detected'] += 1
                            else:
                                stats['real_detected'] += 1

                # Draw results
                if last_bbox is not None:
                    x1, y1, x2, y2, conf = last_bbox

                    if last_prob is None or len(frame_buffer) < self.num_frames:
                        color = (255, 165, 0)  # Orange
                        text = f"Collecting {len(frame_buffer)}/{self.num_frames}"
                    else:
                        color = (0, 255, 0) if last_label == "REAL" else (0, 0, 255)
                        text = f"{last_label} ({last_prob:.3f})"

                    # Draw bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0] + 10, y1),
                        color, -1
                    )
                    cv2.putText(frame, text, (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Update FPS
                current_time = time.time()
                if current_time - last_fps_update >= 0.5:
                    elapsed = current_time - start_time
                    display_fps = total_frames / elapsed
                    last_fps_update = current_time

                # Draw FPS
                cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show frame
                cv2.imshow("Face Anti-Spoofing Detection", frame)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("\n" + "=" * 50)
                    print("STATISTICS")
                    print("=" * 50)
                    print(f"Total inferences: {stats['total_inferences']}")
                    print(f"REAL detected: {stats['real_detected']}")
                    print(f"FAKE detected: {stats['fake_detected']}")
                    if stats['total_inferences'] > 0:
                        real_rate = (stats['real_detected'] / stats['total_inferences']) * 100
                        fake_rate = (stats['fake_detected'] / stats['total_inferences']) * 100
                        print(f"REAL rate: {real_rate:.1f}%")
                        print(f"FAKE rate: {fake_rate:.1f}%")
                    print("=" * 50 + "\n")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            print(f"\n✓ Processed {total_frames} frames in {elapsed:.2f}s")
            print(f"✓ Average FPS: {total_frames / elapsed:.2f}")
            print("\nFinal Statistics:")
            print(f"  Total inferences: {stats['total_inferences']}")
            print(f"  REAL detected: {stats['real_detected']}")
            print(f"  FAKE detected: {stats['fake_detected']}")


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # OPTION 1: FULL CHECKPOINT (RECOMMENDED - 1 file)
    # ============================================================
    detector = AntiSpoofingDetector(
        model_path="antispoofing_full.pth",
        yolo_model_path="yolov8s-face-lindevs.onnx",
        device="cuda",
        num_frames=10,
        threshold=0.2
    )

    # Example 1: Predict single image
    # result = detector.predict_image("test_image.jpg")
    # print(result)

    # Example 2: Predict video
    # result = detector.predict_video("test_video.mp4")
    # print(result)

    # Example 3: Run webcam
    detector.run_webcam(camera_id=0, frame_skip=2)