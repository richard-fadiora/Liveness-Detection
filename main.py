import cv2
import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from typing import List
from PIL import Image
import torchvision.transforms as transforms
from fastapi.middleware.cors import CORSMiddleware
from Antispoofing.inference import DeepFakeModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LivenessSDK_Brain:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepFakeModel(backbone_ckpt=None).to(self.device)
        self.load_weights()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_weights(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(parent_dir, "Antispoofing", "antispoofing_full.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def check_glare_and_texture(self, frame_rgb):
        # Convert to Gray for texture analysis
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Glare Ratio: Screens reflect light in large white blobs
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        glare_ratio = (cv2.countNonZero(binary) / gray.size) * 100
        
        # 2. Laplacian: Real skin has high frequency "pore" texture
        # Screens are either too blurry or have a "Moiré" grid pattern
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"DEBUG Texture -> Glare: {glare_ratio:.2f}%, Var: {laplacian_var:.2f}")

        # TIGHTER THRESHOLDS: Adjust these based on your camera quality
        if glare_ratio > 12:  # Lowered from 30 to catch phone screens
            return False, f"SCREEN_REFLECTION_DETECTED ({glare_ratio:.1f}%)"
        
        if laplacian_var < 40: # Catching blurry video playback
            return False, f"SPOOF_BLUR_DETECTED ({laplacian_var:.1f})"
            
        if laplacian_var > 500: # Catching screen pixel grids (Moiré)
            return False, f"SCREEN_MOIRE_PATTERN_DETECTED ({laplacian_var:.1f})"

        return True, "OK"

    def check_motion_coherence(self, frames_rgb):
        # Videos played back often have "perfect" or "robotic" movement
        # Static images have ZERO movement.
        diffs = []
        for i in range(len(frames_rgb) - 1):
            d = cv2.absdiff(frames_rgb[i], frames_rgb[i+1])
            diffs.append(np.mean(d))
        
        avg_motion = np.mean(diffs)
        print(f"DEBUG Motion -> Avg Diff: {avg_motion:.4f}")
        
        if avg_motion < 0.15: # It's a static image or a very still screen
            return False, "STATIC_IMAGE_OR_SCREEN_DETECTION"
        
        return True, "OK"

brain = None

@app.post("/v1/verify")
async def verify(files: List[UploadFile] = File(...)):
    if brain == None:
        brain = LivenessSDK_Brain()
    cv_frames = []
    for file in files:
        nparr = np.frombuffer(await file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            # We need RGB for both the Model and the Texture check
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv_frames.append(img_rgb)

    if not cv_frames:
        return {"is_live": False, "reason": "NO_IMAGE_DATA"}

    # 1. Texture & Glare Check (Passive Defense)
    tex_ok, tex_msg = brain.check_glare_and_texture(cv_frames[-1])
    if not tex_ok:
        return {"is_live": False, "reason": tex_msg, "status": "fail"}

    # 2. Motion Check (Catching static spoofs)
    motion_ok, motion_msg = brain.check_motion_coherence(cv_frames)
    if not motion_ok:
        return {"is_live": False, "reason": motion_msg, "status": "fail"}

    # 3. AI Model Inference
    tensors = [brain.transform(Image.fromarray(f)) for f in cv_frames]
    batch = torch.stack(tensors).unsqueeze(0).to(brain.device) 

    with torch.no_grad():
        logits = brain.model(batch)
        # We take the sigmoid of the logits to get 0.0 to 1.0
        scores = torch.sigmoid(logits)
        # Take the maximum or average score of the batch
        skin_score = torch.mean(scores).item()

    print(f"--- FINAL DECISION ---")
    print(f"Skin Score: {skin_score}")

    # Threshold: High for security, lower if you get too many false rejections
    is_live = skin_score > 0.75 

    return {
        "is_live": is_live,
        "skin_confidence": round(skin_score, 4),
        "status": "success",
        "reason": "OK" if is_live else "AI_SPOOF_DETECTION"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
