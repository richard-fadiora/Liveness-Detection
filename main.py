import cv2
import os
import torch
import numpy as np
import gc
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
        
        # 1. Glare Ratio
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        glare_ratio = (cv2.countNonZero(binary) / gray.size) * 100
        
        # 2. Laplacian Variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Clean up grayscale intermediate images
        del gray
        del binary

        if glare_ratio > 12:
            return False, f"SCREEN_REFLECTION_DETECTED ({glare_ratio:.1f}%)"
        if laplacian_var < 40:
            return False, f"SPOOF_BLUR_DETECTED ({laplacian_var:.1f})"
        if laplacian_var > 500:
            return False, f"SCREEN_MOIRE_PATTERN_DETECTED ({laplacian_var:.1f})"

        return True, "OK"

    def check_motion_coherence(self, frames_rgb):
        if len(frames_rgb) < 2:
            return True, "OK"
            
        diffs = []
        for i in range(len(frames_rgb) - 1):
            d = cv2.absdiff(frames_rgb[i], frames_rgb[i+1])
            diffs.append(np.mean(d))
            del d # Free the difference frame immediately
        
        avg_motion = np.mean(diffs)
        if avg_motion < 0.15:
            return False, "STATIC_IMAGE_OR_SCREEN_DETECTION"
        
        return True, "OK"

brain = LivenessSDK_Brain()

@app.post("/v1/verify")
async def verify(files: List[UploadFile] = File(...)):
    cv_frames = []
    tensors = []
    
    try:
        # Limit to first 10 frames to prevent RAM exhaustion attacks
        processing_files = files[:10]

        for file in processing_files:
            # Step 1: Read bytes and convert to Numpy
            file_bytes = await file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            del file_bytes # Free raw bytes
            
            # Step 2: Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            del nparr # Free buffer
            
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                del img # Free BGR version
                
                # Step 3: Create tensor and move to GPU/Device immediately 
                # (Moves data out of System RAM)
                tensor = brain.transform(Image.fromarray(img_rgb)).to(brain.device)
                tensors.append(tensor)
                
                # Keep RGB only for the texture/motion checks
                cv_frames.append(img_rgb)
            
            await file.close()

        if not cv_frames:
            return {"is_live": False, "reason": "NO_IMAGE_DATA", "status": "fail"}

        # Passive Defense Checks
        tex_ok, tex_msg = brain.check_glare_and_texture(cv_frames[-1])
        if not tex_ok:
            return {"is_live": False, "reason": tex_msg, "status": "fail"}

        motion_ok, motion_msg = brain.check_motion_coherence(cv_frames)
        if not motion_ok:
            return {"is_live": False, "reason": motion_msg, "status": "fail"}

        # Clear RGB frames from System RAM before Model Inference
        del cv_frames

        # AI Model Inference
        batch = torch.stack(tensors).unsqueeze(0) 
        with torch.no_grad():
            logits = brain.model(batch)
            scores = torch.sigmoid(logits)
            skin_score = torch.mean(scores).item()

        is_live = skin_score > 0.75 

        return {
            "is_live": is_live,
            "skin_confidence": round(skin_score, 4),
            "status": "success",
            "reason": "OK" if is_live else "AI_SPOOF_DETECTION"
        }

    except Exception as e:
        return {"is_live": False, "reason": f"SERVER_ERROR: {str(e)}", "status": "error"}

    finally:
        # Final cleanup for the Garbage Collector
        if 'tensors' in locals(): del tensors
        if 'batch' in locals(): del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
