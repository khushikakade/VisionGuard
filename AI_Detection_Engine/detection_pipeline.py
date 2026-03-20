import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from scenarios import SCENARIOS
import time

class DetectionPipeline:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading model {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Pre-calculate text embeddings for scenarios
        self.scenario_embeddings = {}
        self.scenario_names = list(SCENARIOS.keys())
        self.descriptions = [SCENARIOS[name]["description"] for name in self.scenario_names]
        
        # Temporal buffer: {scenario_name: count}
        self.detection_buffer = {name: 0 for name in self.scenario_names}
        self.buffer_threshold = 2 # Must detect in 2 consecutive samples
        
        with torch.no_grad():
            inputs = self.processor(text=self.descriptions, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            # Handle different transformers output formats
            text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            self.text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            
        print("Model loaded and scenarios initialized.")

    def detect_scenarios(self, frame):
        # Convert OpenCV BGR to RGB for PIL
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        with torch.no_grad():
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            # Handle different transformers output formats
            image_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            
            # Cosine similarity
            similarities = (image_features @ self.text_features.T).squeeze(0)
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Winner-Takes-All Logic: Find the highest score first
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item()
        best_scenario = self.scenario_names[best_idx]
        
        # 2. Threshold Check
        threshold = SCENARIOS[best_scenario]["threshold"]
        
        detected_alerts = []
        
        # 3. Temporal Filtering
        if best_score > threshold:
            self.detection_buffer[best_scenario] += 1
            
            # Only alert if the same scenario is detected consistently
            if self.detection_buffer[best_scenario] >= self.buffer_threshold:
                alert_info = {
                    "timestamp": timestamp,
                    "scenario_key": best_scenario,
                    "scenario": best_scenario.replace("_", " ").title(),
                    "score": round(float(best_score), 4)
                }
                print(f"[{timestamp}] CONFIRMED ALERT: {best_scenario} (Score: {best_score:.4f})")
                detected_alerts.append(alert_info)
        else:
            # Decay buffer if not detected
            for name in self.detection_buffer:
                self.detection_buffer[name] = max(0, self.detection_buffer[name] - 1)
        
        return detected_alerts

if __name__ == "__main__":
    # You can pass a video file path here or 0 for webcam
    pipeline = DetectionPipeline()
    pipeline.process_video(video_source=0, sample_rate=1.5)
