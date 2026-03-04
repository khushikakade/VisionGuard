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
        
        with torch.no_grad():
            inputs = self.processor(text=self.descriptions, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            # Ensure we have a tensor (handle different transformers versions)
            text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            self.text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            
        print("Model loaded and scenarios initialized.")

    def process_video(self, video_source=0, sample_rate=1.0):
        """
        Processes video from a source (file or camera).
        sample_rate: seconds between processed frames
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30 # Default for webcams
        
        frame_interval = int(fps * sample_rate)
        frame_count = 0
        
        print(f"Starting detection on {video_source}... Press 'q' to stop.")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    self.detect_scenarios(frame)
                
                # Show video feed (optional, for local debugging)
                cv2.imshow('Smart CCTV Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def detect_scenarios(self, frame):
        # Convert OpenCV BGR to RGB for PIL
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        with torch.no_grad():
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            image_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            
            # Cosine similarity
            similarities = (image_features @ self.text_features.T).squeeze(0)
            
        timestamp = time.strftime("%H:%M:%S")
        
        for i, score in enumerate(similarities):
            scenario_name = self.scenario_names[i]
            threshold = SCENARIOS[scenario_name]["threshold"]
            
            if score > threshold:
                print(f"[{timestamp}] ALERT DETECTED: {scenario_name} (Score: {score:.4f})")
                # In a real system, you'd trigger a callback or send to a message queue here

if __name__ == "__main__":
    # You can pass a video file path here or 0 for webcam
    pipeline = DetectionPipeline()
    pipeline.process_video(video_source=0, sample_rate=1.5)
