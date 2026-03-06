
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from scenarios import SCENARIOS

def test_snapshot_accuracy():
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    scenario_names = list(SCENARIOS.keys())
    descriptions = [SCENARIOS[name]["description"] for name in scenario_names]
    
    snapshot_dir = "AI_Detection_Engine/snapshots"
    snapshots = [f for f in os.listdir(snapshot_dir) if f.endswith(".jpg")]
    
    if not snapshots:
        print("No snapshots found to test.")
        return

    # Test the first 3 snapshots
    for snapshot_name in snapshots[:3]:
        img_path = os.path.join(snapshot_dir, snapshot_name)
        image = Image.open(img_path)
        
        print(f"\nTesting Snapshot: {snapshot_name}")
        
        with torch.no_grad():
            inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            
            # Use logits_per_image which are scaled cosine similarities
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            similarities = (outputs.logits_per_image / model.logit_scale.exp()).cpu().numpy()[0]
            
        # Get top 3 predictions
        top_indices = similarities.argsort()[-3:][::-1]
        for idx in top_indices:
            name = scenario_names[idx]
            score = similarities[idx]
            prob = probs[idx]
            threshold = SCENARIOS[name]["threshold"]
            status = "PASS" if score > threshold else "FAIL"
            print(f"  - {name:<25}: Sim {score:.4f} | Prob {prob:.4f} (Thresh: {threshold}) [{status}]")

if __name__ == "__main__":
    test_snapshot_accuracy()
