
import torch
from transformers import CLIPProcessor, CLIPModel
from scenarios import SCENARIOS

def test_scenario_separation():
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    scenario_names = list(SCENARIOS.keys())
    descriptions = [SCENARIOS[name]["description"] for name in scenario_names]
    
    with torch.no_grad():
        inputs = processor(text=descriptions, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        
        # Calculate cross-similarity between all scenario descriptions
        similarity_matrix = (text_features @ text_features.T).cpu().numpy()
        
    print(f"{'Scenario 1':<25} | {'Scenario 2':<25} | {'Similarity'}")
    print("-" * 65)
    
    high_overlaps = []
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            sim = similarity_matrix[i][j]
            if sim > 0.8: # High similarity threshold
                high_overlaps.append((scenario_names[i], scenario_names[j], sim))
                print(f"{scenario_names[i]:<25} | {scenario_names[j]:<25} | {sim:.4f}")
    
    if not high_overlaps:
        print("No high overlaps found between scenario descriptions.")
    else:
        print(f"\nFound {len(high_overlaps)} highly overlapping scenario descriptions.")

if __name__ == "__main__":
    test_scenario_separation()
