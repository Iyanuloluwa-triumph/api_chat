import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Load Vision Model ONCE on module import
print("Loading MobileNetV2 for Image Search...")
vision_weights = MobileNet_V2_Weights.DEFAULT
vision_model = mobilenet_v2(weights=vision_weights)
vision_model.eval()

preprocess = vision_weights.transforms()
categories = vision_weights.meta["categories"]
print("MobileNetV2 loaded successfully!")

def extract_image_description(image_bytes: bytes) -> str:
    """Processes image bytes into a (3, 224, 224) RGB tensor and returns top 3 ImageNet tags."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = vision_model(input_tensor)
        
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    predicted_tags = [categories[cat_id] for cat_id in top3_catid]
    return ", ".join(predicted_tags)

# Prevent test code from running when app.py imports this module
if __name__ == "__main__":
    # Optional local testing block
    pass