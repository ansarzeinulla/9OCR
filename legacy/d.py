import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys

def main(image_path):
    # 1. SETUP DEVICE
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 2. LOAD CLASS MAPPING
    # Reverse the mapping from { "19x": 5 } to { 5: "19x" }
    with open("class_mapping.json", "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    NUM_CLASSES = len(idx_to_class)

    # 3. LOAD THE MODEL
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load("togyzkumalak_model.pth", map_location=device))
    model = model.to(device)
    model.eval() # Set to evaluation mode!

    # 4. PREPARE THE IMAGE
    # Notice we removed ColorJitter because we don't augment during inference!
    transform = transforms.Compose([
        transforms.Resize((40, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    image = Image.open(image_path).convert('RGB') # Ensure it's 3-channel
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    # 5. PREDICT
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply Softmax to convert raw logits into percentages (0 to 1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # 6. GET TOP 3 PREDICTIONS
    top_probs, top_classes = torch.topk(probabilities, 3)

    print(f"\n--- AI PREDICTIONS FOR '{image_path}' ---")
    for i in range(3):
        prob = top_probs[i].item() * 100
        class_idx = top_classes[i].item()
        class_name = idx_to_class[class_idx]
        print(f"Choice {i+1}: Move '{class_name}' with {prob:.2f}% confidence")
    print("------------------------------------------\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        main(sys.argv[1])