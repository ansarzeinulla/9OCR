import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
import cv2
import numpy as np

# --- EXACT SAME FUNCTION FROM b2.py ---
def crop_and_center_ink(cv2_gray_img, target_w=80, target_h=40, margin=4):
    ink_mask = cv2.bitwise_not(cv2_gray_img)
    coords = cv2.findNonZero(ink_mask)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 0 and h > 0:
            crop = cv2_gray_img[y:y+h, x:x+w]
            scale = min((target_w - 2*margin) / w, (target_h - 2*margin) / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            
            resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_crop
            return canvas
            
    return np.full((target_h, target_w), 255, dtype=np.uint8)

# ... (imports and crop_and_center_ink stay the same) ...

def preprocess_for_ai(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use a sharper threshold to keep lines crisp
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 7)
    
    # Clean up small noise dots ONLY
    ink_is_white = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(ink_is_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 15: # Smaller threshold for noise
            cv2.drawContours(ink_is_white, [cnt], -1, 0, -1)

    # DO NOT dilate here. Let the centering handle the size.
    final_img = cv2.bitwise_not(ink_is_white)

    # Center it
    centered_np = crop_and_center_ink(final_img, target_w=80, target_h=40)
    
    # Optional: Apply a standard 2x2 dilation to match training "thick" mode
    # if the handwriting is very thin.
    # centered_np = cv2.erode(centered_np, np.ones((2,2), np.uint8)) 

    pil_img = Image.fromarray(centered_np).convert('RGB')
    pil_img.save("debug_AI_eyes.png")
    return pil_img

# ... (main function stays same as previous) ...

def main(image_path):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    with open("class_mapping.json", "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    NUM_CLASSES = len(idx_to_class)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("togyzkumalak_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    pil_image = preprocess_for_ai(image_path)

    transform = transforms.Compose([
        transforms.Resize((40, 80)), # MUST match the 2:1 ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_classes = torch.topk(probabilities, 3)

    print(f"\n--- AI PREDICTIONS FOR '{image_path}' ---")
    for i in range(3):
        print(f"Choice {i+1}: Move '{idx_to_class[top_classes[i].item()]}' with {top_probs[i].item() * 100:.2f}% conf")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python d2.py <path_to_image>")
    else:
        main(sys.argv[1])