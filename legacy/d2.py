import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import json
import sys
import cv2
import numpy as np

def preprocess_for_ai(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur (Smooths out paper texture before thresholding)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. FIX HOLLOW LINES: Increase block size to 35, constant to 10.
    # This forces it to see the whole pen stroke, not just the edges!
    # 1. Lower the constant from 10 to 5 (This tells the math: "Don't be so aggressive at deleting faint ink!")
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        35, 5  # <-- Changed 10 to 5
    )

    ink_is_white = cv2.bitwise_not(thresh)

    # 2. Bigger kernel to fill the white holes inside the '9'
    kernel_close = np.ones((5,5), np.uint8)
    closed_ink = cv2.morphologyEx(ink_is_white, cv2.MORPH_CLOSE, kernel_close)

    # 3. DILATION (THE NEW FIX): Thicken the ink to bridge the gap in the '3'
    kernel_dilate = np.ones((3,3), np.uint8)
    thick_ink = cv2.dilate(closed_ink, kernel_dilate, iterations=1)

    # 4. Remove noise dots (same as before, but on thick_ink)
    contours, _ = cv2.findContours(thick_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:  
            cv2.drawContours(thick_ink, [cnt], -1, 0, -1)

    # Invert back to black ink on white paper
    final_img = cv2.bitwise_not(thick_ink)

    # --- ALIGNMENT FIX (b2.py vs d2.py) ---
    # b2.py creates un-stretched characters accurately placed in a 120x40 (3:1) bounding box.
    # To prevent PyTorch from squashing arbitrary crops, we pad it to exactly 120x40 first!
    pil_img = Image.fromarray(final_img).convert('RGB')
    pil_img = ImageOps.pad(pil_img, (120, 40), color=(255, 255, 255))
    # ---------------------------------------

    # Save debug image strictly as the AI sees it
    pil_img.save("debug_AI_eyes.png")

    return pil_img

def main(image_path):
    # SETUP DEVICE
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # LOAD CLASS MAPPING
    with open("class_mapping.json", "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    NUM_CLASSES = len(idx_to_class)

    # LOAD MODEL
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load("togyzkumalak_model_OBED.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # PREPARE IMAGE USING OPENCV PREPROCESSING
    pil_image = preprocess_for_ai(image_path)

    transform = transforms.Compose([
        transforms.Resize((40, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # PREDICT
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_classes = torch.topk(probabilities, 3)

    print(f"\n--- AI PREDICTIONS FOR '{image_path}' ---")
    for i in range(3):
        prob = top_probs[i].item() * 100
        class_idx = top_classes[i].item()
        class_name = idx_to_class[class_idx]
        print(f"Choice {i+1}: Move '{class_name}' with {prob:.2f}% confidence")
    print("------------------------------------------\n")
    print("Check 'debug_AI_eyes.png' in your folder to see the preprocessed image!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python d2.py <path_to_image>")
    else:
        main(sys.argv[1])