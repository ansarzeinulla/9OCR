import os
import random
import glob
import numpy as np
from PIL import Image, ImageFilter
import cv2

INGREDIENTS_PATH = "ingredients"
OUTPUT_PATH = "train_data2"
BOX_HEIGHT = 40
BOX_WIDTH = 80
SAMPLES_PER_CLASS = 300 

move_classes = [f"{s}{e}" for s in range(1, 10) for e in range(1, 10)] + \
               [f"{s}{e}x" for s in range(1, 10) for e in range(1, 10)]
classes = move_classes + ['empty']

os.makedirs(OUTPUT_PATH, exist_ok=True)

def crop_and_center_ink(cv2_gray_img, target_w=80, target_h=40, margin=4):
    ink_mask = cv2.bitwise_not(cv2_gray_img)
    coords = cv2.findNonZero(ink_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        crop = cv2_gray_img[y:y+h, x:x+w]
        scale = min((target_w - 2*margin) / w, (target_h - 2*margin) / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized_crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
        start_x, start_y = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_crop
        return canvas
    return np.full((target_h, target_w), 255, dtype=np.uint8)

def create_move_image(class_name):
    if class_name == 'empty':
        return Image.new('L', (BOX_WIDTH, BOX_HEIGHT), color=255)

    # Start with a clean canvas
    img = Image.new('L', (200, 200), color=255)
    current_x = 40 
    for char in class_name:
        files = glob.glob(os.path.join(INGREDIENTS_PATH, char, "*.png"))
        char_img = Image.open(random.choice(files))
        size = random.randint(22, 28) if char == 'x' else random.randint(32, 42)
        char_img = char_img.resize((size, size), Image.Resampling.LANCZOS)
        char_img = char_img.rotate(random.randint(-12, 12), expand=True, fillcolor=0)
        bbox = char_img.getbbox()
        if bbox: char_img = char_img.crop(bbox)
        
        ink_layer = Image.new('L', char_img.size, color=0)
        paste_y = 100 - (char_img.height // 2) + random.randint(-5, 5)
        img.paste(ink_layer, (current_x, paste_y), mask=char_img)
        current_x += char_img.width + random.randint(-1, 4)

    # --- THE CRITICAL FIX: Center FIRST, then Dilate ---
    temp_arr = np.array(img)
    centered = crop_and_center_ink(temp_arr, BOX_WIDTH, BOX_HEIGHT)
    
    # Randomly vary thickness (Some thin, some thick)
    ink_is_white = cv2.bitwise_not(centered)
    thickness = random.choice([1, 2, 2]) # Weight it towards thicker lines
    kernel = np.ones((thickness, thickness), np.uint8)
    thick_ink = cv2.dilate(ink_is_white, kernel, iterations=1)
    
    # Add a tiny bit of blur/noise to simulate real camera focus
    final_img = cv2.bitwise_not(thick_ink)
    pil_final = Image.fromarray(final_img)
    if random.random() > 0.5:
        pil_final = pil_final.filter(ImageFilter.GaussianBlur(radius=0.3))
        
    return pil_final

# ... Execution loop remains same as previous ...
print("Generating centered data with variable thickness...")
for cls in classes:
    class_dir = os.path.join(OUTPUT_PATH, cls)
    os.makedirs(class_dir, exist_ok=True)
    for i in range(SAMPLES_PER_CLASS):
        create_move_image(cls).save(os.path.join(class_dir, f"{cls}_{i}.png"))