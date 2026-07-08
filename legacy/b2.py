import os
import random
import glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
# --- CONFIGURATION ---
INGREDIENTS_PATH = "ingredients"
OUTPUT_PATH = "train_data2"
BOX_HEIGHT = 40
BOX_WIDTH = 120  # 3:1 Proportion
SAMPLES_PER_CLASS = 300 # Adjust based on your disk space

# 1. Generate the 163 Class Names
move_classes = []
for start_hole in range(1, 10):
    for end_hole in range(1, 10):
        move_classes.append(f"{start_hole}{end_hole}")   # e.g., "72"
        move_classes.append(f"{start_hole}{end_hole}x")  # e.g., "72x"

classes = move_classes + ['empty']

os.makedirs(OUTPUT_PATH, exist_ok=True)

def get_random_ingredient(char):
    files = glob.glob(os.path.join(INGREDIENTS_PATH, char, "*.png"))
    if not files:
        raise ValueError(f"No images found for character: {char}")
    return Image.open(random.choice(files))

def create_move_image(class_name):
    # 1. Create a pure, solid white paper background (255)
    # This matches the clean white background of the post-processed real photos
    img = Image.new('L', (BOX_WIDTH, BOX_HEIGHT), color=255)
    
    if class_name == 'empty':
        return img

    chars_to_draw = list(class_name)
    prepared_chars = []
    total_width = 0
    gaps = []

    # --- STEP 1: PREPARE AND MEASURE ALL CHARACTERS ---
    for i, char in enumerate(chars_to_draw):
        char_img = get_random_ingredient(char)
        
        # Make 'x' slightly smaller than numbers
        if char == 'x':
            size = random.randint(20, 26)
        else:
            size = random.randint(28, 36)
            
        char_img = char_img.resize((size, size), Image.Resampling.LANCZOS)
        char_img = char_img.rotate(random.randint(-10, 10), expand=False, fillcolor=0)
        
        # Crop the black space around the EMNIST digit
        bbox = char_img.getbbox()
        if bbox:
            char_img = char_img.crop(bbox)
            
        prepared_chars.append(char_img)
        total_width += char_img.width
        
        # Calculate random gap
        if i < len(chars_to_draw) - 1:
            gap = random.randint(-2, 5) 
            gaps.append(gap)
            total_width += gap

    # --- STEP 2: RANDOM TRANSLATION ---
    max_start_x = BOX_WIDTH - total_width
    if max_start_x <= 2: 
        start_x = 2 
    else:
        start_x = random.randint(2, max_start_x - 2)

    # --- STEP 3: PASTE THE CHARACTERS (Using Solid Black Ink) ---
    current_x = start_x
    for i, char_img in enumerate(prepared_chars):
        # We use solid pure black ink (0) to match post-threshold images
        ink_layer = Image.new('L', char_img.size, color=0)
        
        max_y = BOX_HEIGHT - char_img.height
        paste_y = random.randint(2, max(2, max_y - 2))
        
        img.paste(ink_layer, (current_x, paste_y), mask=char_img)
        
        if i < len(gaps):
            current_x += char_img.width + gaps[i]

    # --- STEP 4: DILATION ALIGNMENT ---
    # Convert to numpy array for OpenCV
    arr = np.array(img)
    
    # Invert the image so the ink is white (required for dilation)
    ink_is_white = cv2.bitwise_not(arr)
    
    # Apply a 3x3 dilation kernel to thicken the strokes
    # This matches the dilated thickness of the real-world processed pen strokes!
    kernel = np.ones((3,3), np.uint8)
    thick_ink = cv2.dilate(ink_is_white, kernel, iterations=1)
    
    # Invert back: Ink is Black (0), Background is Pure White (255)
    final_img = cv2.bitwise_not(thick_ink)

    return Image.fromarray(final_img)

# --- EXECUTION ---
print(f"Generating {len(classes)} classes with Dynamic Spacing...")

for cls in classes:
    class_dir = os.path.join(OUTPUT_PATH, cls)
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(SAMPLES_PER_CLASS):
        box_img = create_move_image(cls)
        box_img.save(os.path.join(class_dir, f"{cls}_{i}.png"))
    
    # Optional print to track progress
    if (classes.index(cls) + 1) % 10 == 0:
        print(f"Generated {classes.index(cls) + 1}/{len(classes)} classes...")

print(f"\nSuccess! Generated {len(classes)} folders in {OUTPUT_PATH}")