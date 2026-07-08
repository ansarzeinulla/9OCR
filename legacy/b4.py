import os
import random
import glob
import numpy as np
from PIL import Image, ImageFilter
import cv2

# --- CONFIGURATION ---
INGREDIENTS_PATH = "ingredients"
OUTPUT_PATH = "train_data5"
BOX_HEIGHT = 40
BOX_WIDTH = 80  
SAMPLES_PER_CLASS = 300 

# 1. Generate the 163 Class Names
move_classes = []
for start_hole in range(1, 10):
    for end_hole in range(1, 10):
        move_classes.append(f"{start_hole}{end_hole}")   
        move_classes.append(f"{start_hole}{end_hole}x")  

classes = move_classes + ['empty']
os.makedirs(OUTPUT_PATH, exist_ok=True)

def get_random_ingredient(char):
    files = glob.glob(os.path.join(INGREDIENTS_PATH, char, "*.png"))
    if not files:
        raise ValueError(f"No images found for character: {char}")
    # Convert to grayscale to act perfectly as an alpha mask
    return Image.open(random.choice(files)).convert('L')

def add_camera_noise(img_array):
    """Simulates rough paper texture and camera sensor noise"""
    noise = np.random.randint(0, 25, img_array.shape, dtype='uint8')
    # Subtracting noise makes random pixels slightly darker, like paper grain
    noisy_img = np.clip(img_array.astype(int) - noise, 0, 255).astype('uint8')
    return noisy_img

def create_move_image(class_name):
    # --- PHILOSOPHY 1: IMPERFECT PAPER BACKGROUND ---
    # Random RGB values mimicking paper under different lighting (off-white, warm, cool)
    bg_gray = random.randint(200, 255)

    img = Image.new('L', (BOX_WIDTH, BOX_HEIGHT), color=bg_gray)
    
    if class_name == 'empty':
        img_arr = add_camera_noise(np.array(img))
        img = Image.fromarray(img_arr)
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))

    # --- PHILOSOPHY 2: REAL PEN INK COLORS ---
    # Randomly pick black, dark blue, or bright blue ink

    ink_color = random.choice([0,15,30,45])

    chars_to_draw = list(class_name)
    prepared_chars = []
    total_width = 0
    gaps = []

    for i, char in enumerate(chars_to_draw):
        char_img = get_random_ingredient(char)
        
        # Sizing
        size = random.randint(14, 18) if char == 'x' else random.randint(20, 26)
        char_img = char_img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Rotation
        char_img = char_img.rotate(random.randint(-15, 15), expand=True, fillcolor=0)
        
        # --- PHILOSOPHY 3: VARIABLE PEN PRESSURE ---
        # Randomly thicken or thin the stroke using morphology
        char_arr = np.array(char_img)
        kernel = np.ones((2, 2), np.uint8)
        thickness_op = random.choice(['dilate', 'erode', 'none', 'dilate']) # Bias slightly towards thicker
        
        if thickness_op == 'dilate':
            char_arr = cv2.dilate(char_arr, kernel, iterations=1)
        elif thickness_op == 'erode':
            char_arr = cv2.erode(char_arr, kernel, iterations=1)
            
        char_img = Image.fromarray(char_arr)

        # Crop tight around the character
        bbox = char_img.getbbox()
        if bbox:
            char_img = char_img.crop(bbox)
            
        prepared_chars.append(char_img)
        total_width += char_img.width
        
        # Gaps (Allowing negative numbers means strokes might overlap naturally!)
        if i < len(chars_to_draw) - 1:
            gap = random.randint(-6, 1) 
            gaps.append(gap)
            total_width += gap

    # Translation
    max_start_x = BOX_WIDTH - total_width
    start_x = random.randint(2, max(2, max_start_x - 2))

    # Paste using the EMNIST mask
    current_x = start_x
    for i, char_mask in enumerate(prepared_chars):
        # Create a solid block of our chosen ink color
        ink_layer = Image.new('RGB', char_mask.size, color=ink_color)
        
        max_y = BOX_HEIGHT - char_mask.height
        paste_y = random.randint(1, max(1, max_y - 1))
        
        # Paste the ink onto the paper, using the white EMNIST digit as the stencil
        img.paste(ink_layer, (current_x, paste_y), mask=char_mask)
        
        if i < len(gaps):
            current_x += char_mask.width + gaps[i]

    # --- PHILOSOPHY 4: THE "DIRTY" REALITY ---
    # 1. Add noise
    img_arr = np.array(img)
    img_arr = add_camera_noise(img_arr)
    final_img = Image.fromarray(img_arr)
    
    # 2. Add random camera blur
    blur_radius = random.uniform(0.1, 0.8)
    final_img = final_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return final_img

# --- EXECUTION ---
print(f"Generating {len(classes)} classes with Real-World Domain Shift...")

for cls in classes:
    class_dir = os.path.join(OUTPUT_PATH, cls)
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(SAMPLES_PER_CLASS):
        box_img = create_move_image(cls)
        box_img.save(os.path.join(class_dir, f"{cls}_{i}.png"))
    
    if (classes.index(cls) + 1) % 10 == 0:
        print(f"Generated {classes.index(cls) + 1}/{len(classes)} classes...")

print(f"\nSuccess! Generated highly robust dataset in {OUTPUT_PATH}")