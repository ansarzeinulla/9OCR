import os
import random
import glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# --- CONFIGURATION ---
INGREDIENTS_PATH = "ingredients"
OUTPUT_PATH = "train_data"
BOX_HEIGHT = 40
BOX_WIDTH = 120  # 3:1 Proportion
SAMPLES_PER_CLASS = 300 # Adjust based on your disk space

# 1. Generate the 163 Class Names
# Format: "72" (no x) or "72x" (with x)
move_classes = []
for start_hole in range(1, 10):
    for end_hole in range(1, 10):
        move_classes.append(f"{start_hole}{end_hole}")   # e.g., "72"
        move_classes.append(f"{start_hole}{end_hole}x")  # e.g., "72x"

classes = move_classes + ['empty']

os.makedirs(OUTPUT_PATH, exist_ok=True)

def get_random_ingredient(char):
    # char will be '1'-'9' or 'x'
    files = glob.glob(os.path.join(INGREDIENTS_PATH, char, "*.png"))
    if not files:
        raise ValueError(f"No images found for character: {char}")
    return Image.open(random.choice(files))

def create_move_image(class_name):
    # 1. Create the 3:1 paper background (light gray/off-white)
    bg_color = random.randint(220, 250) 
    img = Image.new('L', (BOX_WIDTH, BOX_HEIGHT), color=bg_color)
    
    if class_name == 'empty':
        return img

    chars_to_draw = list(class_name)
    
    for slot in range(len(chars_to_draw)):
        char = chars_to_draw[slot]
        char_img = get_random_ingredient(char) # This is White-on-Black
        
        # Resize
        size = random.randint(28, 36)
        char_img = char_img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Rotate
        char_img = char_img.rotate(random.randint(-10, 10), expand=False, fillcolor=0)

        # --- THE FIX: MASKED PASTING ---
        # Instead of inverting the whole square, we use the original 
        # White-on-Black image as a "mask". 
        
        # Create a solid black square of the same size
        ink_color = random.randint(0, 50) # Dark gray to black ink
        ink_layer = Image.new('L', (size, size), color=ink_color)
        
        # Position
        slot_center_x = (slot * 40) + 20
        paste_x = slot_center_x - (size // 2) + random.randint(-4, 4)
        paste_y = (BOX_HEIGHT // 2) - (size // 2) + random.randint(-3, 3)
        
        # We paste the "ink_layer" onto the "img" ONLY where "char_img" is white.
        img.paste(ink_layer, (paste_x, paste_y), mask=char_img)

    # 4. Final touch: Add a little bit of noise to the whole box
    # This makes the "pure" background look more like paper texture
    arr = np.array(img)
    noise = np.random.randint(-5, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)

# --- EXECUTION ---
print(f"Generating {len(classes)} classes...")

for cls in classes:
    class_dir = os.path.join(OUTPUT_PATH, cls)
    os.makedirs(class_dir, exist_ok=True)
    
    # Use fewer samples if you are just testing, increase for final training
    for i in range(SAMPLES_PER_CLASS):
        box_img = create_move_image(cls)
        # We save as '72x_1.png' etc.
        box_img.save(os.path.join(class_dir, f"{cls}_{i}.png"))
    
    print(f"Class {cls} generated.")

print(f"\nSuccess! Generated {len(classes)} folders in {OUTPUT_PATH}")