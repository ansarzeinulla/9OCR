import os
import torchvision
from PIL import Image

# 1. Create the folder structure
base_dir = "ingredients"
classes_we_want = {
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 
    6: "6", 7: "7", 8: "8", 9: "9", 33: "x"
}

for folder_name in classes_we_want.values():
    os.makedirs(os.path.join(base_dir, folder_name), exist_ok=True)

# 2. Load the dataset
dataset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True, download=True)

print("Extracting and FIXING rotation... please wait.")

counts = {k: 0 for k in classes_we_want.keys()}
limit_per_class = 2000 

for i in range(len(dataset)):
    img, label = dataset[i] # img is a PIL Image
    
    if label in classes_we_want:
        if counts[label] < limit_per_class:
            label_name = classes_we_want[label]
            file_path = os.path.join(base_dir, label_name, f"{label_name}_{counts[label]}.png")
            
            # --- THE FIX ---
            # EMNIST is stored (width, height) instead of (height, width)
            # We transpose it to make it human-readable
            fixed_img = img.transpose(Image.TRANSPOSE)
            # ---------------
            
            fixed_img.save(file_path)
            counts[label] += 1

    if all(c >= limit_per_class for c in counts.values()):
        break

print("Success! Check your 'ingredients' folder now. They should be upright.")