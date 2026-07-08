import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import json
import os
import time

# 1. HYPERPARAMETERS
BATCH_SIZE = 256
EPOCHS = 6 # Lowered to 6 because you were already at 95% by epoch 2!
LEARNING_RATE = 0.001
DATA_DIR = "train_data3" 

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_pin_memory = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # Apple Silicon Mac!
        use_pin_memory = False       # MPS doesn't support pin_memory yet
    else:
        device = torch.device("cpu")
        use_pin_memory = False
        
    print(f"Using device: {device}")

    # 2. DATA TRANSFORMATIONS 
    # Change THIS line in c.py before you retrain!
    transform = transforms.Compose([
        transforms.Resize((40, 80)),  # <--- Changed from 120 to 80
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # 3. LOAD DATASET
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    NUM_CLASSES = len(full_dataset.classes)
    class_names = full_dataset.classes

    print(f"Total classes detected: {NUM_CLASSES}")
    
    with open("class_mapping.json", "w") as f:
        json.dump(full_dataset.class_to_idx, f)
    print("Class mapping saved to class_mapping.json")

    # 4. TRAIN/VALIDATION SPLIT
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # The workers are safely spawned now!
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=use_pin_memory)

    # 5. INITIALIZE ResNet-18
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # 6. LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. TRAINING LOOP
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # VALIDATION
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        elapsed = time.time() - start_time
        print(f"[{elapsed:.2f}s] Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} Val Acc: {100*correct/total:.2f}%")

    # 8. SAVE MODEL
    torch.save(model.state_dict(), "togyzkumalak_model.pth")
    print("Model saved to togyzkumalak_model.pth. Done!")

# THIS IS THE MAGIC LINE THAT FIXES THE CRASH
if __name__ == '__main__':
    main()