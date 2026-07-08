import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import torchvision.transforms as T
from PIL import Image
from fastai.vision.all import load_learner
import pickle
import plum._resolver

# --- THE MODEL LOADING PATCH ---
class PatchedResolver(plum._resolver.Resolver):
    def __setstate__(self, state):
        if isinstance(state, dict):
            for k, v in state.items():
                try: setattr(self, k, v)
                except AttributeError: pass
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, dict):
                    for k, v in item.items():
                        try: setattr(self, k, v)
                        except AttributeError: pass

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Resolver' and 'plum' in module:
            return PatchedResolver
        return super().find_class(module, name)

class SafePickle:
    Unpickler = SafeUnpickler
# -------------------------------

# 1. Load the model using our safe unpickler
print("Loading model...")
learn = load_learner('handwriting_classifier.pkl', pickle_module=SafePickle)
print("Model loaded successfully!")

# 2. Extract the raw PyTorch model and the classes (vocab)
pytorch_model = learn.model
pytorch_model.eval()            # Set model to evaluation mode
vocab = list(learn.dls.vocab)    # Get the list of your 163 classes

# 3. Standard PyTorch Image Preprocessing (Bypasses FastAI transforms entirely)
# This perfectly matches the Resize and standard ImageNet normalization used by ResNet
preprocess = T.Compose([
    T.Resize((40, 120)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],  # Standard ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Open the image using Pillow and preprocess it
img = Image.open('IMG_1521 (4) (1).jpg').convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension [1, 3, 40, 120]

# 5. Raw PyTorch Forward Pass
print("Running raw PyTorch inference...")
with torch.no_grad():
    outputs = pytorch_model(input_tensor)
    # Apply softmax to convert raw outputs to probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# 6. Extract the top 3 predictions
top_prob, top_catid = torch.topk(probabilities, 3)

print("\nPrediction Results:")
for i in range(top_prob.size(0)):
    class_name = vocab[top_catid[i].item()]
    confidence = top_prob[i].item() * 100
    print(f"Class {class_name}: {confidence:.2f}%")