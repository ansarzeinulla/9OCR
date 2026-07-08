from fastai.vision.all import load_learner, PILImage
from PIL import ImageOps

# 1. Load the model
print("Loading model...")
learn = load_learner('handwriting_classifier_best.pkl')

# 2. Open the image and FIX the iPhone rotation bug!
raw_img = PILImage.create('IMG_1521 (4) (1).jpg')

# FastAI's PILImage.create actually attempts to transpose EXIF under the hood,
# but let's double check by physically saving what the model sees:
raw_img.save("what_the_model_actually_sees.png")
print("Saved 'what_the_model_actually_sees.png'. Please open this file and check if it is upright!")

# 3. Predict using FastAI's official pipeline
pred, pred_idx, probs = learn.predict(raw_img)

print(f"\n--- Prediction ---")
print(f"Predicted Class: {pred}")
print(f"Confidence: {probs[pred_idx]*100:.2f}%")

# Show top 3
class_probabilities = dict(zip(learn.dls.vocab, probs.tolist()))
sorted_guesses = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
print("\nTop 3 Guesses:")
for cls, prob in sorted_guesses[:3]:
    print(f"Class {cls}: {prob * 100:.2f}%")