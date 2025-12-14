import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms

try:
    from inception_resnet_v1 import InceptionResnetV1
    print("[OK] Imported InceptionResnetV1 architecture.")
except ImportError:
    print("[FAIL] Could not import InceptionResnetV1. Is 'inception_resnet_v1.py' in this folder?")
    exit()

MODEL_PATH = "face_recognizer_model.pt"
TEST_IMAGE_PATH = "cropped_faces/Yash Saini/Yash Saini_face_0.jpg"

CLASS_NAMES = ['Anthony Burden', 'Jackson Boccanfuso', 'Jacob Brueck', 'Joshua Justice', 'Toshiro Gibson', 'Yash Saini']

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class ConvertPilToRawTensor:
    def __call__(self, pil_img):
        return torch.tensor(np.array(pil_img), dtype=torch.float32).permute(2, 0, 1)

def main():
    print("--- Starting Model Verification ---")

    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Model file '{MODEL_PATH}' not found.")
        return

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[FAIL] Test image '{TEST_IMAGE_PATH}' not found.")
        print("Please update TEST_IMAGE_PATH in the script to point to a valid image file.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    try:
        model = InceptionResnetV1(classify=True, num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print(f"[OK] Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        return

    try:
        img = Image.open(TEST_IMAGE_PATH).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            ConvertPilToRawTensor(),
            transforms.Lambda(fixed_image_standardization)
        ])
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        print(f"[OK] Image '{TEST_IMAGE_PATH}' processed.")
    except Exception as e:
        print(f"[FAIL] Error processing image: {e}")
        return

    try:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_name = CLASS_NAMES[predicted_idx.item()]
            conf_score = confidence.item()
            
            print("\n--- Prediction Results ---")
            print(f"Predicted Class: {predicted_name}")
            print(f"Confidence:      {conf_score:.4f} ({conf_score*100:.2f}%)")
            
            if conf_score > 0.5:
                print("\n[SUCCESS] Model is making confident predictions.")
            else:
                print("\n[WARNING] Prediction confidence is low. Verify the test image is a clear face.")
                
    except Exception as e:
        print(f"[FAIL] Error during inference: {e}")

if __name__ == "__main__":
    main()