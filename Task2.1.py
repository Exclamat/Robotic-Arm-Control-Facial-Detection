import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from mtcnn.mtcnn import MTCNN

try:
    from inception_resnet_v1 import InceptionResnetV1
except ImportError:
    exit()

TRAINED_MODEL_PATH = "face_recognizer_model.pt"
DATA_DIR = "cropped_faces"
FRAME_SKIP = 4
CONFIDENCE_THRESHOLD = 0.7
SYMMETRY_THRESHOLD = 0.7

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class ConvertPilToRawTensor:
    def __call__(self, pil_img):
        return torch.tensor(np.array(pil_img), dtype=torch.float32).permute(2, 0, 1)

def get_gaze_symmetry(keypoints):
    try:
        nose = np.array(keypoints['nose'])
        left_eye = np.array(keypoints['left_eye'])
        right_eye = np.array(keypoints['right_eye'])
        mouth_left = np.array(keypoints['mouth_left'])
        mouth_right = np.array(keypoints['mouth_right'])
        
        nose_to_left_eye = np.linalg.norm(nose - left_eye)
        nose_to_right_eye = np.linalg.norm(nose - right_eye)
        eye_ratio = nose_to_left_eye / nose_to_right_eye
        
        nose_to_left_mouth = np.linalg.norm(nose - mouth_left)
        nose_to_right_mouth = np.linalg.norm(nose - mouth_right)
        mouth_ratio = nose_to_left_mouth / nose_to_right_mouth
        
        is_symmetric = (SYMMETRY_THRESHOLD < eye_ratio < (1/SYMMETRY_THRESHOLD)) and \
                         (SYMMETRY_THRESHOLD < mouth_ratio < (1/SYMMETRY_THRESHOLD))
                         
        return is_symmetric
    except:
        return False

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(DATA_DIR):
        return
    
    dataset = datasets.ImageFolder(DATA_DIR)
    class_names = dataset.classes
    num_classes = len(class_names)
    if num_classes == 0:
        return

    resnet = InceptionResnetV1(
        classify=True,
        num_classes=num_classes
    ).to(device)

    try:
        resnet.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device, weights_only=True))
    except:
        return
        
    resnet.eval()

    data_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        ConvertPilToRawTensor(),
        transforms.Lambda(fixed_image_standardization)
    ])

    mtcnn = MTCNN()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
        
    frame_count = 0
    last_known_results = [] 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if frame_count % FRAME_SKIP == 0:
            current_results = []
            
            detections = mtcnn.detect_faces(frame_rgb)
            
            for det in detections:
                try:
                    x, y, w, h = det['box']
                    x, y = max(0, x), max(0, y)
                    x2, y2 = x + w, y + h
                    keypoints = det['keypoints']
                    
                    face_crop = frame_rgb[y:y2, x:x2]
                    face_pil = Image.fromarray(face_crop)
                    face_tensor = data_transform(face_pil).to(device).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = resnet(face_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence_score, pred_idx = torch.max(probabilities, 1)
                        
                    confidence = confidence_score.item()
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        person_name = class_names[pred_idx.item()]
                        text = f"{person_name} ({confidence*100:.0f}%)"
                    else:
                        person_name = "Unknown"
                        text = "Unknown"
                    
                    is_looking = get_gaze_symmetry(keypoints)
                    
                    current_results.append((x, y, x2, y2, text, is_looking, person_name))

                except:
                    pass
            
            last_known_results = current_results
        
        for (x, y, x2, y2, text, is_looking, person_name) in last_known_results:
            
            color = (0, 255, 0)
            if "Unknown" in text:
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        frame_height, frame_width, _ = frame.shape
        if last_known_results:
            first_person = last_known_results[0]
            is_looking = first_person[5]
            person_name = first_person[6]
            
            if is_looking:
                if person_name != "Unknown":
                    gaze_status_text = f"{person_name} is looking at me!"
                else:
                    gaze_status_text = "Looking at me!"
                
                cv2.putText(frame, gaze_status_text, (10, frame_height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()