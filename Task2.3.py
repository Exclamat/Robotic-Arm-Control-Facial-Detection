import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from mtcnn.mtcnn import MTCNN
from flask import Flask, Response

app = Flask(__name__)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

TRAINED_MODEL_PATH = "face_recognizer_model.pt"
DATA_DIR = "cropped_faces"
FRAME_SKIP = 4
CONFIDENCE_THRESHOLD = 0.7
SYMMETRY_THRESHOLD = 0.7

TRACKING_DEADZONE = 40 
MAX_DEGREE_STEP_J1 = 2.0 
MAX_DEGREE_STEP_J2 = 2.0

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

def calculate_joint_movement(error_pixels, frame_dimension, deadzone, max_step):
    if abs(error_pixels) < deadzone:
        return 0.0

    half_screen = frame_dimension / 2
    adjusted_error = error_pixels - (np.sign(error_pixels) * deadzone)
    norm_error = adjusted_error / (half_screen - deadzone)
    
    norm_error = max(-1.0, min(1.0, norm_error))

    scaled_value = np.sign(norm_error) * (norm_error ** 2)

    movement = scaled_value * max_step
    
    return movement

try:
    from inception_resnet_v1 import InceptionResnetV1
except ImportError:
    print("Error: inception_resnet_v1.py not found.")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.isdir(DATA_DIR):
    print(f"Error: {DATA_DIR} not found")
    exit()

dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
num_classes = len(class_names)

resnet = InceptionResnetV1(classify=True, num_classes=num_classes).to(device)

try:
    if device.type == 'cpu':
            resnet.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location='cpu', weights_only=True))
    else:
            resnet.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device, weights_only=True))
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
    
resnet.eval()

data_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    ConvertPilToRawTensor(),
    transforms.Lambda(fixed_image_standardization)
])

mtcnn = MTCNN()

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_count = 0
    last_known_results = [] 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % FRAME_SKIP == 0:
            current_results = []
            try:
                detections = mtcnn.detect_faces(frame_rgb)
            except:
                detections = []
            
            if detections:
                largest_face = None
                max_area = 0

                for det in detections:
                    _, _, w, h = det['box']
                    if w * h > max_area:
                        max_area = w * h
                        largest_face = det
                
                if largest_face:
                    det = largest_face
                    try:
                        x, y, w, h = det['box']
                        x, y = max(0, x), max(0, y)
                        x2, y2 = x + w, y + h
                        keypoints = det['keypoints']
                        
                        face_crop = frame_rgb[y:y2, x:x2]
                        if face_crop.size != 0: 
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

                            face_center_x = x + w // 2
                            face_center_y = y + h // 2
                            error_x = face_center_x - frame_center_x
                            error_y = face_center_y - frame_center_y

                            j1_move = calculate_joint_movement(error_x, frame_width, TRACKING_DEADZONE, MAX_DEGREE_STEP_J1)
                            
                            j2_move = calculate_joint_movement(error_y, frame_height, TRACKING_DEADZONE, MAX_DEGREE_STEP_J2)

                            print(f"Scaled Joint Movements -> J1: {j1_move:.4f}, J2: {j2_move:.4f}")

                            current_results.append((x, y, x2, y2, text, is_looking, person_name, error_x, error_y, j1_move, j2_move))
                    except:
                        pass
                
                last_known_results = current_results
            else:
                last_known_results = []
        
        cv2.line(frame, (frame_center_x - 10, frame_center_y), (frame_center_x + 10, frame_center_y), (0, 255, 255), 1)
        cv2.line(frame, (frame_center_x, frame_center_y - 10), (frame_center_x, frame_center_y + 10), (0, 255, 255), 1)
        cv2.rectangle(frame, 
                      (frame_center_x - TRACKING_DEADZONE, frame_center_y - TRACKING_DEADZONE),
                      (frame_center_x + TRACKING_DEADZONE, frame_center_y + TRACKING_DEADZONE),
                      (100, 100, 100), 1)

        for (x, y, x2, y2, text, is_looking, person_name, error_x, error_y, j1_move, j2_move) in last_known_results:
            color = (0, 255, 0)
            if "Unknown" in text: color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if person_name != "Unknown":
                face_center_x = x + (x2-x)//2
                face_center_y = y + (y2-y)//2
                cv2.line(frame, (face_center_x, face_center_y), (frame_center_x, frame_center_y), (0, 255, 255), 1)

                cmd_text = f"J1(Base): {j1_move:+.2f} | J2(Shldr): {j2_move:+.2f}"
                cv2.putText(frame, cmd_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if last_known_results:
            first_person = last_known_results[0]
            is_looking = first_person[5]
            person_name = first_person[6]
            
            if is_looking:
                gaze_status_text = f"{person_name} is looking at me!" if person_name != "Unknown" else "Looking at me!"
                cv2.putText(frame, gaze_status_text, (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_count += 1

    cap.release()

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Stream</title>
            <style>body { background-color: #333; color: white; text-align: center; font-family: sans-serif; }</style>
        </head>
        <body>
            <h1>Tracking Feed</h1>
            <p></p>
            <img src="/video_feed" style="width: 80%; border: 2px solid #555;">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)