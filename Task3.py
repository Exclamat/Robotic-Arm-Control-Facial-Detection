import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from mtcnn.mtcnn import MTCNN
from flask import Flask, Response
import math
import time
import serial
import serial.tools.list_ports
import re
from typing import List, Dict, Optional, NamedTuple

class TeensyLink:
    def __init__(self, port: str = None, baud: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self._open()

    def _detect_port(self) -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "ACM" in p.device or "USB" in p.device:
                return p.device
            if "usbmodem" in p.device:
                return p.device
            if "COM" in p.device:
                return p.device
        return None

    def _open(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        if not self.port:
            found = self._detect_port()
            if not found:
                self.port = '/dev/ttyACM0' 
            else:
                self.port = found
        
        print(f"[TeensyLink] Opening {self.port} @ {self.baud}...")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            print(f"[TeensyLink] Connection failed: {e}")
            self.ser = None

    def close(self):
        if self.ser:
            self.ser.close()
            self.ser = None

    def send_command(self, cmd: str, quiet_gap: float = 0.1, overall_timeout: float = 2.0) -> List[str]:
        if not self.ser:
            self._open()
            if not self.ser:
                return []

        payload = (cmd.strip() + "\n").encode("utf-8")
        try:
            self.ser.write(payload)
            self.ser.flush()
        except serial.SerialException:
            self._open()
            if self.ser:
                self.ser.write(payload)
                self.ser.flush()
            else:
                return []

        lines = []
        t_start = time.time()
        t_last = time.time()

        while True:
            if (time.time() - t_start) > overall_timeout:
                break
            
            if self.ser.in_waiting > 0:
                try:
                    chunk = self.ser.readline()
                    if chunk:
                        s = chunk.decode("utf-8", errors="replace").strip()
                        if s:
                            lines.append(s)
                        t_last = time.time()
                except Exception:
                    pass
            else:
                if lines and (time.time() - t_last > quiet_gap):
                    break
                time.sleep(0.01)
        
        return lines

    def send_line_noreply(self, line: str) -> None:
        if not self.ser:
            self._open()
            if not self.ser:
                return

        payload = (line.strip() + "\n").encode("utf-8")
        try:
            self.ser.write(payload)
            self.ser.flush()
        except serial.SerialException:
            self._open()
            if self.ser:
                self.ser.write(payload)
                self.ser.flush()

class MoveResult(NamedTuple):
    ok: bool
    reply: List[str]

_J1 = re.compile(r"\[J1\].*raw_deg=([-\d\.]+).*math_deg=([-\d\.]+)", re.IGNORECASE)
_J2 = re.compile(r"\[J2\].*raw_deg=([-\d\.]+).*math_deg=([-\d\.]+)", re.IGNORECASE)
_EE = re.compile(r"\[EE\].*x=([-\d\.]+).*y=([-\d\.]+)", re.IGNORECASE)

class Arm2D:
    def __init__(self):
        self.link = TeensyLink()

    def close(self):
        self.link.close()

    def initialize(self) -> MoveResult:
        lines = self.link.send_command("h", overall_timeout=15.0)
        success = any("homing complete" in line.lower() for line in lines)
        return MoveResult(ok=success, reply=lines)

    def set_velocity_math(self, j1_deg_per_s: float, j2_deg_per_s: float) -> None:
        cmd = f"V {j1_deg_per_s:.3f} {j2_deg_per_s:.3f}"
        self.link.send_line_noreply(cmd)

    def status(self):
        lines = self.link.send_command("s")
        data = {}
        for line in lines:
            m1 = _J1.search(line)
            if m1:
                data['j1_raw'] = float(m1.group(1))
                data['j1_math'] = float(m1.group(2))
            m2 = _J2.search(line)
            if m2:
                data['j2_raw'] = float(m2.group(1))
                data['j2_math'] = float(m2.group(2))
        return data

app = Flask(__name__)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

TRAINED_MODEL_PATH = "face_recognizer_model.pt"
DATA_DIR = "cropped_faces"
FRAME_SKIP = 4
CONFIDENCE_THRESHOLD = 0.7

TRACKING_DEADZONE = 40 
MAX_VELOCITY_J1 = 20.0 
MAX_VELOCITY_J2 = 10.0

SEARCH_AMPLITUDE_J1 = 15.0
SEARCH_SPEED_J1 = 0.5
SEARCH_AMPLITUDE_J2 = 5.0
SEARCH_SPEED_J2 = 0.2

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class ConvertPilToRawTensor:
    def __call__(self, pil_img):
        return torch.tensor(np.array(pil_img), dtype=torch.float32).permute(2, 0, 1)

def calculate_velocity_command(error_pixels, frame_dimension, deadzone, max_velocity, invert=False):
    if abs(error_pixels) < deadzone:
        return 0.0

    half_screen = frame_dimension / 2
    adjusted_error = error_pixels - (np.sign(error_pixels) * deadzone)
    norm_error = adjusted_error / (half_screen - deadzone)
    
    norm_error = max(-1.0, min(1.0, norm_error))

    scaled_value = np.sign(norm_error) * (norm_error ** 2)

    velocity = scaled_value * max_velocity
    
    if invert:
        velocity = -velocity
        
    return velocity

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

print("Initializing Robot Connection...")
try:
    arm = Arm2D()
    print("Robot connected successfully via Arm2D.")
except Exception as e:
    print(f"Failed to connect to robot: {e}")
    arm = None

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_count = 0
    last_known_results = [] 
    search_time = 0.0
    
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
                            
                            face_center_x = x + w // 2
                            face_center_y = y + h // 2
                            error_x = face_center_x - frame_center_x
                            error_y = face_center_y - frame_center_y

                            j1_vel = calculate_velocity_command(error_x, frame_width, TRACKING_DEADZONE, MAX_VELOCITY_J1, invert=False)
                            j2_vel = calculate_velocity_command(error_y, frame_height, TRACKING_DEADZONE, MAX_VELOCITY_J2, invert=True)

                            print(f"Tracking -> ErrX:{error_x} VJ1:{j1_vel:.2f} | ErrY:{error_y} VJ2:{j2_vel:.2f}")

                            if arm:
                                arm.set_velocity_math(j1_vel, j2_vel)

                            current_results.append((x, y, x2, y2, text, person_name, error_x, error_y, j1_vel, j2_vel))
                    except:
                        pass
                
                last_known_results = current_results
            else:
                last_known_results = []
                # Safety Stop: If we were tracking but lost the face, STOP immediately
                if arm:
                    arm.set_velocity_math(0.0, 0.0)
        
        if not last_known_results:
            search_time += 0.1 
            
            j1_search_vel = SEARCH_AMPLITUDE_J1 * math.sin(search_time * SEARCH_SPEED_J1)
            j2_search_vel = SEARCH_AMPLITUDE_J2 * math.cos(search_time * SEARCH_SPEED_J2)
            
            print(f"Searching -> Vel J1: {j1_search_vel:.2f}, J2: {j2_search_vel:.2f}")
            
            if arm:
                arm.set_velocity_math(j1_search_vel, j2_search_vel)
            
            cv2.putText(frame, "SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cmd_text = f"J1_Vel: {j1_search_vel:.2f} | J2_Vel: {j2_search_vel:.2f}"
            cv2.putText(frame, cmd_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
             pass

        cv2.line(frame, (frame_center_x - 10, frame_center_y), (frame_center_x + 10, frame_center_y), (0, 255, 255), 1)
        cv2.line(frame, (frame_center_x, frame_center_y - 10), (frame_center_x, frame_center_y + 10), (0, 255, 255), 1)
        cv2.rectangle(frame, 
                      (frame_center_x - TRACKING_DEADZONE, frame_center_y - TRACKING_DEADZONE),
                      (frame_center_x + TRACKING_DEADZONE, frame_center_y + TRACKING_DEADZONE),
                      (100, 100, 100), 1)

        for (x, y, x2, y2, text, person_name, error_x, error_y, j1_vel, j2_vel) in last_known_results:
            color = (0, 255, 0)
            if "Unknown" in text: color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if person_name != "Unknown":
                face_center_x = x + (x2-x)//2
                face_center_y = y + (y2-y)//2
                cv2.line(frame, (face_center_x, face_center_y), (frame_center_x, frame_center_y), (0, 255, 255), 1)

                cmd_text = f"J1_Vel: {j1_vel:.2f} | J2_Vel: {j2_vel:.2f}"
                cv2.putText(frame, cmd_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_count += 1

    cap.release()
    if arm:
        arm.close()

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Stream</title>
            <style>body { background-color: #333; color: white; text-align: center; font-family: sans-serif; }</style>
        </head>
        <body>
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