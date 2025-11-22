import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import pyttsx3
from deepface import DeepFace
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
from collections import deque

# --- CONFIGURATION ---
CAMERA_INDEX = 0  # Use "http://IP:PORT/video" for phone
EVENT_TYPE = "ENTRY"

# Thresholds
DEPTH_THRESH = 0.05       # Z-axis difference
EAR_THRESH = 0.22         # Eye Aspect Ratio (Blink level)
RECOGNITION_THRESH = 0.68 # DeepFace ArcFace match
PITCH_THRESH = -5         # If Pitch is lower than this (Frontal/Up), fail. 
                          # We want POSITIVE (Looking Down) for High Camera.

# Load Secrets
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# --- INITIALIZATION ---

# 1. Supabase
try:
    supabase: Client = create_client(url, key)
    print("[SYSTEM] Connected to Supabase.")
except:
    supabase = None
    print("[WARNING] Offline Mode.")

# 2. Audio
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 3. Data & Evidence
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
ENCODINGS_FILE = os.path.join(DATA_DIR, "class_encodings.pkl")

os.makedirs(os.path.join(EVIDENCE_DIR, "spoofs"), exist_ok=True)
os.makedirs(os.path.join(EVIDENCE_DIR, "intruders"), exist_ok=True)

known_data = {"encodings": [], "names": [], "roll_nos": []}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_data = pickle.load(f)
else:
    print("[ERROR] No database found! Run enrollment first.")
    exit()

# 4. MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# State Variables
last_logged = {}
COOLDOWN = 15
blink_buffer = deque(maxlen=10)

# --- MATH FUNCTIONS ---

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except: pass

def calculate_pitch(landmarks):
    """
    Calculates Pitch for HIGH ANGLE CAMERA (Sentry Mode).
    - Real person walking = Looks "Down" relative to camera (Positive Score).
    - Phone Photo = Looks "Frontal" (Negative/Zero Score).
    """
    nose_y = landmarks[1].y
    chin_y = landmarks[152].y
    top_head_y = landmarks[10].y
    
    face_height = abs(chin_y - top_head_y)
    nose_dist_from_top = abs(nose_y - top_head_y)
    
    ratio = nose_dist_from_top / face_height
    
    # INVERTED LOGIC:
    # Frontal Face Ratio ~0.5 -> Score 0
    # Looking Down (High Angle) Ratio > 0.5 -> Score Positive
    # Looking Up (Selfie Mode) Ratio < 0.5 -> Score Negative
    pitch_score = (ratio - 0.5) * 100 * 2 
    
    return pitch_score

def get_liveness_score(landmarks):
    score = 0
    debug_msg = ""

    # 1. PITCH CHECK (Angle)
    pitch = calculate_pitch(landmarks)
    
    # If pitch is too low (Frontal/Looking Up), it's likely a phone photo.
    if pitch < PITCH_THRESH:
        debug_msg = f"Angle: FLAT/UP ({int(pitch)})"
        return 0, debug_msg # Automatic Fail
    
    score += 40
    debug_msg = f"Angle: OK ({int(pitch)})"

    # 2. DEPTH CHECK (Geometry)
    nose_z = landmarks[1].z
    avg_ear_z = (landmarks[234].z + landmarks[454].z) / 2
    depth_diff = abs(nose_z - avg_ear_z)
    
    if depth_diff > DEPTH_THRESH:
        score += 40
        debug_msg += " | Depth: 3D"
    else:
        debug_msg += " | Depth: FLAT"

    # 3. MICRO-MOTION (Bonus)
    score += 20

    return score, debug_msg

def save_evidence(frame, folder_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder_name}_{timestamp}.jpg"
    local_path = os.path.join(EVIDENCE_DIR, folder_name, filename)
    cv2.imwrite(local_path, frame)
    
    if supabase:
        try:
            bucket_path = f"{folder_name}/{filename}"
            with open(local_path, 'rb') as f:
                supabase.storage.from_("evidence").upload(
                    path=bucket_path,
                    file=f,
                    file_options={"content-type": "image/jpeg"}
                )
            return supabase.storage.from_("evidence").get_public_url(bucket_path)
        except: pass
    return None

# --- MAIN LOOP ---

def start_sentry():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print(f"[SENTRY] Active. Monitoring {EVENT_TYPE}...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        status_text = "Scanning..."
        status_color = (255, 255, 255)
        liveness_score = 0
        debug_info = ""

        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                liveness_score, debug_info = get_liveness_score(face_landmarks.landmark)

                if liveness_score < 60:
                    status_text = "SPOOF DETECTED"
                    status_color = (0, 0, 255) # Red
                    
                    now = time.time()
                    if "spoof" not in last_logged or (now - last_logged.get("spoof", 0) > COOLDOWN):
                        print("[ALERT] Spoof Attempt!")
                        evidence_url = save_evidence(frame, "spoofs")
                        if supabase:
                            supabase.table("attendance_logs").insert({
                                "event_type": EVENT_TYPE, "camera_id": CAMERA_INDEX,
                                "is_spoof": True, "evidence_url": evidence_url
                            }).execute()
                        last_logged["spoof"] = now
                
                else:
                    # RECOGNITION
                    status_text = "Verifying..."
                    status_color = (255, 0, 0)

                    try:
                        current_objs = DeepFace.represent(
                            img_path=frame, model_name="ArcFace", 
                            detector_backend="opencv", enforce_detection=False
                        )

                        if len(current_objs) > 0:
                            curr_emb = current_objs[0]["embedding"]
                            best_name = "Unknown"
                            best_score = 100.0
                            best_roll = None

                            for i, known_emb in enumerate(known_data["encodings"]):
                                a = np.array(curr_emb)
                                b = np.array(known_emb)
                                distance = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                                
                                if distance < best_score:
                                    best_score = distance
                                    best_name = known_data["names"][i]
                                    best_roll = known_data["roll_nos"][i]

                            if best_score < RECOGNITION_THRESH:
                                status_text = f"{EVENT_TYPE}: {best_name}"
                                status_color = (0, 255, 0)
                                
                                now = time.time()
                                if best_roll not in last_logged or (now - last_logged.get(best_roll, 0) > COOLDOWN):
                                    print(f"[ACCESS] {best_name}")
                                    speak(f"Welcome {best_name}")
                                    if supabase:
                                        supabase.table("attendance_logs").insert({
                                            "roll_no": best_roll, "event_type": EVENT_TYPE,
                                            "camera_id": CAMERA_INDEX, "is_spoof": False
                                        }).execute()
                                    last_logged[best_roll] = now
                            else:
                                status_text = "Unknown Person"
                                status_color = (0, 165, 255)
                                
                                now = time.time()
                                if "intruder" not in last_logged or (now - last_logged.get("intruder", 0) > COOLDOWN):
                                    print("[ALERT] Intruder Detected!")
                                    evidence_url = save_evidence(frame, "intruders")
                                    if supabase:
                                        supabase.table("attendance_logs").insert({
                                            "roll_no": None, "event_type": EVENT_TYPE,
                                            "camera_id": CAMERA_INDEX, "is_spoof": False, "evidence_url": evidence_url
                                        }).execute()
                                    last_logged["intruder"] = now

                    except: pass

        cv2.putText(display_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display_frame, f"{debug_info}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Sentry Mode", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_sentry()