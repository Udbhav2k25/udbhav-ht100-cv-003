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

# --- CONFIGURATION (Single Entry Sentry) ---
CAMERA_INDEX = 0  
EVENT_TYPE = "ENTRY"

# Thresholds
EAR_THRESH = 0.22         # Below this = Eyes Closed (Blink)
RECOGNITION_THRESH = 0.6 # DeepFace ArcFace match
LIVENESS_TIMEOUT = 3.0   # Seconds to wait for a blink before calling it a Proxy

# Optimization
DEEPFACE_SKIP_FRAMES = 1 

# Load Secrets
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# --- INITIALIZATION ---
try:
    supabase: Client = create_client(url, key)
    print("[SYSTEM] Connected to Supabase.")
except:
    supabase = None
    print("[WARNING] Offline Mode.")

engine = pyttsx3.init()
engine.setProperty('rate', 150)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
ENCODINGS_FILE = os.path.join(DATA_DIR, "class_encodings.pkl")

# Ensure folders exist
os.makedirs(os.path.join(EVIDENCE_DIR, "proxies"), exist_ok=True)
os.makedirs(os.path.join(EVIDENCE_DIR, "intruders"), exist_ok=True)

known_data = {"encodings": [], "names": [], "roll_nos": []}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_data = pickle.load(f)
else:
    print("[ERROR] No database found! Run enrollment first.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# State Variables
last_logged = {}
COOLDOWN = 15
blink_counter = 0 

# Session Tracking
current_session = {
    "name": None,
    "roll": None,
    "start_time": 0,
    "state": "SEARCHING"
}
frame_counter = 0 

# --- FUNCTIONS ---

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except: pass

def calculate_ear(landmarks, indices):
    # Calculates Eye Aspect Ratio (EAR)
    A = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - 
                       np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - 
                       np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
    C = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - 
                       np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
    return (A + B) / (2.0 * C)

def check_blink(landmarks):
    """Checks if a blink event occurred in the current frame."""
    global blink_counter
    # Indices for Left and Right Eye (MediaPipe 468 Map)
    left_indices = [33, 160, 158, 133, 153, 144]
    right_indices = [362, 385, 387, 263, 373, 380]

    ear_left = calculate_ear(landmarks, left_indices)
    ear_right = calculate_ear(landmarks, right_indices)
    avg_ear = (ear_left + ear_right) / 2.0

    if avg_ear < EAR_THRESH:
        blink_counter += 1
    else:
        # If eyes were closed for at least 2 consecutive frames, a blink is confirmed
        if blink_counter >= 2:
            blink_counter = 0
            return True, avg_ear
        blink_counter = 0
    
    return False, avg_ear

def save_evidence(frame, folder_name, roll_no=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{roll_no or folder_name}_{timestamp}.jpg"
    local_path = os.path.join(EVIDENCE_DIR, folder_name, filename)
    cv2.imwrite(local_path, frame)
    
    if supabase:
        try:
            bucket_path = f"{folder_name}/{filename}"
            with open(local_path, 'rb') as f:
                supabase.storage.from_("evidence").upload(
                    path=bucket_path, file=f, file_options={"content-type": "image/jpeg"}
                )
            return supabase.storage.from_("evidence").get_public_url(bucket_path)
        except: pass
    return None

# --- MAIN LOOP ---

def start_entry_sentry():
    global frame_counter
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print(f"[ENTRY SENTRY] Active. Monitoring {EVENT_TYPE}...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        status_text = "Scanning..."
        status_color = (255, 255, 255)
        ear_val = 0.0

        results = face_mesh.process(rgb_frame)
        has_face = bool(results.multi_face_landmarks)

        if not has_face:
            current_session["name"] = None
            current_session["state"] = "SEARCHING"
        
        else:
            landmarks = results.multi_face_landmarks[0].landmark
            is_blinking, ear_val = check_blink(landmarks) # Check blink on every face frame

            if current_session["state"] == "SEARCHING":
                
                # DeepFace runs on every frame (Reliability Mode)
                try:
                    # 1. IDENTITY CHECK (DeepFace)
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
                            # Case A: Strong Match -> Start Liveness Timer
                            current_session["name"] = best_name
                            current_session["roll"] = best_roll
                            current_session["start_time"] = time.time()
                            current_session["state"] = "ANALYZING"
                        else:
                            # Case B: Weak Match / Unknown -> Intruder Alert
                            status_text = "INTRUDER DETECTED"
                            status_color = (0, 165, 255)
                            
                            # CRITICAL: Reset session state immediately after detection
                            current_session["name"] = None 
                            current_session["start_time"] = 0
                            current_session["state"] = "SEARCHING" 

                            now = time.time()
                            if "intruder" not in last_logged or (now - last_logged.get("intruder", 0) > COOLDOWN):
                                print("[ALERT] Intruder Detected!")
                                url = save_evidence(frame, "intruders")
                                if supabase:
                                    supabase.table("attendance_logs").insert({
                                        "event_type": EVENT_TYPE, "camera_id": CAMERA_INDEX,
                                        "is_spoof": False, "evidence_url": url
                                    }).execute()
                                last_logged["intruder"] = now

                except: pass

            elif current_session["state"] == "ANALYZING":
                # LIVENESS CHECK PHASE
                name = current_session["name"]
                elapsed = time.time() - current_session["start_time"]
                remaining = max(0, LIVENESS_TIMEOUT - elapsed)

                status_text = f"Verifying {name}... BLINK NOW ({remaining:.1f}s)"
                status_color = (0, 255, 255) # Yellow Warning

                if is_blinking:
                    # SUCCESS: Blinked in time
                    current_session["state"] = "GRANTED"
                    print(f"[ACCESS] {name} - Liveness Confirmed.")
                    speak(f"Welcome {name}")

                    now = time.time()
                    if name not in last_logged or (now - last_logged.get(name, 0) > COOLDOWN):
                        if supabase:
                            supabase.table("attendance_logs").insert({
                                "roll_no": current_session["roll"],
                                "event_type": EVENT_TYPE,
                                "camera_id": CAMERA_INDEX,
                                "is_spoof": False
                            }).execute()
                        last_logged[name] = now

                # TIMEOUT: Didn't blink in time -> PROXY
                elif elapsed > LIVENESS_TIMEOUT:
                    current_session["state"] = "PROXY"
                    print(f"[ALERT] Proxy Attempt by {name} (Static Face)")

                    now = time.time()
                    if "proxy" not in last_logged or (now - last_logged.get("proxy", 0) > COOLDOWN):
                        url = save_evidence(frame, "proxies", current_session["roll"])
                        if supabase:
                            supabase.table("attendance_logs").insert({
                                "roll_no": current_session["roll"],
                                "event_type": EVENT_TYPE,
                                "camera_id": CAMERA_INDEX,
                                "is_spoof": True,
                                "evidence_url": url
                            }).execute()
                        last_logged["proxy"] = now

            elif current_session["state"] == "GRANTED":
                # Final state: Reset session if face leaves frame (handled by global has_face check)
                status_text = f"ENTRY: {current_session['name']}"
                status_color = (0, 255, 0)

            elif current_session["state"] == "PROXY":
                status_text = "PROXY DETECTED (Static Face)"
                status_color = (0, 0, 255)
                # After proxy log, force reset the entire session immediately
                current_session["name"] = None 
                current_session["start_time"] = 0
                current_session["state"] = "SEARCHING" 


        # UI Overlay
        cv2.putText(display_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Debug EAR is only shown in ANALYZING state
        if current_session["state"] == "ANALYZING":
            cv2.putText(display_frame, f"Eye Ratio: {ear_val:.3f} (Need < {EAR_THRESH})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Entry Sentry (Active Security)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_entry_sentry()