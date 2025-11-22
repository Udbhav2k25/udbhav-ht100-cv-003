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
CAMERA_INDEX = 0
EVENT_TYPE = "ENTRY"

# Thresholds
EAR_THRESH = 0.21         # Below this = Eyes Closed (Blink)
RECOGNITION_THRESH = 0.68 # DeepFace ArcFace match
BLINK_CONSEC_FRAMES = 2   # How fast a blink is (frames)

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

os.makedirs(os.path.join(EVIDENCE_DIR, "spoofs"), exist_ok=True)
os.makedirs(os.path.join(EVIDENCE_DIR, "intruders"), exist_ok=True)

known_data = {"encodings": [], "names": [], "roll_nos": []}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_data = pickle.load(f)
else:
    print("[ERROR] No database found! Run enrollment first.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_landmarks=True # CRITICAL for Iris/Eye points
)

last_logged = {}
COOLDOWN = 15

# --- STATE VARIABLES FOR BLINK ---
# We track the state across frames
blink_counter = 0
is_blinking = False
liveness_verified = False
liveness_timestamp = 0
LIVENESS_VALIDITY_DURATION = 2.0 # How long liveness lasts after a blink (seconds)

# --- HELPER FUNCTIONS ---

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except: pass

def calculate_ear(landmarks, indices):
    """
    Calculates Eye Aspect Ratio (EAR).
    EAR = (Vertical Dist A + Vertical Dist B) / (2 * Horizontal Dist C)
    """
    # Vertical lines
    A = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - 
                       np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
    B = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - 
                       np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
    # Horizontal line
    C = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - 
                       np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
    ear = (A + B) / (2.0 * C)
    return ear

def check_blink(landmarks):
    """Returns True if a blink just finished."""
    global blink_counter, is_blinking

    # Indices for Left and Right Eye (MediaPipe 468 Map)
    # Left Eye: [33, 160, 158, 133, 153, 144]
    # Right Eye: [362, 385, 387, 263, 373, 380]
    left_indices = [33, 160, 158, 133, 153, 144]
    right_indices = [362, 385, 387, 263, 373, 380]

    ear_left = calculate_ear(landmarks, left_indices)
    ear_right = calculate_ear(landmarks, right_indices)
    avg_ear = (ear_left + ear_right) / 2.0

    # Check threshold
    if avg_ear < EAR_THRESH:
        blink_counter += 1
    else:
        # If eyes were closed for enough frames, and now open -> BLINK!
        if blink_counter >= BLINK_CONSEC_FRAMES:
            blink_counter = 0
            return True, avg_ear
        blink_counter = 0
    
    return False, avg_ear

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
    global liveness_verified, liveness_timestamp
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print(f"[SENTRY] Active. Monitoring {EVENT_TYPE}...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        status_text = "WAITING FOR FACE..."
        status_color = (200, 200, 200)
        ear_val = 0.0

        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # 1. CHECK BLINK (Primary Liveness)
                has_blinked, ear_val = check_blink(face_landmarks.landmark)
                
                if has_blinked:
                    liveness_verified = True
                    liveness_timestamp = time.time()
                    print("[LIVENESS] Blink Detected! Unlock.")

                # Check if liveness is still valid (expires after 2 seconds)
                if liveness_verified and (time.time() - liveness_timestamp < LIVENESS_VALIDITY_DURATION):
                    
                    # --- LIVENESS PASSED: RECOGNIZE ---
                    status_text = "Verifying Identity..."
                    status_color = (255, 0, 0) # Blue

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
                                status_color = (0, 255, 0) # Green
                                
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
                                status_color = (0, 165, 255) # Orange
                                # Intruder Logic (Optional: log evidence)

                    except: pass

                else:
                    # --- LIVENESS FAILED / EXPIRED ---
                    status_text = "PLEASE BLINK EYES"
                    status_color = (0, 255, 255) # Yellow
                    liveness_verified = False # Reset

                    # If EAR is consistently high (staring) for too long, it's suspicious
                    # But for now, we just wait for a blink.
                    # A photo will stay in this state forever.

        # UI
        cv2.putText(display_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Debug EAR (Helpful to see if blink is registering)
        # Normal Eye ~0.30, Blink < 0.20
        debug_str = f"Eye Ratio: {ear_val:.3f} (Blink if < {EAR_THRESH})"
        cv2.putText(display_frame, debug_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Sentry Mode", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_sentry()