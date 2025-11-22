import cv2
import numpy as np
import pickle
import os
import time
import pyttsx3
from deepface import DeepFace
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

# --- CONFIGURATION (Exit Camera) ---
CAMERA_INDEX = 1  # IMPORTANT: Change this to your USB camera or Phone 2 URL
EVENT_TYPE = "EXIT" # Logs 'EXIT' event type
RECOGNITION_THRESH = 0.60 # DeepFace ArcFace match

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
    print("[WARNING] Running Offline.")

# Audio
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
ENCODINGS_FILE = os.path.join(DATA_DIR, "class_encodings.pkl")

os.makedirs(os.path.join(EVIDENCE_DIR, "intruders"), exist_ok=True) # Only intruders are logged here

known_data = {"encodings": [], "names": [], "roll_nos": []}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_data = pickle.load(f)
else:
    print("[ERROR] Database empty! Run enrollment.py first.")
    exit()

# Cooldowns
last_logged = {}
COOLDOWN = 10 # Shorter cooldown for exiting quickly

# --- FUNCTIONS ---

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except: pass

def save_evidence(frame, folder_name):
    # This function is simpler, only logs intruders locally/to cloud.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder_name}_{timestamp}.jpg"
    local_path = os.path.join(EVIDENCE_DIR, folder_name, filename)
    cv2.imwrite(local_path, frame)
    
    if supabase:
        try:
            bucket_path = f"{folder_name}/{filename}"
            with open(local_path, 'rb') as f:
                # Assuming 'evidence' is your bucket name
                supabase.storage.from_("evidence").upload(
                    path=bucket_path,
                    file=f,
                    file_options={"content-type": "image/jpeg"}
                )
            return supabase.storage.from_("evidence").get_public_url(bucket_path)
        except: pass
    return None

# --- MAIN LOOP (Passive Mode) ---

def start_exit_sentry():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print(f"[EXIT SENTRY] Active. Monitoring {EVENT_TYPE}...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        status_text = "PASSIVE EXIT SCAN"
        status_color = (100, 100, 255) # Purple/Passive

        # 1. IDENTITY CHECK (DeepFace)
        try:
            current_objs = DeepFace.represent(
                img_path=frame, model_name="ArcFace", 
                detector_backend="opencv", enforce_detection=False # We rely on DeepFace's detector
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
                    # ACCESS GRANTED (PASSIVE EXIT)
                    status_text = f"{EVENT_TYPE}: {best_name}"
                    status_color = (0, 255, 0) # Green
                    
                    now = time.time()
                    if best_roll not in last_logged or (now - last_logged.get(best_roll, 0) > COOLDOWN):
                        print(f"[EXIT LOGGED] {best_name}")
                        speak(f"Goodbye {best_name}")
                        
                        if supabase:
                            supabase.table("attendance_logs").insert({
                                "roll_no": best_roll, "event_type": EVENT_TYPE,
                                "camera_id": CAMERA_INDEX, "is_spoof": False
                            }).execute()
                        last_logged[best_roll] = now
                else:
                    # UNKNOWN PERSON (INTRUDER)
                    status_text = "INTRUDER ALERT (Exit)"
                    status_color = (0, 165, 255) # Orange
                    
                    now = time.time()
                    intruder_key = "intruder_exit"
                    if intruder_key not in last_logged or (now - last_logged.get(intruder_key, 0) > COOLDOWN):
                        print("[ALERT] Intruder Detected at Exit!")
                        save_evidence(frame, "intruders")
                        last_logged[intruder_key] = now

        except Exception: 
            pass

        cv2.putText(display_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.imshow("Exit Sentry (Passive)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_exit_sentry()