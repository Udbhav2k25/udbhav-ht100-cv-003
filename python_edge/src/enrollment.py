import cv2
import pickle
import os
import numpy as np
from deepface import DeepFace
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Setup Configuration
# We look for .env in the parent directory (python_edge/.env)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Initialize Cloud Connection
try:
    supabase: Client = create_client(url, key)
    print("[SYSTEM] Connected to Supabase Cloud.")
except Exception as e:
    print(f"[WARNING] Cloud connection failed: {e}")
    print("Continuing in OFFLINE MODE (Local storage only).")
    supabase = None

# Local Storage Paths
# Go up one level from src to python_edge, then into data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ENCODINGS_FILE = os.path.join(DATA_DIR, "class_encodings.pkl")

# Ensure data folder exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_existing_db():
    """Loads the pickle file if it exists, or creates a new empty one."""
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                return pickle.load(f)
        except EOFError:
            return {"encodings": [], "names": [], "roll_nos": []}
    return {"encodings": [], "names": [], "roll_nos": []}

def save_db(data):
    """Saves the updated face data to disk."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"[SUCCESS] Database saved to {ENCODINGS_FILE}")

def sync_to_cloud(roll_no, name):
    """Sends the student ID to Supabase so the Dashboard knows they exist."""
    if not supabase:
        return
    
    print(f"[CLOUD] Registering {name}...")
    try:
        # Matches the 'students' table we created in SQL
        data = {"roll_no": roll_no, "full_name": name}
        supabase.table("students").upsert(data).execute()
        print("[CLOUD] Sync Successful.")
    except Exception as e:
        print(f"[ERROR] Supabase Sync Failed: {e}")

def run_enrollment():
    print("\n=== BIOMETRIC SENTRY: ENROLLMENT (DeepFace Powered) ===")
    roll_no = input("Enter Roll Number: ").strip()
    name = input("Enter Full Name: ").strip()

    # 1. Register in Cloud
    sync_to_cloud(roll_no, name)

    # 2. Load Local DB
    data = load_existing_db()

    # 3. Start Camera
    print("\n[INSTRUCTION] Opening Camera...")
    print("To make the 'High Angle' detection work, we need 3 views:")
    print("  1. Look STRAIGHT at the camera.")
    print("  2. Look slightly UP (Chin up) - Simulates door cam view.")
    print("  3. Look slightly DOWN or TILTED.")
    print("Press 's' to save a frame. Capture at least 5 frames.")
    print("Press 'q' to finish.\n")

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from webcam.")
            break

        # Copy frame for display
        display_frame = frame.copy()

        # UI Text
        cv2.putText(display_frame, f"Captured: {count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to Save", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Enrollment Mode", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 's' to Save
        if key == ord('s'):
            print("[PROCESSING] Extracting DeepFace Embeddings... (Wait)")
            try:
                # DeepFace Magic: Extract embeddings using ArcFace
                # We enforce detection to ensure a face actually exists
                embedding_objs = DeepFace.represent(
                    img_path=frame, 
                    model_name="ArcFace", 
                    detector_backend="opencv",
                    enforce_detection=True
                )

                # DeepFace returns a list, take the first one
                if len(embedding_objs) > 0:
                    embedding = embedding_objs[0]["embedding"]
                    
                    # Store the data
                    data["encodings"].append(embedding)
                    data["names"].append(name)
                    data["roll_nos"].append(roll_no)
                    count += 1
                    print(f"[SUCCESS] Frame {count} captured!")
            
            except ValueError:
                print("[WARNING] No face detected! Adjust your angle and try again.")
            except Exception as e:
                # Most common error is face not detected
                print(f"[RETRY] Face not clear. Move closer or adjust light.")

        # 'q' to Quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 4. Save to Disk
    if count > 0:
        save_db(data)
        print(f"\n[DONE] Enrolled {name} with {count} embeddings.")
    else:
        print("\n[CANCELLED] No faces saved.")

if __name__ == "__main__":
    run_enrollment()