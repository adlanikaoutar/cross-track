import cv2
import os
import time
from src.gender_detector import GenderDetector
from src.reid_manager import VisualEncoder
from src.global_tracker import GlobalIdManager

# -------------------------------
# 1️⃣ Configuration
# -------------------------------
RTSP_URL = "rtsp://..."
SAVE_DIR = "dataset_frames"
os.makedirs(os.path.join(SAVE_DIR, "male"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "female"), exist_ok=True)

CAPTURE_INTERVAL = 5  # secondes entre captures pour la même personne

# -------------------------------
# 2️⃣ Initialisation modèles
# -------------------------------
reid = VisualEncoder()
global_tracker = GlobalIdManager(visual_encoder=reid, threshold=0.6)
gender_model = GenderDetector()  # ton modèle de genre

# -------------------------------
# 3️⃣ Variables
# -------------------------------
last_capture_time = {}   # dict pour limiter fréquence capture
frame_count = 0

# -------------------------------
# 4️⃣ Connexion caméra
# -------------------------------
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # réduit le lag

if not cap.isOpened():
    print("❌ Impossible d'ouvrir la caméra")
    exit()

# -------------------------------
# 5️⃣ Boucle principale
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erreur de lecture du flux")
        break

    # ---------------------------
    # Détection YOLO (personne)
    # ---------------------------
    # TODO: Remplacez par votre modèle YOLO
    # detections = [[x1, y1, x2, y2], ...]
    # track_ids = [id_local, ...]
    detections = []  # placeholder
    track_ids = []   # placeholder

    # Exemple pour tester tout le frame comme "détection"
    # detections = [[0, 0, frame.shape[1], frame.shape[0]]]
    # track_ids = [0]

    detections_list = []

    for track_id, bbox in zip(track_ids, detections):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        # ---------------------------
        # 6️⃣ Prédiction genre
        # ---------------------------
        gender = gender_model.predict(crop)  # "male" ou "female"

        # ---------------------------
        # 7️⃣ Préparer la détection pour GlobalIdManager
        # ---------------------------
        detections_list.append({
            'local_id': track_id,
            'crop': crop,
            'info': {'gender': gender}
        })

    # ---------------------------
    # 8️⃣ Mise à jour ID global
    # ---------------------------
    global_tracker.update(cam_name="cam1", local_detections=detections_list)

    # ---------------------------
    # 9️⃣ Sauvegarde des frames détectées
    # ---------------------------
    for det in detections_list:
        global_id = det['global_id']
        crop = det['crop']
        gender = det['info']['gender']

        last_time = last_capture_time.get(global_id, 0)
        if time.time() - last_time < CAPTURE_INTERVAL:
            continue  # pas encore le temps de capturer

        last_capture_time[global_id] = time.time()
        frame_count += 1

        filename = os.path.join(SAVE_DIR, gender, f"{global_id}_{frame_count:04d}.jpg")
        cv2.imwrite(filename, crop)
        print(f"✅ Frame sauvegardée : {filename}")

    # ---------------------------
    # 10️⃣ Affichage optionnel
    # ---------------------------
    cv2.imshow("Camera Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()