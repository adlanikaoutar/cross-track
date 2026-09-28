import os
import sys
import torch

try:
    BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
except Exception:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, 'models')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    try:
        torch.backends.cudnn.benchmark = True
        print(f"⚙️ Mode GPU activé : {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"⚠️ Erreur GPU : {e}. Passage en CPU.")
        DEVICE = "cpu"
else:
    print("⚙️ Mode CPU activé")

GENDER_MODEL_PATH = os.path.join(MODELS_DIR, 'efficientnet_gender_v7.1_local.pt')
GENDER_DICT = {0: 'Femme', 1: 'Homme'}

CAMERA_1_SOURCE = "rtsp://..."
CAMERA_2_SOURCE = "rtsp://..."

CONFIDENCE_THRESHOLD = 0.5
GENDER_CONFIDENCE_THRESHOLD = 0.75

REID_VISUAL_THRESHOLD = 0.45
CROSS_CAMERA_THRESHOLD = 0.40
CROWD_THRESHOLD = 6
CROSS_CAM_TIMEOUT = 15.0

# --- RELINKING INTRA-CAMÉRA ---
RELINK_TIMEOUT = 10.0
RELINK_MAX_DISTANCE = 100
RELINK_CLOSE_DISTANCE = 60
RELINK_IOU_THRESHOLD = 0.15
RELINK_REID_THRESHOLD = 0.40
RELINK_HIST_THRESHOLD = 0.35
RELINK_COMBINED_MIN = 0.30

# --- STABILITÉ GENRE ---
GENDER_STABLE_FRAMES = 3
GENDER_HISTORY_LENGTH = 20
GENDER_LOCK_FRAMES = 5

# --- POIDS HYBRIDE CROSS-CAM ---
WEIGHT_REID = 0.40
WEIGHT_HISTOGRAM = 0.20
WEIGHT_DIR = 0.10
WEIGHT_SIZE = 0.05
WEIGHT_SOURCE_OVERLAP = 0.25

# --- BONUS / PÉNALITÉS CROSS-CAM ---
CROSS_CAM_OVERLAP_BONUS = 0.05
CROSS_CAM_ACTIVE_PENALTY = 0.15
CROSS_CAM_PENDING_BONUS = 0.10
CROSS_CAM_LOST_BONUS = 0.03

# --- FILTRE SOURCE OVERLAP ---
CROSS_CAM_REQUIRE_SOURCE_OVERLAP = True
CROSS_CAM_SOURCE_OVERLAP_MIN = 0.5
CROSS_CAM_ACTIVE_SOURCE_OVERLAP_MIN = 0.5

# FALLBACK : Si aucun candidat ne passe le filtre SOURCE_OVERLAP,
# on fait un 2ème passage sans ce filtre mais avec un seuil plus élevé
CROSS_CAM_FALLBACK_THRESHOLD = 0.50        # Seuil plus élevé pour le fallback

# --- SEUILS VISUELS MINIMUM ---
CROSS_CAM_MIN_REID = 0.45
CROSS_CAM_MIN_HIST = 0.20

# --- GENRE (PÉNALITÉ SEULEMENT, PAS DE BLOCAGE) ---
CROSS_CAM_GENDER_MATCH_BONUS = 0.05
CROSS_CAM_GENDER_MISMATCH_PENALTY = 0.15
CROSS_CAM_GENDER_LOCKED_MISMATCH_PENALTY = 0.30

# --- PLAFOND BONUS ---
CROSS_CAM_MAX_FINAL_SCORE_FROM_BONUS = 0.20

# --- POST-MATCH VERIFICATION ---
CROSS_CAM_POST_MATCH_MIN_REID = 0.40

# --- ANTI-SWITCH ---
VISUAL_CONFIRMATION_THRESHOLD = 0.25
MIN_ID_STABLE_SECONDS = 1.5

# --- LOST LOGGING ---
LOST_GRACE_SECONDS = 2.0
AUTO_PENDING_OVERLAP_DISTANCE = 300      