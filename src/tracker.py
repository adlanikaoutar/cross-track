import os
from ultralytics import YOLO
from src.config import DEVICE

class PersonTracker:
    def __init__(self):
        print(f"[Tracker] Initialisation YOLOv8 sur {DEVICE}...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        custom_config_path = os.path.join(project_root, 'custom_tracker_config.yaml')

        self.model = YOLO('yolov8n.pt') 
        
        # Utilise le device défini dans config
        self.model.to(DEVICE)
        
 

        if os.path.exists(custom_config_path):
            self.tracker_config = custom_config_path
        else:
            self.tracker_config = 'botsort.yaml' 
            
        print("[Tracker] ✅ Prêt.")

    def track(self, frame):
        results = self.model.track(
            frame, 
            persist=True, 
            classes=[0], 
            tracker=self.tracker_config, 
            verbose=False,
            iou=0.5,
            device=DEVICE,
            imgsz=416,         # Résolution réduite pour la vitesse
            half=False         # ❌ Désactivé pour stabilité (correction bug)
        )
        return results