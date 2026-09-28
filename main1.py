import cv2
import sys
import os
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.config import (REID_VISUAL_THRESHOLD, GENDER_CONFIDENCE_THRESHOLD,
                        DEVICE, GENDER_STABLE_FRAMES)
from src.gender_detector import GenderDetector
from src.tracker import PersonTracker
from src.utils import get_body_crop, draw_detection
from src.logger import Logger
from src.global_tracker import GlobalIdManager
from src.reid_manager import VisualEncoder
from src.color_detector import analyze_person_colors
from src.zones import (get_zone_name, is_in_entry_zone, get_histogram,
                        draw_zones, _normalize_cam_name, ZONES)


TARGET_DISPLAY_WIDTH = 1280  

WINDOW_NAME = "Tracking Multi-Cam"


def process_frame(frame, tracker, gender_det, cam_name, global_manager):
    if frame is None:
        return None

    results = tracker.track(frame)
    local_detections = []

    if results and len(results) > 0 \
            and results[0].boxes is not None \
            and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, local_id in zip(boxes, ids):
            crop = get_body_crop(frame, box)
            gender = "Inconnu"

            if crop is not None:
                pred_gender, conf = gender_det.predict(crop)
                if conf > GENDER_CONFIDENCE_THRESHOLD:
                    gender = pred_gender

            color_top, color_bottom = "Inconnu", "Inconnu"
            if crop is not None:
                color_top, color_bottom = analyze_person_colors(frame, box)

            histogram = get_histogram(crop)
            zone_name = get_zone_name(box, cam_name)
            is_entry = is_in_entry_zone(box, cam_name)
            area = (box[2] - box[0]) * (box[3] - box[1])

            det_info = {
                'local_id': local_id,
                'gender': gender,
                'box': box,
                'crop': crop,
                'info': {
                    'gender': gender,
                    'color_top': color_top,
                    'area': area,
                    'zone_name': zone_name,
                    'is_entry': is_entry,
                    'histogram': histogram
                }
            }
            local_detections.append(det_info)

    global_manager.update(cam_name, local_detections)
    frame = draw_zones(frame, cam_name)

    for det in local_detections:
        gid = det.get('global_id')
        if gid is not None:
            stable_frames = det.get('stable_frames', 0)
            stable_gender = det.get('stable_gender', 'Inconnu')
            event = det.get('event', 'UPDATE')
            frame = draw_detection(frame, det['box'], gid, stable_gender,
                                   stable_frames, GENDER_STABLE_FRAMES, event)

    return frame


def main():
    print(f"🎬 DÉMARRAGE DU SYSTÈME MULTI-CAM")
    print(f"⚙️  Device : {DEVICE}")

    # Debug : afficher les clés de zones
    print(f"📍 Clés zones disponibles : {list(ZONES.keys())}")
    print(f"📍 'Cam 1' normalisé → '{_normalize_cam_name('Cam 1')}'")
    print(f"📍 'Cam 2' normalisé → '{_normalize_cam_name('Cam 2')}'")

    video_path = "video/9.1.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la vidéo")
        return

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"📹 Vidéo : {video_w}x{video_h} @ {video_fps:.1f}fps")

    gender_det = GenderDetector()
    visual_encoder = VisualEncoder()
    tracker_cam1 = PersonTracker()
    tracker_cam2 = PersonTracker()
    logger = Logger("detections_log.csv")

    global_manager = GlobalIdManager(
        visual_encoder=visual_encoder,
        threshold=REID_VISUAL_THRESHOLD,
        logger=logger
    )

    print("🚀 Lancement du tracking...")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    half_w = video_w // 2
    combined_w = half_w * 2
    scale_factor = TARGET_DISPLAY_WIDTH / combined_w
    display_w = TARGET_DISPLAY_WIDTH
    display_h = int(video_h * scale_factor)

    print(f"🖥️  Fenêtre : {display_w}x{display_h}")
    cv2.resizeWindow(WINDOW_NAME, display_w, display_h)

    is_fullscreen = False
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📽️ Fin de la vidéo.")
            break

        h, w = frame.shape[:2]
        mid = w // 2
        frame1 = frame[:, :mid]
        frame2 = frame[:, mid:]

        try:
            frame1 = process_frame(frame1, tracker_cam1, gender_det,
                                   "Cam 1", global_manager)
            frame2 = process_frame(frame2, tracker_cam2, gender_det,
                                   "Cam 2", global_manager)
        except Exception as e:
            print(f"❌ Erreur : {e}")
            import traceback
            traceback.print_exc()
            continue

        combined = np.hstack((frame1, frame2))

        # Labels caméras
        label_scale = max(0.8, combined.shape[0] / 600.0)
        font_thick = max(2, int(2 * label_scale))
        cv2.putText(combined, "CAM 1", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                    (0, 255, 0), font_thick)
        cv2.putText(combined, "CAM 2", (mid + 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                    (0, 255, 0), font_thick)

        # Ligne séparatrice
        cv2.line(combined, (mid, 0), (mid, h), (255, 255, 255), 2)

        # Redimensionnement
        h_disp, w_disp = combined.shape[:2]
        s = TARGET_DISPLAY_WIDTH / w_disp
        target_h = int(h_disp * s)

        combined_resized = cv2.resize(combined,
                                       (TARGET_DISPLAY_WIDTH, target_h),
                                       interpolation=cv2.INTER_LINEAR)

        # FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            fps_text = f"FPS: {fps:.1f} | IDs: {len(global_manager.visual_db)} | " \
                       f"Pending: {len(global_manager.pending_crossings)} | " \
                       f"Lost: {len(global_manager.cross_camera_lost)}"
            cv2.putText(combined_resized, fps_text,
                        (20, combined_resized.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, combined_resized)

        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"⏱️ {time.strftime('%H:%M:%S')} | FPS: {fps:.1f} | "
                  f"IDs: {len(global_manager.visual_db)} | "
                  f"Pending: {len(global_manager.pending_crossings)} | "
                  f"Lost: {len(global_manager.cross_camera_lost)} | "
                  f"RelinkQueue: {len(global_manager.recently_lost)}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME,
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME,
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, display_w, display_h)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Fin du programme.")


if __name__ == "__main__":
    main()