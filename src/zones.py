import cv2
import numpy as np
import pickle
import os

ZONE_FILE = os.path.join(os.getcwd(), "zones_data.pkl")
ZONES = {}

if os.path.exists(ZONE_FILE):
    with open(ZONE_FILE, 'rb') as f:
        ZONES = pickle.load(f)
    print(f"[Zones] {len(ZONES)} elements charges.")
    for k in ZONES:
        print(f"  → Clé zone: '{k}'")
else:
    print("[Zones] ⚠️ Fichier introuvable.")


def _normalize_cam_name(cam_name):
    """Convertit 'Cam 1' → 'CAM1', 'Cam 2' → 'CAM2'"""
    return cam_name.replace(" ", "").upper()


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def has_crossed_line(prev_pos, curr_pos, line_start, line_end):
    if prev_pos is None or curr_pos is None:
        return False
    return segments_intersect(prev_pos, curr_pos, line_start, line_end)


def point_to_line_distance(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def get_overlap_line(cam_name):
    cam_key = _normalize_cam_name(cam_name)
    line_key = f"{cam_key}_OVERLAP_LINE"
    if line_key in ZONES and isinstance(ZONES[line_key], tuple):
        return ZONES[line_key]
    return None


def is_near_overlap_line(box, cam_name, max_distance=80):
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    line_data = get_overlap_line(cam_name)
    if line_data is None:
        return False
    p1, p2 = line_data
    dist = point_to_line_distance((cx, cy), p1, p2)
    return dist < max_distance


def get_zone_name(box, cam_name):
    cam_key = _normalize_cam_name(cam_name)
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for z_name, z_data in ZONES.items():
        if z_name.startswith(cam_key) and not isinstance(z_data, tuple):
            if cv2.pointPolygonTest(z_data, (cx, cy), False) >= 0:
                return z_name
    return None


def is_in_entry_zone(box, cam_name):
    name = get_zone_name(box, cam_name)
    return name and "ENTRY" in name.upper()


def get_histogram(crop):
    if crop is None or crop.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    except:
        return None


def get_histogram_score(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 0.0
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return (score + 1.0) / 2.0


def draw_zones(frame, cam_name):
    cam_key = _normalize_cam_name(cam_name)
    for z_name, z_data in ZONES.items():
        if z_name.startswith(cam_key):
            if isinstance(z_data, tuple):
                cv2.line(frame, tuple(z_data[0]), tuple(z_data[1]),
                         (255, 0, 0), 2)
                cv2.putText(frame, z_name, tuple(z_data[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                cv2.polylines(frame, [z_data], True, (0, 255, 0), 2)
                cv2.putText(frame, z_name, tuple(z_data[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame