import cv2
import numpy as np


def get_body_crop(frame, box):
    x1, y1, x2, y2 = map(int, box)
    margin = int((y2 - y1) * 0.20)

    y1 = max(0, y1 - margin)
    y2 = min(frame.shape[0], y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(frame.shape[1], x2 + margin)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


def calculate_center_distance(boxA, boxB):
    centerA = ((boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2)
    centerB = ((boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2)
    return np.sqrt((centerA[0] - centerB[0]) ** 2 +
                   (centerA[1] - centerB[1]) ** 2)


def draw_detection(frame, box, global_id, gender, stable_gender_frame=0,
                   stable_frames_req=3, event="UPDATE"):
    x1, y1, x2, y2 = map(int, box)

    # Échelle adaptative
    h_frame = frame.shape[0]
    scale = max(0.5, h_frame / 700.0)
    font_scale = 0.55 * scale
    thickness = max(1, int(1.5 * scale))
    box_thickness = max(2, int(2 * scale))

    # Couleur selon événement
    if event == "PAIR ASSIGN":
        color = (0, 200, 255)     # Orange → Match inter-cam
        box_thickness = max(3, int(3 * scale))
    elif event == "RELINK":
        color = (255, 200, 0)     # Cyan → Relink
        box_thickness = max(3, int(3 * scale))
    elif event == "NEW SHARED":
        color = (0, 255, 255)     # Jaune → Nouveau
    else:
        color = (0, 255, 0)       # Vert → Update normal

    padding = int(10 * scale)
    draw_x1 = max(0, x1 - padding)
    draw_y1 = max(0, y1 - padding)
    draw_x2 = min(frame.shape[1], x2 + padding)
    draw_y2 = min(frame.shape[0], y2 + padding)

    cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2),
                  color, box_thickness)

    display_gender = "Inconnu"
    if gender != "Inconnu" and stable_gender_frame >= stable_frames_req:
        display_gender = gender

    if display_gender != "Inconnu":
        label = f"GID:{global_id} {display_gender}"
    else:
        label = f"GID:{global_id}"

    if event == "PAIR ASSIGN":
        label += " [CROSS]"
    elif event == "RELINK":
        label += " [RELINK]"

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                 font_scale, thickness)
    cv2.rectangle(frame,
                  (draw_x1, draw_y1 - h - int(10 * scale)),
                  (draw_x1 + w, draw_y1), color, -1)
    cv2.putText(frame, label,
                (draw_x1, draw_y1 - int(5 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness)

    return frame