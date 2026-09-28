import cv2
import numpy as np


def get_color_name(hsv_value):
    h, s, v = hsv_value

    if v < 50:
        return "Noir"
    if s < 30 and v > 50:
        return "Blanc" if v > 200 else "Gris"

    if 0 <= h < 10 or 170 <= h <= 180:
        return "Rouge"
    if 10 <= h < 25:
        return "Orange"
    if 25 <= h < 35:
        return "Jaune"
    if 35 <= h < 85:
        return "Vert"
    if 85 <= h < 130:
        return "Bleu"
    if 130 <= h < 160:
        return "Violet"
    return "Inconnu"


def analyze_person_colors(frame, box):
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1

    if height < 50:
        return "Inconnu", "Inconnu"

    crop_top = frame[y1:int(y1 + height * 0.40), x1:x2]
    crop_bottom = frame[int(y2 - height * 0.40):y2, x1:x2]

    def get_dominant(crop):
        if crop is None or crop.size == 0:
            return "Inconnu"
        try:
            from sklearn.cluster import KMeans
            small = cv2.resize(crop, (50, 50))
            pixels = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            kmeans = KMeans(n_clusters=1, n_init=5).fit(pixels)
            return get_color_name(kmeans.cluster_centers_[0])
        except:
            avg = np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), axis=(0, 1))
            return get_color_name(avg)

    return get_dominant(crop_top), get_dominant(crop_bottom)