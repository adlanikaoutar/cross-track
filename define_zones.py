import cv2
import numpy as np
import pickle
import os

ZONES_TO_DEFINE = [
    {"name": "CAM1_ENTRY_PORTES", "type": "ZONE",
     "desc": "1. Dessine la ZONE des portes (Polygone)"},
    {"name": "CAM1_OVERLAP_LINE", "type": "LINE",
     "desc": "2. Dessine la LIGNE de chevauchement (2 points)"},
    {"name": "CAM2_ENTRY_SORTIE", "type": "ZONE",
     "desc": "3. Dessine la ZONE de sortie (Polygone)"},
    {"name": "CAM2_OVERLAP_LINE", "type": "LINE",
     "desc": "4. Dessine la LIGNE de reception (2 points)"},
]

IMAGE_PATH = "reference_frame.png"
OUTPUT_FILE = "zones_data.pkl"
TARGET_WIDTH = 1280

current_points = []
zones = {}
img_display = None
original_img = None


def mouse_callback(event, x, y, flags, param):
    global current_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        if len(current_points) > 1:
            cv2.line(img_display, current_points[-2],
                     current_points[-1], (0, 255, 0), 2)


def draw_text_centered(img, text, y_pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    cv2.rectangle(img, (text_x - 10, y_pos - 30),
                  (text_x + text_size[0] + 10, y_pos + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (text_x, y_pos), font, 0.7, (255, 255, 255), 2)


def define_loop(zone_name, zone_type, description):
    global current_points, img_display
    current_points = []

    print(f"\n{'='*60}")
    print(f" ZONE ACTUELLE : {zone_name}")
    print(f" TYPE : {zone_type}")
    print(f"{'='*60}")

    cv2.namedWindow("Define Zone", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Define Zone", mouse_callback)

    while True:
        temp_display = img_display.copy()

        if zone_type == "LINE":
            help_text = "MODE LIGNE : Clique 2 FOIS. Appuie sur 'c' pour valider."
        else:
            help_text = "MODE ZONE : Clique pour creer un polygone. Appuie sur 'c' pour fermer."

        draw_text_centered(temp_display, help_text, 50)
        draw_text_centered(temp_display, description, 90)

        cv2.imshow("Define Zone", temp_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if zone_type == "LINE":
                if len(current_points) == 2:
                    zones[zone_name] = (np.array(current_points[0]),
                                        np.array(current_points[1]))
                    print(f"   ✅ Ligne sauvegarde.")
                    break
                else:
                    print("   ❌ Il faut exactement 2 points pour une ligne !")
            else:
                if len(current_points) > 2:
                    zones[zone_name] = np.array(current_points, np.int32)
                    print(f"   ✅ Zone sauvegardee.")
                    break
                else:
                    print("   ❌ Il faut au moins 3 points pour une zone !")

        elif key == ord('r'):
            current_points = []

        elif key == 27:
            return False

    cv2.destroyWindow("Define Zone")
    return True


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"ERREUR: Image '{IMAGE_PATH}' introuvable.")
        input("Appuie sur Entree pour quitter...")
        exit()

    original_img = cv2.imread(IMAGE_PATH)
    h, w = original_img.shape[:2]
    scale = TARGET_WIDTH / w
    new_h = int(h * scale)
    original_img = cv2.resize(original_img, (TARGET_WIDTH, new_h))
    img_display = original_img.copy()
    print(f"Image redimensionnee a : {TARGET_WIDTH}x{new_h}")

    for z in ZONES_TO_DEFINE:
        img_display = original_img.copy()
        for z_name, z_data in zones.items():
            if isinstance(z_data, tuple):
                cv2.line(img_display, tuple(z_data[0]),
                         tuple(z_data[1]), (0, 0, 255), 5)
            else:
                cv2.polylines(img_display, [z_data], True, (0, 0, 255), 2)

        success = define_loop(z["name"], z["type"], z["desc"])
        if not success:
            break

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(zones, f)
    print(f"\n✅ Toutes les zones sont sauvegardees dans '{OUTPUT_FILE}'")

    result_img = original_img.copy()
    for z_name, z_data in zones.items():
        if isinstance(z_data, tuple):
            cv2.line(result_img, tuple(z_data[0]),
                     tuple(z_data[1]), (0, 255, 0), 3)
            cv2.putText(result_img, z_name, tuple(z_data[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.polylines(result_img, [z_data], True, (0, 255, 0), 2)
            cv2.putText(result_img, z_name, tuple(z_data[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("zones_result.png", result_img)
    cv2.imshow("Resultat Final", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()