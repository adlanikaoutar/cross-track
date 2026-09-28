import csv
import os
from datetime import datetime


class Logger:
    def __init__(self, filename="detections_log.csv"):
        base_dir = os.getcwd()
        self.log_path = os.path.join(base_dir, filename)

        print("=" * 60)
        print(f" [LOGGER] Fichier CSV cible :")
        print(f" [LOGGER] {self.log_path}")
        print("=" * 60)

        self.headers = ["Date", "Time", "Global_ID", "Genre", "Zone", "Event"]

        if not os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'w', newline='',
                          encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
                print(f"[Logger] Nouveau fichier CSV créé.")
            except Exception as e:
                print(f"[Logger] Erreur création CSV : {e}")
        else:
            print(f"[Logger] Fichier CSV existant trouvé.")

    def log_person(self, global_id, gender, zone, event):
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H:%M:%S.%f")[:-3]

        try:
            with open(self.log_path, 'a', newline='',
                      encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([date_str, time_str, global_id,
                                 gender, zone, event])
                f.flush()
        except PermissionError:
            print(f"[Logger] ERREUR : CSV ouvert dans Excel. Fermez-le.")
        except Exception as e:
            print(f"[Logger] Erreur écriture : {e}")