import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from src.config import GENDER_MODEL_PATH, DEVICE, GENDER_DICT


class GenderDetector:
    def __init__(self):
        self.device = DEVICE
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self):
        print(f"[GenderDetector] Chargement sur {self.device}...")
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)

        try:
            model.load_state_dict(
                torch.load(GENDER_MODEL_PATH,
                           map_location=self.device,
                           weights_only=True))
            model.to(self.device)
            model.eval()
            print("[GenderDetector]  Modèle chargé.")
            return model
        except Exception as e:
            print(f"[ERREUR] Chargement modèle genre : {e}")
            raise

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return "Inconnu", 0.0
        try:
            img_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            return GENDER_DICT[predicted.item()], confidence.item()
        except Exception as e:
            print(f"[GenderDetector] Erreur inference: {e}")
            return "Erreur", 0.0