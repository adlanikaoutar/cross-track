import torch
import torchreid
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from src.config import DEVICE, MODELS_DIR


class VisualEncoder:
    def __init__(self):
        self.device = DEVICE
        self._debug_done = False
        print(f"[ReID] Initialisation OSNet-x1.0 sur {self.device}...")

        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=False
        )

        if hasattr(self.model, 'classifier'):
            print(f"[ReID] Classificateur supprimé")
            del self.model.classifier

        possible_names = ['osnet_x1_0_msmt17.pt', 'osnet_x1_0_msmt17.pth']
        weight_path = None
        for name in possible_names:
            path = os.path.join(MODELS_DIR, name)
            if os.path.exists(path):
                weight_path = path
                break

        weights_loaded = False

        if weight_path:
            print(f"[ReID] Chargement depuis : {weight_path}")
            try:
                # Sauvegarder l'état AVANT le chargement
                initial_state = {}
                for k, v in self.model.state_dict().items():
                    initial_state[k] = v.clone()

                # Charger sur CPU d'abord
                state_dict = torch.load(
                    weight_path,
                    map_location='cpu',
                    weights_only=True
                )

                model_keys = set(self.model.state_dict().keys())
                weight_keys = set(state_dict.keys())
                matching = weight_keys & model_keys
                extra = weight_keys - model_keys

                print(f"[ReID] Clés fichier    : {len(weight_keys)}")
                print(f"[ReID] Clés modèle     : {len(model_keys)}")
                print(f"[ReID] Correspondances : {len(matching)}")
                if extra:
                    print(f"[ReID] Extra (ignorées): {len(extra)}")

                result = self.model.load_state_dict(state_dict, strict=False)

                # Comparer AVANT vs APRÈS
                loaded_count = 0
                for k in matching:
                    if k in initial_state:
                        after_val = self.model.state_dict()[k]
                        if initial_state[k].shape == after_val.shape:
                            if not torch.equal(initial_state[k], after_val):
                                loaded_count += 1

                print(f"[ReID] ✅ SUCCÈS : {loaded_count}/{len(matching)} "
                      f"couches effectivement mises à jour")
                weights_loaded = True

            except Exception as e:
                print(f"[ReID] ❌ VRAIE ERREUR : {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[ReID] ⚠️ Fichier introuvable dans {MODELS_DIR}")

        # Déplacer sur GPU APRÈS le chargement CPU
        self.model.to(self.device)
        self.model.eval()

        if weights_loaded:
            print(f"[ReID] Modèle prêt sur {self.device}")
        else:
            print(f"[ReID] ⚠️ ATTENTION : Poids non chargés !")

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def get_embedding(self, crop):
        if crop is None:
            return None
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(tensor)

            vec = features.cpu().numpy().flatten()

            if not self._debug_done:
                self._debug_done = True
                self._print_debug(vec, crop)

            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            return vec
        except Exception as e:
            print(f"[ReID] Erreur inference: {e}")
            return None

    def _print_debug(self, vec, crop):
        norm = np.linalg.norm(vec)
        print(f"\n{'='*60}")
        print(f"[ReID] 🔍 DEBUG PREMIÈRE INFÉRENCE :")
        print(f"  Feature shape    : (1, {len(vec)})")
        print(f"  Vector mean      : {np.mean(vec):.6f}")
        print(f"  Vector std       : {np.std(vec):.6f}")
        print(f"  Vector L2 norm   : {norm:.6f}")
        print(f"  Non-zero         : {np.count_nonzero(vec)}/{len(vec)}")

        vec2 = self.get_embedding(crop)
        if vec2 is not None:
            n1 = vec / (np.linalg.norm(vec) + 1e-8)
            n2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
            sim = float(np.dot(n1, n2))
            status = "✅ OK" if sim > 0.99 else "❌ PROBLÈME !"
            print(f"  Self-similarity  : {sim:.6f} {status}")
        print(f"{'='*60}\n")