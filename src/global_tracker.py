import time
import cv2
import csv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from src.config import (REID_VISUAL_THRESHOLD,
                        GENDER_HISTORY_LENGTH, WEIGHT_REID, WEIGHT_HISTOGRAM,
                        WEIGHT_DIR, WEIGHT_SIZE, WEIGHT_SOURCE_OVERLAP,
                        CROWD_THRESHOLD, CROSS_CAM_TIMEOUT, CROSS_CAMERA_THRESHOLD,
                        RELINK_TIMEOUT, RELINK_MAX_DISTANCE, RELINK_CLOSE_DISTANCE,
                        RELINK_IOU_THRESHOLD, RELINK_REID_THRESHOLD,
                        RELINK_HIST_THRESHOLD, RELINK_COMBINED_MIN,
                        GENDER_STABLE_FRAMES, GENDER_LOCK_FRAMES,
                        CROSS_CAM_OVERLAP_BONUS, MIN_ID_STABLE_SECONDS,
                        VISUAL_CONFIRMATION_THRESHOLD,
                        CROSS_CAM_MIN_REID, CROSS_CAM_MIN_HIST,
                        LOST_GRACE_SECONDS,
                        CROSS_CAM_ACTIVE_PENALTY,
                        CROSS_CAM_PENDING_BONUS,
                        CROSS_CAM_LOST_BONUS,
                        CROSS_CAM_GENDER_MATCH_BONUS,
                        CROSS_CAM_GENDER_MISMATCH_PENALTY,
                        CROSS_CAM_GENDER_LOCKED_MISMATCH_PENALTY,
                        CROSS_CAM_REQUIRE_SOURCE_OVERLAP,
                        CROSS_CAM_SOURCE_OVERLAP_MIN,
                        CROSS_CAM_ACTIVE_SOURCE_OVERLAP_MIN,
                        CROSS_CAM_MAX_FINAL_SCORE_FROM_BONUS,
                        CROSS_CAM_POST_MATCH_MIN_REID,
                        CROSS_CAM_FALLBACK_THRESHOLD,
                        AUTO_PENDING_OVERLAP_DISTANCE)

from src.zones import (has_crossed_line, ZONES, get_histogram_score,
                        is_near_overlap_line, get_overlap_line,
                        _normalize_cam_name)
from src.utils import calculate_iou, calculate_center_distance


class GlobalIdManager:
    def __init__(self, visual_encoder, threshold=0.6, logger=None):
        self.visual_encoder = visual_encoder
        self.threshold = threshold
        self.logger = logger

        self.next_global_id = self._load_last_id()

        self.local_to_global = {}
        self.local_to_global_time = {}

        self.visual_db = {}

        self.gender_history = {}
        self.locked_gender = {}

        self.last_seen_local = {}
        self.first_lost_time = {}

        self.cross_camera_lost = []
        self.pending_crossings = {}

        self.recently_lost = []

        self.prev_centers = {}
        self.last_known_global_pos = {}
        self.last_seen_zone = {}

        self.active_per_cam = {}
        self._active_gids_this_frame = set()
        self._lost_logged = set()
        self._cross_lost_added = set()

    def _load_last_id(self):
        log_path = os.path.join(os.getcwd(), "detections_log.csv")
        if not os.path.exists(log_path):
            return 1
        try:
            max_id = 0
            with open(log_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) > 2:
                        try:
                            val = int(row[2])
                        except:
                            continue
                        if val > max_id:
                            max_id = val
            print(f"[GlobalID] Reprise compteur. Prochain ID : {max_id + 1}")
            return max_id + 1
        except Exception:
            return 1

    def _get_stable_gender(self, gid, current_gender):
        if gid not in self.gender_history:
            self.gender_history[gid] = []
        hist = self.gender_history[gid]
        hist.append(current_gender)
        if len(hist) > GENDER_HISTORY_LENGTH:
            hist.pop(0)

        if gid in self.locked_gender:
            return self.locked_gender[gid]

        valid_votes = [g for g in hist if g != "Inconnu"]
        if not valid_votes:
            return "Inconnu"

        counts = Counter(valid_votes)
        most_common, num_votes = counts.most_common(1)[0]
        stable = most_common if (num_votes / len(valid_votes) > 0.5) else "Inconnu"

        if stable != "Inconnu" and num_votes >= GENDER_LOCK_FRAMES:
            self.locked_gender[gid] = stable
            print(f"🔒 GENDER LOCKED: GID {gid} → {stable} "
                  f"(après {num_votes} votes)")

        return stable

    def _adaptive_smooth(self, old_vec, new_vec, crop):
        if new_vec is None:
            return old_vec
        if old_vec is None:
            return new_vec
        if crop is not None:
            h, w = crop.shape[:2]
            area = h * w
            if area > 10000:
                alpha_old = 0.60
            elif area > 4000:
                alpha_old = 0.72
            else:
                alpha_old = 0.85
        else:
            alpha_old = 0.80
        blended = alpha_old * old_vec + (1.0 - alpha_old) * new_vec
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 1e-6 else blended

    def _compute_visual_scores(self, det, db_entry):
        crop = det.get('crop')
        if crop is None:
            return 0.0, 0.0
        vec = self.visual_encoder.get_embedding(crop)
        if vec is None or db_entry['vector'] is None:
            return 0.0, 0.0
        score_reid = cosine_similarity([vec], [db_entry['vector']])[0][0]
        hist1 = det['info'].get('histogram')
        hist2 = db_entry['info'].get('histogram')
        score_hist = 0.0
        if hist1 is not None and hist2 is not None:
            score_hist = get_histogram_score(hist1, hist2)
        return score_reid, score_hist

    def _log_event(self, global_id, gender, cam_name, event):
        if self.logger:
            self.logger.log_person(global_id, gender, cam_name, event)

    def _is_gid_already_active_in_cam(self, gid, cam_name, exclude_key=None):
        for k, v in self.local_to_global.items():
            if v == gid and k[0] == cam_name and k != exclude_key:
                return True
        return False

    def _add_to_cross_camera_lost(self, gid, cam_name, now):
        pair = (gid, cam_name)
        if pair in self._cross_lost_added:
            return
        if gid not in self.visual_db:
            return
        pos_data = self.last_known_global_pos.get(gid)
        stored_box = pos_data['box'] if pos_data else [0, 0, 0, 0]
        self.cross_camera_lost.append({
            'gid': gid,
            'cam': cam_name,
            'time': now,
            'gender': self.locked_gender.get(gid,
                    self.visual_db[gid]['info'].get('gender', 'Inconnu')),
            'vector': self.visual_db[gid]['vector'],
            'box': stored_box,
            'hist': self.visual_db[gid]['info'].get('histogram')
        })
        self._cross_lost_added.add(pair)

    def _was_near_source_overlap(self, gid, source_cam, max_distance=120):
        pos_data = self.last_known_global_pos.get(gid)
        if pos_data is None:
            return 0.0
        box = pos_data.get('box')
        if box is None:
            return 0.0
        if is_near_overlap_line(box, source_cam, max_distance=120):
            return 1.0
        elif is_near_overlap_line(box, source_cam, max_distance=250):
            return 0.5
        return 0.0

    def _evaluate_candidate(self, gid, cand_status, det, cam_name, source_cam,
                            check_source_overlap=True):
        """Évalue un candidat et retourne les scores ou None si rejeté."""
        box = det['box']
        info = det['info']
        raw_gender = info.get('gender', 'Inconnu')

        db_entry = self.visual_db[gid]
        locked_g = self.locked_gender.get(gid)
        db_gender = locked_g if locked_g else \
            db_entry['info'].get('gender', 'Inconnu')

        # Scores visuels
        score_deep, score_hist = self._compute_visual_scores(det, db_entry)

        if score_deep < CROSS_CAM_MIN_REID:
            return {'reason': f'REID_LOW({score_deep:.3f}<{CROSS_CAM_MIN_REID})'}
        if score_hist < CROSS_CAM_MIN_HIST:
            return {'reason': f'HIST_LOW({score_hist:.3f})'}

        # Source overlap
        source_overlap_score = self._was_near_source_overlap(
            gid, source_cam, max_distance=120)

        if check_source_overlap and CROSS_CAM_REQUIRE_SOURCE_OVERLAP:
            if cand_status == 'pending':
                pass
            elif cand_status == 'lost':
                if source_overlap_score < CROSS_CAM_SOURCE_OVERLAP_MIN:
                    return {'reason': f'SOURCE_OVERLAP_TOO_FAR(SO={source_overlap_score:.1f})'}
            else:  # active
                if source_overlap_score < CROSS_CAM_ACTIVE_SOURCE_OVERLAP_MIN:
                    return {'reason': f'ACTIVE_OVERLAP_TOO_FAR(SO={source_overlap_score:.1f})'}

        # Score direction
        prev_pos_data = self.last_known_global_pos.get(gid)
        score_dir = 0.0
        if prev_pos_data:
            prev_box = prev_pos_data.get('box', box)
            dist_px = calculate_center_distance(box, prev_box)
            score_dir = max(0, 1.0 - (dist_px / 1000.0))

        # Score taille
        area_curr = (box[2] - box[0]) * (box[3] - box[1])
        prev_box = prev_pos_data.get('box', box) if prev_pos_data else box
        area_prev = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
        score_size = 1.0 - abs(area_curr - area_prev) / \
            (area_curr + area_prev + 1e-6)

        # Bonus overlap cible
        target_overlap_bonus = 0.0
        if is_near_overlap_line(box, cam_name, max_distance=80):
            target_overlap_bonus = CROSS_CAM_OVERLAP_BONUS

        # Statut
        if cand_status == 'pending':
            status_bonus = CROSS_CAM_PENDING_BONUS
            status_penalty = 0.0
        elif cand_status == 'lost':
            status_bonus = CROSS_CAM_LOST_BONUS
            status_penalty = 0.0
        else:
            status_bonus = 0.0
            status_penalty = CROSS_CAM_ACTIVE_PENALTY

        # Genre
        gender_bonus = 0.0
        gender_penalty = 0.0

        if raw_gender != "Inconnu" and db_gender != "Inconnu":
            if raw_gender == db_gender:
                gender_bonus = CROSS_CAM_GENDER_MATCH_BONUS
            else:
                if locked_g:
                    gender_penalty = CROSS_CAM_GENDER_LOCKED_MISMATCH_PENALTY
                else:
                    gender_penalty = CROSS_CAM_GENDER_MISMATCH_PENALTY
        elif raw_gender != "Inconnu" and db_gender == "Inconnu":
            gender_bonus = 0.02
        elif raw_gender == "Inconnu" and db_gender != "Inconnu":
            gender_bonus = 0.03

        # Score de base
        base_score = (WEIGHT_REID * score_deep) + \
                     (WEIGHT_HISTOGRAM * score_hist) + \
                     (WEIGHT_DIR * score_dir) + \
                     (WEIGHT_SIZE * score_size) + \
                     (WEIGHT_SOURCE_OVERLAP * source_overlap_score)

        total_bonus = min(status_bonus + gender_bonus + target_overlap_bonus,
                          CROSS_CAM_MAX_FINAL_SCORE_FROM_BONUS)
        total_penalty = status_penalty + gender_penalty

        final_score = base_score + total_bonus - total_penalty

        return {
            'gid': gid,
            'status': cand_status,
            'locked_g': locked_g or '-',
            'db_g': db_gender,
            'raw_g': raw_gender,
            'reid': score_deep,
            'hist': score_hist,
            'dir': score_dir,
            'size': score_size,
            'SO': source_overlap_score,
            'base': base_score,
            'bonus': total_bonus,
            'penalty': total_penalty,
            'final': final_score,
            'db_gender': db_gender,
            'locked_g': locked_g,
            'score_deep': score_deep,
            'score_hist': score_hist,
            'score_dir': score_dir,
            'score_size': score_size,
            'source_overlap_score': source_overlap_score,
            'base_score': base_score,
            'total_bonus': total_bonus,
            'total_penalty': total_penalty,
            'final_score': final_score,
            'reason': None
        }

    # ════════════════════════════════════════════════════════════
    # MÉTHODE PRINCIPALE
    # ════════════════════════════════════════════════════════════
    def update(self, cam_name, local_detections):
        now = time.time()
        current_local_ids = set()
        processed_global_ids = set()
        self._active_gids_this_frame = set()

        prev_active = self.active_per_cam.get(cam_name, set())

        for det in local_detections:
            lid = det['local_id']
            box = det['box']
            info = det['info']
            raw_gender = info.get('gender', 'Inconnu')
            current_zone = info.get('zone_name')

            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            curr_pos = (cx, cy)
            key = (cam_name, lid)

            line_data = get_overlap_line(cam_name)
            has_crossed = False
            if line_data is not None:
                prev_pos = self.prev_centers.get(key)
                if prev_pos and has_crossed_line(prev_pos, curr_pos,
                                                  line_data[0], line_data[1]):
                    has_crossed = True

            self.prev_centers[key] = curr_pos
            current_local_ids.add(lid)
            self.last_seen_local[key] = now

            if key in self.first_lost_time:
                del self.first_lost_time[key]

            # ══════════════════════════════════════════════════
            # CAS 1 : ID LOCAL DÉJÀ CONNU → UPDATE
            # ══════════════════════════════════════════════════
            if key in self.local_to_global:
                gid = self.local_to_global[key]

                old_vector = self.visual_db[gid]['vector']
                new_vector = self.visual_encoder.get_embedding(det['crop'])
                current_visual_score = 0.0
                if new_vector is not None and old_vector is not None:
                    current_visual_score = cosine_similarity(
                        [new_vector], [old_vector])[0][0]

                force_new_id = False
                if current_visual_score < VISUAL_CONFIRMATION_THRESHOLD:
                    id_age = now - self.local_to_global_time.get(key, now)
                    if current_visual_score < 0.15 and id_age > MIN_ID_STABLE_SECONDS:
                        del self.local_to_global[key]
                        if key in self.local_to_global_time:
                            del self.local_to_global_time[key]
                        force_new_id = True

                if not force_new_id:
                    prev_zone = self.last_seen_zone.get(gid)

                    if prev_zone != current_zone:
                        if current_zone and "OVERLAP" in current_zone.upper():
                            self._log_event(gid, raw_gender, cam_name,
                                            f"ENTERS {current_zone}")
                        elif prev_zone and "OVERLAP" in prev_zone.upper() \
                                and current_zone != prev_zone:
                            self._log_event(gid, raw_gender, cam_name,
                                            f"LEAVES {prev_zone}")

                    if has_crossed and gid not in self.pending_crossings:
                        self.pending_crossings[gid] = now
                        self._log_event(gid, raw_gender, cam_name,
                                        "CROSS_OVERLAP_LINE")
                        print(f"🚪 GID {gid} franchi overlap de {cam_name}")

                    locked_g = self.locked_gender.get(gid)
                    stable_gender = locked_g if locked_g else \
                        self._get_stable_gender(gid, raw_gender)
                    display_gender = locked_g if locked_g else stable_gender

                    self.visual_db[gid]['last_seen'] = now
                    self.visual_db[gid]['last_seen_cam'] = cam_name
                    self.visual_db[gid]['info']['zone_name'] = current_zone
                    self.visual_db[gid]['info']['gender'] = display_gender
                    self.visual_db[gid]['info']['histogram'] = info.get('histogram')

                    if 'stable_frames_count' not in self.visual_db[gid]['info']:
                        self.visual_db[gid]['info']['stable_frames_count'] = 0
                    if display_gender != "Inconnu":
                        self.visual_db[gid]['info']['stable_frames_count'] += 1
                    else:
                        self.visual_db[gid]['info']['stable_frames_count'] = 0

                    if new_vector is not None:
                        self.visual_db[gid]['vector'] = self._adaptive_smooth(
                            self.visual_db[gid]['vector'], new_vector, det['crop'])

                    self.last_known_global_pos[gid] = {'box': box, 'center': curr_pos}
                    self.last_seen_zone[gid] = current_zone

                    self.cross_camera_lost = [
                        x for x in self.cross_camera_lost
                        if not (x['gid'] == gid and x['cam'] == cam_name)]
                    self._cross_lost_added.discard((gid, cam_name))
                    self._lost_logged.discard((gid, cam_name))

                    det['global_id'] = gid
                    det['stable_frames'] = self.visual_db[gid]['info'].get(
                        'stable_frames_count', 0)
                    det['stable_gender'] = display_gender
                    det['event'] = "UPDATE"
                    processed_global_ids.add(gid)
                    self._active_gids_this_frame.add(gid)
                    continue

            # ══════════════════════════════════════════════════
            # CAS 2 : RELINKING INTRA-CAMÉRA
            # ══════════════════════════════════════════════════
            best_relid_gid = None
            best_relink_score = 0.0

            for lost in self.recently_lost:
                if lost['cam'] != cam_name:
                    continue
                if (now - lost['time']) > RELINK_TIMEOUT:
                    continue

                gid = lost['gid']

                if gid in self._active_gids_this_frame:
                    continue
                if self._is_gid_already_active_in_cam(gid, cam_name):
                    continue

                lost_box = lost.get('box')
                if lost_box is None:
                    continue

                iou_score = calculate_iou(box, lost_box)
                dist_score = calculate_center_distance(box, lost_box)

                vec_current = self.visual_encoder.get_embedding(det['crop'])
                reid_score = 0.0
                if vec_current is not None and lost['vector'] is not None:
                    reid_score = cosine_similarity(
                        [vec_current], [lost['vector']])[0][0]

                hist_score = get_histogram_score(info.get('histogram'), lost['hist'])

                locked_g = self.locked_gender.get(gid)
                if locked_g and raw_gender != "Inconnu" and raw_gender != locked_g:
                    continue

                if dist_score > RELINK_MAX_DISTANCE:
                    continue

                if dist_score < RELINK_CLOSE_DISTANCE:
                    if reid_score < RELINK_REID_THRESHOLD:
                        continue
                else:
                    if reid_score < 0.55 or hist_score < 0.45:
                        continue

                dist_norm = max(0.0, 1.0 - dist_score / RELINK_MAX_DISTANCE)
                combined = (iou_score * 0.20) + (dist_norm * 0.15) + \
                           (reid_score * 0.40) + (hist_score * 0.25)

                if combined < RELINK_COMBINED_MIN:
                    continue

                if combined > best_relink_score:
                    best_relink_score = combined
                    best_relid_gid = gid

            if best_relid_gid:
                print(f"🔄 RELINK: GID {best_relid_gid} "
                      f"(score={best_relink_score:.3f})")

                self.local_to_global[key] = best_relid_gid
                self.local_to_global_time[key] = now
                self.visual_db[best_relid_gid]['last_seen'] = now
                self.visual_db[best_relid_gid]['last_seen_cam'] = cam_name
                self.visual_db[best_relid_gid]['info']['histogram'] = info.get('histogram')
                self.recently_lost = [x for x in self.recently_lost
                                      if x['gid'] != best_relid_gid]
                self.cross_camera_lost = [x for x in self.cross_camera_lost
                                          if x['gid'] != best_relid_gid]
                self._cross_lost_added.discard((best_relid_gid, cam_name))
                self._lost_logged.discard((best_relid_gid, cam_name))

                self._log_event(best_relid_gid, raw_gender, cam_name, "RELINK")

                det['global_id'] = best_relid_gid
                det['stable_frames'] = self.visual_db[best_relid_gid]['info'].get(
                    'stable_frames_count', 0)
                det['stable_gender'] = self.locked_gender.get(best_relid_gid,
                    self.visual_db[best_relid_gid]['info'].get('gender', 'Inconnu'))
                det['event'] = "RELINK"
                processed_global_ids.add(best_relid_gid)
                self._active_gids_this_frame.add(best_relid_gid)
                continue

            # ══════════════════════════════════════════════════
            # CAS 3 : CROSS-CAMERA MATCHING
            # ══════════════════════════════════════════════════
            source_cam = "Cam 2" if cam_name == "Cam 1" else "Cam 1"

            candidate_info = {}

            for gid, cross_time in self.pending_crossings.items():
                if (now - cross_time) < CROSS_CAM_TIMEOUT:
                    candidate_info[gid] = 'pending'

            for lost_entry in self.cross_camera_lost:
                if lost_entry['cam'] == source_cam \
                        and (now - lost_entry['time']) < CROSS_CAM_TIMEOUT:
                    if lost_entry['gid'] not in candidate_info:
                        candidate_info[lost_entry['gid']] = 'lost'

            for gid, data in self.visual_db.items():
                if data.get('last_seen_cam') == source_cam:
                    time_since = now - data['last_seen']
                    if time_since < CROSS_CAM_TIMEOUT:
                        if gid not in candidate_info:
                            candidate_info[gid] = 'active'

            best_gid = None
            best_final_score = 0.0
            best_match_details = {}
            all_scores = []

            # ══════════════════════════════════════════════════
            # 1er PASSAGE : filtre SOURCE_OVERLAP strict
            # ══════════════════════════════════════════════════
            for gid, cand_status in candidate_info.items():
                if gid in processed_global_ids:
                    continue
                if gid in self._active_gids_this_frame:
                    continue
                if self._is_gid_already_active_in_cam(gid, cam_name):
                    continue
                if gid not in self.visual_db:
                    continue

                result = self._evaluate_candidate(
                    gid, cand_status, det, cam_name, source_cam,
                    check_source_overlap=True)

                if result.get('reason'):
                    all_scores.append({
                        'gid': gid, 'status': cand_status,
                        'reason': result['reason'],
                        'pass': 1
                    })
                    continue

                final_score = result['final']

                all_scores.append({
                    'gid': gid,
                    'status': cand_status,
                    'locked_g': result.get('locked_g', '-'),
                    'db_g': result.get('db_g', '-'),
                    'raw_g': result.get('raw_g', '-'),
                    'reid': result['reid'],
                    'hist': result['hist'],
                    'SO': result['SO'],
                    'base': result['base'],
                    'bonus': result['bonus'],
                    'penalty': result['penalty'],
                    'final': final_score,
                    'reason': None,
                    'pass': 1
                })

                if final_score > best_final_score and final_score > CROSS_CAMERA_THRESHOLD:
                    best_final_score = final_score
                    best_gid = gid
                    best_match_details = result

            # ══════════════════════════════════════════════════
            # 🆕 2ème PASSAGE : FALLBACK sans filtre SOURCE_OVERLAP
            # (uniquement si aucun match trouvé dans le 1er passage)
            # ══════════════════════════════════════════════════
            if best_gid is None:
                for gid, cand_status in candidate_info.items():
                    if gid in processed_global_ids:
                        continue
                    if gid in self._active_gids_this_frame:
                        continue
                    if self._is_gid_already_active_in_cam(gid, cam_name):
                        continue
                    if gid not in self.visual_db:
                        continue

                    # Déjà évalué avec succès dans le 1er passage ?
                    already_evaluated = any(
                        s['gid'] == gid and s.get('reason') is None
                        for s in all_scores)
                    if already_evaluated:
                        continue

                    result = self._evaluate_candidate(
                        gid, cand_status, det, cam_name, source_cam,
                        check_source_overlap=False)  # 🔧 Pas de filtre SO

                    if result.get('reason'):
                        all_scores.append({
                            'gid': gid, 'status': cand_status,
                            'reason': result['reason'],
                            'pass': 2
                        })
                        continue

                    final_score = result['final']

                    # 🔧 Seuil plus élevé pour le fallback
                    if final_score < CROSS_CAM_FALLBACK_THRESHOLD:
                        all_scores.append({
                            'gid': gid, 'status': cand_status,
                            'reid': result['reid'],
                            'final': final_score,
                            'reason': f'FALLBACK_LOW({final_score:.3f}<{CROSS_CAM_FALLBACK_THRESHOLD})',
                            'pass': 2
                        })
                        continue

                    all_scores.append({
                        'gid': gid,
                        'status': cand_status,
                        'locked_g': result.get('locked_g', '-'),
                        'db_g': result.get('db_g', '-'),
                        'raw_g': result.get('raw_g', '-'),
                        'reid': result['reid'],
                        'hist': result['hist'],
                        'SO': result['SO'],
                        'base': result['base'],
                        'bonus': result['bonus'],
                        'penalty': result['penalty'],
                        'final': final_score,
                        'reason': None,
                        'pass': 2  # 🆕 Indique fallback
                    })

                    if final_score > best_final_score:
                        best_final_score = final_score
                        best_gid = gid
                        best_match_details = result

            # Post-match verification
            if best_gid:
                current_crop = det.get('crop')
                db_vec = self.visual_db[best_gid]['vector']

                if current_crop is not None and db_vec is not None:
                    current_vec = self.visual_encoder.get_embedding(current_crop)
                    if current_vec is not None:
                        post_match_reid = cosine_similarity(
                            [current_vec], [db_vec])[0][0]

                        if post_match_reid < CROSS_CAM_POST_MATCH_MIN_REID:
                            print(f"❌ POST-MATCH REJECT: GID {best_gid} "
                                  f"post_ReID={post_match_reid:.3f}")
                            best_gid = None
                            best_final_score = 0.0

            # Résumé détaillé
            if all_scores:
                all_scores.sort(key=lambda x: x.get('final', -1), reverse=True)
                print(f"\n{'─'*70}")
                print(f"[CROSS] {cam_name} ← {source_cam} | "
                      f"raw_g={raw_gender} | "
                      f"{len(all_scores)} cand.")
                for i, s in enumerate(all_scores[:6]):
                    if s.get('reason'):
                        pass_label = f"P{s.get('pass',1)}"
                        print(f"  {i+1}. GID {s['gid']:3d} "
                              f"({s['status']:7s},{pass_label}): "
                              f"❌ {s['reason']}")
                    else:
                        marker = " ✅" if s['gid'] == best_gid else ""
                        pass_label = f"P{s.get('pass',1)}"
                        print(f"  {i+1}. GID {s['gid']:3d} "
                              f"({s['status']:7s},{pass_label},g={s.get('db_g','-'):5s}): "
                              f"R={s['reid']:.2f} H={s['hist']:.2f} "
                              f"SO={s.get('SO',0):.1f} "
                              f"base={s['base']:.2f} "
                              f"+{s['bonus']:.2f}-{s['penalty']:.2f} "
                              f"→ {s['final']:.3f}{marker}")
                print(f"{'─'*70}")

            if best_gid:
                d = best_match_details
                ts_str = time.strftime("%H:%M:%S", time.localtime(now))
                locked_g = d.get('locked_g') or "N/A"
                is_fallback = d.get('SO', 0) < CROSS_CAM_SOURCE_OVERLAP_MIN
                fallback_tag = " [FALLBACK]" if is_fallback else ""
                print(f"\n🔗 MATCH{fallback_tag}: ({ts_str}, GID:{d['gid']}, "
                      f"CAM:{cam_name}, POS:{curr_pos}, "
                      f"raw_g={raw_gender}, lock_g={locked_g}, "
                      f"R={d['score_deep']:.3f} H={d['score_hist']:.3f} "
                      f"SO={d['source_overlap_score']:.1f} "
                      f"base={d['base_score']:.2f} "
                      f"+{d['total_bonus']:.2f}-{d['total_penalty']:.2f} "
                      f"FINAL={d['final_score']:.3f})\n")

                self.local_to_global[key] = best_gid
                self.local_to_global_time[key] = now
                self.visual_db[best_gid]['last_seen'] = now
                self.visual_db[best_gid]['last_seen_cam'] = cam_name
                self.visual_db[best_gid]['info']['histogram'] = info.get('histogram')
                self.visual_db[best_gid]['info']['zone_name'] = current_zone

                match_gender = self.locked_gender.get(best_gid)
                if match_gender is None:
                    match_gender = d.get('db_gender', raw_gender)
                    if match_gender != "Inconnu":
                        self.locked_gender[best_gid] = match_gender
                        print(f"🔒 LOCKED AFTER MATCH: GID {best_gid} → {match_gender}")

                self.visual_db[best_gid]['info']['gender'] = match_gender

                if best_gid in self.pending_crossings:
                    del self.pending_crossings[best_gid]

                best_is_active = d['status'] == 'active'
                if not best_is_active:
                    self.cross_camera_lost = [
                        x for x in self.cross_camera_lost
                        if x['gid'] != best_gid]
                    self._cross_lost_added = {
                        (x['gid'], x['cam'])
                        for x in self.cross_camera_lost}

                self._lost_logged.discard((best_gid, cam_name))
                self._lost_logged.discard((best_gid, source_cam))

                self.last_known_global_pos[best_gid] = {
                    'box': box, 'center': curr_pos}
                self.last_seen_zone[best_gid] = current_zone

                det['global_id'] = best_gid
                det['stable_frames'] = 0
                det['stable_gender'] = match_gender
                det['event'] = "PAIR ASSIGN"
                self._log_event(best_gid, match_gender, cam_name,
                                f"MATCH FROM {source_cam}")
                processed_global_ids.add(best_gid)
                self._active_gids_this_frame.add(best_gid)

            else:
                # NOUVELLE PERSONNE
                gid = self.next_global_id
                self.next_global_id += 1

                self.local_to_global[key] = gid
                self.local_to_global_time[key] = now
                stable_gender = self._get_stable_gender(gid, raw_gender)

                vec = self.visual_encoder.get_embedding(det['crop'])
                if vec is not None:
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm

                self.visual_db[gid] = {
                    'vector': vec,
                    'info': {
                        'gender': stable_gender,
                        'zone_name': current_zone,
                        'histogram': info.get('histogram'),
                        'stable_frames_count': 0
                    },
                    'last_seen': now,
                    'last_seen_cam': cam_name,
                }
                self.gender_history[gid] = [raw_gender]

                self.last_known_global_pos[gid] = {'box': box, 'center': curr_pos}
                self.last_seen_zone[gid] = current_zone

                det['global_id'] = gid
                det['stable_frames'] = 0
                det['stable_gender'] = stable_gender
                det['event'] = "NEW SHARED"
                self._log_event(gid, raw_gender, cam_name, "NEW PERSON DETECTED")
                processed_global_ids.add(gid)
                self._active_gids_this_frame.add(gid)

        # ══════════════════════════════════════════════════════
        # DÉTECTION DE PERTE
        # ══════════════════════════════════════════════════════
        lost_ids = prev_active - current_local_ids
        for lid in lost_ids:
            key = (cam_name, lid)
            if key not in self.local_to_global:
                continue

            gid = self.local_to_global[key]

            if gid in self._active_gids_this_frame:
                continue

            # Ajout IMMÉDIAT
            self._add_to_cross_camera_lost(gid, cam_name, now)

            if gid not in self.pending_crossings:
                pos_data = self.last_known_global_pos.get(gid)
                if pos_data:
                    stored_box = pos_data.get('box', [0, 0, 0, 0])
                    if is_near_overlap_line(stored_box, cam_name,
                            max_distance=AUTO_PENDING_OVERLAP_DISTANCE):
                        self.pending_crossings[gid] = now
                        locked_g = self.locked_gender.get(gid, "?")
                        self._log_event(gid, locked_g, cam_name,
                                        "AUTO_PENDING_OVERLAP")
                        print(f"📌 AUTO-PENDING: GID {gid} près "
                              f"de l'overlap de {cam_name}")

            already_in_lost = any(
                x['gid'] == gid and x['cam'] == cam_name
                for x in self.recently_lost)
            if not already_in_lost and gid in self.visual_db:
                pos_data = self.last_known_global_pos.get(gid)
                stored_box = pos_data['box'] if pos_data else [0, 0, 0, 0]
                self.recently_lost.append({
                    'gid': gid,
                    'cam': cam_name,
                    'time': now,
                    'vector': self.visual_db[gid]['vector'],
                    'hist': self.visual_db[gid]['info'].get('histogram'),
                    'box': stored_box
                })

            if key not in self.first_lost_time:
                self.first_lost_time[key] = now

            time_since_lost = now - self.first_lost_time[key]
            if time_since_lost >= LOST_GRACE_SECONDS:
                lost_pair = (gid, cam_name)
                if lost_pair not in self._lost_logged:
                    self._lost_logged.add(lost_pair)
                    locked_g = self.locked_gender.get(gid, "?")
                    self._log_event(gid, locked_g, cam_name,
                                    f"LOST from {cam_name}")

        # ══════════════════════════════════════════════════════
        # CLEANUP
        # ══════════════════════════════════════════════════════
        keys_to_remove = []
        for k in list(self.local_to_global.keys()):
            if k[0] != cam_name:
                continue
            if k[1] not in current_local_ids:
                last_t = self.last_seen_local.get(k, 0)
                if now - last_t > RELINK_TIMEOUT:
                    keys_to_remove.append(k)
                    if k in self.last_seen_local:
                        del self.last_seen_local[k]
                    if k in self.local_to_global_time:
                        del self.local_to_global_time[k]
                    if k in self.first_lost_time:
                        del self.first_lost_time[k]

        for k in keys_to_remove:
            del self.local_to_global[k]

        self.recently_lost = [
            x for x in self.recently_lost
            if (now - x['time']) < RELINK_TIMEOUT]

        self.cross_camera_lost = [
            x for x in self.cross_camera_lost
            if (now - x['time']) < CROSS_CAM_TIMEOUT]
        self._cross_lost_added = {
            (x['gid'], x['cam']) for x in self.cross_camera_lost}

        expired = [gid for gid, t in self.pending_crossings.items()
                   if (now - t) > CROSS_CAM_TIMEOUT]
        for gid in expired:
            del self.pending_crossings[gid]

        all_gids = set()
        for k in self.local_to_global.values():
            all_gids.add(k)
        for x in self.recently_lost:
            all_gids.add(x['gid'])
        for x in self.cross_camera_lost:
            all_gids.add(x['gid'])
        self._lost_logged = {
            (g, c) for g, c in self._lost_logged if g in all_gids}

        self.active_per_cam[cam_name] = current_local_ids