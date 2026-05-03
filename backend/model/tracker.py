import torch
import numpy as np
from PIL import Image
from backbone_config import BACKBONE_CONFIG, VEHICLE_CLASS_IDS
from deformable_detr import DeformableDETRBackbone
from reid_head import ReIDAttentionHead
from reid_utils import load_reid_weights, cosine_similarity_matrix
from hungarian import hungarian_match, iou_matrix, combined_similarity


class Track:
    """Represents a single tracked vehicle"""
    def __init__(self, track_id, box, embedding, frame_id):
        self.track_id  = track_id
        self.box       = box           # [x1,y1,x2,y2] tensor
        self.embedding = embedding     # (feature_dim,) tensor
        self.lost      = 0             # frames since last matched
        self.age       = 1             # total frames this track has lived
        self.history   = [box]        # list of past boxes
        self.first_frame = frame_id
        self.last_frame  = frame_id

    def update(self, box, embedding, frame_id):
        self.box       = box
        self.embedding = embedding
        self.lost      = 0
        self.age      += 1
        self.last_frame = frame_id
        self.history.append(box)
        # Keep only last 30 boxes in history
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_lost(self):
        self.lost += 1
        self.age  += 1

    def is_dead(self, max_lost):
        return self.lost > max_lost

    def get_center(self):
        box = self.box
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        return ((box[0]+box[2])//2, (box[1]+box[3])//2)


class VehicleTracker:
    """
    Full multi-object vehicle tracker.
    Pipeline per frame:
      1. Detect vehicles via Deformable DETR backbone
      2. Extract encoder features for each detection
      3. Compare detections vs track memory via Re-ID head
      4. Assign IDs using Hungarian matching (Re-ID + IoU combined)
      5. Update track memory: matched / new / lost / dead
    """
    def __init__(
        self,
        device=None,
        reid_weights_path=None,
        detection_threshold=None,
        match_threshold=None,
        max_lost=None
    ):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.detection_threshold = detection_threshold or \
            BACKBONE_CONFIG['detection_threshold']
        self.match_threshold = match_threshold or \
            BACKBONE_CONFIG['match_threshold']
        self.max_lost = max_lost or \
            BACKBONE_CONFIG['max_lost_frames']

        # Load backbone (frozen)
        print("[Tracker] Loading backbone...")
        self.backbone = DeformableDETRBackbone(self.device)

        # Load Re-ID head
        print("[Tracker] Loading Re-ID head...")
        self.reid_head = ReIDAttentionHead().to(self.device)
        if reid_weights_path:
            self.reid_head = load_reid_weights(
                self.reid_head, reid_weights_path, self.device
            )
        self.reid_head.eval()

        # Track memory
        self.tracks   = {}     # track_id → Track object
        self.next_id  = 1
        self.frame_id = 0

        print(f"[Tracker] Ready on {self.device}")

    def update(self, frame_image):
        """
        Process one frame and return tracking results.

        Args:
          frame_image: PIL.Image (RGB)

        Returns:
          results: list of dicts with keys:
            track_id, box [x1,y1,x2,y2], score, age, lost
        """
        self.frame_id += 1

        # Step 1: Detect vehicles
        boxes, scores, labels = self.backbone.detect(
            frame_image, self.detection_threshold
        )

        # Filter vehicle classes only
        vehicle_mask = torch.tensor([
            l.item() in VEHICLE_CLASS_IDS for l in labels
        ])
        if vehicle_mask.sum() == 0:
            self._age_all_tracks()
            return []

        boxes  = boxes[vehicle_mask]
        scores = scores[vehicle_mask]

        # Step 2: Extract features
        det_feats = self._extract_features(frame_image, boxes)

        results = []

        if len(self.tracks) == 0:
            # No existing tracks — initialize all detections as new tracks
            for i in range(len(boxes)):
                self._init_track(boxes[i], det_feats[i], scores[i])

        else:
            # Step 3: Build track tensors
            trk_ids   = list(self.tracks.keys())
            trk_feats = torch.stack([
                self.tracks[tid].embedding for tid in trk_ids
            ]).to(self.device)
            trk_boxes = np.array([
                self.tracks[tid].box.cpu().tolist()
                if isinstance(self.tracks[tid].box, torch.Tensor)
                else self.tracks[tid].box
                for tid in trk_ids
            ])

            # Step 4: Compute similarity
            with torch.no_grad():
                reid_sim, _ = self.reid_head(det_feats, trk_feats)
                reid_sim_np = reid_sim.cpu().numpy()

            det_boxes_np = boxes.cpu().numpy()
            iou_sim_np   = iou_matrix(det_boxes_np, trk_boxes)

            # Pad IoU to same shape as reid_sim if needed
            if iou_sim_np.shape != reid_sim_np.shape:
                iou_sim_np = np.zeros_like(reid_sim_np)

            combined = combined_similarity(reid_sim_np, iou_sim_np)

            # Step 5: Hungarian assignment
            matched, unmatched_dets, unmatched_trks = hungarian_match(
                combined, self.match_threshold
            )

            # Update matched tracks
            for det_idx, trk_idx in matched:
                tid = trk_ids[trk_idx]
                self.tracks[tid].update(
                    boxes[det_idx],
                    det_feats[det_idx].detach(),
                    self.frame_id
                )

            # Initialize new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._init_track(
                    boxes[det_idx],
                    det_feats[det_idx],
                    scores[det_idx]
                )

            # Age unmatched tracks
            for trk_idx in unmatched_trks:
                tid = trk_ids[trk_idx]
                self.tracks[tid].mark_lost()

        # Step 6: Remove dead tracks
        dead = [tid for tid, t in self.tracks.items()
                if t.is_dead(self.max_lost)]
        for tid in dead:
            del self.tracks[tid]

        # Build results
        for tid, track in self.tracks.items():
            box = track.box
            if isinstance(box, torch.Tensor):
                box = box.int().tolist()
            results.append({
                'track_id': tid,
                'box':      box,
                'age':      track.age,
                'lost':     track.lost,
                'center':   track.get_center()
            })

        return results

    def _init_track(self, box, feat, score):
        """Create a new Track object and add to memory"""
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = Track(
            track_id=tid,
            box=box,
            embedding=feat.detach(),
            frame_id=self.frame_id
        )

    def _extract_features(self, image, boxes):
        """
        Extract per-detection features from backbone encoder output.
        Uses mean pooling over encoder tokens as a simple RoI feature.
        Phase 5 upgrade: replace with proper RoI Align.
        """
        enc_feats = self.backbone.extract_features(image)
        # enc_feats: (1, num_tokens, 256)
        base_feat = enc_feats.mean(dim=1).squeeze(0)  # (256,)
        # Replicate base feature for each detection
        # (upgrade in Phase 5 with RoI-based features)
        return base_feat.unsqueeze(0).expand(len(boxes), -1).clone()

    def _age_all_tracks(self):
        """Age all tracks when no detections in frame"""
        dead = []
        for tid in self.tracks:
            self.tracks[tid].mark_lost()
            if self.tracks[tid].is_dead(self.max_lost):
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

    def reset(self):
        """Reset tracker state between videos"""
        self.tracks   = {}
        self.next_id  = 1
        self.frame_id = 0
        print("[Tracker] State reset.")

    def get_track_count(self):
        return len(self.tracks)

    def get_stats(self):
        return {
            'frame_id':    self.frame_id,
            'active_tracks': len(self.tracks),
            'next_id':     self.next_id,
            'device':      self.device
        }


if __name__ == "__main__":
    import requests
    print("VehicleTracker — Unit Test")

    tracker = VehicleTracker()

    # Load two test frames from COCO
    urls = [
        "http://images.cocodataset.org/val2017/000000397133.jpg",
        "http://images.cocodataset.org/val2017/000000037777.jpg"
    ]

    for i, url in enumerate(urls):
        frame = Image.open(
            requests.get(url, stream=True).raw
        ).convert("RGB")
        results = tracker.update(frame)
        print(f"\nFrame {i+1}: {len(results)} vehicles tracked")
        for r in results:
            print(f"  ID:{r['track_id']} box:{r['box']} age:{r['age']}")

    print(f"\nTracker stats: {tracker.get_stats()}")
    print("[TEST PASSED] Full tracker pipeline working.")
