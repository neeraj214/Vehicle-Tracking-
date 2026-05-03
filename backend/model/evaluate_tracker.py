import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from tracker import VehicleTracker
from hungarian import iou_matrix

def load_gt_sequence(img_dir, ann_xml_path):
    """Load ground truth frames and annotations from UA-DETRAC XML"""
    tree = ET.parse(ann_xml_path)
    root = tree.getroot()
    sequence = []

    for frame in root.findall('frame'):
        frame_num = int(frame.get('num'))
        img_path  = os.path.join(img_dir, f"img{frame_num:05d}.jpg")
        if not os.path.exists(img_path):
            continue

        gt_boxes, gt_ids = [], []
        for target in frame.findall('.//target'):
            box = target.find('box')
            x = float(box.get('left'))
            y = float(box.get('top'))
            w = float(box.get('width'))
            h = float(box.get('height'))
            gt_boxes.append([x, y, x+w, y+h])
            gt_ids.append(int(target.get('id')))

        sequence.append({
            'frame_num': frame_num,
            'img_path':  img_path,
            'gt_boxes':  np.array(gt_boxes),
            'gt_ids':    gt_ids
        })

    return sequence


def compute_metrics(sequence, tracker, iou_threshold=0.5):
    """
    Compute MOTA, MOTP, ID Switches over a sequence.

    MOTA = 1 - (FP + FN + ID_SW) / GT_total
    MOTP = mean IoU of matched pairs
    """
    total_gt    = 0
    total_fp    = 0
    total_fn    = 0
    total_idsw  = 0
    total_iou   = []
    prev_matches = {}   # gt_id → track_id from previous frame

    for sample in sequence:
        img   = Image.open(sample['img_path']).convert('RGB')
        gt_b  = sample['gt_boxes']
        gt_ids = sample['gt_ids']
        total_gt += len(gt_ids)

        # Run tracker
        results  = tracker.update(img)
        pred_boxes = np.array([r['box'] for r in results]) \
                     if results else np.zeros((0,4))
        pred_ids   = [r['track_id'] for r in results]

        if len(gt_b) == 0 and len(pred_boxes) == 0:
            continue

        if len(pred_boxes) == 0:
            total_fn += len(gt_ids)
            continue

        if len(gt_b) == 0:
            total_fp += len(pred_boxes)
            continue

        # Match GT to predictions via IoU
        iou = iou_matrix(gt_b, pred_boxes)
        matched_gt, matched_pred = [], []
        used_pred = set()

        for g_idx in range(len(gt_ids)):
            best_iou  = iou_threshold
            best_pidx = -1
            for p_idx in range(len(pred_ids)):
                if p_idx not in used_pred and iou[g_idx, p_idx] > best_iou:
                    best_iou  = iou[g_idx, p_idx]
                    best_pidx = p_idx
            if best_pidx >= 0:
                matched_gt.append(g_idx)
                matched_pred.append(best_pidx)
                used_pred.add(best_pidx)
                total_iou.append(best_iou)

        # FP, FN
        total_fp += len(pred_ids) - len(matched_pred)
        total_fn += len(gt_ids)  - len(matched_gt)

        # ID Switches
        curr_matches = {}
        for g_idx, p_idx in zip(matched_gt, matched_pred):
            gt_id   = gt_ids[g_idx]
            pred_id = pred_ids[p_idx]
            curr_matches[gt_id] = pred_id
            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                total_idsw += 1

        prev_matches = curr_matches

    # Compute final metrics
    mota = 1 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)
    motp = float(np.mean(total_iou)) if total_iou else 0.0

    return {
        'MOTA':        round(mota * 100, 2),
        'MOTP':        round(motp * 100, 2),
        'ID_Switches': total_idsw,
        'FP':          total_fp,
        'FN':          total_fn,
        'GT_total':    total_gt,
    }


if __name__ == "__main__":
    # Update these paths to your UA-DETRAC sequence
    IMG_DIR  = "data/UA-DETRAC/images/MVI_20011"
    ANN_XML  = "data/UA-DETRAC/annotations/MVI_20011.xml"

    if not os.path.exists(ANN_XML):
        print("[Skip] UA-DETRAC data not found.")
        print("       Update IMG_DIR and ANN_XML paths after downloading.")
        print("       Download from: https://detrac-db.rit.albany.edu/")
    else:
        print("Loading sequence...")
        sequence = load_gt_sequence(IMG_DIR, ANN_XML)
        print(f"Loaded {len(sequence)} frames")

        tracker = VehicleTracker()
        print("Running evaluation...")
        metrics = compute_metrics(sequence, tracker)

        print("\n" + "="*40)
        print("TRACKING METRICS")
        print("="*40)
        for k, v in metrics.items():
            print(f"  {k:<15}: {v}")
        print("="*40)
