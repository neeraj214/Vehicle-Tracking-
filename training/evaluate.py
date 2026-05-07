import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'model'))
from tracker import VehicleTracker
from evaluate_tracker import load_gt_sequence, compute_metrics
from backbone_config import BACKBONE_CONFIG


def run_evaluation(img_dir, ann_xml, reid_weights=None, output_json=None):
    print("=" * 55)
    print("VEHICLE TRACKER — EVALUATION")
    print("=" * 55)

    if not os.path.exists(ann_xml):
        print(f"[Error] Annotation file not found: {ann_xml}")
        return

    print(f"Loading sequence from: {img_dir}")
    sequence = load_gt_sequence(img_dir, ann_xml)
    print(f"Loaded {len(sequence)} frames\n")

    tracker = VehicleTracker(
        reid_weights_path=reid_weights
    )

    print("Running tracker on sequence...")
    metrics = compute_metrics(sequence, tracker)

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"  MOTA         : {metrics['MOTA']}%")
    print(f"  MOTP         : {metrics['MOTP']}%")
    print(f"  ID Switches  : {metrics['ID_Switches']}")
    print(f"  False Pos    : {metrics['FP']}")
    print(f"  False Neg    : {metrics['FN']}")
    print(f"  GT Total     : {metrics['GT_total']}")
    print("=" * 55)

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Saved] Metrics → {output_json}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",  required=True,
                        help="Path to sequence image folder")
    parser.add_argument("--ann_xml",  required=True,
                        help="Path to UA-DETRAC XML annotation")
    parser.add_argument("--weights",  default=None,
                        help="Path to trained Re-ID weights (.pth)")
    parser.add_argument("--output",   default=None,
                        help="Save metrics to JSON file")
    args = parser.parse_args()

    run_evaluation(
        args.img_dir,
        args.ann_xml,
        args.weights,
        args.output
    )
