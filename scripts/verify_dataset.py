import os
import sys
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DATA_ROOT = "data/UA-DETRAC"
IMG_DIR   = os.path.join(DATA_ROOT, "images")
ANN_DIR   = os.path.join(DATA_ROOT, "annotations")


def verify_structure():
    print("="*55)
    print("UA-DETRAC DATASET VERIFICATION")
    print("="*55)

    errors   = []
    warnings = []

    # Check root dirs
    for d in [IMG_DIR, ANN_DIR]:
        if not os.path.exists(d):
            errors.append(f"Missing directory: {d}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return False

    # Get sequences
    sequences = sorted([
        d for d in os.listdir(IMG_DIR)
        if os.path.isdir(os.path.join(IMG_DIR, d))
    ])
    xml_files = sorted([
        f.replace(".xml", "") for f in os.listdir(ANN_DIR)
        if f.endswith(".xml")
    ])

    print(f"\n[1] Sequences found : {len(sequences)}")
    print(f"[2] XML files found : {len(xml_files)}")

    # Check matching
    missing_xml = set(sequences) - set(xml_files)
    missing_img = set(xml_files) - set(sequences)
    if missing_xml:
        warnings.append(f"Sequences without XML: {missing_xml}")
    if missing_img:
        warnings.append(f"XMLs without image folder: {missing_img}")

    # Per-sequence stats
    total_frames = 0
    total_boxes  = 0
    bad_images   = 0

    print(f"\n{'Sequence':<20} {'Frames':>8} {'Boxes':>8} {'Status'}")
    print("-" * 55)

    for seq in sequences[:10]:   # check first 10
        img_folder = os.path.join(IMG_DIR, seq)
        xml_path   = os.path.join(ANN_DIR, f"{seq}.xml")

        frames = sorted([
            f for f in os.listdir(img_folder)
            if f.endswith(".jpg")
        ])
        n_frames = len(frames)

        # Parse XML
        n_boxes = 0
        status  = "[OK]"
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for frame in root.findall("frame"):
                    for _ in frame.findall(".//target"):
                        n_boxes += 1
            except Exception as ex:
                status = f"[ERROR] XML error: {ex}"
                errors.append(f"{seq}: {ex}")
        else:
            status = "[WARN] No XML"
            warnings.append(f"{seq}: missing XML")

        # Check first image readable
        if frames:
            try:
                img = Image.open(
                    os.path.join(img_folder, frames[0])
                )
                _ = img.size
            except Exception:
                bad_images += 1
                status = "[ERROR] Bad image"

        total_frames += n_frames
        total_boxes  += n_boxes

        print(f"{seq:<20} {n_frames:>8} {n_boxes:>8}  {status}")

    if len(sequences) > 10:
        print(f"  ... and {len(sequences)-10} more sequences")

    # Summary
    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    print(f"  Total sequences  : {len(sequences)}")
    print(f"  Total frames     : {total_frames}")
    print(f"  Total boxes      : {total_boxes}")
    print(f"  Bad images       : {bad_images}")
    print(f"  Warnings         : {len(warnings)}")
    print(f"  Errors           : {len(errors)}")

    if warnings:
        print(f"\nWARNINGS:")
        for w in warnings:
            print(f"  [WARN]  {w}")

    if errors:
        print(f"\nERRORS:")
        for e in errors:
            print(f"  [ERROR] {e}")
        return False

    print(f"\n[OK] Dataset is ready for training!")
    return True


def test_dataset_loader():
    """Quick test of the PyTorch dataset loader"""
    print(f"\n{'='*55}")
    print("TESTING DATASET LOADER")
    print(f"{'='*55}")

    try:
        sys.path.append("training")
        from dataset import UATRACDataset
        import torch

        ds = UATRACDataset(IMG_DIR, ANN_DIR)
        print(f"  Dataset size: {len(ds)} samples")

        if len(ds) == 0:
            print("  [ERROR] Dataset is empty — check paths")
            return False

        # Load first sample
        img, boxes, ids = ds[0]
        print(f"  Sample 0:")
        print(f"    Image shape : {img.shape}")
        print(f"    Boxes       : {len(boxes)} detections")
        print(f"    Track IDs   : {ids[:5]}")

        # Load a few more
        for i in [1, 5, 10]:
            if i < len(ds):
                img2, b2, i2 = ds[i]
                assert img2.shape[0] == 3, "Image must be 3-channel"

        print(f"\n  [OK] DataLoader working correctly!")
        return True

    except ImportError as e:
        print(f"  [WARN]  Could not import dataset.py: {e}")
        print(f"      Run from project root directory")
        return False
    except Exception as e:
        print(f"  [ERROR] DataLoader error: {e}")
        return False


if __name__ == "__main__":
    ok = verify_structure()
    if ok:
        test_dataset_loader()
    else:
        print("\n[ERROR] Fix errors above before training.")
        sys.exit(1)
