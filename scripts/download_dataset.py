import os
import json
import shutil
import requests
import zipfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG — update YOUR_ROBOFLOW_API_KEY below
# Get free API key at: https://app.roboflow.com
# ─────────────────────────────────────────────
ROBOFLOW_API_KEY   = "YOUR_ROBOFLOW_API_KEY"
WORKSPACE          = "vehicle-detection-loakn"
PROJECT            = "ua-detrac-10k-sample"
VERSION            = 1
FORMAT             = "coco"

DATA_ROOT          = "data/UA-DETRAC"
IMG_DIR            = os.path.join(DATA_ROOT, "images")
ANN_DIR            = os.path.join(DATA_ROOT, "annotations")
RAW_DIR            = os.path.join(DATA_ROOT, "raw_coco")

SEQUENCES_PER_SPLIT = 5   # number of fake sequences to group images into


def download_from_roboflow():
    """Download dataset via Roboflow API"""
    print("[1/4] Fetching download URL from Roboflow...")

    url = (
        f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/{FORMAT}"
        f"?api_key={ROBOFLOW_API_KEY}"
    )
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"[Error] Roboflow API response: {resp.status_code}")
        print(resp.text[:300])
        raise SystemExit(1)

    data     = resp.json()
    dl_url   = data.get("export", {}).get("link") or data.get("link")
    if not dl_url:
        print(f"[Error] Could not find download link in response: {data}")
        raise SystemExit(1)

    print(f"[1/4] Download URL obtained.")
    print(f"[2/4] Downloading dataset zip...")

    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "dataset.zip")

    r = requests.get(dl_url, stream=True, timeout=120)
    total = int(r.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[2/4] Downloaded → {zip_path}")
    return zip_path


def extract_zip(zip_path):
    """Extract downloaded zip"""
    print("[3/4] Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    print(f"[3/4] Extracted to {RAW_DIR}")


def find_coco_files():
    """Find all _annotations.coco.json files in extracted folder"""
    coco_files = list(Path(RAW_DIR).rglob("*_annotations.coco.json"))
    if not coco_files:
        # fallback: any .json file
        coco_files = list(Path(RAW_DIR).rglob("*.json"))
    return coco_files


def coco_to_detrac_xml(images, annotations, seq_name):
    """
    Convert a list of COCO images + annotations to UA-DETRAC XML format.

    UA-DETRAC XML structure:
    <sequence name="MVI_XXXXX">
      <frame num="1">
        <target_list>
          <target id="1">
            <box left="x" top="y" width="w" height="h" />
          </target>
        </target_list>
      </frame>
    </sequence>
    """
    # Build image_id → annotation list map
    ann_map = {}
    for ann in annotations:
        iid = ann["image_id"]
        ann_map.setdefault(iid, []).append(ann)

    root = ET.Element("sequence", name=seq_name)

    for frame_num, img in enumerate(images, start=1):
        frame_el = ET.SubElement(root, "frame", num=str(frame_num))
        tlist    = ET.SubElement(frame_el, "target_list")

        for track_id, ann in enumerate(
            ann_map.get(img["id"], []), start=1
        ):
            bbox = ann["bbox"]   # COCO: [x, y, w, h]
            target = ET.SubElement(tlist, "target", id=str(track_id))
            ET.SubElement(target, "box",
                left=str(round(bbox[0], 2)),
                top=str(round(bbox[1], 2)),
                width=str(round(bbox[2], 2)),
                height=str(round(bbox[3], 2))
            )

    # Pretty print
    xmlstr = minidom.parseString(
        ET.tostring(root, encoding="unicode")
    ).toprettyxml(indent="  ")
    return xmlstr


def organize_dataset():
    """
    Main conversion:
    1. Find COCO JSON files
    2. Group images into sequences
    3. Copy images to data/UA-DETRAC/images/SEQ_NAME/
    4. Write XML to data/UA-DETRAC/annotations/SEQ_NAME.xml
    """
    print("[4/4] Converting COCO → UA-DETRAC format...")

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(ANN_DIR, exist_ok=True)

    coco_files = find_coco_files()
    if not coco_files:
        print("[Error] No COCO annotation files found in downloaded data.")
        print(f"        Check contents of: {RAW_DIR}")
        raise SystemExit(1)

    print(f"  Found {len(coco_files)} annotation file(s)")

    total_imgs = 0
    total_seqs = 0

    for coco_path in coco_files:
        split_name = coco_path.stem.replace(
            "_annotations.coco", ""
        ).replace("_annotations", "")

        with open(coco_path) as f:
            coco = json.load(f)

        images      = coco.get("images", [])
        annotations = coco.get("annotations", [])

        print(f"\n  Processing split: {split_name}")
        print(f"  Images: {len(images)} | Annotations: {len(annotations)}")

        if not images:
            print("  [Skip] No images found.")
            continue

        # Group images into sequences of equal size
        chunk_size = max(1, len(images) // SEQUENCES_PER_SPLIT)
        chunks     = [
            images[i:i+chunk_size]
            for i in range(0, len(images), chunk_size)
        ]

        # Find image source folder
        img_source_dirs = list(Path(RAW_DIR).rglob(split_name))
        img_source_dir  = img_source_dirs[0] if img_source_dirs else None

        for seq_idx, chunk in enumerate(chunks):
            seq_name   = f"MVI_{20011 + total_seqs:05d}"
            seq_img_dir = os.path.join(IMG_DIR, seq_name)
            os.makedirs(seq_img_dir, exist_ok=True)

            # Copy images
            for frame_num, img_info in enumerate(
                tqdm(chunk, desc=f"  Copying {seq_name}"), start=1
            ):
                fname    = img_info["file_name"]
                dst_name = f"img{frame_num:05d}.jpg"
                dst      = os.path.join(seq_img_dir, dst_name)

                # Try to find source image
                src = None
                if img_source_dir:
                    candidate = img_source_dir / fname
                    if candidate.exists():
                        src = str(candidate)

                if src is None:
                    # Search recursively
                    matches = list(Path(RAW_DIR).rglob(fname))
                    src     = str(matches[0]) if matches else None

                if src and os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    # Create blank placeholder if image not found
                    from PIL import Image as PILImage
                    PILImage.new("RGB", (960, 540), color=(50,50,50)).save(dst)

            # Write XML annotation
            seq_anns = [
                a for a in annotations
                if a["image_id"] in {img["id"] for img in chunk}
            ]
            xml_str  = coco_to_detrac_xml(chunk, seq_anns, seq_name)
            xml_path = os.path.join(ANN_DIR, f"{seq_name}.xml")
            with open(xml_path, "w") as f:
                f.write(xml_str)

            total_imgs += len(chunk)
            total_seqs += 1
            print(f"  ✅ {seq_name}: {len(chunk)} frames → {xml_path}")

    print(f"\n{'='*50}")
    print(f"DATASET SETUP COMPLETE")
    print(f"  Total sequences : {total_seqs}")
    print(f"  Total frames    : {total_imgs}")
    print(f"  Images dir      : {IMG_DIR}")
    print(f"  Annotations dir : {ANN_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("="*50)
    print("UA-DETRAC Dataset Downloader + Converter")
    print("="*50)

    if ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY":
        print("\n[ERROR] Please set your Roboflow API key first!")
        print("  1. Go to https://app.roboflow.com")
        print("  2. Sign up free → Settings → API Keys")
        print("  3. Copy your key and paste it into this script")
        print("     ROBOFLOW_API_KEY = 'your_actual_key_here'")
        raise SystemExit(1)

    zip_path = download_from_roboflow()
    extract_zip(zip_path)
    organize_dataset()
