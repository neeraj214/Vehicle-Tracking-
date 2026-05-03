import cv2
import sys
import os
import time
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from tracker import VehicleTracker
from backbone_config import BACKBONE_CONFIG, VEHICLE_CLASS_IDS

# Color palette for track IDs (BGR)
COLORS = [
    (0,255,0),(0,128,255),(255,0,128),(0,255,255),
    (255,255,0),(128,0,255),(255,128,0),(0,200,150),
    (200,0,200),(150,200,0),(0,150,200),(200,150,0)
]

def get_color(track_id):
    return COLORS[track_id % len(COLORS)]


def draw_trail(frame, track_history, track_id, color):
    """Draw motion trail from past centers"""
    history = track_history.get(track_id, [])
    for i in range(1, len(history)):
        alpha = i / len(history)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, history[i-1], history[i], color, thickness)


def annotate_frame(frame, results, track_history):
    """Draw boxes, IDs, centers and trails on frame"""
    for r in results:
        tid   = r['track_id']
        box   = r['box']
        age   = r['age']
        lost  = r['lost']
        color = get_color(tid)

        x1,y1,x2,y2 = [int(v) for v in box]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # Update trail history
        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append((cx, cy))
        if len(track_history[tid]) > 40:
            track_history[tid].pop(0)

        # Draw trail
        draw_trail(frame, track_history, tid, color)

        # Draw box
        thickness = 1 if lost > 0 else 2
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

        # Draw center dot
        cv2.circle(frame, (cx,cy), 4, color, -1)

        # Draw label
        label = f"ID:{tid} age:{age}"
        if lost > 0:
            label += f" lost:{lost}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+4,y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

    return frame


def run_inference(video_path, output_path=None, max_frames=None):
    """
    Run full vehicle tracking inference on a video file.

    Args:
      video_path:  path to input video
      output_path: path to save annotated video (optional)
      max_frames:  limit frames for testing (None = full video)
    """
    if not os.path.exists(video_path):
        print(f"[Error] Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Video] {video_path}")
    print(f"        {W}x{H} @ {FPS:.1f}fps | {TOTAL} frames")

    # Setup output writer
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))
        print(f"[Output] Saving to {output_path}")

    # Init tracker
    tracker       = VehicleTracker()
    track_history = {}

    # Stats
    frame_idx       = 0
    total_time      = 0
    max_active      = 0
    total_track_ids = set()

    print("\n[Processing...]")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        t0 = time.time()

        # Convert BGR → PIL RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # Track
        results = tracker.update(pil_frame)

        elapsed     = time.time() - t0
        total_time += elapsed
        frame_idx  += 1

        # Update stats
        active = len(results)
        if active > max_active:
            max_active = active
        for r in results:
            total_track_ids.add(r['track_id'])

        # Annotate frame
        annotated = annotate_frame(frame_bgr.copy(), results, track_history)

        # Overlay stats
        cv2.putText(
            annotated,
            f"Frame:{frame_idx} | Active:{active} | FPS:{1/elapsed:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2
        )

        if writer:
            writer.write(annotated)

        # Console progress every 10 frames
        if frame_idx % 10 == 0:
            print(
                f"  Frame {frame_idx}/{TOTAL} | "
                f"Active:{active} | "
                f"{elapsed*1000:.0f}ms/frame"
            )

    cap.release()
    if writer:
        writer.release()

    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(f"\n{'='*50}")
    print(f"TRACKING COMPLETE")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Avg speed        : {avg_fps:.1f} FPS")
    print(f"  Total tracks     : {len(total_track_ids)}")
    print(f"  Max active       : {max_active}")
    if output_path:
        print(f"  Output saved     : {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vehicle Tracker Inference")
    parser.add_argument("--video",   required=True, help="Input video path")
    parser.add_argument("--output",  default="outputs/tracked.mp4",
                        help="Output video path")
    parser.add_argument("--frames",  type=int, default=None,
                        help="Max frames to process (default: all)")
    args = parser.parse_args()

    run_inference(args.video, args.output, args.frames)
