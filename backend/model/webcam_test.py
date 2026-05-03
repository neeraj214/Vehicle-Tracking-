import cv2
import sys
import os
from PIL import Image
import time

sys.path.append(os.path.dirname(__file__))
from tracker import VehicleTracker
from backbone_config import BACKBONE_CONFIG

COLORS = [
    (0,255,0),(0,128,255),(255,0,128),(0,255,255),
    (255,255,0),(128,0,255),(255,128,0),(0,200,150),
]

def get_color(tid):
    return COLORS[tid % len(COLORS)]


def run_webcam(source=0):
    """
    Live vehicle tracking from webcam or video file.
    Press Q to quit. Press R to reset tracker.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Cannot open source: {source}")
        return

    tracker = VehicleTracker()
    track_history = {}

    print("[Webcam] Starting live tracking. Press Q=quit, R=reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        pil_img = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        results = tracker.update(pil_img)
        fps = 1 / (time.time() - t0)

        # Draw results
        for r in results:
            tid   = r['track_id']
            box   = r['box']
            color = get_color(tid)
            x1,y1,x2,y2 = [int(v) for v in box]
            cx,cy = (x1+x2)//2, (y1+y2)//2

            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append((cx,cy))
            if len(track_history[tid]) > 30:
                track_history[tid].pop(0)

            # Trail
            hist = track_history[tid]
            for i in range(1, len(hist)):
                cv2.line(frame, hist[i-1], hist[i], color, 2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # HUD
        cv2.putText(frame,
            f"Vehicles:{len(results)} FPS:{fps:.1f}",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2
        )

        cv2.imshow("Vehicle Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker.reset()
            track_history.clear()
            print("[Reset] Tracker cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Webcam tracking stopped.")


if __name__ == "__main__":
    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam(source)
