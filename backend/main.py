import os
import sys
import time
import uuid
import shutil
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from tracker import VehicleTracker
from backbone_config import BACKBONE_CONFIG

app = FastAPI(
    title="Vehicle Tracker API",
    description="Multi-Object Vehicle Tracking using Deformable DETR + Re-ID",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tracker once at startup
print("[API] Initializing tracker...")
tracker = VehicleTracker(
    reid_weights_path=BACKBONE_CONFIG.get('reid_weights_path')
)
print("[API] Tracker ready.")

COLORS = [
    (0,255,0),(0,128,255),(255,0,128),(0,255,255),
    (255,255,0),(128,0,255),(255,128,0),(0,200,150),
]

def get_color(tid):
    return COLORS[tid % len(COLORS)]


@app.get("/health")
def health():
    return {"status": "ok", "model": "Deformable DETR + Re-ID Head"}


@app.get("/stats")
def stats():
    return tracker.get_stats()


@app.post("/track-frame")
async def track_frame(file: UploadFile = File(...)):
    """
    Upload a single image frame.
    Returns: list of detected + tracked vehicles with boxes and IDs.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    contents = await file.read()
    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")

    with open(tmp_path, "wb") as f:
        f.write(contents)

    try:
        image = Image.open(tmp_path).convert("RGB")
        results = tracker.update(image)
        return JSONResponse({
            "frame_id":   tracker.frame_id,
            "detections": len(results),
            "tracks": [
                {
                    "track_id": r["track_id"],
                    "box":      r["box"],
                    "age":      r["age"],
                    "center":   list(r["center"])
                }
                for r in results
            ]
        })
    finally:
        os.remove(tmp_path)


@app.post("/track-video")
async def track_video(file: UploadFile = File(...)):
    """
    Upload a video file.
    Returns: annotated video with track IDs and trail lines.
    """
    allowed = [".mp4", ".avi", ".mov", ".mkv"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Allowed formats: {allowed}")

    job_id   = str(uuid.uuid4())
    in_path  = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}_tracked.mp4")

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        tracker.reset()
        cap = cv2.VideoCapture(in_path)

        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

        track_history = {}
        frame_idx     = 0
        t0            = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pil_img = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            results = tracker.update(pil_img)
            frame_idx += 1

            # Annotate
            for r in results:
                tid   = r["track_id"]
                box   = r["box"]
                color = get_color(tid)
                x1,y1,x2,y2 = [int(v) for v in box]
                cx,cy = (x1+x2)//2, (y1+y2)//2

                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > 40:
                    track_history[tid].pop(0)

                # Trail
                hist = track_history[tid]
                for i in range(1, len(hist)):
                    cv2.line(frame, hist[i-1], hist[i], color, 2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(
                    frame, f"ID:{tid}",
                    (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            cv2.putText(
                frame,
                f"Frame:{frame_idx}/{TOTAL} Tracks:{len(results)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2
            )
            writer.write(frame)

        cap.release()
        writer.release()

        elapsed = time.time() - t0
        return FileResponse(
            out_path,
            media_type="video/mp4",
            filename="tracked_output.mp4",
            headers={
                "X-Frames-Processed": str(frame_idx),
                "X-Processing-Time":  f"{elapsed:.2f}s",
                "X-Total-Tracks":     str(tracker.next_id - 1)
            }
        )
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
