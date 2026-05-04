# backbone_config.py 
# Central config for Deformable DETR backbone 
# Import this in reid_head.py, tracker.py, train.py 
  
BACKBONE_CONFIG = { 
    # HuggingFace model name 
    "model_name": "SenseTime/deformable-detr", 
  
    # Encoder output dimension — input to Re-ID head 
    "feature_dim": 256, 
  
    # Re-ID embedding space dimension — output of Re-ID head 
    "embed_dim": 128, 
  
    # Number of attention heads in Re-ID cross-attention 
    "num_heads": 4, 
  
    # Detection confidence threshold 
    "detection_threshold": 0.5, 
  
    # Max frames a track can be lost before deletion 
    "max_lost_frames": 30, 
  
    # Hungarian matching similarity threshold 
    "match_threshold": 0.5, 
  
    # Re-ID head learning rate 
    "reid_lr": 1e-4, 
  
    # Training epochs for Re-ID head 
    "reid_epochs": 20, 
  
    # Batch size 
    "batch_size": 4, 
  
    # Input image size for processor 
    "image_size": (800, 800), 
  
    # Dataset paths (update before training) 
    "ua_detrac_img_dir": "data/UA-DETRAC/images",
    "ua_detrac_ann_dir": "data/UA-DETRAC/annotations",
  
    # Output paths 
    "reid_weights_path": "backend/model/weights/reid_head.pth", 
    "output_video_path": "outputs/tracked_output.mp4", 
} 
  
# COCO vehicle class IDs detected by Deformable DETR 
VEHICLE_CLASS_IDS = { 
    2: "bicycle", 
    3: "car", 
    4: "motorcycle", 
    6: "bus", 
    8: "truck", 
} 
  
if __name__ == "__main__": 
    print("Backbone Config:") 
    for k, v in BACKBONE_CONFIG.items(): 
        print(f"  {k}: {v}") 
    print(f"\nTracking vehicle classes: {list(VEHICLE_CLASS_IDS.values())}") 
