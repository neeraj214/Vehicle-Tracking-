import cv2 
import requests 
import numpy as np 
import torch 
from PIL import Image 
from deformable_detr import DeformableDETRBackbone 
  
# COCO class names (first 10 relevant ones) 
COCO_CLASSES = { 
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 
    5: "airplane", 6: "bus", 7: "train", 8: "truck", 
    9: "boat", 10: "traffic light" 
} 
  
def draw_detections(image_pil, boxes, scores, labels): 
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR) 
    for box, score, label in zip(boxes, scores, labels): 
        x1, y1, x2, y2 = box.int().tolist() 
        class_name = COCO_CLASSES.get(label.item(), f"cls_{label.item()}") 
        color = (0, 255, 0) 
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) 
        cv2.putText( 
            img, 
            f"{class_name} {score:.2f}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, color, 2 
        ) 
    return img 
  
if __name__ == "__main__": 
    # Load a sample car image from COCO 
    url = "http://images.cocodataset.org/val2017/000000397133.jpg" 
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB") 
    print(f"Image size: {image.size}") 
  
    # Load backbone 
    backbone = DeformableDETRBackbone() 
  
    # Extract features 
    feats = backbone.extract_features(image) 
    print(f"Encoder feature shape: {feats.shape}") 
  
    # Run detection 
    boxes, scores, labels = backbone.detect(image, threshold=0.4) 
    print(f"Total detections: {len(boxes)}") 
  
    # Draw and save 
    result_img = draw_detections(image, boxes, scores, labels) 
    cv2.imwrite("output_test.jpg", result_img) 
    print("Saved detection result to output_test.jpg") 
  
    # Summary 
    for box, score, label in zip(boxes, scores, labels): 
        cls = COCO_CLASSES.get(label.item(), f"cls_{label.item()}") 
        print(f"  {cls}: {score:.2f} @ {box.int().tolist()}") 
