from transformers import AutoImageProcessor, DeformableDetrForObjectDetection 
import torch 
from PIL import Image 
  
class DeformableDETRBackbone: 
    def __init__(self, device=None): 
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu') 
        print(f"[Backbone] Loading on device: {self.device}") 
  
        self.processor = AutoImageProcessor.from_pretrained( 
            "SenseTime/deformable-detr" 
        ) 
        self.model = DeformableDetrForObjectDetection.from_pretrained( 
            "SenseTime/deformable-detr" 
        ).to(self.device) 
  
        # Freeze all backbone parameters — only Re-ID head will be trained 
        for param in self.model.parameters(): 
            param.requires_grad = False 
  
        self.model.eval() 
        print("[Backbone] Loaded and frozen successfully.") 
  
    def extract_features(self, images): 
        """ 
        Extract encoder hidden states from one or more PIL images. 
        Args: 
            images: list of PIL.Image or single PIL.Image 
        Returns: 
            Tensor of shape (batch, num_queries, hidden_dim=256) 
        """ 
        if isinstance(images, Image.Image): 
            images = [images] 
  
        inputs = self.processor( 
            images=images, 
            return_tensors="pt" 
        ).to(self.device) 
  
        with torch.no_grad(): 
            outputs = self.model( 
                **inputs, 
                output_hidden_states=True 
            ) 
  
        # encoder_last_hidden_state: (batch, num_queries, 256) 
        return outputs.encoder_last_hidden_state 
  
    def detect(self, image, threshold=0.5): 
        """ 
        Run object detection on a single PIL image. 
        Args: 
            image: PIL.Image 
            threshold: confidence score cutoff 
        Returns: 
            boxes  — Tensor (N, 4) in [x1, y1, x2, y2] format 
            scores — Tensor (N,) 
            labels — Tensor (N,) 
        """ 
        if not isinstance(image, Image.Image): 
            raise TypeError("Input must be a PIL.Image") 
  
        inputs = self.processor( 
            images=image, 
            return_tensors="pt" 
        ).to(self.device) 
  
        with torch.no_grad(): 
            outputs = self.model(**inputs) 
  
        target_size = torch.tensor([image.size[::-1]])  # (H, W) 
        results = self.processor.post_process_object_detection( 
            outputs, 
            threshold=threshold, 
            target_sizes=target_size 
        )[0] 
  
        return results['boxes'], results['scores'], results['labels'] 
  
    def get_feature_dim(self): 
        """Returns the hidden dimension size of encoder output (256)""" 
        return self.model.config.d_model 
  
  
if __name__ == "__main__": 
    from PIL import Image 
    import requests 
  
    # Quick sanity test using a sample image 
    url = "http://images.cocodataset.org/val2017/000000039769.jpg" 
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB") 
  
    backbone = DeformableDETRBackbone() 
  
    # Test feature extraction 
    feats = backbone.extract_features(image) 
    print(f"Feature shape: {feats.shape}")  # Expected: (1, num_queries, 256) 
  
    # Test detection 
    boxes, scores, labels = backbone.detect(image, threshold=0.5) 
    print(f"Detected {len(boxes)} objects") 
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)): 
        print(f"  [{i}] label={label.item()} score={score:.2f} box={box.int().tolist()}") 
  
    print(f"Feature dim: {backbone.get_feature_dim()}") 
    print("[TEST PASSED] Backbone working correctly.") 
