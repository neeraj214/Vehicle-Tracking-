import torch 
from PIL import Image 
import requests 
from deformable_detr import DeformableDETRBackbone 
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection 
  
def explore_model_features(): 
    print("=" * 60) 
    print("DEFORMABLE DETR — FEATURE EXPLORER") 
    print("=" * 60) 
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
  
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr") 
    model = DeformableDetrForObjectDetection.from_pretrained( 
        "SenseTime/deformable-detr" 
    ).to(device) 
    model.eval() 
  
    # Load sample image 
    url = "http://images.cocodataset.org/val2017/000000039769.jpg" 
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB") 
    inputs = processor(images=image, return_tensors="pt").to(device) 
  
    with torch.no_grad(): 
        outputs = model(**inputs, output_hidden_states=True) 
  
    print(f"\n[1] Encoder last hidden state shape:") 
    print(f"    {outputs.encoder_last_hidden_state.shape}") 
    print(f"    → (batch, num_encoder_tokens, d_model)") 
  
    print(f"\n[2] Decoder hidden states (last layer):") 
    dec = outputs.decoder_hidden_states 
    if dec: 
        print(f"    {dec[-1].shape}") 
        print(f"    → (batch, num_queries, d_model)") 
    else: 
        print("    Not available") 
  
    print(f"\n[3] Logits shape (class predictions):") 
    print(f"    {outputs.logits.shape}") 
    print(f"    → (batch, num_queries, num_classes+1)") 
  
    print(f"\n[4] Pred boxes shape:") 
    print(f"    {outputs.pred_boxes.shape}") 
    print(f"    → (batch, num_queries, 4) in cxcywh format") 
  
    print(f"\n[5] Model config — key values:") 
    print(f"    d_model (hidden dim):  {model.config.d_model}") 
    print(f"    num_queries:           {model.config.num_queries}") 
    print(f"    encoder_layers:        {model.config.encoder_layers}") 
    print(f"    decoder_layers:        {model.config.decoder_layers}") 
    print(f"    num_labels:            {model.config.num_labels}") 
  
    print(f"\n[6] Total parameters: {sum(p.numel() for p in model.parameters()):,}") 
    print(f"    Trainable params:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}") 
  
    print("\n" + "=" * 60) 
    print("KEY INSIGHT FOR RE-ID HEAD (Phase 3):") 
    print(f"  Input dim to Re-ID head = d_model = {model.config.d_model}") 
    print(f"  You will project {model.config.d_model}D → 128D embedding space") 
    print("=" * 60) 
  
if __name__ == "__main__": 
    explore_model_features() 
