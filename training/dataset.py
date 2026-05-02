import os 
import xml.etree.ElementTree as ET 
from torch.utils.data import Dataset 
from PIL import Image 
import torchvision.transforms as T 
  
class UATRACDataset(Dataset): 
    def __init__(self, img_dir, ann_dir, transform=None): 
        self.samples = [] 
        self.transform = transform or T.ToTensor() 
  
        for xml_file in os.listdir(ann_dir): 
            if not xml_file.endswith('.xml'): 
                continue 
            seq = xml_file.replace('.xml', '') 
            tree = ET.parse(os.path.join(ann_dir, xml_file)) 
            root = tree.getroot() 
  
            for frame in root.findall('frame'): 
                frame_num = int(frame.get('num')) 
                img_path = os.path.join( 
                    img_dir, seq, f"img{frame_num:05d}.jpg" 
                ) 
                boxes, track_ids = [], [] 
                for target in frame.findall('.//target'): 
                    box = target.find('box') 
                    boxes.append([ 
                        float(box.get('left')), 
                        float(box.get('top')), 
                        float(box.get('width')), 
                        float(box.get('height')) 
                    ]) 
                    track_ids.append(int(target.get('id'))) 
  
                if os.path.exists(img_path): 
                    self.samples.append({ 
                        'img_path': img_path, 
                        'boxes': boxes, 
                        'track_ids': track_ids, 
                        'frame': frame_num, 
                        'seq': seq 
                    }) 
  
    def __len__(self): 
        return len(self.samples) 
  
    def __getitem__(self, idx): 
        s = self.samples[idx] 
        img = Image.open(s['img_path']).convert('RGB') 
        return self.transform(img), s['boxes'], s['track_ids'] 
  
if __name__ == "__main__": 
    # Quick test — update paths before running 
    dataset = UATRACDataset( 
        img_dir="data/UA-DETRAC/images", 
        ann_dir="data/UA-DETRAC/annotations" 
    ) 
    print(f"Total frames loaded: {len(dataset)}") 
    img, boxes, ids = dataset[0] 
    print(f"Image shape: {img.shape}") 
    print(f"Boxes: {boxes[:2]}") 
    print(f"Track IDs: {ids[:2]}")
