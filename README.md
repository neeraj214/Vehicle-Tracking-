# 🚗 Vehicle Tracking System 
> Multi-Object Vehicle Tracker using Deformable DETR + Custom Re-ID Attention Head 
  
## 🔍 Overview 
Tracks vehicles across highway CCTV video frames using a pretrained Deformable DETR backbone with a custom cross-attention Re-ID head for multi-object tracking on the UA-DETRAC dataset. 
  
## 🧱 Architecture 
- **Backbone**: Deformable DETR (pretrained, frozen) 
- **Custom Head**: Re-ID Cross-Attention Module (trained) 
- **Assignment**: Hungarian Algorithm 
- **Dataset**: UA-DETRAC 
  
## 📦 Tech Stack 
- PyTorch, HuggingFace Transformers 
- FastAPI backend 
- React + TailwindCSS frontend 
- Deployed on Hugging Face Spaces + Vercel 
  
## 🚀 Setup 
```bash
pip install -r backend/requirements.txt 
python backend/main.py 
```
  
## 📁 Project Structure 
```
vehicle-tracker/ 
├── backend/ 
│   ├── model/ 
│   │   ├── __init__.py 
│   │   ├── deformable_detr.py 
│   │   ├── reid_head.py 
│   │   ├── hungarian.py 
│   │   └── tracker.py 
│   ├── main.py 
│   └── requirements.txt 
├── training/ 
│   ├── dataset.py 
│   ├── train.py 
│   └── evaluate.py 
├── notebooks/ 
│   └── explore_dataset.ipynb 
├── .gitignore 
└── README.md 
```
  
## 📊 Metrics 
- MOTA (Multi-Object Tracking Accuracy)
- MOTP (Multi-Object Tracking Precision)
- IDF1 (ID F1-Score)
- ID Switches 
  
## 👤 Author 
**Neeraj** — MCA Student | ML + Full Stack Developer 
GitHub: [https://github.com/neeraj214](https://github.com/neeraj214) 
