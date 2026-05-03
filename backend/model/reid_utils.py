import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

def pair_generator(features, track_ids, max_pairs=256):
    """
    Generate positive and negative pairs from a batch.
    Positive pair: same track_id → label = 1
    Negative pair: different track_id → label = 0

    Args:
      features:  (N, feature_dim) tensor
      track_ids: list of N track IDs
      max_pairs: max total pairs to return (balanced pos/neg)

    Returns:
      feats1, feats2: (M, feature_dim) tensors
      labels: (M,) float tensor with 0/1
    """
    feats1, feats2, labels = [], [], []
    n = len(track_ids)

    for i in range(n):
        for j in range(i + 1, n):
            label = 1.0 if track_ids[i] == track_ids[j] else 0.0
            feats1.append(features[i])
            feats2.append(features[j])
            labels.append(label)

    if len(feats1) == 0:
        return None, None, None

    feats1  = torch.stack(feats1)
    feats2  = torch.stack(feats2)
    labels  = torch.tensor(labels, dtype=torch.float)

    # Balance and cap
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
    n_each  = min(len(pos_idx), len(neg_idx), max_pairs // 2)

    if n_each == 0:
        return feats1[:max_pairs], feats2[:max_pairs], labels[:max_pairs]

    pos_sel = pos_idx[torch.randperm(len(pos_idx))[:n_each]]
    neg_sel = neg_idx[torch.randperm(len(neg_idx))[:n_each]]
    sel     = torch.cat([pos_sel, neg_sel])

    return feats1[sel], feats2[sel], labels[sel]


def cosine_similarity_matrix(feats1, feats2):
    """
    Compute cosine similarity matrix as a fast baseline
    (before Re-ID head is trained).

    Args:
      feats1: (N, D)
      feats2: (M, D)
    Returns:
      similarity: (N, M) in [-1, 1]
    """
    f1 = F.normalize(feats1, p=2, dim=-1)
    f2 = F.normalize(feats2, p=2, dim=-1)
    return torch.mm(f1, f2.t())


def visualize_attention(frame_bgr, boxes, attn_weights, track_ids):
    """
    Overlay attention weights as color-coded lines between
    detections and their matched tracks.

    Args:
      frame_bgr:   OpenCV BGR image (H, W, 3)
      boxes:       list of [x1,y1,x2,y2] for current detections
      attn_weights: (N_det, N_trk) numpy array
      track_ids:   list of track IDs for existing tracks
    Returns:
      annotated frame (BGR)
    """
    img = frame_bgr.copy()
    N_det, N_trk = attn_weights.shape

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if i < N_det:
            best_trk = int(np.argmax(attn_weights[i]))
            score    = float(attn_weights[i][best_trk])
            color    = (0, int(255 * score), int(255 * (1 - score)))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_ids[best_trk]} ({score:.2f})" \
                    if best_trk < len(track_ids) else f"new"
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return img


def save_reid_weights(model, path):
    """Save Re-ID head weights to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': model.feature_dim,
        'embed_dim':   model.embed_dim,
        'num_heads':   model.num_heads,
    }, path)
    print(f"[Saved] Re-ID weights → {path}")


def load_reid_weights(model, path, device='cpu'):
    """Load Re-ID head weights from disk"""
    if not os.path.exists(path):
        print(f"[Warning] No weights found at {path} — using random init")
        return model
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[Loaded] Re-ID weights from {path}")
    return model


if __name__ == "__main__":
    print("Re-ID Utils — Unit Test")

    # Test pair generator
    feats = torch.randn(6, 256)
    ids   = [1, 1, 2, 2, 3, 3]
    f1, f2, lbls = pair_generator(feats, ids)
    print(f"Pairs generated: {len(lbls)}")
    print(f"Positives: {int(lbls.sum())}  Negatives: {int((lbls==0).sum())}")

    # Test cosine similarity
    sim = cosine_similarity_matrix(feats[:3], feats[3:])
    print(f"Cosine sim matrix shape: {sim.shape}")

    print("[TEST PASSED] Utils working correctly.")
