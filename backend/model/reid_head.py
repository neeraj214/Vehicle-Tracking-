import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone_config import BACKBONE_CONFIG

class DetectionProjection(nn.Module):
    """Projects raw backbone features into Re-ID embedding space"""
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class TrackMemoryProjection(nn.Module):
    """Projects stored track embeddings into Re-ID space"""
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionReID(nn.Module):
    """
    Cross-attention module:
    - Query = current frame detections
    - Key/Value = stored track embeddings
    Each detection attends to ALL tracks to find the best match.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, det_emb, trk_emb):
        """
        det_emb: (N_det, embed_dim)
        trk_emb: (N_trk, embed_dim)
        Returns: (N_det, embed_dim) — attended detection features
        """
        # Add batch dimension for MultiheadAttention
        det_emb_b = det_emb.unsqueeze(0)   # (1, N_det, embed_dim)
        trk_emb_b = trk_emb.unsqueeze(0)   # (1, N_trk, embed_dim)

        attended, attn_weights = self.attention(
            query=det_emb_b,
            key=trk_emb_b,
            value=trk_emb_b
        )  # (1, N_det, embed_dim)

        # Residual connection + norm
        out = self.norm(det_emb_b + self.dropout(attended))
        return out.squeeze(0), attn_weights.squeeze(0)


class SimilarityHead(nn.Module):
    """
    Computes pairwise similarity score between each
    detection and each track. Output range: [0, 1]
    1 = same vehicle, 0 = different vehicle
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, det_emb, trk_emb):
        """
        det_emb: (N_det, embed_dim)
        trk_emb: (N_trk, embed_dim)
        Returns: similarity_matrix (N_det, N_trk)
        """
        N_det = det_emb.shape[0]
        N_trk = trk_emb.shape[0]

        # Expand for pairwise computation
        det_exp = det_emb.unsqueeze(1).expand(-1, N_trk, -1)  # (N_det, N_trk, embed_dim)
        trk_exp = trk_emb.unsqueeze(0).expand(N_det, -1, -1)  # (N_det, N_trk, embed_dim)

        # Concatenate and score
        pair_feats = torch.cat([det_exp, trk_exp], dim=-1)    # (N_det, N_trk, embed_dim*2)
        scores = self.scorer(pair_feats).squeeze(-1)           # (N_det, N_trk)
        return scores


class ReIDAttentionHead(nn.Module):
    """
    Full Re-ID Attention Head — assembled from all sub-modules.

    Forward pass:
      1. Project detection features → embed space
      2. Project track features → embed space
      3. Cross-attention: detections attend to tracks
      4. Compute pairwise similarity scores
      5. Return similarity matrix for Hungarian matching

    Args:
      feature_dim: backbone encoder output dim (default 256)
      embed_dim:   Re-ID embedding space dim   (default 128)
      num_heads:   cross-attention heads        (default 4)
    """
    def __init__(
        self,
        feature_dim=BACKBONE_CONFIG['feature_dim'],
        embed_dim=BACKBONE_CONFIG['embed_dim'],
        num_heads=BACKBONE_CONFIG['num_heads']
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads

        self.det_proj   = DetectionProjection(feature_dim, embed_dim)
        self.trk_proj   = TrackMemoryProjection(feature_dim, embed_dim)
        self.cross_attn = CrossAttentionReID(embed_dim, num_heads)
        self.sim_head   = SimilarityHead(embed_dim)

    def forward(self, detection_feats, track_feats):
        """
        detection_feats: (N_det, feature_dim) — current frame backbone features
        track_feats:     (N_trk, feature_dim) — stored track backbone features

        Returns:
          similarity_matrix: (N_det, N_trk) — used for Hungarian matching
          attn_weights:      (N_det, N_trk) — for visualization/debugging
        """
        # Step 1: Project to embedding space
        det_emb = self.det_proj(detection_feats)   # (N_det, embed_dim)
        trk_emb = self.trk_proj(track_feats)       # (N_trk, embed_dim)

        # Step 2: Normalize embeddings
        det_emb = F.normalize(det_emb, p=2, dim=-1)
        trk_emb = F.normalize(trk_emb, p=2, dim=-1)

        # Step 3: Cross-attention
        attended_det, attn_weights = self.cross_attn(det_emb, trk_emb)

        # Step 4: Similarity scoring
        similarity = self.sim_head(attended_det, trk_emb)

        return similarity, attn_weights

    def get_embedding(self, features, mode='detection'):
        """
        Get normalized embedding for a single set of features.
        Used during inference to store track embeddings.
        mode: 'detection' or 'track'
        """
        with torch.no_grad():
            if mode == 'detection':
                emb = self.det_proj(features)
            else:
                emb = self.trk_proj(features)
            return F.normalize(emb, p=2, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 50)
    print("Re-ID Attention Head — Unit Test")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ReIDAttentionHead().to(device)

    print(f"Trainable parameters: {model.count_parameters():,}")
    print(f"feature_dim : {model.feature_dim}")
    print(f"embed_dim   : {model.embed_dim}")
    print(f"num_heads   : {model.num_heads}")

    # Simulate: 5 detections, 3 existing tracks
    N_det, N_trk = 5, 3
    det_feats = torch.randn(N_det, 256).to(device)
    trk_feats = torch.randn(N_trk, 256).to(device)

    sim_matrix, attn_weights = model(det_feats, trk_feats)

    print(f"\nInput  — det_feats : {det_feats.shape}")
    print(f"Input  — trk_feats : {trk_feats.shape}")
    print(f"Output — similarity: {sim_matrix.shape}  (should be {N_det}x{N_trk})")
    print(f"Output — attn_weights: {attn_weights.shape}")
    print(f"\nSimilarity matrix:\n{sim_matrix.detach().cpu()}")
    print(f"\nAll values in [0,1]: {(sim_matrix >= 0).all() and (sim_matrix <= 1).all()}")
    print("\n[TEST PASSED] Re-ID Head working correctly.")
