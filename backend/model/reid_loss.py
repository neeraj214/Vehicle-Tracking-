import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Re-ID training.
    Same vehicle (label=1): minimize distance
    Different vehicle (label=0): maximize distance beyond margin
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, labels):
        """
        embeddings1: (N, embed_dim)
        embeddings2: (N, embed_dim)
        labels: (N,) — 1 if same track, 0 if different
        """
        dist = F.pairwise_distance(embeddings1, embeddings2)
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        loss = (pos_loss + neg_loss).mean()
        return loss


class FocalBCELoss(nn.Module):
    """
    Focal BCE loss for similarity matrix training.
    Reduces weight of easy negatives — focuses on hard examples.
    gamma=2 is standard, alpha handles class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        predictions: (N_det, N_trk) similarity scores in [0,1]
        targets:     (N_det, N_trk) binary labels
        """
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = (focal_weight * bce).mean()
        return loss


class ReIDLoss(nn.Module):
    """
    Combined loss for Re-ID head training.
    total_loss = w1 * focal_bce + w2 * contrastive
    """
    def __init__(self, w_focal=1.0, w_contrastive=0.5, margin=1.0):
        super().__init__()
        self.w_focal       = w_focal
        self.w_contrastive = w_contrastive
        self.focal_loss    = FocalBCELoss()
        self.contrastive   = ContrastiveLoss(margin=margin)

    def forward(self, sim_matrix, det_emb, trk_emb, pair_labels):
        """
        sim_matrix:  (N_det, N_trk) predicted similarity
        det_emb:     (N, embed_dim) detection embeddings
        trk_emb:     (N, embed_dim) track embeddings (matched pairs)
        pair_labels: (N,) 1=same track, 0=different (for contrastive)
        """
        # Focal BCE on full similarity matrix
        label_matrix = self._build_label_matrix(
            pair_labels, sim_matrix.shape
        )
        focal = self.focal_loss(sim_matrix, label_matrix)

        # Contrastive on paired embeddings
        contrastive = self.contrastive(det_emb, trk_emb, pair_labels)

        total = self.w_focal * focal + self.w_contrastive * contrastive
        return total, focal.item(), contrastive.item()

    def _build_label_matrix(self, pair_labels, shape):
        """Build (N_det, N_trk) binary label matrix from pair labels"""
        N_det, N_trk = shape
        matrix = torch.zeros(N_det, N_trk, device=pair_labels.device)
        for i in range(min(N_det, len(pair_labels))):
            if pair_labels[i] == 1 and i < N_trk:
                matrix[i, i] = 1.0
        return matrix


if __name__ == "__main__":
    print("Re-ID Loss — Unit Test")

    # Simulate
    N_det, N_trk, embed_dim = 5, 3, 128
    sim_matrix  = torch.rand(N_det, N_trk)
    det_emb     = F.normalize(torch.randn(N_det, embed_dim), dim=-1)
    trk_emb     = F.normalize(torch.randn(N_det, embed_dim), dim=-1)
    pair_labels = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float)

    criterion = ReIDLoss()
    total, focal, contrastive = criterion(sim_matrix, det_emb, trk_emb, pair_labels)

    print(f"Total loss:       {total:.4f}")
    print(f"Focal BCE loss:   {focal:.4f}")
    print(f"Contrastive loss: {contrastive:.4f}")
    print("[TEST PASSED] Loss functions working correctly.")
