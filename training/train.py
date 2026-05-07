import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'model'))
from reid_head   import ReIDAttentionHead
from reid_loss   import ReIDLoss
from reid_utils  import pair_generator, save_reid_weights
from deformable_detr import DeformableDETRBackbone
from backbone_config import BACKBONE_CONFIG
from dataset import UATRACDataset


def collate_fn(batch):
    imgs, boxes, track_ids = zip(*batch)
    return list(imgs), list(boxes), list(track_ids)


def extract_batch_features(backbone, imgs, device):
    """Extract mean-pooled encoder features for a batch of images"""
    all_feats = []
    for img in imgs:
        enc = backbone.extract_features(img)        # (1, tokens, 256)
        feat = enc.mean(dim=1).squeeze(0)           # (256,)
        all_feats.append(feat)
    return torch.stack(all_feats).to(device)        # (B, 256)


def train_one_epoch(backbone, reid_head, optimizer,
                    criterion, loader, device, epoch):
    reid_head.train()
    total_loss = 0.0
    total_focal = 0.0
    total_cont  = 0.0
    skipped = 0

    for batch_idx, (imgs, boxes, track_ids) in enumerate(loader):
        # Flatten track_ids per batch item
        flat_ids = []
        for ids in track_ids:
            flat_ids.extend(ids)

        # Extract features for all images in batch
        feats = extract_batch_features(backbone, imgs, device)

        # Generate positive / negative pairs
        f1, f2, labels = pair_generator(feats, flat_ids)
        if f1 is None:
            skipped += 1
            continue

        f1      = f1.to(device)
        f2      = f2.to(device)
        labels  = labels.to(device)

        # Build small sim matrix for loss
        sim_matrix, _ = reid_head(f1, f2)

        # Get embeddings for contrastive loss
        det_emb = reid_head.det_proj(f1)
        trk_emb = reid_head.trk_proj(f2)

        loss, focal, cont = criterion(
            sim_matrix, det_emb, trk_emb, labels
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reid_head.parameters(), 1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_focal += focal
        total_cont  += cont

        if batch_idx % 20 == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss:{loss.item():.4f} "
                f"Focal:{focal:.4f} Cont:{cont:.4f}"
            )

    n = max(len(loader) - skipped, 1)
    return total_loss/n, total_focal/n, total_cont/n


def validate(backbone, reid_head, criterion, loader, device):
    reid_head.eval()
    total_loss = 0.0
    skipped = 0

    with torch.no_grad():
        for imgs, boxes, track_ids in loader:
            flat_ids = []
            for ids in track_ids:
                flat_ids.extend(ids)

            feats = extract_batch_features(backbone, imgs, device)
            f1, f2, labels = pair_generator(feats, flat_ids)
            if f1 is None:
                skipped += 1
                continue

            f1     = f1.to(device)
            f2     = f2.to(device)
            labels = labels.to(device)

            sim_matrix, _ = reid_head(f1, f2)
            det_emb = reid_head.det_proj(f1)
            trk_emb = reid_head.trk_proj(f2)

            loss, _, _ = criterion(sim_matrix, det_emb, trk_emb, labels)
            total_loss += loss.item()

    n = max(len(loader) - skipped, 1)
    return total_loss / n


def train(
    img_dir=None,
    ann_dir=None,
    epochs=None,
    batch_size=None,
    lr=None,
    save_path=None,
    val_split=0.1
):
    # Load config defaults
    img_dir    = img_dir    or BACKBONE_CONFIG['ua_detrac_img_dir']
    ann_dir    = ann_dir    or BACKBONE_CONFIG['ua_detrac_ann_dir']
    epochs     = epochs     or BACKBONE_CONFIG['reid_epochs']
    batch_size = batch_size or BACKBONE_CONFIG['batch_size']
    lr         = lr         or BACKBONE_CONFIG['reid_lr']
    save_path  = save_path  or BACKBONE_CONFIG['reid_weights_path']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Train] Device: {device}")
    print(f"[Train] Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    # Dataset
    if not os.path.exists(img_dir):
        print(f"[Error] UA-DETRAC images not found at: {img_dir}")
        print("        Download from https://detrac-db.rit.albany.edu/")
        return

    dataset  = UATRACDataset(img_dir, ann_dir)
    n_val    = int(len(dataset) * val_split)
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    print(f"[Train] {n_train} train | {n_val} val samples")

    # Models
    backbone  = DeformableDETRBackbone(device)
    reid_head = ReIDAttentionHead().to(device)
    criterion = ReIDLoss()
    optimizer = torch.optim.AdamW(
        reid_head.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*55}")
        print(f"EPOCH {epoch}/{epochs}")
        print(f"{'='*55}")

        train_loss, focal, cont = train_one_epoch(
            backbone, reid_head, optimizer,
            criterion, train_loader, device, epoch
        )
        val_loss = validate(
            backbone, reid_head, criterion, val_loader, device
        )
        scheduler.step()

        print(f"\n  Train Loss : {train_loss:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  LR         : {scheduler.get_last_lr()[0]:.6f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_reid_weights(reid_head, save_path)
            print(f"  [Saved] Best model → {save_path}")

    print(f"\n[Done] Training complete. Best val loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",  default=None)
    parser.add_argument("--ann_dir",  default=None)
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--batch",    type=int, default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--save",     default=None)
    args = parser.parse_args()

    train(
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        save_path=args.save
    )
