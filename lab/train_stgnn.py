"""QuakeWatch — Spatio-Temporal Graph Neural Network Training

Architecture:
  1. Per-station temporal encoder (1D conv → GRU) processes 6h of 5-min features
  2. Graph Attention Network (GAT) propagates signals between nearby stations
  3. Global features (solar wind, Kp, Dst, etc.) fused after graph convolution
  4. Per-zone prediction heads output M5+ probability within 1 hour

Target: binary classification — will an M5.0+ event occur in zone Z
within the next 60 minutes?
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

torch.set_float32_matmul_precision("medium")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.stgnn_data import (
    build_dataset, create_sequences, temporal_split,
    NUM_STATIONS, NUM_NODE_FEATURES, NUM_GLOBAL_FEATURES, NUM_ZONES,
    LOOKBACK_STEPS, HORIZON_STEPS, ZONES,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeismicGraphDataset(Dataset):
    def __init__(self, X_node, X_global, Y, edge_index, edge_weight):
        self.X_node = torch.from_numpy(X_node)        # (S, T, N, F)
        self.X_global = torch.from_numpy(X_global)     # (S, T, F_g)
        self.Y = torch.from_numpy(Y)                   # (S, Z)
        self.edge_index = torch.from_numpy(edge_index)  # (2, E)
        self.edge_weight = torch.from_numpy(edge_weight) # (E,)

    def __len__(self):
        return len(self.X_node)

    def __getitem__(self, idx):
        return self.X_node[idx], self.X_global[idx], self.Y[idx]


class TemporalEncoder(nn.Module):
    """Per-station temporal feature extractor: 1D conv → GRU."""
    def __init__(self, in_features, hidden_dim=64, num_layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, F) for a single station
        B, T, F = x.shape
        # Conv expects (B, F, T)
        h = F.gelu(self.conv1(x.transpose(1, 2)))
        h = F.gelu(self.conv2(h))
        # Back to (B, T, H)
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        return self.norm(h[:, -1, :])  # (B, H) — last hidden state


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention with edge weights."""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        # x: (B, N, F), edge_index: (2, E), edge_weight: (E,)
        B, N, _ = x.shape
        h = self.W(x)  # (B, N, out_dim)

        src, dst = edge_index[0], edge_index[1]
        # Attention scores
        e_src = self.a_src(h[:, src])  # (B, E, 1)
        e_dst = self.a_dst(h[:, dst])  # (B, E, 1)
        e = self.leaky(e_src + e_dst).squeeze(-1)  # (B, E)

        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(0)  # scale by distance weight

        # Softmax per destination node
        alpha = torch.full((B, N, N), float('-inf'), device=x.device)
        alpha[:, dst, src] = e
        alpha = torch.softmax(alpha, dim=-1)
        alpha = self.dropout(alpha)

        out = torch.bmm(alpha, h)  # (B, N, out_dim)
        return out


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention Network."""
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, head_dim, dropout) for _ in range(num_heads)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        head_outs = [head(x, edge_index, edge_weight) for head in self.heads]
        out = torch.cat(head_outs, dim=-1)  # (B, N, hidden_dim)
        return self.norm(self.dropout(out) + x if out.shape == x.shape else self.norm(out))


class STGNN(nn.Module):
    """Spatio-Temporal Graph Neural Network for earthquake prediction."""
    def __init__(self, hidden_dim=64, num_gat_heads=4, num_gat_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-station temporal encoder
        self.temporal_enc = TemporalEncoder(NUM_NODE_FEATURES, hidden_dim)

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            MultiHeadGAT(hidden_dim, hidden_dim, num_gat_heads, dropout)
            for _ in range(num_gat_layers)
        ])

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(NUM_GLOBAL_FEATURES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Fusion: station embeddings + global → zone predictions
        # Pool station embeddings per zone using attention
        self.zone_query = nn.Parameter(torch.randn(NUM_ZONES, hidden_dim))
        self.zone_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Final classifier per zone
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_node, x_global, edge_index, edge_weight=None):
        """
        x_node:   (B, T, N, F_node)
        x_global: (B, T, F_global)
        edge_index: (2, E)
        edge_weight: (E,)
        """
        B, T, N, F = x_node.shape

        # 1. Temporal encoding per station
        # Reshape to (B*N, T, F) for temporal encoder
        x_flat = x_node.permute(0, 2, 1, 3).reshape(B * N, T, F)
        station_emb = self.temporal_enc(x_flat)  # (B*N, H)
        station_emb = station_emb.view(B, N, self.hidden_dim)  # (B, N, H)

        # 2. Graph attention — propagate between stations
        for gat in self.gat_layers:
            station_emb = station_emb + gat(station_emb, edge_index, edge_weight)

        # 3. Global feature encoding (use last timestep)
        global_emb = self.global_proj(x_global[:, -1, :])  # (B, H)

        # 4. Zone-level aggregation via cross-attention
        # Query: zone embeddings, Key/Value: station embeddings
        zone_q = self.zone_query.unsqueeze(0).expand(B, -1, -1)  # (B, Z, H)
        zone_emb, _ = self.zone_attn(zone_q, station_emb, station_emb)  # (B, Z, H)

        # 5. Fuse with global and classify
        global_exp = global_emb.unsqueeze(1).expand(-1, NUM_ZONES, -1)  # (B, Z, H)
        fused = torch.cat([zone_emb, global_exp], dim=-1)  # (B, Z, 2H)
        logits = self.classifier(fused).squeeze(-1)  # (B, Z)

        return logits


class FocalLoss(nn.Module):
    """Focal loss for handling severe class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def train_epoch(model, loader, optimizer, criterion, edge_index, edge_weight):
    model.train()
    total_loss = 0
    for x_node, x_global, y in loader:
        x_node = x_node.to(DEVICE)
        x_global = x_global.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_node, x_global, edge_index, edge_weight)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x_node.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, edge_index, edge_weight):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for x_node, x_global, y in loader:
        x_node = x_node.to(DEVICE)
        x_global = x_global.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x_node, x_global, edge_index, edge_weight)
        loss = criterion(logits, y)
        total_loss += loss.item() * x_node.size(0)

        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    metrics = {"loss": avg_loss}

    # Global (any zone) metrics
    any_pred = preds.max(axis=1)
    any_target = targets.max(axis=1)
    if any_target.sum() > 0 and any_target.sum() < len(any_target):
        metrics["auc"] = roc_auc_score(any_target, any_pred)
        binary_pred = (any_pred > 0.5).astype(int)
        metrics["precision"] = precision_score(any_target, binary_pred, zero_division=0)
        metrics["recall"] = recall_score(any_target, binary_pred, zero_division=0)
        metrics["f1"] = f1_score(any_target, binary_pred, zero_division=0)
    else:
        metrics["auc"] = 0.0

    # Per-zone AUC
    zone_aucs = {}
    for zi in range(NUM_ZONES):
        if targets[:, zi].sum() > 0 and targets[:, zi].sum() < len(targets):
            zone_aucs[ZONES[zi]["id"]] = roc_auc_score(targets[:, zi], preds[:, zi])
    metrics["zone_aucs"] = zone_aucs

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train ST-GNN earthquake forecaster")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--skip-data-build", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  SPATIO-TEMPORAL GNN — QUAKEWATCH")
    print("=" * 60)

    npz_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "stgnn_dataset.npz")

    if args.skip_data_build and os.path.exists(npz_path):
        print(f"\n  Loading cached dataset from {npz_path}")
        cached = np.load(npz_path, allow_pickle=True)
        dataset = {k: cached[k] for k in cached.files}
    else:
        print(f"\n  Building dataset...")
        dataset = build_dataset(save_path=npz_path)

    print(f"\n  Creating sequences...")
    X_node, X_global, Y = create_sequences(dataset)
    print(f"  Total sequences: {X_node.shape[0]:,}")
    print(f"  Positive rate: {(Y.max(axis=1) > 0).mean():.3f}")

    # Temporal split
    splits = temporal_split(X_node.shape[0])
    print(f"  Train: {splits['train'][0]}–{splits['train'][1]}")
    print(f"  Val:   {splits['val'][0]}–{splits['val'][1]}")
    print(f"  Test:  {splits['test'][0]}–{splits['test'][1]}")

    edge_index = dataset["edge_index"]
    edge_weight = dataset["edge_weight"]

    # Create datasets
    def make_ds(split_key):
        s, e = splits[split_key]
        return SeismicGraphDataset(
            X_node[s:e], X_global[s:e], Y[s:e], edge_index, edge_weight
        )

    train_ds = make_ds("train")
    val_ds = make_ds("val")
    test_ds = make_ds("test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    # Move graph structure to device
    ei = torch.from_numpy(edge_index).to(DEVICE)
    ew = torch.from_numpy(edge_weight).to(DEVICE)

    # Model
    model = STGNN(
        hidden_dim=args.hidden,
        num_gat_heads=args.gat_heads,
        num_gat_layers=args.gat_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"  Hidden: {args.hidden}, GAT heads: {args.gat_heads}, GAT layers: {args.gat_layers}")
    print(f"  Device: {DEVICE}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    best_val_auc = 0
    patience_counter = 0
    best_path = os.path.join(MODEL_DIR, "stgnn_best.pt")

    print(f"\n  Training for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"  {'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Val AUC':>8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'LR':>10}")
    print("  " + "-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, ei, ew)
        val_metrics = evaluate(model, val_loader, criterion, ei, ew)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"  {epoch:>5d} {train_loss:>11.5f} {val_metrics['loss']:>9.5f} "
              f"{val_metrics.get('auc', 0):>8.4f} "
              f"{val_metrics.get('precision', 0):>6.3f} "
              f"{val_metrics.get('recall', 0):>7.3f} "
              f"{val_metrics.get('f1', 0):>6.3f} "
              f"{lr:>10.6f}  ({elapsed:.1f}s)")

        if val_metrics.get("auc", 0) > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, best_path)
            print(f"         ↑ new best (AUC={best_val_auc:.4f}), saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Evaluate on test set
    print(f"\n  Loading best model (AUC={best_val_auc:.4f})...")
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, criterion, ei, ew)

    print(f"\n  ── ST-GNN Test Results ──")
    print(f"  AUC:       {test_metrics.get('auc', 0):.4f}")
    print(f"  Precision: {test_metrics.get('precision', 0):.3f}")
    print(f"  Recall:    {test_metrics.get('recall', 0):.3f}")
    print(f"  F1:        {test_metrics.get('f1', 0):.3f}")

    if test_metrics.get("zone_aucs"):
        print(f"\n  Per-zone AUC:")
        for zone_id, auc in sorted(test_metrics["zone_aucs"].items(), key=lambda x: -x[1]):
            print(f"    {zone_id:20s} {auc:.4f}")

    # Save results
    results = {
        "test_auc": test_metrics.get("auc", 0),
        "test_precision": test_metrics.get("precision", 0),
        "test_recall": test_metrics.get("recall", 0),
        "test_f1": test_metrics.get("f1", 0),
        "best_val_auc": best_val_auc,
        "best_epoch": ckpt["epoch"],
        "zone_aucs": test_metrics.get("zone_aucs", {}),
        "n_params": n_params,
        "args": vars(args),
    }

    results_path = os.path.join(MODEL_DIR, "stgnn_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    print(f"  Model saved: {best_path}")


if __name__ == "__main__":
    main()
