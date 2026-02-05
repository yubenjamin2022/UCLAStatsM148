#!/usr/bin/env python3
"""
Time-aware RNN (GRU) to predict probability of success (purchase completion) from user journeys.

Expected input (from your notebook):
  - data/cleaned_dat_train1.csv with columns:
      id, ed_id, event_name, event_timestamp

What this does:
  1) Reads cleaned event log
  2) Builds journeys by grouping by id and sorting by timestamp
  3) Defines SUCCESS as presence of event_name == "order_shipped"
  4) Trains a GRU that consumes [action_embedding ; log1p(delta_t_seconds)] per step
  5) Trains on random prefixes (leakage-safe: never include the success event itself)
  6) Evaluates with logloss + (optional) ROC-AUC / PR-AUC if scikit-learn is available

Run:
  python train_purchase_rnn.py
"""

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Optional metrics (recommended; script still runs without sklearn)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


# ============================================================
# CONFIG (edit these directly)
# ============================================================

CSV_PATH = "data/cleaned_dat_train1.csv"
OUT_PATH = "purchase_rnn.pt"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Success definition
SUCCESS_EVENT_NAME = "order_shipped"   # change if your success event differs

# Split
VAL_FRAC = 0.10

# Sequence handling
MAX_SEQ_LEN = 200                      # keep most recent MAX_SEQ_LEN events in a prefix
CAP_DELTA_SECONDS = 86400.0            # cap time gaps before log1p (default: 1 day)
# If you want no cap, set CAP_DELTA_SECONDS = None

# Model
EMB_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.10

# Training
EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# ============================================================


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Vocab (ed_id -> compact ids)
# -------------------------
@dataclass
class Vocab:
    pad_id: int
    bos_id: int
    unk_id: int
    token_to_id: Dict[int, int]
    id_to_token: Dict[int, int]

    @property
    def size(self) -> int:
        return len(self.id_to_token)


def build_vocab(all_sequences: List[List[int]]) -> Vocab:
    PAD, BOS, UNK = 0, 1, 2
    unique = sorted({t for seq in all_sequences for t in seq})
    token_to_id = {}
    id_to_token = {PAD: -1, BOS: -2, UNK: -3}
    nxt = 3
    for tok in unique:
        token_to_id[tok] = nxt
        id_to_token[nxt] = tok
        nxt += 1
    return Vocab(PAD, BOS, UNK, token_to_id, id_to_token)


def encode_actions(ed_ids: List[int], vocab: Vocab) -> List[int]:
    # Include BOS to help the model at the first step
    out = [vocab.bos_id]
    out.extend(vocab.token_to_id.get(t, vocab.unk_id) for t in ed_ids)
    return out


# -------------------------
# Data: read + build journeys
# -------------------------
def read_cleaned_csv(csv_path: str) -> pd.DataFrame:
    usecols = ["id", "ed_id", "event_name", "event_timestamp"]
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["event_timestamp"], low_memory=False)

    # Basic sanitation
    df["id"] = df["id"].astype(str)
    df["ed_id"] = pd.to_numeric(df["ed_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ed_id", "event_timestamp"])
    df["ed_id"] = df["ed_id"].astype(int)

    df = df.sort_values(["id", "event_timestamp"])
    return df


def build_journeys(df: pd.DataFrame, success_event_name: str) -> pd.DataFrame:
    """
    Returns DataFrame with:
      - id
      - events: List[int] (ed_id)
      - event_names: List[str]
      - timestamps: List[pd.Timestamp]
      - success: bool (contains success_event_name)
      - first_success_idx: int (index in events list; -1 if none)
    """
    g = df.groupby("id", sort=False)

    journeys = pd.DataFrame({
        "id": g["id"].first(),
        "events": g["ed_id"].apply(list),
        "event_names": g["event_name"].apply(list),
        "timestamps": g["event_timestamp"].apply(list),
    }).reset_index(drop=True)

    def compute_success_and_first_idx(names: List[str]) -> Tuple[bool, int]:
        try:
            idx = names.index(success_event_name)
            return True, idx
        except ValueError:
            return False, -1

    tmp = journeys["event_names"].apply(compute_success_and_first_idx)
    journeys["success"] = tmp.apply(lambda x: x[0])
    journeys["first_success_idx"] = tmp.apply(lambda x: x[1])

    journeys["journey_length"] = journeys["events"].apply(len)
    journeys = journeys[journeys["journey_length"] > 0].reset_index(drop=True)
    return journeys


# -------------------------
# Time features
# -------------------------
def compute_log_deltas_seconds(timestamps: List[pd.Timestamp], cap_seconds: Optional[float]) -> np.ndarray:
    """
    Returns shape [T] float array aligned to events (NOT including BOS).
    delta[0] = 0.0
    delta[i] = log1p(seconds between event i and i-1), optionally capped before log1p.
    """
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if ts.isna().any():
        ts = ts.ffill().bfill()

    diffs = ts.diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=np.float32)
    diffs = np.maximum(diffs, 0.0)
    if cap_seconds is not None:
        diffs = np.minimum(diffs, float(cap_seconds))
    return np.log1p(diffs).astype(np.float32)


# -------------------------
# Dataset: random prefix sampling (leakage-safe)
# -------------------------
class PrefixPurchaseDataset(Dataset):
    """
    Each item returns one training example built from a random prefix of a journey.

    For success journeys: prefix end <= first_success_idx (so we never include the success event itself).
    For failure journeys: prefix end <= journey_length.

    Output:
      actions: [L] long (includes BOS at position 0)
      deltas:  [L] float (aligned to actions; deltas[0]=0 for BOS)
      label:   float (1 if success eventually, else 0)
    """
    def __init__(
        self,
        journeys: pd.DataFrame,
        vocab: Vocab,
        max_seq_len: int = 200,
        cap_delta_seconds: Optional[float] = 86400.0,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.cap_delta_seconds = cap_delta_seconds

        self.events = journeys["events"].tolist()
        self.timestamps = journeys["timestamps"].tolist()
        self.success = journeys["success"].astype(bool).to_numpy()
        self.first_success_idx = journeys["first_success_idx"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        ev = self.events[idx]
        ts = self.timestamps[idx]
        is_pos = bool(self.success[idx])
        first_pos = int(self.first_success_idx[idx])

        T = len(ev)

        # For positives, do not include the success event itself (avoid trivial leakage)
        max_end = first_pos if is_pos else T
        if max_end <= 0:
            max_end = min(1, T)

        # Sample prefix length in [1, max_end] (on the event sequence)
        end = random.randint(1, max_end) if T > 0 else 0

        ev_pref = ev[:end]
        ts_pref = ts[:end]

        # Truncate to keep recent history
        if len(ev_pref) > self.max_seq_len:
            ev_pref = ev_pref[-self.max_seq_len:]
            ts_pref = ts_pref[-self.max_seq_len:]

        actions = encode_actions(ev_pref, self.vocab)

        deltas_events = compute_log_deltas_seconds(ts_pref, self.cap_delta_seconds)  # len == len(ev_pref)
        deltas = np.concatenate([np.array([0.0], dtype=np.float32), deltas_events], axis=0)

        label = 1.0 if is_pos else 0.0
        return (
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(deltas, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_pad(batch, pad_id: int):
    actions, deltas, labels = zip(*batch)
    lengths = torch.tensor([len(a) for a in actions], dtype=torch.long)
    max_len = int(lengths.max().item())

    A = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    D = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, (a, d) in enumerate(zip(actions, deltas)):
        A[i, : len(a)] = a
        D[i, : len(d)] = d

    Y = torch.stack(labels, dim=0)
    return A, D, lengths, Y


# -------------------------
# Model
# -------------------------
class GRUTimePurchase(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.gru = nn.GRU(
            input_size=emb_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actions: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(actions)  # [B,T,E]
        x = torch.cat([emb, deltas.unsqueeze(-1)], dim=-1)  # [B,T,E+1]

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)    # h: [L,B,H]
        h_last = h[-1]             # [B,H]
        logits = self.head(h_last).squeeze(-1)  # [B]
        return logits


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    bce_sum = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    n = 0
    probs_all = []
    labels_all = []

    for actions, deltas, lengths, y in loader:
        actions = actions.to(device)
        deltas = deltas.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(actions, deltas, lengths)
        loss = bce_sum(logits, y)

        total_loss += loss.item()
        n += y.numel()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.append(probs)
        labels_all.append(y.detach().cpu().numpy())

    probs = np.concatenate(probs_all) if probs_all else np.array([])
    labels = np.concatenate(labels_all) if labels_all else np.array([])

    out = {"logloss": float(total_loss / max(n, 1))}

    if roc_auc_score is not None and len(np.unique(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["pr_auc"] = float(average_precision_score(labels, probs))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


# -------------------------
# Train
# -------------------------
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find CSV at: {CSV_PATH}")

    df = read_cleaned_csv(CSV_PATH)
    journeys = build_journeys(df, SUCCESS_EVENT_NAME)

    print(f"Journeys: {len(journeys)}")
    print(f"Journey-level success rate: {journeys['success'].mean():.4f}")

    # Split by journey (id-level split)
    n = len(journeys)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - VAL_FRAC))
    train_idx, val_idx = idx[:split], idx[split:]
    train_j = journeys.iloc[train_idx].reset_index(drop=True)
    val_j = journeys.iloc[val_idx].reset_index(drop=True)

    # Vocab from train only (avoid minor leakage)
    vocab = build_vocab(train_j["events"].tolist())
    print(f"Vocab size (incl PAD/BOS/UNK): {vocab.size}")

    train_ds = PrefixPurchaseDataset(
        train_j, vocab,
        max_seq_len=MAX_SEQ_LEN,
        cap_delta_seconds=CAP_DELTA_SECONDS,
    )
    val_ds = PrefixPurchaseDataset(
        val_j, vocab,
        max_seq_len=MAX_SEQ_LEN,
        cap_delta_seconds=CAP_DELTA_SECONDS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_pad(b, vocab.pad_id),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_pad(b, vocab.pad_id),
        pin_memory=True,
    )

    model = GRUTimePurchase(
        vocab_size=vocab.size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_id=vocab.pad_id,
    ).to(DEVICE)

    # Class imbalance handling via pos_weight (computed from journey-level labels)
    pos = float(train_j["success"].sum())
    neg = float(len(train_j) - pos)
    pos_weight = (neg / pos) if pos > 0 else 1.0
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
    print(f"Using pos_weight = {pos_weight:.4f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for actions, deltas, lengths, y in train_loader:
            actions = actions.to(DEVICE)
            deltas = deltas.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(actions, deltas, lengths)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item() * y.numel()
            total_n += y.numel()

        train_logloss = total_loss / max(total_n, 1)
        val_metrics = eval_epoch(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:02d} | "
            f"train_logloss={train_logloss:.4f} | "
            f"val_logloss={val_metrics['logloss']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']} | "
            f"val_pr_auc={val_metrics['pr_auc']}"
        )

        # Save best by validation logloss (good for probabilities/calibration)
        if val_metrics["logloss"] < best_val:
            best_val = val_metrics["logloss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab.__dict__,
                    "config": {
                        "emb_dim": EMB_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "max_seq_len": MAX_SEQ_LEN,
                        "cap_delta_seconds": CAP_DELTA_SECONDS,
                        "success_event_name": SUCCESS_EVENT_NAME,
                    },
                    "notes": (
                        "Prefix training; leakage-safe (never includes success event). "
                        "Time feature is log1p(delta_seconds) with optional cap."
                    ),
                },
                OUT_PATH,
            )
            print(f"  ✓ Saved best checkpoint to {OUT_PATH}")

    print("Done.")


# -------------------------
# Optional: simple inference helper
# -------------------------
@torch.no_grad()
def predict_success_probability(
    ckpt_path: str,
    prefix_ed_ids: List[int],
    prefix_timestamps: List[pd.Timestamp],
) -> float:
    """
    Load checkpoint and score a single prefix.
    prefix_ed_ids and prefix_timestamps should be aligned lists (same length).
    Returns P(success).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = Vocab(**ckpt["vocab"])
    cfg = ckpt["config"]

    model = GRUTimePurchase(
        vocab_size=vocab.size,
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_id=vocab.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Truncate to match training behavior (keep recent history)
    if len(prefix_ed_ids) > cfg["max_seq_len"]:
        prefix_ed_ids = prefix_ed_ids[-cfg["max_seq_len"]:]
        prefix_timestamps = prefix_timestamps[-cfg["max_seq_len"]:]

    actions = encode_actions(prefix_ed_ids, vocab)
    deltas_events = compute_log_deltas_seconds(prefix_timestamps, cfg["cap_delta_seconds"])
    deltas = np.concatenate([np.array([0.0], dtype=np.float32), deltas_events], axis=0)

    A = torch.tensor(actions, dtype=torch.long).unsqueeze(0)      # [1,T]
    D = torch.tensor(deltas, dtype=torch.float32).unsqueeze(0)    # [1,T]
    lengths = torch.tensor([A.size(1)], dtype=torch.long)         # [1]

    logits = model(A, D, lengths)
    prob = torch.sigmoid(logits).item()
    return float(prob)


if __name__ == "__main__":
    main()
