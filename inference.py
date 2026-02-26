"""
RNN Inference Script for Order Shipped Prediction
Loads a trained BiLSTM model and generates predictions on a test dataset.

Usage:
    python rnn_inference.py --weights_path <path_to_weights.pt> --csv_path <path_to_test.csv> --output_path <output.csv>
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Vocab:
    """Vocabulary for encoding event IDs"""
    pad_id: int
    bos_id: int
    unk_id: int
    token_to_id: Dict[int, int]
    id_to_token: Dict[int, int]

    @property
    def size(self) -> int:
        return len(self.id_to_token)


class BiLSTMTimePurchase(nn.Module):
    """Bidirectional LSTM model with time features for purchase prediction"""
    
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

        self.lstm = nn.LSTM(
            input_size=emb_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional => concat forward + backward states => 2*hidden_dim
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actions: torch.Tensor, deltas: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(actions)  # [B,T,E]
        x = torch.cat([emb, deltas.unsqueeze(-1)], dim=-1)  # [B,T,E+1]

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h, c) = self.lstm(packed)  # h: [2*num_layers, B, H]

        # Last layer: forward is at index 2*(num_layers-1), backward at 2*(num_layers-1)+1
        fwd = h[2 * (self.lstm.num_layers - 1)]       # [B,H]
        bwd = h[2 * (self.lstm.num_layers - 1) + 1]   # [B,H]
        h_last = torch.cat([fwd, bwd], dim=-1)        # [B,2H]

        logits = self.head(h_last).squeeze(-1)  # [B]
        return logits

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

def encode_actions(ed_ids: List[int], vocab: Vocab) -> List[int]:
    """Encode a sequence of event IDs using the vocabulary"""
    out = [vocab.bos_id]
    out.extend(vocab.token_to_id.get(t, vocab.unk_id) for t in ed_ids)
    return out


def compute_log_deltas_seconds(timestamps: List[pd.Timestamp], cap_seconds: Optional[float]) -> np.ndarray:
    """
    Compute log1p of time deltas between consecutive events.
    
    Returns:
        Array of shape [T] where delta[0] = 0.0 and delta[i] = log1p(seconds between event i and i-1)
    """
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if ts.isna().any():
        ts = ts.ffill().bfill()

    diffs = ts.diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=np.float32)
    diffs = np.maximum(diffs, 0.0)
    if cap_seconds is not None:
        diffs = np.minimum(diffs, float(cap_seconds))
    return np.log1p(diffs).astype(np.float32)


def read_cleaned_csv(csv_path: str) -> pd.DataFrame:
    """Read and preprocess the test CSV file"""
    usecols = ["id", "ed_id", "event_name", "event_timestamp"]
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["event_timestamp"], low_memory=False)

    # Basic sanitation
    df["id"] = df["id"].astype(str)
    df["ed_id"] = pd.to_numeric(df["ed_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ed_id", "event_timestamp"])
    df["ed_id"] = df["ed_id"].astype(int)

    df = df.sort_values(["id", "event_timestamp"])
    return df


def build_journeys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user journeys from event data.
    
    Returns:
        DataFrame with columns: id, events, timestamps, journey_length
    """
    g = df.groupby("id", sort=False)

    journeys = pd.DataFrame({
        "id": g["id"].first(),
        "events": g["ed_id"].apply(list),
        "timestamps": g["event_timestamp"].apply(list),
    }).reset_index(drop=True)

    journeys["journey_length"] = journeys["events"].apply(len)
    journeys = journeys[journeys["journey_length"] > 0].reset_index(drop=True)
    return journeys


class InferenceDataset(Dataset):
    """
    Dataset for inference - uses full sequences (no random prefixing).
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

        self.ids = journeys["id"].tolist()
        self.events = journeys["events"].tolist()
        self.timestamps = journeys["timestamps"].tolist()

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        ev = self.events[idx]
        ts = self.timestamps[idx]
        user_id = self.ids[idx]

        # Use full sequence (truncate if too long)
        if len(ev) > self.max_seq_len:
            ev = ev[-self.max_seq_len:]
            ts = ts[-self.max_seq_len:]

        # Encode actions
        actions = encode_actions(ev, self.vocab)

        # Compute time deltas
        deltas_events = compute_log_deltas_seconds(ts, self.cap_delta_seconds)
        deltas = np.concatenate([np.array([0.0], dtype=np.float32), deltas_events], axis=0)

        return (
            user_id,
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(deltas, dtype=torch.float32),
        )


def collate_pad_inference(batch, pad_id: int):
    """Collate function for inference batches"""
    ids, actions, deltas = zip(*batch)
    lengths = torch.tensor([len(a) for a in actions], dtype=torch.long)
    max_len = int(lengths.max().item())

    A = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    D = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, (a, d) in enumerate(zip(actions, deltas)):
        A[i, : len(a)] = a
        D[i, : len(d)] = d

    return ids, A, D, lengths


def load_model_and_vocab(weights_path: str, device: str = "cpu"):
    """
    Load trained model weights and vocabulary.
    
    Returns:
        model, vocab, config
    """
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Reconstruct vocab
    vocab_dict = checkpoint["vocab"]
    vocab = Vocab(
        pad_id=vocab_dict["pad_id"],
        bos_id=vocab_dict["bos_id"],
        unk_id=vocab_dict["unk_id"],
        token_to_id=vocab_dict["token_to_id"],
        id_to_token={int(k): v for k, v in vocab_dict["id_to_token"].items()},
    )
    
    # Get config
    config = checkpoint["config"]
    
    # Build model
    model = BiLSTMTimePurchase(
        vocab_size=vocab.size,
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_id=vocab.pad_id,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    return model, vocab, config


@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run inference on a data loader.
    
    Returns:
        List of (id, probability) tuples
    """
    model.eval()
    results = []
    
    for ids, actions, deltas, lengths in loader:
        actions = actions.to(device)
        deltas = deltas.to(device)
        lengths = lengths.to(device)
        
        logits = model(actions, deltas, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        for user_id, prob in zip(ids, probs):
            results.append((user_id, float(prob)))
    
    return results


def main(weights_path: str, csv_path: str, output_path: str = "predictions.csv", 
         batch_size: int = 256, device: str = None):
    """
    Main inference pipeline.
    
    Args:
        weights_path: Path to trained model weights (.pt file)
        csv_path: Path to test CSV file
        output_path: Path to save predictions CSV
        batch_size: Batch size for inference
        device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Loading model from: {weights_path}")
    
    # Load model and config
    model, vocab, config = load_model_and_vocab(weights_path, device)
    print(f"Model loaded. Vocab size: {vocab.size}")
    print(f"Config: {config}")
    
    # Read test data
    print(f"\nReading test data from: {csv_path}")
    df = read_cleaned_csv(csv_path)
    print(f"Total events: {len(df)}")
    
    # Build journeys
    journeys = build_journeys(df)
    print(f"Total journeys: {len(journeys)}")
    
    # Create dataset and loader
    test_ds = InferenceDataset(
        journeys,
        vocab,
        max_seq_len=config["max_seq_len"],
        cap_delta_seconds=config["cap_delta_seconds"],
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_pad_inference(b, vocab.pad_id),
    )
    
    # Run inference
    print("\nRunning inference...")
    results = run_inference(model, test_loader, device)
    
    # Create output dataframe
    output_df = pd.DataFrame(results, columns=["id", "order_shipped"])
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total predictions: {len(output_df)}")
    print(f"\nSample predictions:")
    print(output_df.head(10))
    print(f"\nProbability statistics:")
    print(output_df["order_shipped"].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RNN inference for order shipped prediction")
    parser.add_argument("--weights_path", type=str, help="Path to trained model weights (.pt file)", default='purchase_rnn.pt')
    parser.add_argument("--csv_path", type=str, help="Path to test CSV file", default = './data/open_journeys1.csv')
    parser.add_argument("--output_path", type=str, default="predictions_LSTM.csv", help="Path to save predictions CSV")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu). Auto-detected if not specified")
    
    args = parser.parse_args()
    
    main(
        weights_path=args.weights_path,
        csv_path=args.csv_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=args.device,
    )