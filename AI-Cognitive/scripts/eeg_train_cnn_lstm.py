from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FS = 250.0
WINDOW_SIZE = int(4 * FS)  # 4-second windows (must match eeg_make_window_features)
STEP_SIZE = int(2 * FS)


class EEGWindowDataset(Dataset):
    """
    Dataset that loads 4s windows directly from eeg_all_samples.csv
    and returns tensors of shape (channels, time), plus label index.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, label_to_idx: dict[str, int]):
        self.df = df
        self.target_col = target_col
        self.label_to_idx = label_to_idx

        # numeric value columns (v0..vN-1)
        self.value_cols: List[str] = [
            c for c in df.columns if c.startswith("v") and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not self.value_cols:
            raise ValueError("No numeric value columns starting with 'v' were found.")

        self.windows: list[Tuple[int, int, str]] = []
        group_cols = ["file_name", "subject_id", "task_type", "condition_4", "load_3"]

        for _, g in df.groupby(group_cols, sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            start = 0
            while start + WINDOW_SIZE <= n:
                end = start + WINDOW_SIZE
                label = str(g.loc[0, target_col])
                self.windows.append((g.index[0] + start, g.index[0] + end, label))
                start += STEP_SIZE

        if not self.windows:
            raise RuntimeError("No windows generated for CNN-LSTM dataset.")

        # Pre-normalize per-channel using global mean/std over the subset
        data = df[self.value_cols].to_numpy().astype(np.float32)
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-6

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end, label = self.windows[idx]
        segment = self.df.loc[start:end - 1, self.value_cols].to_numpy().astype(np.float32)
        # normalize per-channel
        segment = (segment - self.mean) / self.std
        # shape: (time, channels) -> (channels, time)
        x = torch.from_numpy(segment).transpose(0, 1)
        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return x, y


class CnnLstmNet(nn.Module):
    """
    Simple 1D CNN + LSTM over time dimension.
    Input: (batch, channels, time)
    """

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        z = self.conv(x)  # (batch, 64, time')
        z = z.transpose(1, 2)  # (batch, time', 64)
        out, _ = self.lstm(z)  # (batch, time', 128)
        # take last time step
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def load_eeg_samples() -> pd.DataFrame:
    csv = PROJECT_ROOT / "csv" / "eeg_all_samples.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found. Run 'eeg_convert_raw_to_clean.py' first.")
    df = pd.read_csv(csv)
    required = {"file_name", "condition_4", "load_3", "task_type", "subject_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    df = df[df["condition_4"] != "unknown"].copy()
    df = df[df["load_3"] != "unknown"].copy()
    df = df.reset_index(drop=True)
    return df


def split_files_for_target(df: pd.DataFrame, target_col: str, test_size: float = 0.2, seed: int = 42):
    file_level = (
        df.groupby("file_name")[target_col]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    train_files, test_files = train_test_split(
        file_level["file_name"],
        test_size=test_size,
        random_state=seed,
        stratify=file_level[target_col],
    )
    train_mask = df["file_name"].isin(train_files)
    test_mask = df["file_name"].isin(test_files)
    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def train_cnn_lstm(
    target_col: str = "load_3",
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
) -> None:
    df = load_eeg_samples()
    df_train, df_test = split_files_for_target(df, target_col=target_col)

    classes = sorted(df_train[target_col].unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    train_ds = EEGWindowDataset(df_train, target_col, label_to_idx)
    test_ds = EEGWindowDataset(df_test, target_col, label_to_idx)

    n_channels = len(train_ds.value_cols)
    n_classes = len(classes)

    # Selección de dispositivo: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS (Metal)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    print(f"Using device for CNN+LSTM: {device} ({device_name})")
    model = CnnLstmNet(n_channels=n_channels, n_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print("\n=== CNN+LSTM training ===")
    print(f"Training CNN+LSTM on {len(train_ds)} windows, testing on {len(test_ds)} windows")
    print(f"Classes: {classes}")

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}] starting...")
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"[Epoch {epoch}/{epochs}] train loss: {avg_loss:.4f}", flush=True)

    # Evaluation
    model.eval()
    all_y: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_y.extend(y.numpy().tolist())

    y_true_labels = [idx_to_label[i] for i in all_y]
    y_pred_labels = [idx_to_label[i] for i in all_pred]

    print("\n=== CNN+LSTM (PyTorch) on windows ===")
    print("Classification report:")
    print(classification_report(y_true_labels, y_pred_labels))
    print("Confusion matrix:")
    print(confusion_matrix(y_true_labels, y_pred_labels, labels=classes))


def main() -> None:
    # Default: 10 epochs for a first comparison (adjust as needed)
    train_cnn_lstm(target_col="load_3", epochs=10)


if __name__ == "__main__":
    main()

