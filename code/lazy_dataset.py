import numpy as np
import torch
from torch.utils.data import Dataset


class LazyEEGDataset(Dataset):

    def __init__(
        self,
        npz_path,
        indices,
        median,
        iqr,
        channels=None,
        augment: bool = False,
        clip_val: float = 3.0,
    ):
        # On charge en mode lazy
        self.data = np.load(npz_path, mmap_mode="r")
        self.X = self.data["X"]   # attend (N, C, T)
        self.y = self.data["y"]

        self.indices = indices
        self.augment = augment
        self.clip_val = clip_val

        # median, iqr viennent de compute_median_iqr (C, 1)
        self.median = median
        self.iqr = iqr
        self.channels = channels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        x = self.X[real_idx]      # numpy (C, T) après correction dans main_kfold
        y = int(self.y[real_idx])

        # Normalisation robuste par canal
        x = (x - self.median) / self.iqr      # (C, T) - (C,1) : OK, broadcast
        x = np.clip(x, -self.clip_val, self.clip_val)

        # Sélection de sous-ensemble de canaux (top-32, top-16, etc.)
        if self.channels is not None:
            x = x[self.channels]

        # Passage en tenseurs
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # Data augmentation
        if self.augment:
            if torch.rand(1).item() < 0.45:
                x = x + 0.02 * torch.randn_like(x)
            if torch.rand(1).item() < 0.40:
                shift = torch.randint(-8, 9, (1,)).item()
                x = torch.roll(x, shifts=shift, dims=1)
            if torch.rand(1).item() < 0.30:
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                x = x * scale
            if torch.rand(1).item() < 0.30:
                L = torch.randint(8, 21, (1,)).item()
                start = torch.randint(0, x.shape[1] - L + 1, (1,)).item()
                x[:, start:start + L] = 0.0
            if torch.rand(1).item() < 0.20:
                C = x.shape[0]
                k = max(1, int(0.1 * C))
                idxs = torch.randperm(C)[:k]
                perm = idxs[torch.randperm(k)]
                x[idxs] = x[perm]

        return x, y
