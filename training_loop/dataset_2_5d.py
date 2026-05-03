from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def find_dim_min(vec, thr):
    high = vec.detach().cpu().numpy() >= thr
    return np.argmax(high)


def find_dim_max(vec, thr):
    high = vec.detach().cpu().numpy() >= thr
    return len(high) - np.argmax(high[::-1])


def crop_volume(volume, thr=1e-6):
    dims_x = torch.sum(torch.sum(volume, 1), -1) / np.prod(volume.shape)
    dims_y = torch.sum(torch.sum(volume, 0), -1) / np.prod(volume.shape)
    dims_z = torch.sum(torch.sum(volume, 0), 0) / np.prod(volume.shape)
    return volume[
        find_dim_min(dims_x, thr): find_dim_max(dims_x, thr),
        find_dim_min(dims_y, thr): find_dim_max(dims_y, thr),
        find_dim_min(dims_z, thr): find_dim_max(dims_z, thr),
    ]


def resize_slice_keep_aspect(slice_2d, size=224):
    h, w = slice_2d.shape
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = F.interpolate(
        slice_2d.unsqueeze(0).unsqueeze(0).float(),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    result = torch.zeros((size, size), dtype=resized.dtype)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    result[y0: y0 + new_h, x0: x0 + new_w] = resized
    return result


def make_2_5d_windows(volume, image_size=224, num_windows=128):
    # NIfTI volume is loaded as (X, Y, Z). We use Z as the slice axis.
    volume = volume.permute(2, 0, 1).contiguous()
    depth = volume.shape[0]

    centers = torch.linspace(0, depth - 1, steps=num_windows).round().long()
    windows = []

    for center in centers:
        center = int(center.item())
        indices = [
            max(0, center - 1),
            center,
            min(depth - 1, center + 1),
        ]
        slices = [resize_slice_keep_aspect(volume[idx], size=image_size) for idx in indices]
        windows.append(torch.stack(slices, dim=0))

    return torch.stack(windows, dim=0)


class CTWindowsDataset(Dataset):
    def __init__(self, table_path, images_dir, label_columns, image_size=224, num_windows=128):
        self.table_path = Path(table_path)
        self.images_dir = Path(images_dir)
        self.label_columns = list(label_columns)
        self.image_size = int(image_size)
        self.num_windows = int(num_windows)

        self.samples_df = pd.read_csv(self.table_path)
        self.samples_df["study_uid"] = self.samples_df["study_uid"].astype(str)
        self.samples_df["image_path"] = self.samples_df["study_uid"].apply(
            lambda uid: str(self.images_dir / f"{uid}.nii.gz")
        )

        self.samples_df = self.samples_df[self.samples_df["image_path"].apply(lambda p: Path(p).exists())]
        self.samples_df = self.samples_df.dropna(subset=self.label_columns).reset_index(drop=True)

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        row = self.samples_df.iloc[idx]

        volume = nib.load(row["image_path"]).get_fdata(dtype="float16")
        volume = torch.from_numpy(volume)
        volume = crop_volume(volume, thr=1e-6)
        windows = make_2_5d_windows(
            volume,
            image_size=self.image_size,
            num_windows=self.num_windows,
        )

        labels = torch.tensor(
            row[self.label_columns].to_numpy(dtype=np.float16),
            dtype=torch.float32,
        )

        return windows.float(), labels
