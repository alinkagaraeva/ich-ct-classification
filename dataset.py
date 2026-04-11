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


def _tuple_int(t):
    return tuple(t.numpy().astype(int))


def crop_volume(volume, thr=1e-6):
    dims_x = torch.sum(torch.sum(volume, 1), -1) / np.prod(volume.shape)
    dims_y = torch.sum(torch.sum(volume, 0), -1) / np.prod(volume.shape)
    dims_z = torch.sum(torch.sum(volume, 0), 0) / np.prod(volume.shape)
    return volume[
        find_dim_min(dims_x, thr): find_dim_max(dims_x, thr),
        find_dim_min(dims_y, thr): find_dim_max(dims_y, thr),
        find_dim_min(dims_z, thr): find_dim_max(dims_z, thr),
    ]


def resize_volume(volume, size=256):
    shape_old = torch.tensor(volume.shape)
    shape_new = torch.tensor([size] * 3)
    scale = torch.max(shape_old.to(float) / shape_new)
    shape_scale = shape_old / scale
    vol_ = F.interpolate(
        volume.unsqueeze(0).unsqueeze(0),
        size=_tuple_int(shape_scale),
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    offset = _tuple_int((shape_new - shape_scale) / 2)
    volume = torch.zeros(*_tuple_int(shape_new), dtype=volume.dtype)
    shape_scale = _tuple_int(shape_scale)
    volume[
        offset[0]: offset[0] + shape_scale[0],
        offset[1]: offset[1] + shape_scale[1],
        offset[2]: offset[2] + shape_scale[2],
    ] = vol_
    return volume


class CTVolumeDataset(Dataset):
    def __init__(
        self,
        xlsx_path,
        images_dir,
        label_columns,
        target_size=128,
        sheet_name="Sheet1",
    ):
        self.xlsx_path = Path(xlsx_path)
        self.images_dir = Path(images_dir)
        self.label_columns = list(label_columns)
        self.target_size = int(target_size)

        xl_file = pd.ExcelFile(self.xlsx_path)
        self.samples_df = xl_file.parse(sheet_name)

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
        volume = resize_volume(volume, size=self.target_size)

        volume = np.transpose(volume, (2, 0, 1))
        volume = volume.unsqueeze(0).float()

        labels = torch.tensor(
            row[self.label_columns].to_numpy(dtype=np.float16),
            dtype=torch.float16,
        )

        return volume, labels
