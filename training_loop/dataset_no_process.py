from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def resize_volume(volume, size=256):
    return F.interpolate(
        volume.unsqueeze(0).unsqueeze(0),
        size=(size, size, size),
        mode="trilinear",
        align_corners=False,
    )[0, 0]


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
        volume = resize_volume(volume, size=self.target_size)

        volume = np.transpose(volume, (2, 0, 1))
        volume = volume.unsqueeze(0).float()

        labels = torch.tensor(
            row[self.label_columns].to_numpy(dtype=np.float16),
            dtype=torch.float16,
        )

        return volume, labels
