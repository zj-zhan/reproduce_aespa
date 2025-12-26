import numpy as np
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from utils import ifft2_np, fft2_np, kspace_to_target
from fastmrislice import FastMRISliceNPY

class MRISliceDataset(Dataset):
    def __init__(self, 
                 root:Union[str, Path] = "../../../../data/fastmri_knee_mc",
                 target_size: int = 320,
                ):
        self.target_size = target_size
        self.crop_shape = (self.target_size, self.target_size)
        self.slicedata = FastMRISliceNPY(root)
        
        print(f"Total slices: {len(self.slicedata)}")

    def _crop_if_needed(self, image):
        # expect input shape [Ncoil, Nread, Npe]
        if self.crop_shape[0] < image.shape[-2]:
            h_from = (image.shape[-2] - self.crop_shape[0]) // 2
            h_to = h_from + self.crop_shape[0]
        else:
            h_from = 0
            h_to = image.shape[-2]

        if self.crop_shape[1] < image.shape[-1]:
            w_from = (image.shape[-1] - self.crop_shape[1]) // 2
            w_to = w_from + self.crop_shape[1]
        else:
            w_from = 0
            w_to = image.shape[-1]

        return image[:, h_from:h_to, w_from:w_to]

    def _pad_if_needed(self, image):
        # expect input shape [Ncoil, Nread, Npe]
        pad_h = self.crop_shape[0] - image.shape[-2]
        pad_w = self.crop_shape[1] - image.shape[-1]

        if pad_w > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            pad_left = pad_right = 0

        if pad_h > 0:
            pad_up = pad_h // 2
            pad_down = pad_h - pad_up
        else:
            pad_up = pad_down = 0

        pad_width = ((0, 0), (pad_up, pad_down), (pad_left, pad_right))
        image = np.pad(image, pad_width, mode='reflect')
        return image

    def _to_uniform_size(self, image):
        # expect input shape [Ncoil, Nread, Npe]
        image = self._crop_if_needed(image)
        shape_raw = np.array([image.shape[-2], image.shape[-1]])
        image = self._pad_if_needed(image)
        return image, shape_raw

    def __len__(self):
        return len(self.slicedata.raw_samples)

    def __getitem__(self, idx):
        kspace_raw, max_value = self.slicedata[idx]
        fname = Path(self.slicedata.raw_samples[idx][0]).name
        mctgt = ifft2_np(kspace_raw)
        mctgt, shape_raw = self._to_uniform_size(mctgt)
        kspace = fft2_np(mctgt)
        norm_factor = 1.0 / max_value
        mctgt = mctgt * norm_factor
        max_value = max_value * norm_factor

        kspace = fft2_np(mctgt)
        tgt = kspace_to_target(kspace)

        return {
            'idx': idx,
            'fname': fname,
            'kspace': kspace,
            "rss": tgt,
            "max_val": max_value,
            "shape_raw": shape_raw,
        }