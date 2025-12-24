import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
from torch.utils.data import Dataset
from utils import ifft2_np, fft2_np, kspace_to_target
from fastmrislice import FastMRISliceH5PY

class MRIDataset(Dataset):
    avg_max_value = {
        "knee": 0.0004,
        "brain": 0.000835,
        "stanford2d": 1348138.72,
        "stanford3dsag": 1441552.74,
        "stanford3dcor": 1537736.81,
        "skmteasage1": 104226972.9,
        "skmteasage2": 82136858.47,
        "ccbrainax": 30488595.85,
        "ccbrainsag": 30620699.87,
        "aheadaxe1": 13859689984.0,
        "aheadaxe2": 25906087987.2,
        "aheadaxe3": 16932815411.2,
        "aheadaxe4": 12756087961.6,
        "aheadaxe5": 11186915737.6,
        "m4raw": 141.31,
        "m4rawgre": 313.52,
        "cmrxrecon": 0.0035,
        "ocmr3.0t": 0.00525,
        "ocmr1.5t": 0.00275,
        "ocmr0.55t": 0.00037,
    }
    def __init__(self, 
                 root: Union[str, Path] = "../../../../data/fastmri_knee_mc", 
                 target_size: int = 320,
                 which_data: str ="knee",
                 data_norm_type: str = "volume_max",
                 device: torch.device = torch.device('cpu')): 
        
        self.target_size = target_size
        self.which_data = which_data
        self.crop_shape = (self.target_size, self.target_size)
        self.data_norm_type = data_norm_type
        self.device = device
        self.slicedata = FastMRISliceH5PY(root,"multicoil", False)
        
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

    def __len__(self) -> int:
        return len(self.slicedata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        kspace_raw, max_value = self.slicedata.__getitem__(idx)

        fname = Path(self.slicedata.raw_samples[idx][0]).name
        mctgt = ifft2_np(kspace_raw) # [Ncoil, Nread, Npe] complex
        mctgt, shape_raw = self._to_uniform_size(mctgt) # [Ncoil, Ny, Ny] complex
        kspace = fft2_np(mctgt)

        if self.data_norm_type == "volume_max":
            norm_factor = 1.0 / max_value
        elif self.data_norm_type == "slice_max":
            norm_factor = 1.0 / torch.max(torch.sqrt(torch.sum(mctgt**2, dim=(0, -1))))
        elif self.data_norm_type == "avg":
            norm_factor = 1.0 / self.avg_max_value[self.which_data]
        elif self.data_norm_type == "none":
            norm_factor = 1.0

        mctgt = mctgt * norm_factor
        max_value = max_value * norm_factor

        kspace = fft2_np(mctgt)
        tgt = kspace_to_target(kspace)

        return {
            "kspace": kspace,
            "rss": tgt,
            "max_val": max_value,
            "idx": idx,
            "shape_raw": shape_raw,
            "fname": fname
        }