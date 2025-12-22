import os, shutil
import h5py
import numpy as np
from pathlib import Path
from typing import Union
from fastmri.data import SliceDataset
from torchvision.datasets.vision import VisionDataset

def fastmri_h5py_to_npy(rootin: str, rootout: str, is_multicoil: bool = True):
    """
    Reformat FastMRI data files from h5py (each volume) to npy (each slice) for
    efficient data loading.

    Only kspace is stored as a complex np.ndarray with shape as below
        single coil data: [Nread, Npe]
        multi coil data: [Ncoil, Nread, Npe]

    Parameters:
        rootin (str): directory where raw data files exist
        rootout (str): directory to put new data files
        is_multicoil (bool): true for multi-coil, false for single-coil
    """
    challenge = "multicoil" if is_multicoil else "singlecoil"
    dataset = SliceDataset(root=Path(rootin), challenge=challenge, use_dataset_cache=False)
    N = len(dataset)

    if os.path.exists(rootout) and os.path.isdir(rootout):
        shutil.rmtree(rootout)
    os.makedirs(rootout)

    for i in range(N):
        print(f"{rootin}: {i} / {N}")
        fname, dataslice, metadata = dataset.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            max_value = attrs["max"]

        max_value = f"{max_value:.10f}"[2:]
        fname = Path(fname).stem
        subject_dir = Path(rootout) / fname
        subject_dir.mkdir(exist_ok=True, parents=True)
        savepath = subject_dir / f"{dataslice:03d}_{max_value}.npy"
        np.save(savepath, kspace)

    return None

def check_file_type(root, file_ext):
    """return True if all files within root have extension of file_ext
    """
    if os.path.exists(root) and os.path.isdir(root):
        for rt, dirs, files in os.walk(root):
            for f in files:
                if Path(f).suffix != file_ext:
                    return False
    else:
        return False
    return True

class FastMRISliceH5PY(SliceDataset):
    def __init__(self,
                 root: Union[str, Path],
                 challenge: str = "multicoil",
                 use_dataset_cache: bool = False,
                 ):
        assert check_file_type(root, ".h5")
        super().__init__(root=Path(root), challenge=challenge, use_dataset_cache=use_dataset_cache)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, index: int):
        fname, dataslice, metadata = self.raw_samples[index]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            attrs = dict(hf.attrs)
            attrs.update(metadata)
            max_value = attrs["max"]
        return kspace, max_value

class FastMRISliceNPY(VisionDataset):
    def __init__(self, root: Union[str, Path]):
        assert check_file_type(root, ".npy")
        self.raw_samples = sorted(self._get_filename(root, ".npy"))

    def _get_filename(self, dirname, file_ext):
        filename_list = []
        for rt, dirs, files in os.walk(dirname):
            for f in files:
                if Path(f).suffix == file_ext:
                    filename_list.append(os.path.join(rt, f))
        return filename_list

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, index: int):
        kspace = np.load(self.raw_samples[index])
        fname = os.path.splitext(os.path.basename(self.raw_samples[index]))[0]
        max_value = "0." + fname.split("_")[-1]
        max_value = float(max_value)
        return kspace, max_value