import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from fastmrislice import FastMRISliceNPY, FastMRISliceH5PY
from fastmrislice import fastmri_h5py_to_npy

def test_fastmri_slice_npy(root_h5py, root_npy, challenge):
    dataset_new = FastMRISliceNPY(root_npy)
    dataset_raw = FastMRISliceH5PY(root_h5py, challenge)
    assert len(dataset_new) == len(dataset_raw)
    for i in range(len(dataset_new)):
        k0, val0 = dataset_raw.__getitem__(i)
        k1, val1 = dataset_new.__getitem__(i)
        assert np.allclose(k0, k1, rtol=0, atol=1e-20)
        assert np.allclose(val0, val1, rtol=0, atol=1e-8)
    print("all data from npy match with h5py")
    return None

if __name__ == '__main__':
    # generate raw data files for FastMRISliceNPY, and verify
    # fastmri knee multi coil
    rootin = Path("/data0/zijian/data/Aespa_data/example_pdfs_data")
    rootout = Path("/data0/zijian/data/Aespa_data/fmknee_pdfs_npy")
    fastmri_h5py_to_npy(rootin, rootout, True)
