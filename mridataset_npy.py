import os
import numpy as np
from torch.utils.data import Dataset

class MRISliceDataset(Dataset):
    def __init__(self, root_dir):
        """
        极简 MRI Slice 加载器：只读数据，不裁剪，不检查 valid
        """
        self.root_dir = root_dir
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
            
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])

        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        file_path = os.path.join(self.root_dir, fname)
        k = np.squeeze(np.load(file_path))
        return {
            'k_space': k,      # 原始尺寸 K 空间
            'fname': fname,    # 文件名
            'idx': idx         # 索引
        }