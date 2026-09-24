import os
import torch
import numpy as np
from torch.utils import data
import tifffile as tiff
from skimage.transform import resize


class Dataset(data.Dataset):
    """
    用于 PyTorch 训练的数据集类。
    集成了 3D 图像的读取和固定尺寸/倍率的重采样功能。
    """
    def __init__(self, list_IDs, image_root, target_shape=None, scale_factor=None):
        self.list_IDs = list_IDs
        self.data_root = image_root
        self.target_shape = target_shape
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.list_IDs)

    def _resample_image(self, image):
        """内部函数：执行图像重采样"""
        if self.target_shape is not None:
            # 保证输出尺寸固定，供网络处理
            image = resize(image, self.target_shape, order=1, preserve_range=True, anti_aliasing=True)
        elif self.scale_factor is not None:
            from scipy.ndimage import zoom
            zoom_factors = (self.scale_factor, self.scale_factor, self.scale_factor)
            image = zoom(image, zoom=zoom_factors, order=1)
        return image

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        fixed_path = os.path.join(self.data_root, f"{ID}_1.tif")
        moving_path = os.path.join(self.data_root, f"{ID}_2.tif")

        # 使用 tifffile 读取 (支持大规模3D数据)
        fixed_image = tiff.imread(fixed_path).astype(np.float32)
        moving_image = tiff.imread(moving_path).astype(np.float32)

        # 执行重采样 (如果在参数中配置了)
        fixed_image = self._resample_image(fixed_image)
        moving_image = self._resample_image(moving_image)

        # 转换为 Tensor 并增加 Channel 维度 -> (1, Z, Y, X)
        fixed_tensor = torch.from_numpy(fixed_image).unsqueeze(0)
        moving_tensor = torch.from_numpy(moving_image).unsqueeze(0)

        return fixed_tensor, moving_tensor