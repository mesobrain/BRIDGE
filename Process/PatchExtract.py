import math
import numpy as np


class PatchExtractor:
    """
    带重叠区域的 3D ROI 图像裁剪器
    """

    def __init__(self, patch_size=(128, 128, 128), overlap_ratio=0.10):
        """
        参数:
            patch_size (tuple): 裁剪的图像块大小 (Z, Y, X)
            overlap_ratio (float): 重叠比例 (0.0 到 0.99)
        """
        assert 0 <= overlap_ratio < 1, "重叠比例必须在 [0, 1) 之间"
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio

    def extract_patches(self, image):
        """
        对输入的 3D 图像进行滑动窗口裁剪。
        为了保证边缘数据不丢失，会对图像右侧/下侧/后侧进行反射填充(Padding)。

        参数:
            image (numpy.ndarray): 输入的 3D 图像 (Z, Y, X)

        返回:
            patches (list of ndarray): 裁剪出的图像块列表
            positions (list of tuple): 每个块在 Padded 图像中的起始坐标 (Z, Y, X)，用于后期还原
            padded_shape (tuple): Padding 后的总体图像尺寸
        """
        z_dim, y_dim, x_dim = image.shape
        pz, py, px = self.patch_size

        # 计算每次滑动的步长 (Stride)
        sz = max(1, int(pz * (1 - self.overlap_ratio)))
        sy = max(1, int(py * (1 - self.overlap_ratio)))
        sx = max(1, int(px * (1 - self.overlap_ratio)))

        # 计算在每个维度上需要滑动的次数
        nz = math.ceil(max(z_dim - pz, 0) / sz) + 1
        ny = math.ceil(max(y_dim - py, 0) / sy) + 1
        nx = math.ceil(max(x_dim - px, 0) / sx) + 1

        # 计算为了让最后一次裁剪不越界，需要 Padding 的大小
        pad_z = max((nz - 1) * sz + pz - z_dim, 0)
        pad_y = max((ny - 1) * sy + py - y_dim, 0)
        pad_x = max((nx - 1) * sx + px - x_dim, 0)

        # 对图像进行边缘反射填充 (避免黑边对配准网络产生干扰)
        print(f"原图尺寸: {image.shape}, 补齐边缘尺寸: (Z:+{pad_z}, Y:+{pad_y}, X:+{pad_x})")
        padded_image = np.pad(image, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='reflect')

        patches = []
        positions = []

        print(f"开始裁剪，步长: {(sz, sy, sx)}，预计生成 {nz * ny * nx} 个图像块...")

        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    # 计算当前块的起始坐标
                    z_start = z * sz
                    y_start = y * sy
                    x_start = x * sx

                    # 切片提取
                    patch = padded_image[z_start:z_start + pz, y_start:y_start + py, x_start:x_start + px]

                    patches.append(patch)
                    positions.append((z_start, y_start, x_start))

        return patches, positions, padded_image.shape