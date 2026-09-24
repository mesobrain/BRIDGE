import gc
import os.path
import tifffile
import numpy as np
from skimage.transform import resize
from scipy.ndimage import zoom
import torch
from tqdm import tqdm
import torch.nn.functional as F
import math
import SimpleITK as sitk


class FastImageResampler:
    """
    针对 uint16 显微图像优化的极速分步下采样器。
    保留了 skimage 的抗混叠优势，通过将 3D 运算拆解为 2D 运算来突破速度瓶颈。
    """
    def __init__(self, target_shape=(128, 128, 128), scale_factors=[0.5, 0.25], order=1):
        self.target_shape = target_shape
        self.scale_factors = scale_factors
        self.order = order
        self.image_shape = (1, 1, 1)
        self.scales_list = {}

    def load_and_resize(self, image_root, ):
        if os.path.isfile(image_root):
            # 处理单一文件
            pyramid = self._load_sitk_3d_resize(image_root)
            return pyramid
        else:
            # 处理单帧存储文件，采用分离式策略降低下采样计算显存
            image_list = sorted([os.path.join(image_root, image_name) for image_name in os.listdir(image_root) if 'tif' in image_name])
            # return self._load_fast_3d_resize(image_list)
            scales_list = {}
            z_in = len(image_list)
            y_in, x_in = tifffile.imread(image_list[0]).shape
            self.image_shape = tuple([z_in, y_in, x_in])
            scales_list[0] = [self.target_shape[0] / x_in, self.target_shape[1] / y_in, self.target_shape[2] / z_in]
            for i, scale_i in enumerate(self.scale_factors):
                scales_list[i+1] = [scale_i for _ in range(3)]
            self.scales_list = scales_list
            down_pyramid = self._load_sitk_3d_frame_resize(image_list, scales_list)
            fixed_block = down_pyramid[0]
            pyramid = {}
            for i, scale_i in enumerate(self.scale_factors):
                pyramid[scale_i] = down_pyramid[i+1]
            return fixed_block, pyramid

    def _load_sitk_3d_resize(self, image_path, ):
        # 提前初始化字典结构，更简洁
        image = sitk.ReadImage(image_path)
        image_size = image.GetSize()
        self.image_shape = tuple((image_size[2], image_size[1], image_size[0]))
        scales_list = {}
        scales_list[0] = [self.target_shape[0] / image_size[2], self.target_shape[1] / image_size[1],
                          self.target_shape[2] / image_size[0]]
        for i, scale_i in enumerate(self.scale_factors):
            scales_list[i + 1] = [scale_i for _ in range(3)]
        self.scales_list = scales_list
        pyramid = {scale: [] for scale in scales_list.keys()}

        dim = image.GetDimension()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        pixel_id = image.GetPixelID()

        # 将图像各维度的 spacing 初始化为 1 (延续你原有的逻辑)
        image.SetSpacing([1.0] * dim)

        for scale_name, scale_factors in scales_list.items():
            orig_size = image.GetSize()

            if dim == 3:
                # 获取3个维度的缩放因子。如果传入的字典只给了2个(XY)，Z轴默认设为不缩放(1.0)
                sx = scale_factors[0]
                sy = scale_factors[1]
                sz = scale_factors[2] if len(scale_factors) > 2 else 1.0

                # ================= X, Y, Z 轴下采样  =================
                size_xyz = [int(orig_size[0] * sx), int(orig_size[1] * sy), int(orig_size[2] * sz)]
                spacing_xyz = [1.0 / sx, 1.0 / sy, 1.0 / sz]

                new_img = sitk.Resample(
                    image, size_xyz, sitk.Transform(), sitk.sitkLinear,
                    origin, spacing_xyz, direction, 0.0, pixel_id
                )

            else:
                # ================= 保留原有的 2D 下采样逻辑 =================
                sx = scale_factors[0]
                sy = scale_factors[1]
                size_xy = [int(orig_size[0] * sx), int(orig_size[1] * sy)]
                spacing_xy = [1.0 / sx, 1.0 / sy]

                new_img = sitk.Resample(
                    image, size_xy, sitk.Transform(), sitk.sitkLinear,
                    origin, spacing_xy, direction, 0.0, pixel_id
                )

            pyramid[scale_name].append(new_img)

        # 处理完单个图像后，从内存中释放
        del image

        return pyramid

    def _load_sitk_3d_frame_resize(self, image_name_list, scales_list:dict):
        pyramid = {}
        for scale in scales_list.keys():
            pyramid[scale] = []
        for image_name in tqdm(image_name_list):
            image = sitk.ReadImage(image_name)
            origin = image.GetOrigin()
            image.SetSpacing([1 for _ in range(2)])
            for scale in scales_list.keys():
                image_size = [int(image.GetSize()[i] * scales_list[scale][i]) for i in range(2)]

                new_img = sitk.Resample(image, image_size, sitk.Transform(), sitk.sitkLinear,
                                        origin, [1 / scales_list[scale][i] for i in range(2)],
                                        image.GetDirection(), 0.0, image.GetPixelID())
                # new_img = sitk.Resample(image, image_size, sitk.Transform(), sitk.sitkNearestNeighbor, [0, 0], [scales_list[scale][i] for i in range(2)])
                pyramid[scale].append(new_img)
            del image

        down_pyramid = {}
        for scale in scales_list.keys():
            sample_image = sitk.JoinSeries(pyramid[scale])
            # print(sample_image.GetSize())
            sample_image.SetSpacing([1 / scales_list[scale][0], 1 / scales_list[scale][1], 1])
            image_size = [sample_image.GetSize()[0], sample_image.GetSize()[1], int(sample_image.GetSize()[2] * scales_list[scale][2])]
            down_pyramid[scale] = sitk.GetArrayFromImage(sitk.Resample(sample_image, image_size, sitk.Transform(), sitk.sitkLinear, [0, 0, 0],
                                     [1 / scales_list[scale][i] for i in range(3)]))
            print('Scale {}: Before Z sample {}; After Z sample {}'.format(scale,
                    sample_image.GetSize(), down_pyramid[scale].shape))
        return down_pyramid

    def _load_fast_3d_resize(self, image_list):
        z_in = len(image_list)
        y_in, x_in = tifffile.imread(image_list[0]).shape
        image_block = []
        pyramid = {}
        for i in self.scale_factors:
            pyramid[i] = []
        for image_path in tqdm(image_list):
            image = tifffile.imread(image_path)
            image_block.append(resize(
                image,
                (self.target_shape[1], self.target_shape[2]),
                order=self.order,
                preserve_range=True,
                anti_aliasing=(self.order > 0)
            ))
            for scale_i in self.scale_factors:
                y_out = max(1, math.floor(image.shape[0] * scale_i))
                x_out = max(1, math.floor(image.shape[1] * scale_i))
                pyramid[i].append(resize(image,
                                  (y_out, x_out),
                                  order=self.order,
                                  preserve_range=True,
                                  anti_aliasing=(self.order > 0)
                                  ))
        image_block = np.array(image_block)

        down_image_block = np.zeros((self.target_shape[0], self.target_shape[1], self.target_shape[2]),
                                    dtype=image_block.dtype)
        for y in range(self.target_shape[1]):
            down_image_block[:, y, :] = resize(image_block[:, y, :], (self.target_shape[0], self.target_shape[2]),
                                               order=self.order,
                                               preserve_range=True,
                                               anti_aliasing=(self.order > 0)
                                               )
        del image_block
        gc.collect()
        down_pyramid = {}
        for scale_i in self.scale_factors:
            z_out = max(1, math.floor(image.shape[0] * scale_i))
            y_out = max(1, math.floor(image.shape[1] * scale_i))
            x_out = max(1, math.floor(image.shape[2] * scale_i))
            image_py = np.array(pyramid[scale_i])
            down_image_py = np.zeros((z_out, y_out, x_out), dtype=down_image_block.dtype)
            for y in range(y_out):
                down_image_py[:, y, :] = resize(image_py[:, y, :], (z_out, x_out),
                                                order=self.order,
                                                preserve_range=True,
                                                anti_aliasing=(self.order > 0)
                                                )
            down_pyramid[scale_i] = down_image_py
        return down_image_block, down_pyramid


    def _fast_3d_resize(self, image, target_shape):
        """
        核心加速逻辑：将 3D 重采样拆分为 2D(XY) + 1D(Z) 的独立操作
        """
        z_in, y_in, x_in = image.shape
        z_out, y_out, x_out = target_shape

        # 记录原始数据类型 (通常为 uint16)
        orig_dtype = image.dtype

        # ---------------------------------------------------------
        # 第一步：沿 Z 轴遍历，对每个 XY 平面进行 2D 下采样
        # ---------------------------------------------------------
        # 预分配中间内存块 (Z_in, Y_out, X_out)
        temp_vol = np.zeros((z_in, y_out, x_out), dtype=orig_dtype)

        for i in range(z_in):
            # 对单张 XY 切片进行 skimage resize
            # 2D 的抗混叠计算速度极快
            resized_slice = resize(
                image[i],
                (y_out, x_out),
                order=self.order,
                preserve_range=True,
                anti_aliasing=(self.order > 0)
            )
            temp_vol[i] = resized_slice.astype(orig_dtype)

        # 如果 Z 轴不需要改变尺寸，直接返回
        if z_in == z_out:
            return temp_vol

        # ---------------------------------------------------------
        # 第二步：沿 Y 轴遍历，对每个 ZX 平面进行下采样 (实现 Z 轴缩放)
        # ---------------------------------------------------------
        # 预分配最终内存块 (Z_out, Y_out, X_out)
        final_vol = np.zeros((z_out, y_out, x_out), dtype=orig_dtype)

        for i in range(y_out):
            # 取出 ZX 平面：temp_vol[:, i, :] 的形状为 (Z_in, X_out)
            # 我们需要将其 resize 为 (Z_out, X_out)
            resized_zx = resize(
                temp_vol[:, i, :],
                (z_out, x_out),
                order=self.order,
                preserve_range=True,
                anti_aliasing=(self.order > 0)
            )
            final_vol[:, i, :] = resized_zx.astype(orig_dtype)

        return final_vol

    def generate_fixed_size_block(self, image):
        print(f"正在极速生成固定尺寸图像块 {self.target_shape} ...")
        return self._fast_3d_resize(image, self.target_shape)

    def generate_scale_pyramid(self, image):
        pyramid = {}
        for scale in self.scale_factors:
            print(f"正在极速生成 {scale} 倍率下采样图像 ...")
            z_out = max(1, math.floor(image.shape[0] * scale))
            y_out = max(1, math.floor(image.shape[1] * scale))
            x_out = max(1, math.floor(image.shape[2] * scale))
            target_shape = (z_out, y_out, x_out)

            pyramid[scale] = self._fast_3d_resize(image, target_shape)

        return pyramid

    def process_volume(self, image):
        fixed_block = self.generate_fixed_size_block(image)
        pyramid = self.generate_scale_pyramid(image)
        return fixed_block, pyramid


class ImageResampler:
    """
    图像下采样与金字塔生成器
    """

    def __init__(self, target_shape=(128, 128, 128), scale_factors=[0.5, 0.25], order=1):
        """
        参数:
            target_shape (tuple): 固定的全局图像尺寸 (Z, Y, X)
            scale_factors (list): 多倍下采样的倍率列表，例如 [0.5, 0.25] 表示 2倍和4倍下采样
        """
        self.target_shape = target_shape
        self.scale_factors = scale_factors
        self.order = order

    def generate_fixed_size_block(self, image):
        """
        将整幅图像下采样成一个固定尺寸的图像块
        """
        print(f"正在生成固定尺寸图像块 {self.target_shape} ...")
        # 使用 skimage 的 resize，自带抗混叠
        fixed_img = resize(
            image,
            self.target_shape,
            order=self.order,
            preserve_range=True,
            anti_aliasing=(self.order > 0)
        )
        return fixed_img.astype(image.dtype)

    def generate_scale_pyramid(self, image):
        """
        根据 scale_factors 生成多倍下采样图像金字塔

        返回:
            dict: 键为倍率，值为对应的下采样图像
        """
        pyramid = {}
        for scale in self.scale_factors:
            print(f"正在生成 {scale} 倍率下采样图像 ...")
            zoom_factors = (scale, scale, scale)
            resampled_img = zoom(image, zoom=zoom_factors, order=self.order)
            pyramid[scale] = resampled_img.astype(image.dtype)
        return pyramid

    def process_volume(self, image):
        """
        一键执行：返回固定尺寸块和金字塔字典
        """
        fixed_block = self.generate_fixed_size_block(image)
        pyramid = self.generate_scale_pyramid(image)
        return fixed_block, pyramid



def warp_arbitrary_roi(img_large_cpu, affine_matrix, original_shape, roi_start, roi_size, device='cuda'):
    """
    对大图的任意局部 ROI 进行仿射变换提取，只截取所需的最小内存块。

    参数:
        img_large_cpu: 原始大图像 Tensor (1, 1, Z_orig, Y_orig, X_orig)，在 CPU
        affine_matrix: 3x4 仿射矩阵 (1, 3, 4)，在 GPU
        original_shape: 原图全局尺寸 (Z, Y, X)
        roi_start: 目标 ROI 的起始坐标 (z_start, y_start, x_start)
        roi_size: 目标 ROI 的大小 (z_size, y_size, x_size)

    返回:
        warped_roi: 仅包含该 ROI 的形变后图像 Tensor
    """
    Z_orig, Y_orig, X_orig = original_shape
    z0, y0, x0 = roi_start
    dz, dy, dx = roi_size

    # 1. 直接生成目标 ROI 的局部网格
    z_coords = torch.arange(z0, z0 + dz, device=device)
    y_coords = torch.arange(y0, y0 + dy, device=device)
    x_coords = torch.arange(x0, x0 + dx, device=device)

    mesh_z, mesh_y, mesh_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # 2. 映射到全局 [-1, 1] 坐标系
    norm_z = 2.0 * mesh_z / max(Z_orig - 1, 1) - 1.0
    norm_y = 2.0 * mesh_y / max(Y_orig - 1, 1) - 1.0
    norm_x = 2.0 * mesh_x / max(X_orig - 1, 1) - 1.0

    # 构建齐次坐标 (4, N)
    flatten_x, flatten_y, flatten_z = norm_x.flatten(), norm_y.flatten(), norm_z.flatten()
    coords_homo = torch.stack([flatten_x, flatten_y, flatten_z, torch.ones_like(flatten_x)], dim=0)

    # 3. 矩阵乘法算出在原图中的真实采样位置
    mat = affine_matrix.to(device).squeeze(0)
    sampled_coords = torch.matmul(mat, coords_homo).view(3, dz, dy, dx)
    sampled_x, sampled_y, sampled_z = sampled_coords[0], sampled_coords[1], sampled_coords[2]

    # 4. 计算 3D 边界框 (Bounding Box)，并向外扩展 1 个像素以防插值边缘伪影
    def get_bounds(sampled_dim, max_dim):
        abs_coords = (sampled_dim + 1.0) / 2.0 * (max_dim - 1)
        min_val = max(0, math.floor(abs_coords.min().item()) - 1)
        max_val = min(max_dim, math.ceil(abs_coords.max().item()) + 2)
        return abs_coords, min_val, max_val

    abs_x, min_x, max_x = get_bounds(sampled_x, X_orig)
    abs_y, min_y, max_y = get_bounds(sampled_y, Y_orig)
    abs_z, min_z, max_z = get_bounds(sampled_z, Z_orig)

    # 5. 从 CPU 内存中“精准抠图”送入 GPU
    # 注意：这里我们同时在 Z, Y, X 三个维度上进行裁剪，极大节省了显存！
    img_chunk_gpu = img_large_cpu[:, :, min_z:max_z, min_y:max_y, min_x:max_x].to(device).float()

    # 6. 将绝对坐标重新归一化到我们刚刚抠出来的小块内部的 [-1, 1] 坐标系
    local_norm_x = 2.0 * (abs_x - min_x) / max(max_x - min_x - 1, 1) - 1.0
    local_norm_y = 2.0 * (abs_y - min_y) / max(max_y - min_y - 1, 1) - 1.0
    local_norm_z = 2.0 * (abs_z - min_z) / max(max_z - min_z - 1, 1) - 1.0

    grid_for_img = torch.stack([local_norm_x, local_norm_y, local_norm_z], dim=-1).unsqueeze(0)

    # 7. 局部形变
    with torch.no_grad():
        warped_roi = F.grid_sample(img_chunk_gpu, grid_for_img, align_corners=True, mode='bilinear')

    return warped_roi

# --- 使用示例 ---
# 假设你想在对齐后的图像中，提取一个以 (Z=100, Y=500, X=500) 为起点的 64x64x64 图像块：
# roi_warped = warp_arbitrary_roi(
#     img_large_cpu, affine_matrix, original_shape=(500, 1024, 1024),
#     roi_start=(100, 500, 500), roi_size=(64, 64, 64)
# )


def pipeline(image_root, save_root, target_shape=(128, 128, 128), scale_factors=[0.25, 0.125, 0.0625]):
    os.makedirs(save_root, exist_ok=True)
    resampler = FastImageResampler(target_shape=target_shape, scale_factors=scale_factors)
    print('downsample {}'.format(image_root))
    fixed_block, pyramid = resampler.load_and_resize(image_root)
    tifffile.imwrite(os.path.join(save_root, 'image_mapping.tif'), fixed_block, compression='lzw')
    for scale_i in scale_factors:
        tifffile.imwrite(os.path.join(save_root, 'image_{}.tif'.format(scale_i)), pyramid[scale_i], compression='lzw')


if __name__ == '__main__':
    root_list = [
                 r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\Raw\WT2\R1_4.0',
                 r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\Raw\WT2\R2_4.0',]
    for root in root_list:
        image_root = root
        save_root = root + '_Downsample'
        pipeline(image_root, save_root, scale_factors=[0.5, 0.125, 0.0625])