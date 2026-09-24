import numpy as np
import torch


def unify_tensor_shape(tensor: torch.Tensor, target_shape: tuple, pad_value=0.0) -> torch.Tensor:
    """
    统一 PyTorch 张量的尺寸：中心裁剪与边缘填充 (Zero-Padding)

    参数:
        tensor (torch.Tensor): 输入的 PyTorch 张量 (N维)
        target_shape (tuple): 目标形状，长度必须与 tensor.dim() 一致
        pad_value (float or int): 填充的数值，默认为 0.0

    返回:
        torch.Tensor: 形状严格等于 target_shape 的新张量
    """
    current_shape = tensor.shape

    if len(current_shape) != len(target_shape):
        raise ValueError(f"输入张量的维度数 ({len(current_shape)}) 与目标形状 ({len(target_shape)}) 不匹配！")

    # PyTorch F.pad 的要求: padding 参数是从最后一个维度开始向前计算的
    # 格式为: (pad_last_dim_before, pad_last_dim_after, pad_second_last_before, pad_second_last_after, ...)
    pad_tuple = []
    slices = []

    # 按照从前向后的正常顺序遍历维度
    for c_size, t_size in zip(current_shape, target_shape):
        diff = t_size - c_size

        if diff > 0:
            # 目标尺寸更大 -> 需要填充 (Padding)
            pad_before = diff // 2
            pad_after = diff - pad_before

            # 使用 insert(0, ...) 倒序插入，以满足 PyTorch F.pad 的特殊要求
            pad_tuple.insert(0, pad_after)
            pad_tuple.insert(0, pad_before)
            slices.append(slice(None))  # 填充后，该维度的切片全部保留

        elif diff < 0:
            # 目标尺寸更小 -> 需要中心裁剪 (Cropping)
            pad_tuple.insert(0, 0)
            pad_tuple.insert(0, 0)

            crop_start = abs(diff) // 2
            crop_end = crop_start + t_size
            slices.append(slice(crop_start, crop_end))

        else:
            # 尺寸刚好相等
            pad_tuple.insert(0, 0)
            pad_tuple.insert(0, 0)
            slices.append(slice(None))

    # 1. 执行 PyTorch 原生填充 (仅作用于尺寸较小的维度)
    padded_tensor = torch.nn.functional.pad(tensor, tuple(pad_tuple), mode='constant', value=pad_value)

    # 2. 执行张量切片裁剪 (仅作用于尺寸较大的维度)
    final_tensor = padded_tensor[tuple(slices)]

    return final_tensor

def unify_image_shape(image: np.ndarray, target_shape: tuple, pad_value=0) -> np.ndarray:
    """
    统一图像的尺寸：
    如果图像在某个维度上大于 target_shape，则进行中心裁剪。
    如果图像在某个维度上小于 target_shape，则在两端对称填充 pad_value (默认值为0)。

    参数:
        image (np.ndarray): 输入的图像数组 (N维)
        target_shape (tuple): 目标形状，长度必须与 image.ndim 一致
        pad_value (int or float): 填充的数值，默认为 0

    返回:
        np.ndarray: 形状严格等于 target_shape 的新图像
    """
    current_shape = image.shape

    if len(current_shape) != len(target_shape):
        raise ValueError(f"输入图像的维度数 ({len(current_shape)}) 与目标形状的维度数 ({len(target_shape)}) 不匹配！")

    pad_width = []
    slices = []

    for c_size, t_size in zip(current_shape, target_shape):
        diff = t_size - c_size

        if diff > 0:
            # 目标尺寸更大 -> 需要填充 (Padding)
            pad_before = diff // 2  # 左/上/前 侧填充量
            pad_after = diff - pad_before  # 右/下/后 侧填充量 (处理奇数差异)

            pad_width.append((pad_before, pad_after))
            slices.append(slice(None))  # 填充后该维度全部保留

        elif diff < 0:
            # 目标尺寸更小 -> 需要中心裁剪 (Cropping)
            pad_width.append((0, 0))  # 不需要填充

            crop_start = abs(diff) // 2
            crop_end = crop_start + t_size
            slices.append(slice(crop_start, crop_end))

        else:
            # 尺寸刚好相等
            pad_width.append((0, 0))
            slices.append(slice(None))

    # 1. 先进行边缘填充 (只影响需要变大的维度)
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=pad_value)

    # 2. 再进行中心裁剪 (只影响需要变小的维度)
    final_image = padded_image[tuple(slices)]

    return final_image


import numpy as np

def crop_3d_volume_with_overlap_generator(volume, crop_size=(64, 128, 128), overlap_ratio=0.1):
    """
    【生成器版本】将三维体数据裁剪为固定大小，包含指定重叠率。
    每次迭代使用 yield 吐出一个 Patch，极大地节省内存。

    返回 (Generator yields):
        patch: 当前的 3D 图像块
        core_coords: 当前块对应的核心区域坐标 (z1, z2, y1, y2, x1, x2)
        padding_info: 记录填充的像素 (pad_d, pad_h, pad_w)
    """
    D, H, W = volume.shape[:3]
    CD, CH, CW = crop_size

    # --- 1. 自动填充 (Padding) ---
    pad_d = max(0, CD - D)
    pad_h = max(0, CH - H)
    pad_w = max(0, CW - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        if volume.ndim == 4:  # 处理带通道的数据 (D, H, W, C)
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w), (0, 0))
        else:  # 处理单通道数据 (D, H, W)
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))

        volume = np.pad(volume, pad_width)
        D, H, W = volume.shape[:3]  # 更新填充后的尺寸

    # --- 2. 计算步长与滑动窗口起点 ---
    OD = int(CD * overlap_ratio)
    OH = int(CH * overlap_ratio)
    OW = int(CW * overlap_ratio)

    SD = CD - OD  # Z方向步长
    SH = CH - OH  # Y方向步长
    SW = CW - OW  # X方向步长

    # 生成 Z, Y, X 方向的起点，并确保覆盖边缘
    z_starts = list(range(0, D - CD + 1, SD))
    if not z_starts or z_starts[-1] != D - CD:
        z_starts.append(D - CD)

    y_starts = list(range(0, H - CH + 1, SH))
    if not y_starts or y_starts[-1] != H - CH:
        y_starts.append(H - CH)

    x_starts = list(range(0, W - CW + 1, SW))
    if not x_starts or x_starts[-1] != W - CW:
        x_starts.append(W - CW)

    # 提取公共变量，避免在循环内重复创建
    padding_info = (pad_d, pad_h, pad_w)

    # --- 3. 三层嵌套执行切块 (使用 yield 替代 append) ---
    for k, z in enumerate(z_starts):
        for i, y in enumerate(y_starts):
            for j, x in enumerate(x_starts):
                # 提取当前 3D 图像块 (这里只是获取切片视图，占用极少内存)
                patch = volume[z:z + CD, y:y + CH, x:x + CW]

                # 【核心逻辑】计算 Z, Y, X 三个方向的无重叠核心区域
                # 深度 Z 方向
                core_z1 = z if k == 0 else (z_starts[k - 1] + CD + z) // 2
                core_z2 = z + CD if k == len(z_starts) - 1 else (z + CD + z_starts[k + 1]) // 2

                # 高度 Y 方向
                core_y1 = y if i == 0 else (y_starts[i - 1] + CH + y) // 2
                core_y2 = y + CH if i == len(y_starts) - 1 else (y + CH + y_starts[i + 1]) // 2

                # 宽度 X 方向
                core_x1 = x if j == 0 else (x_starts[j - 1] + CW + x) // 2
                core_x2 = x + CW if j == len(x_starts) - 1 else (x + CW + x_starts[j + 1]) // 2

                core_coords = (core_z1, core_z2, core_y1, core_y2, core_x1, core_x2)

                # 【关键修改】：不再把 patch 和 coords 塞进列表，而是直接 yield 吐出去
                yield patch, core_coords, padding_info

def crop_3d_volume_with_overlap(volume, crop_size=(64, 128, 128), overlap_ratio=0.1):
    """
    将三维体数据裁剪为固定大小，包含指定重叠率，并记录核心无重叠区域的坐标。

    参数:
        volume: 输入的三维数据，NumPy 数组 (D, H, W) 或带通道 (D, H, W, C)
        crop_size: 裁剪尺寸 (D_crop, H_crop, W_crop)
        overlap_ratio: 重叠比例，默认 0.1 (10%)

    返回:
        patches: 裁剪出的 3D 图像块列表
        core_coords: 每个块对应的核心区域坐标 [(z1, z2, y1, y2, x1, x2), ...]
        padding_info: 记录填充的像素 (pad_d, pad_h, pad_w)
    """
    D, H, W = volume.shape[:3]
    CD, CH, CW = crop_size

    # --- 1. 自动填充 (Padding) ---
    # 如果原图的任意维度小于裁剪尺寸，进行边缘反射填充
    pad_d = max(0, CD - D)
    pad_h = max(0, CH - H)
    pad_w = max(0, CW - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        if volume.ndim == 4:  # 处理带通道的数据 (D, H, W, C)
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w), (0, 0))
        else:  # 处理单通道数据 (D, H, W)
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))

        volume = np.pad(volume, pad_width)
        D, H, W = volume.shape[:3]  # 更新填充后的尺寸

    # --- 2. 计算步长与滑动窗口起点 ---
    OD = int(CD * overlap_ratio)
    OH = int(CH * overlap_ratio)
    OW = int(CW * overlap_ratio)

    SD = CD - OD  # Z方向步长
    SH = CH - OH  # Y方向步长
    SW = CW - OW  # X方向步长

    # 生成 Z, Y, X 方向的起点，并确保覆盖边缘
    z_starts = list(range(0, D - CD + 1, SD))
    if not z_starts or z_starts[-1] != D - CD:
        z_starts.append(D - CD)

    y_starts = list(range(0, H - CH + 1, SH))
    if not y_starts or y_starts[-1] != H - CH:
        y_starts.append(H - CH)

    x_starts = list(range(0, W - CW + 1, SW))
    if not x_starts or x_starts[-1] != W - CW:
        x_starts.append(W - CW)

    patches = []
    core_coords = []

    # --- 3. 三层嵌套执行切块并计算核心坐标 ---
    for k, z in enumerate(z_starts):
        for i, y in enumerate(y_starts):
            for j, x in enumerate(x_starts):
                # 提取当前 3D 图像块
                patch = volume[z:z + CD, y:y + CH, x:x + CW]

                patches.append(patch)

                # 【核心逻辑】计算 Z, Y, X 三个方向的无重叠核心区域
                # 深度 Z 方向
                core_z1 = z if k == 0 else (z_starts[k - 1] + CD + z) // 2
                core_z2 = z + CD if k == len(z_starts) - 1 else (z + CD + z_starts[k + 1]) // 2

                # 高度 Y 方向
                core_y1 = y if i == 0 else (y_starts[i - 1] + CH + y) // 2
                core_y2 = y + CH if i == len(y_starts) - 1 else (y + CH + y_starts[i + 1]) // 2

                # 宽度 X 方向
                core_x1 = x if j == 0 else (x_starts[j - 1] + CW + x) // 2
                core_x2 = x + CW if j == len(x_starts) - 1 else (x + CW + x_starts[j + 1]) // 2

                core_coords.append((core_z1, core_z2, core_y1, core_y2, core_x1, core_x2))

    return patches, core_coords, (pad_d, pad_h, pad_w)


# ==========================================
# 🌟 测试与验证 (模拟医学图像)
# ==========================================
if __name__ == "__main__":
    # 模拟一个 100 张切片，每张 200x200 像素的 3D 数据 (D, H, W)
    dummy_3d_volume = np.random.randint(0, 255, (128, 128, 128), dtype=np.uint8)

    print(f"原 3D 图像尺寸: {dummy_3d_volume.shape}")

    # 执行三维裁剪 (固定大小 64x128x128, 重叠 10%)
    patches, core_coords, padding = crop_3d_volume_with_overlap(
        dummy_3d_volume,
        crop_size=(128, 128, 128),
        overlap_ratio=0.1
    )

    print(f"✅ 共裁剪出 {len(patches)} 个 3D 图像块")
    print(f"💡 边缘填充信息 (pad_d, pad_h, pad_w): {padding}")

    # 打印前 3 个块的核心坐标示例
    print("\n📊 前 3 个图像块的核心区域坐标 (z1, z2, y1, y2, x1, x2):")
    for idx in range(min(3, len(core_coords))):
        coords = core_coords[idx]
        print(f"  - 块 {idx}: {coords}")
        print(
            f"    有效深度: {coords[1] - coords[0]}, 有效高度: {coords[3] - coords[2]}, 有效宽度: {coords[5] - coords[4]}")