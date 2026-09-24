import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm


class SpatialTransformer(nn.Module):
    """
    N-D 空间形变应用模块 (基于 VoxelMorph 架构的经典实现)
    支持将形变场应用到图像上。
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        # 创建标准的网格坐标系 (Identity Grid)
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        self.grid = grid.type(torch.FloatTensor)
        self.register_buffer('identity_grid', self.grid)

    def forward(self, src, flow):
        """
        参数:
            src: 输入图像 Tensor (B, C, Z, Y, X)
            flow: 密集形变场 Tensor (B, 3, Z, Y, X) - 假设通道顺序为 (Z, Y, X)
        """
        # 将新形变场加到标准网格上
        new_locs = self.identity_grid + flow
        shape = flow.shape[2:]

        # 将坐标标准化到 [-1, 1] 区间供 grid_sample 使用
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # grid_sample 期望的 grid 形状为 (B, Z, Y, X, 3)，并且最后一维的坐标顺序为 (X, Y, Z)
        # 如果你的 flow 通道顺序是 Z, Y, X，这里需要做翻转并调整维度
        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        else:
            raise NotImplementedError("目前仅支持 3D 图像")

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


def upscale_deformation_field(disp_field, target_shape):
    """
    将下采样产生的固定尺寸形变场还原为原始尺寸的形变场。

    参数:
        disp_field: 神经网络输出的形变场 Tensor (B, 3, Z_fixed, Y_fixed, X_fixed)
        target_shape: 原始图像尺寸 tuple (Z_orig, Y_orig, X_orig)

    返回:
        upscaled_field: 原始尺寸的形变场，位移值已按比例放大
    """
    b, c, z_f, y_f, x_f = disp_field.shape
    z_o, y_o, x_o = target_shape

    # 1. 空间维度上采样 (双三次/三线性插值保证平滑)
    upscaled_field = F.interpolate(
        disp_field,
        size=target_shape,
        mode='trilinear',
        align_corners=True
    )

    # 2. 缩放位移值本身
    # 假设通道的顺序对应空间维度：0->Z, 1->Y, 2->X
    scale_factors = [z_o / z_f, y_o / y_f, x_o / x_f]
    scale_tensor = torch.tensor(scale_factors, device=disp_field.device).view(1, 3, 1, 1, 1)

    upscaled_field = upscaled_field * scale_tensor

    return upscaled_field


def adjust_affine_matrix(affine_matrix, fixed_shape, target_shape):
    """
    调整仿射变换矩阵，使其适配还原后的原始大尺寸图像。
    原理： T_orig = S * T_fixed * S^-1

    参数:
        affine_matrix: (B, 3, 4) 齐次仿射矩阵 (去除最后一行 [0,0,0,1])
        fixed_shape: 训练时的尺寸 (Z_fixed, Y_fixed, X_fixed)
        target_shape: 原始尺寸 (Z_orig, Y_orig, X_orig)

    返回:
        adjusted_affine: (B, 3, 4) 适用于大尺寸的仿射矩阵
    """
    device = affine_matrix.device
    b = affine_matrix.shape[0]

    z_f, y_f, x_f = fixed_shape
    z_o, y_o, x_o = target_shape

    # 构造缩放矩阵及其逆矩阵
    S = torch.tensor([
        [x_o / x_f, 0, 0, 0],
        [0, y_o / y_f, 0, 0],
        [0, 0, z_o / z_f, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    S_inv = torch.tensor([
        [x_f / x_o, 0, 0, 0],
        [0, y_f / y_o, 0, 0],
        [0, 0, z_f / z_o, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    # 补齐齐次矩阵的最后一行以便做矩阵乘法
    homo_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 4).repeat(b, 1, 1)
    full_affine = torch.cat([affine_matrix, homo_row], dim=1)  # (B, 4, 4)

    # 矩阵变换: S @ Affine @ S_inv
    adjusted_full = torch.bmm(S.unsqueeze(0).repeat(b, 1, 1), full_affine)
    adjusted_full = torch.bmm(adjusted_full, S_inv.unsqueeze(0).repeat(b, 1, 1))

    # 返回 (B, 3, 4) 格式
    return adjusted_full[:, 0:3, :]


def create_3d_grid(shape=(32, 128, 128), spacing=16):
    """
    生成一个 3D 网格图像，方便肉眼观察形变效果。
    """
    grid = torch.zeros(shape)
    # 在 Z, Y, X 三个维度上每隔 spacing 画一条线
    grid[::spacing, :, :] = 1.0
    grid[:, ::spacing, :] = 1.0
    grid[:, :, ::spacing] = 1.0

    # 为了让边缘更清晰，给网格加上一个边框
    grid[0, :, :] = 1;
    grid[-1, :, :] = 1
    grid[:, 0, :] = 1;
    grid[:, -1, :] = 1
    grid[:, :, 0] = 1;
    grid[:, :, -1] = 1
    return grid


class DistributedWarper:
    """
    针对大规模 3D 图像的低显存分块形变器。
    巧妙利用局部坐标归一化，彻底避免全局高分辨率形变场的生成。
    """

    def __init__(self, original_shape, fixed_shape, device='cuda'):
        """
        参数:
            original_shape: 原始大图像的尺寸 (Z_orig, Y_orig, X_orig)
            fixed_shape: 配准网络输出的小形变场尺寸 (Z_fixed, Y_fixed, X_fixed)
            device: 运算设备
        """
        self.original_shape = original_shape
        self.fixed_shape = fixed_shape
        self.device = torch.device(device)

        # 计算各维度的放大倍率
        self.scale_z = original_shape[0] / fixed_shape[0]
        self.scale_y = original_shape[1] / fixed_shape[1]
        self.scale_x = original_shape[2] / fixed_shape[2]

    def warp_chunk_by_chunk(self, img_large_cpu, flow_fixed, chunk_size=32):
        """
        沿 Z 轴分块执行形变。

        参数:
            img_large_cpu: 原始大尺寸图像 Tensor，存放在 CPU 内存中 (1, 1, Z, Y, X)
                           建议使用 tifffile.memmap 读取以应对超大文件
            flow_fixed: 网络输出的低分辨率形变场 (1, 3, Z_f, Y_f, X_f)，通道序约定为 Z, Y, X
            chunk_size: 每次送入 GPU 处理的 Z 轴厚度

        返回:
            warped_large: 完整尺寸的形变后图像 Tensor (在 CPU 上)
        """
        Z_orig, Y_orig, X_orig = self.original_shape

        # 初始化输出容器 (放在 CPU 上)
        warped_large = torch.zeros_like(img_large_cpu)
        flow_fixed = flow_fixed.to(self.device)

        print(f"开始执行 Z 轴分块分布式形变 (总层数: {Z_orig}, 分块大小: {chunk_size})...")

        for z_start in tqdm(range(0, Z_orig, chunk_size), desc="Warping Chunks"):
            z_end = min(z_start + chunk_size, Z_orig)
            z_len = z_end - z_start

            # ========================================================
            # 步骤 1: 即时生成当前块的高分辨率形变场 (不生成全局形变场)
            # ========================================================
            # 生成目标输出块的绝对坐标网格
            z_coords = torch.arange(z_start, z_end, device=self.device)
            y_coords = torch.arange(0, Y_orig, device=self.device)
            x_coords = torch.arange(0, X_orig, device=self.device)

            mesh_z, mesh_y, mesh_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

            # 将网格归一化到 [-1, 1] 相对整个大图的尺寸，用于采样 flow_fixed
            norm_z_flow = 2.0 * mesh_z / max(Z_orig - 1, 1) - 1.0
            norm_y_flow = 2.0 * mesh_y / max(Y_orig - 1, 1) - 1.0
            norm_x_flow = 2.0 * mesh_x / max(X_orig - 1, 1) - 1.0

            grid_for_flow = torch.stack([norm_x_flow, norm_y_flow, norm_z_flow], dim=-1).unsqueeze(0)

            # 从小尺寸形变场中提取并放大当前块的形变场
            flow_chunk_unscaled = F.grid_sample(flow_fixed, grid_for_flow, align_corners=True, mode='trilinear')

            # 按物理比例放大位移值
            flow_chunk_z = flow_chunk_unscaled[:, 0, ...] * self.scale_z
            flow_chunk_y = flow_chunk_unscaled[:, 1, ...] * self.scale_y
            flow_chunk_x = flow_chunk_unscaled[:, 2, ...] * self.scale_x

            # ========================================================
            # 步骤 2: 计算动态边界并截取输入图像
            # ========================================================
            # 目标体素在原图中的绝对采样坐标 = 网格坐标 + 位移量
            sample_z_abs = mesh_z + flow_chunk_z
            sample_y_abs = mesh_y + flow_chunk_y
            sample_x_abs = mesh_x + flow_chunk_x

            # 查看当前块的 Z 轴极限偏移，决定需要向上下多读取多少层
            min_z_req = math.floor(sample_z_abs.min().item())
            max_z_req = math.ceil(sample_z_abs.max().item())

            # 限定在合法范围内
            in_z_start = max(0, min_z_req)
            in_z_end = min(Z_orig, max_z_req + 1)

            # 从 CPU 内存中截取“带 Padding”的图像切片，并送入 GPU
            img_chunk_gpu = img_large_cpu[:, :, in_z_start:in_z_end, :, :].to(self.device).float()

            # ========================================================
            # 步骤 3: 基于局部切片重映射坐标系并执行 Warp
            # ========================================================
            # PyTorch 的 grid_sample 要求坐标是基于当前传入的图像张量归一化到 [-1, 1] 的
            # 因为传入的 img_chunk_gpu 是被截断过的，所以需要重新归一化 Z 坐标
            norm_z_img = 2.0 * (sample_z_abs - in_z_start) / max(in_z_end - in_z_start - 1, 1) - 1.0
            norm_y_img = 2.0 * sample_y_abs / max(Y_orig - 1, 1) - 1.0
            norm_x_img = 2.0 * sample_x_abs / max(X_orig - 1, 1) - 1.0

            grid_for_img = torch.stack([norm_x_img, norm_y_img, norm_z_img], dim=-1).unsqueeze(0)

            # 执行形变
            with torch.no_grad():
                warped_chunk = F.grid_sample(img_chunk_gpu, grid_for_img, align_corners=True, mode='bilinear')

            # 将结果放回 CPU 的完整容器中
            warped_large[:, :, z_start:z_end, :, :] = warped_chunk.cpu()

            # 释放显存
            del img_chunk_gpu, warped_chunk, grid_for_img, grid_for_flow, flow_chunk_unscaled
            torch.cuda.empty_cache()

        return warped_large


def validate_deformation_pipeline():
    print("=== 开始形变场与上采样逻辑验证 ===")

    # 尽可能使用 CPU 进行验证，方便没有 GPU 的环境也能跑通
    device = torch.device('cpu')

    # 1. 设定测试尺寸 (大图和网络输入的小图)
    original_shape = (512, 680, 512)  # Z, Y, X (大尺寸)
    fixed_shape = (64, 64, 64)  # Z, Y, X (下采样后输入网络的固定尺寸)

    print(f"原图尺寸 (Target): {original_shape}")
    print(f"网络输入尺寸 (Fixed): {fixed_shape}")

    # 计算放大倍率 (Z: 2倍, Y: 4倍, X: 4倍)
    scale_z = original_shape[0] / fixed_shape[0]
    scale_y = original_shape[1] / fixed_shape[1]
    scale_x = original_shape[2] / fixed_shape[2]

    # 2. 生成原始大图的网格 (Batch=1, Channel=1, Z, Y, X)
    image_large = create_3d_grid(original_shape, spacing=32).unsqueeze(0).unsqueeze(0).to(device)

    # 3. 构建一个模拟的网络输出形变场 (Batch=1, Channels=3, Z_f, Y_f, X_f)
    # 通道顺序约定为: 0->Z, 1->Y, 2->X
    flow_fixed = torch.zeros((1, 3, *fixed_shape), device=device)

    # 假设在小图上，所有体素沿 X 轴(通道2)向右平移 5 个像素，沿 Y 轴(通道1)向下平移 2 个像素
    shift_x_fixed = 5.0
    shift_y_fixed = 2.0
    flow_fixed[:, 2, ...] = shift_x_fixed
    flow_fixed[:, 1, ...] = shift_y_fixed

    print(f"\n[模拟形变场] 在 {fixed_shape} 尺寸下: X轴平移 {shift_x_fixed} 像素, Y轴平移 {shift_y_fixed} 像素")

    # 4. 执行形变场上采样
    print("\n正在执行形变场上采样...")
    flow_upscaled = upscale_deformation_field(flow_fixed, original_shape)

    # --- 数学断言验证 ---
    # 断言1：尺寸必须完全匹配原图
    assert flow_upscaled.shape == (1, 3, *original_shape), f"尺寸还原错误: {flow_upscaled.shape}"

    # 断言2：位移值必须按比例放大
    expected_shift_x_large = shift_x_fixed * scale_x  # 5.0 * 4 = 20.0
    expected_shift_y_large = shift_y_fixed * scale_y  # 2.0 * 4 = 8.0

    actual_shift_x = flow_upscaled[0, 2, 0, 0, 0].item()
    actual_shift_y = flow_upscaled[0, 1, 0, 0, 0].item()

    print(f"[验证] 还原后 X 轴位移: 期望 {expected_shift_x_large}, 实际 {actual_shift_x}")
    print(f"[验证] 还原后 Y 轴位移: 期望 {expected_shift_y_large}, 实际 {actual_shift_y}")

    assert torch.isclose(torch.tensor(actual_shift_x), torch.tensor(expected_shift_x_large)), "X轴位移缩放不正确！"
    assert torch.isclose(torch.tensor(actual_shift_y), torch.tensor(expected_shift_y_large)), "Y轴位移缩放不正确！"
    print("✅ 形变场数学上采样断言通过！")

    # 5. 执行形变 (Warping)
    print("\n正在应用形变场到原始大尺寸图像...")
    transformer = SpatialTransformer(original_shape).to(device)
    warped_image_large = transformer(image_large, flow_upscaled)

    # 6. 提取中心切片进行可视化
    print("✅ 形变执行完成，正在生成可视化比对图...")
    img_orig_slice = image_large[0, 0, original_shape[0] // 2, :, :].numpy()
    img_warp_slice = warped_image_large[0, 0, original_shape[0] // 2, :, :].detach().numpy()

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Deformation Field Upscaling & Warping Validation", fontsize=16)

    # 原图
    axes[0].imshow(img_orig_slice, cmap='gray')
    axes[0].set_title(f"Original Grid\n{original_shape}", fontsize=12)
    # 画一条红色的参考线（十字准星）代表网格原始位置
    axes[0].axvline(x=128, color='red', linestyle='--', alpha=0.5)
    axes[0].axhline(y=128, color='red', linestyle='--', alpha=0.5)

    # 形变后的图
    axes[1].imshow(img_warp_slice, cmap='gray')
    axes[1].set_title(f"Warped Grid\n(Shift X: +{expected_shift_x_large}, Y: +{expected_shift_y_large})", fontsize=12)
    # 画同样的参考线查看偏移
    axes[1].axvline(x=128, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=128, color='red', linestyle='--', alpha=0.5)

    # 差异图 (Diff)
    diff = np.abs(img_orig_slice - img_warp_slice)
    axes[2].imshow(diff, cmap='magma')
    axes[2].set_title("Absolute Difference", fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate_deformation_pipeline()