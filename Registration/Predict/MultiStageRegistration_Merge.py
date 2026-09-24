import os
import gc
import time

import SimpleITK as sitk
import tifffile
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as nnf
import numpy as np
from Process.Normalize import normalize_3d_percentile
from Process.Resampler_v2 import image_resample
from Pipeline.config import Config
from Predict.Prediction import Pred
from DatasetConstruction.ImagePatch import crop_3d_volume_with_overlap
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F


def displacement_to_affine_flow(disp_field, sample_stride=4):
    """
    参数:
        disp_field: torch.Tensor, 形状为 (B, 3, D, H, W)
        sample_stride: 降采样步长，加速最小二乘拟合，默认 4
    返回:
        affine_disp: torch.Tensor, 形状为 (B, 3, D, H, W)
        theta: torch.Tensor, 形状为 (B, 3, 4)
    """
    B, C, D, H, W = disp_field.shape
    B, C, D, H, W = int(B), int(C), int(D), int(H), int(W)
    device = disp_field.device
    dtype = disp_field.dtype
    assert C == 3, "形变场通道数必须为 3 (x, y, z)"

    # 1. 构造 [-1, 1] 的基础归一化网格 (Identity Grid)
    identity_theta = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    base_grid = F.affine_grid(
        identity_theta,
        size=[B, 1, D, H, W],
        align_corners=True
    )  # 形状: (B, D, H, W, 3)

    # 2. 将输入的位移场重排为 (B, D, H, W, 3)
    disp_grid = disp_field.permute(0, 2, 3, 4, 1)

    # 3. 归一化位移
    scale = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0], device=device, dtype=dtype)
    disp_norm = disp_grid / scale

    # 目标网格点 Y = 基础点 X + 归一化位移 U
    target_grid = base_grid + disp_norm  # (B, D, H, W, 3)

    # 4. 空间降采样
    sub_X = base_grid[:, ::sample_stride, ::sample_stride, ::sample_stride, :]   # (B, d, h, w, 3)
    sub_Y = target_grid[:, ::sample_stride, ::sample_stride, ::sample_stride, :] # (B, d, h, w, 3)

    # 展平成点集矩阵
    X_pts = sub_X.reshape(B, -1, 3)  # (B, N, 3)
    Y_pts = sub_Y.reshape(B, -1, 3)  # (B, N, 3)

    # 5. 构造正规方程设计矩阵 P = [X, 1]，拟合 Y ≈ P @ affine_param
    N = X_pts.shape[1]
    ones = torch.ones(B, N, 1, device=device, dtype=dtype)
    P = torch.cat([X_pts, ones], dim=-1)  # (B, N, 4)

    # 最小二乘求解: 注意这里的解变量命名为 affine_weights，避免覆盖宽度 W
    affine_weights = torch.linalg.lstsq(P, Y_pts).solution  # (B, 4, 3)

    # 转置成 PyTorch 仿射矩阵格式 theta: (B, 3, 4)
    theta = affine_weights.transpose(1, 2)

    # 6. 生成仿射变换网格 (此时 W 是整数，不会再报错)
    affine_grid = F.affine_grid(theta, size=[B, 1, D, H, W], align_corners=True)  # (B, D, H, W, 3)

    # 7. 计算仿射位移场并恢复尺度
    affine_disp_norm = affine_grid - base_grid
    affine_disp_grid = affine_disp_norm * scale
    affine_disp = affine_disp_grid.permute(0, 4, 1, 2, 3)

    return affine_disp

def resample_sitk(moving_image, displace_field, ref_image=None, scale_factor=(1, 1, 1)):
    moving_image = sitk.GetImageFromArray(np.array(moving_image))
    print(moving_image.GetSize())
    # displace_field = np.array(displace_field).astype(np.float64)
    if isinstance(displace_field, np.ndarray):
        displace_field = sitk.GetImageFromArray(displace_field)
    elif isinstance(displace_field, torch.Tensor):
        displace_field = sitk.GetImageFromArray(np.array(displace_field))

    displace_field = sitk.Compose(sitk.VectorIndexSelectionCast(displace_field, 2),
                                  sitk.VectorIndexSelectionCast(displace_field, 1),
                                  sitk.VectorIndexSelectionCast(displace_field, 0))

    displace_field.SetSpacing(scale_factor)
    # print(displace_field.GetSize())
    # displace_field.SetSpacing(param_scale)
    tf2 = sitk.DisplacementFieldTransform(sitk.Cast(displace_field, sitk.sitkVectorFloat64))
    # print(displace_field.GetSize())
    if ref_image is None:
        ref_image = moving_image
    dst = sitk.Resample(moving_image, ref_image, tf2, sitk.sitkLinear)
    return sitk.GetArrayFromImage(dst), displace_field


def generate_3d_hann_window(patch_shape, device):
    """
    生成 3D 汉明窗，用于局部块的平滑拼接。
    中心权重最高，边缘平滑降为 0。
    """
    z, y, x = patch_shape
    w_z = torch.sin(torch.pi * torch.arange(z) / (z - 1)) ** 2
    w_y = torch.sin(torch.pi * torch.arange(y) / (y - 1)) ** 2
    w_x = torch.sin(torch.pi * torch.arange(x) / (x - 1)) ** 2

    window = w_z.view(-1, 1, 1) * w_y.view(1, -1, 1) * w_x.view(1, 1, -1)
    return window.to(device)


def get_identity_grid(shape, device='cpu'):
    """生成全局归一化基准坐标网格 [-1, 1]"""
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors, indexing='ij')
    grid = torch.stack(grids).unsqueeze(0).to(device).float()
    for i in range(len(shape)):
        grid[:, i, ...] = 2 * (grid[:, i, ...] / (shape[i] - 1) - 0.5)
    return grid[:, [2, 1, 0], ...]  # 匹配 xyz 顺序，形状 (1, 3, Z, Y, X)


def field_transformer(fields_list, scales_list):

    if len(fields_list) == 0:
        displacement_field = sitk.Transform()
    else:
        fields_list = fields_list[::-1]
        scales_list = scales_list[::-1]
        displacement_field = sitk.Transform(3, sitk.sitkComposite)
        for i in range(len(fields_list)):
            fields_list[i] = sitk.Compose(sitk.VectorIndexSelectionCast(fields_list[i], 2) * scales_list[i][0],
                                          sitk.VectorIndexSelectionCast(fields_list[i], 1) * scales_list[i][1],
                                          sitk.VectorIndexSelectionCast(fields_list[i], 0) * scales_list[i][2])
            fields_list[i].SetSpacing(scales_list[i])
            if displacement_field is None:

                # print(fields_list[i].GetPixelID(), fields_list[i].GetSize())
                displacement_field = sitk.DisplacementFieldTransform(
                    sitk.Cast(fields_list[i], sitk.sitkVectorFloat64))
            else:
                displacement_field.AddTransform(
                    sitk.DisplacementFieldTransform(
                        sitk.Cast(fields_list[i], sitk.sitkVectorFloat64)
                    ))
    return displacement_field


def get_dynamic_bspline_kernel_3d(kernel_size: int, device='cpu'):
    """
    动态生成任意奇数尺寸的 3D B-spline 卷积核
    """
    if kernel_size % 2 == 0:
        raise ValueError("卷积核尺寸必须是奇数！")

    radius = kernel_size // 2
    grid = torch.arange(-radius, radius + 1, dtype=torch.float32)
    x = grid * (2.0 / (radius + 0.5))

    abs_x = torch.abs(x)
    w_1d = torch.zeros_like(x)

    mask1 = abs_x < 1
    w_1d[mask1] = (2.0 / 3.0) - (abs_x[mask1] ** 2) + (abs_x[mask1] ** 3 / 2.0)

    mask2 = (abs_x >= 1) & (abs_x < 2)
    w_1d[mask2] = (1.0 / 6.0) * ((2.0 - abs_x[mask2]) ** 3)

    w_1d = w_1d / w_1d.sum()
    w_3d = w_1d.view(-1, 1, 1) * w_1d.view(1, -1, 1) * w_1d.view(1, 1, -1)

    # (in_channels, 1, D, H, W) -> (3, 1, K, K, K)
    b_spline_kernel = w_3d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1, 1)
    return b_spline_kernel.to(device)


def _process_single_patch(task_data):
    """
    单个 Patch 的独立处理逻辑：归一化、网络预测、仿射拟合转换
    """
    idx, fix_patch, mov_patch, predictor, affine_bool, sample_stride = task_data

    # 1. 预处理与归一化
    fix_norm = normalize_3d_percentile(fix_patch, prec=[1, 98], min_thre=[92, 256 + 92], image_range=255)
    mov_norm = normalize_3d_percentile(mov_patch, prec=[1, 98], min_thre=[92, 256 + 92], image_range=255)

    # 2. 推理
    with torch.no_grad():
        aligned_patch, local_flow = predictor.predict_image(fix_norm, mov_norm)

        # 3. 仿射拟合转换
        if affine_bool:
            local_flow = displacement_to_affine_flow(local_flow, sample_stride=sample_stride)

    # 转至 CPU 返回，避免主进程显存积压
    if isinstance(local_flow, torch.Tensor):
        local_flow = local_flow.cpu()

    return idx, local_flow, aligned_patch

def patch_registration(fixed_image, moving_image, predictor,
                       grid_shape: list = (64, 256, 256),
                       patch_shape: tuple = (64, 256, 256),
                       affine_bool=True,
                       affine_grid=None, overlap_ratio=0.1,
                       kernel_size = 7, device='cpu'):
    # 1. 准备全局累加器
    global_flow = torch.zeros((1, 3, *grid_shape), device=device, dtype=torch.float32)
    global_weight = torch.zeros((1, 1, *grid_shape), device=device, dtype=torch.float32)
    # 【新增】：专门用于记录每个体素被 Patch 覆盖的绝对次数，用于精确提取接缝 Mask
    global_overlap_count = torch.zeros((1, 1, *grid_shape), device=device, dtype=torch.int16)
    # 对图像进行小块裁剪
    fixed_volume, fixed_core_coords, fixed_padding = crop_3d_volume_with_overlap(
        fixed_image, crop_size=patch_shape, overlap_ratio=overlap_ratio)
    moving_volume, moving_core_coords, moving_padding = crop_3d_volume_with_overlap(
        moving_image, crop_size=patch_shape, overlap_ratio=overlap_ratio)
    # 生成对应的 3D 汉明窗
    patch_window = generate_3d_hann_window(patch_shape, device).unsqueeze(0).unsqueeze(0)
    for i in tqdm(range(len(fixed_volume)), desc='Predicting patch images'):
        # 调用网络获取局部形变场
        _, local_flow = predictor.predict_image(
            normalize_3d_percentile(fixed_volume[i], prec=[1, 98], min_thre=[140, 140+256], image_range=255),
            normalize_3d_percentile(moving_volume[i], prec=[1, 98], min_thre=[140, 256+140], image_range=255)
        )
        if i == 200:
            tifffile.imwrite(r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\MultiRoundData\R2_aligned\fixed_200.tif',
                             fixed_volume[i])
            tifffile.imwrite(r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\MultiRoundData\R2_aligned\moving_200.tif',
                             moving_volume[i])
            tifffile.imwrite(r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\MultiRoundData\R2_aligned\moving_200_aligned.tif', _)
        if affine_bool:
            local_flow = displacement_to_affine_flow(local_flow, sample_stride=8)
        # 提取当前块在大图中的映射坐标
        core_coords = [int(c) for c in moving_core_coords[i]]
        z0, z1 = max(0, core_coords[0]), min(grid_shape[0], core_coords[1])
        y0, y1 = max(0, core_coords[2]), min(grid_shape[1], core_coords[3])
        x0, x1 = max(0, core_coords[4]), min(grid_shape[2], core_coords[5])
        # 裁剪出有效的窗口部分（处理边界截断）
        valid_z, valid_y, valid_x = z1 - z0, y1 - y0, x1 - x0
        valid_local_flow = local_flow[..., :valid_z, :valid_y, :valid_x]
        valid_patch_window = patch_window[..., :valid_z, :valid_y, :valid_x]
        # 2. 核心魔法：加权累加 (Soft Blending)
        weighted_flow = valid_local_flow * valid_patch_window
        global_flow[0, :, z0:z1, y0:y1, x0:x1] += weighted_flow.squeeze(0)
        global_weight[0, 0, z0:z1, y0:y1, x0:x1] += valid_patch_window.squeeze(0).squeeze(0)
        # 【新增】：硬覆盖计数，只要被 Patch 框住，计数就 +1
        global_overlap_count[0, 0, z0:z1, y0:y1, x0:x1] += 1
    # 3. 计算加权平均，获得初步融合的形变场
    global_weight = torch.clamp(global_weight, min=1e-5)
    raw_global_flow = global_flow / global_weight
    # =========================================================================
    # 4. 【新增高阶优化】：掩膜引导的 B-spline 接缝平滑滤波
    # =========================================================================
    # 设定核尺寸，7x7x7 在处理 Patch 接缝时通常效果最好
    padding = kernel_size // 2
    # 获取动态生成的 3D B-spline 卷积核
    bspline_kernel = get_dynamic_bspline_kernel_3d(kernel_size, device)
    # 对初步融合的全局形变场进行平滑处理 (groups=3 保证 X,Y,Z 方向相互独立)
    smooth_flow = nnf.conv3d(raw_global_flow, bspline_kernel, padding=padding, groups=3)
    # 生成接缝掩膜：只要覆盖次数大于 1，说明该体素由多个 Patch 拼接而成，属于接缝区域
    seam_mask = (global_overlap_count > 1).expand_as(raw_global_flow)
    # 混合替换：接缝处采用 B-spline 平滑结果，非接缝处（中心区域）完美保留网络原始输出
    final_global_flow = torch.where(seam_mask, smooth_flow, raw_global_flow)
    # =========================================================================
    print(

        f"Flow Shape: {final_global_flow.shape}, Max: {torch.max(final_global_flow):.4f}, Min: {torch.min(final_global_flow):.4f}")

    if final_global_flow.shape[-1] == 3:

        final_global_flow = final_global_flow.permute(0, 4, 1, 2, 3)

    # 5. 将拼接好的相对位移 (Flow) 加到全局基准网格上

    final_grid = get_identity_grid(grid_shape, device)

    final_grid[:, 0] += final_global_flow[:, 2] * 2 / grid_shape[2]

    final_grid[:, 1] += final_global_flow[:, 1] * 2 / grid_shape[1]

    final_grid[:, 2] += final_global_flow[:, 0] * 2 / grid_shape[0]

    return final_grid, final_global_flow


def stage_reg(image_pairs: tuple, affine_bool=True, config=None):

    if config is not None:
        predictor = Pred(config['model_config'])
    else:
        predictor = Pred()

    # 分块进行配准
    flow, pixel_grid = patch_registration(image_pairs[0], image_pairs[1], predictor=predictor,
                                    grid_shape=image_pairs[0].shape,
                                    patch_shape= (64, 256, 256), affine_bool=affine_bool)
    pixel_grid = sitk.Cast(sitk.GetImageFromArray(np.array(pixel_grid.permute(0, 2, 3, 4, 1)[0]).astype(np.float32)),
                           sitk.sitkVectorFloat32)
    aligned_image, displacement_field = resample_sitk(image_pairs[1], pixel_grid,
                                  ref_image=[i for i in image_pairs[0].shape].reverse(), scale_factor=(1, 1, 1))

    print(pixel_grid.GetSize())

    return aligned_image, pixel_grid


def multi_stage_reg_after_sample(image_root_pairs: tuple, save_root, namas_list:list, config=None, target_shape=None):
    """generate downsample image before registration"""
    os.makedirs(save_root, exist_ok=True)
    downsample_shape = []
    fields_list = []
    scale_factor_list = []
    for name_i, nama in enumerate(namas_list):
        print('====== Aligning Image {} ====='.format(nama))
        st = time.time()
        if name_i <len(namas_list)-1:
            affine_bool = True
        else:
            affine_bool = False
        scales_list = []
        fixed_image = tifffile.imread(os.path.join(image_root_pairs[0], nama))
        moving_image = tifffile.imread(os.path.join(image_root_pairs[1], nama))
        for i in range(len(downsample_shape)):
            scale = [fixed_image.shape[j] / downsample_shape[i][j] for j in range(len(fixed_image.shape))][::-1]
            scales_list.append(scale)
        displacement_field = field_transformer(fields_list, scales_list)

        init_moving_image = sitk.Resample(sitk.GetImageFromArray(moving_image),
                                                                 [i for i in fixed_image.shape][::-1],
                                          displacement_field, sitk.sitkLinear)
        init_moving_image = sitk.GetArrayFromImage(init_moving_image)
        downsample_shape.append(fixed_image.shape)
        aligned_image, field = stage_reg((fixed_image, init_moving_image), affine_bool=affine_bool, config=config)

        # test
        save_bool = True
        if save_bool:
            os.makedirs(os.path.join(save_root, nama), exist_ok=True)
            tifffile.imwrite(os.path.join(save_root, 'fixed_{}'.format(nama)), fixed_image, compression='lzw')
            tifffile.imwrite(os.path.join(save_root, 'moving_{}'.format(nama)), moving_image, compression='lzw')
            tifffile.imwrite(os.path.join(save_root, 'init_moving_{}'.format(nama)), init_moving_image, compression='lzw')
            tifffile.imwrite(os.path.join(save_root, 'align_moving_{}'.format(nama)), aligned_image, compression='lzw')
        fields_list.append(field)
        if len(fields_list)>=2:
            scales_list = []
            for i in range(len(downsample_shape)):
                scale = [fixed_image.shape[j] / downsample_shape[i][j] for j in range(len(fixed_image.shape))][::-1]
                scales_list.append(scale)
            displacement_field = field_transformer(fields_list, scales_list)
            field = sitk.TransformToDisplacementField(displacement_field,
                                                      size=list(fixed_image.shape)[::-1], )
            # print(field.GetSize())
            sitk.WriteImage(field, os.path.join(save_root, 'field_{}.mha'.format(nama)))
        print('Aligning Image {} test {}s'.format(nama, time.time() - st))
        # if target_shape is None:
        target_shape = fixed_image.shape
        del fixed_image, moving_image, init_moving_image, aligned_image
        gc.collect()

    scales_list = []
    for i in range(len(downsample_shape)):
        scale = [target_shape[j] / downsample_shape[i][j] for j in range(len(target_shape))][::-1]
        scales_list.append(scale)

    displacement_field = field_transformer(fields_list, scales_list)
    field = sitk.TransformToDisplacementField(displacement_field,
                                              size=list(target_shape)[::-1],)
    # print(field.GetSize())
    sitk.WriteImage(field, os.path.join(save_root, 'field.mha'))


def multi_stage_reg_with_sample(image_root_pairs: tuple, save_root, downsample_list:list, config=None, target_shape=None,
                                moving_path_list:list=None, data_image_shape=(64, 256, 256), other_channels_paths=None):
    os.makedirs(save_root, exist_ok=True)
    downsample_shape_list = []
    fields_list = []
    scale_factor_list = []
    fixed_raw_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_root_pairs[0])))
    moving_raw_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_root_pairs[1])))
    for downsample_scale in downsample_list:
        print('====== Aligning Image {} ====='.format(downsample_scale))
        st = time.time()
        scales_list = []
        if downsample_scale == 1:
            fixed_image = fixed_raw_image
            moving_image = moving_raw_image
            affine_bool = False
        else:
            if downsample_scale == 0:
                downsample_shape = data_image_shape
                image_scale = [downsample_shape[i] / fixed_raw_image.shape[i] for i in range(len(data_image_shape))]
            else:
                # image_scale = [downsample_scale for i in range(len(data_image_shape))]
                # print(downsample_scale[i])
                downsample_shape = [max(int(fixed_raw_image.shape[i] * downsample_scale), data_image_shape[i]) for i in range(len(data_image_shape))]
                image_scale = [downsample_shape[i] / fixed_raw_image.shape[i] for i in range(len(data_image_shape))]
            affine_bool = True
            fixed_image = image_resample(fixed_raw_image, image_scale[::-1], downsample_shape[::-1])

            moving_image = image_resample(moving_raw_image, image_scale[::-1], downsample_shape[::-1])
            #$ tifffile.imwrite(os.path.join(save_root, 'fixed_{}'.format(downsample_scale)), fixed_image)
            # tifffile.imwrite(os.path.join(save_root, 'moving_{}'.format(downsample_scale)), moving_image)

        for i in range(len(downsample_shape_list)):
            scale = [fixed_image.shape[j] / downsample_shape_list[i][j] for j in range(len(fixed_image.shape))][::-1]
            scales_list.append(scale)
        displacement_field = field_transformer(fields_list, scales_list)

        init_moving_image = sitk.Resample(sitk.GetImageFromArray(moving_image),
                                                                 [i for i in fixed_image.shape][::-1],
                                          displacement_field, sitk.sitkLinear)
        init_moving_image = sitk.GetArrayFromImage(init_moving_image)
        downsample_shape_list.append(fixed_image.shape)
        aligned_image, field = stage_reg((fixed_image, init_moving_image), affine_bool=affine_bool, config=config)
        save_bool = True
        if save_bool:
            save_path = os.path.join(save_root, '{}'.format(os.path.split(image_root_pairs[1])[-1][:-4]))
            os.makedirs(save_path, exist_ok=True)
            tifffile.imwrite(os.path.join(save_path, 'fixed_{}.tif'.format(downsample_scale)), fixed_image, compression='lzw')
            tifffile.imwrite(os.path.join(save_path, 'moving_{}.tif'.format(downsample_scale)), moving_image, compression='lzw')
            tifffile.imwrite(os.path.join(save_path, 'init_moving_{}.tif'.format(downsample_scale)), init_moving_image, compression='lzw')
            tifffile.imwrite(os.path.join(save_path, 'align_moving_{}.tif'.format(downsample_scale)), aligned_image, compression='lzw')
        fields_list.append(field)

        # if target_shape is None:
        print('Aligning Image {} test {}s'.format(downsample_scale, time.time() - st))
        target_shape = fixed_image.shape
        del fixed_image, moving_image, init_moving_image, aligned_image
        gc.collect()

    scales_list = []
    for i in range(len(downsample_shape_list)):
        scale = [target_shape[j] / downsample_shape_list[i][j] for j in range(len(target_shape))][::-1]
        scales_list.append(scale)

    displacement_field = field_transformer(fields_list, scales_list)
    field = sitk.TransformToDisplacementField(
        displacement_field,
        size=list(fixed_raw_image.shape)[::-1],
    )

    print(field.GetSize())  # 此时尺寸就会与参考图像完全一致
    sitk.WriteImage(field, os.path.join(save_root, 'field.mha'))
    fixed_raw_image = sitk.GetImageFromArray(fixed_raw_image)
    moving_raw_image = sitk.GetImageFromArray(moving_raw_image)
    sitk.WriteImage(fixed_raw_image,
                    os.path.join(save_root, '{}'.format(os.path.split(image_root_pairs[0])[-1])))
    sitk.WriteImage(sitk.Resample(moving_raw_image, fixed_raw_image, displacement_field, sitk.sitkLinear),
                    os.path.join(save_root, '{}'.format(os.path.split(image_root_pairs[1])[-1])))
    for channel_path in other_channels_paths:

        channel_image = sitk.ReadImage(channel_path)
        warped_channel_image = sitk.Resample(channel_image, fixed_raw_image, displacement_field, sitk.sitkLinear)
        sitk.WriteImage(warped_channel_image,
                        os.path.join(save_root, '{}'.format(os.path.split(channel_path)[-1])))
        del channel_image, warped_channel_image
        gc.collect()


if __name__ == '__main__':
    root_patch_pairs = (
        r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Atlas_Data\Brain\WT5.tif',
        r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Atlas_Data\Brain\WT2.tif'
    )
    downsample_list = [0, 1]
    save_root = r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Atlas_Data\Brain\WT2_Reg'
    other_channels_paths = [
        # r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\MultiRoundData\R2\R2_402nm_10X.tif',
    ]
    multi_stage_reg_with_sample(root_patch_pairs, save_root, downsample_list, other_channels_paths=other_channels_paths)

    image_root_pairs = [
        r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\Raw\WT2\R1_4.0_Downsample',
        r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\Raw\WT2\R2_4.0_Downsample'
    ]
    names_list = ['image_mapping.tif', 'image_0.0625.tif', 'image_0.125.tif', 'image_0.25.tif', 'image_1.0.tif']
    save_root = r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\Raw\WT2\R2_aligned_2'
    # multi_stage_reg_after_sample(image_root_pairs, save_root, names_list)


    # 图谱
    name_lists = ['{:03d}'.format(i) for i in range(1, 52)]
    for name in tqdm(name_lists):
        # print(name)
        root_patch_pairs = (
            r'Q:\WangH_lab\BML_data\CellTypeMap\Mouse\MouseBrain_FiveGenes_TwoRounds_202608\20260814_WJY_MHW_WT5_R1_1\Reconstruction\SliceImage\4.0\MHW_WT5_1_{}_405nm_10X.tif'.format(name),
            r'Q:\WangH_lab\BML_data\CellTypeMap\Mouse\MouseBrain_FiveGenes_TwoRounds_202608\20260821_WJY_MHW_WT5_R2_1\Reconstruction\SliceImage\4.0\MHW_WT5_R5_{}_405nm_10X.tif'.format(name),
        )
        downsample_list = [0, 0.125, 0.25, 0.5, 1]
        save_root = r'Q:\WangH_lab\BML_data\CellTypeMap\Mouse\MouseBrain_FiveGenes_TwoRounds_202608\20260821_WJY_MHW_WT5_R2_1\Reconstruction\SliceImage\4.0_aligned'
        other_channels_paths = [
            r'Q:\WangH_lab\BML_data\CellTypeMap\Mouse\MouseBrain_FiveGenes_TwoRounds_202608\20260821_WJY_MHW_WT5_R2_1\Reconstruction\SliceImage\4.0\MHW_WT5_R5_{}_640nm_10X.tif'.format(name), ]
        # multi_stage_reg_with_sample(root_patch_pairs, save_root, downsample_list, other_channels_paths=other_channels_paths)