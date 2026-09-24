import os
import gc
import SimpleITK as sitk
import tifffile
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as nnf
import numpy as np
# from Registration.E2ERegPipeline.Pipeline.config import Config
from Registration.E2ERegPipeline.Registration.Baser import Baser
from Registration.E2ERegPipeline.Registration.Baser import Config
from Registration.E2ERegPipeline.DatasetConstruction.ImagePatch import crop_3d_volume_with_overlap


def resample_sitk(moving_image, displace_field, ref_image=None, scale_factor=(1, 1, 1)):
    moving_image = sitk.GetImageFromArray(np.array(moving_image))
    print(moving_image.GetSize())
    # displace_field = np.array(displace_field).astype(np.float64)
    if isinstance(displace_field, np.ndarray):
        displace_field = sitk.GetImageFromArray(displace_field)
    elif isinstance(displace_field, torch.Tensor):
        displace_field = sitk.GetImageFromArray(np.array(displace_field))
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
        displacement_field = sitk.Transform(3, sitk.sitkComposite)
        for i in range(len(fields_list)):

            fields_list[i].SetSpacing(scales_list[i])
            if displacement_field is None:
                print(fields_list[i].GetPixelID(), fields_list[i].GetSize())
                displacement_field = sitk.DisplacementFieldTransform(
                    sitk.Cast(fields_list[i], sitk.sitkVectorFloat64))
            else:
                displacement_field.AddTransform(
                    sitk.DisplacementFieldTransform(
                        sitk.Cast(fields_list[i], sitk.sitkVectorFloat64)
                    ))
    return displacement_field

def patch_registration(fixed_image, moving_image, predictor,
                       grid_shape: list = (64, 256, 256),
                       patch_shape: tuple = (64, 256, 256),
                       affine_grid=None, overlap_ratio=0.1, device='cpu'):
    # 1. 准备全局的 Flow (位移) 累加器和 Weight (权重) 累加器
    # 注意：拼接必须在位移场 (Flow) 上进行，最后再转换回绝对网格 (Grid)
    global_flow = torch.zeros((1, 3, *grid_shape), device=device, dtype=torch.float32)
    global_weight = torch.zeros((1, 1, *grid_shape), device=device, dtype=torch.float32)


    # init_moving_image = unify_tensor_shape(init_moving_image, fixed_image.shape)

    # 对图像进行小块裁剪
    fixed_volume, fixed_core_coords, fixed_padding = crop_3d_volume_with_overlap(fixed_image, crop_size=patch_shape,
                                                                  overlap_ratio=overlap_ratio)
    moving_volume, moving_core_coords, moving_padding = crop_3d_volume_with_overlap(moving_image, crop_size=patch_shape,
                                                                    overlap_ratio=overlap_ratio)

    # 提前计算好插值后的 Patch 尺寸，并生成对应的 3D 汉明窗
    # scaled_patch_shape = [int(p * scale) for p in patch_shape]
    patch_window = generate_3d_hann_window(patch_shape, device).unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)

    for i in tqdm(range(len(fixed_volume)), desc='Predicting patch images'):
        # 调用网络获取局部形变场 (假设 flow 形状为 1, 3, Z, Y, X)
        _, local_flow = predictor.predict_image(np.clip(fixed_volume[i], 90, 65535)-90,
                                                np.clip(moving_volume[i], 90, 65535)-90)
        patch_identity = get_identity_grid(patch_shape, device)
        patch_identity[:, 0] += local_flow[:, 2] * 2 / patch_shape[2]
        patch_identity[:, 1] += local_flow[:, 1] * 2 / patch_shape[1]
        patch_identity[:, 2] += local_flow[:, 0] * 2 / patch_shape[0]

        # 提取当前块在大图中的映射坐标
        core_coords = [int(c) for c in moving_core_coords[i]]
        z0, z1 = max(0, core_coords[0]), min(grid_shape[0], core_coords[1])
        y0, y1 = max(0, core_coords[2]), min(grid_shape[1], core_coords[3])
        x0, x1 = max(0, core_coords[4]), min(grid_shape[2], core_coords[5])

        # 2. 核心魔法：加权累加 (Soft Blending)

        weighted_flow = local_flow * patch_window
        # weighted_flow = torch.zeros_like(local_flow) # +10
        # weighted_flow[:, 0] += 10
        global_flow[0, :, z0:z1, y0:y1, x0:x1] += weighted_flow.squeeze(0)[..., :z1-z0,
                                                                                :y1-y0,
                                                                                :x1-x0]
        global_weight[0, 0, z0:z1, y0:y1, x0:x1] += patch_window[..., :z1-z0, :y1-y0, :x1-x0].squeeze(0).squeeze(0)

    # 3. 计算加权平均，完美消除接缝
    global_weight = torch.clamp(global_weight, min=1e-5)  # 防止除以 0
    final_global_flow = global_flow / global_weight

    print(final_global_flow.shape, torch.max(final_global_flow), torch.min(final_global_flow))
    if final_global_flow.shape[-1] == 3:
        final_global_flow = final_global_flow.permute(0, 4, 1, 2, 3)
    print(torch.max(final_global_flow), torch.min(final_global_flow))
    # 4. 将拼接好的相对位移 (Flow) 加到全局基准网格上，变回 affine_grid 格式
    final_grid = get_identity_grid(grid_shape, device)
    # final_grid = global_identity
    final_grid[:, 0] += final_global_flow[:, 2] * 2 / grid_shape[2]
    final_grid[:, 1] += final_global_flow[:, 1] * 2 / grid_shape[1]
    final_grid[:, 2] += final_global_flow[:, 0] * 2 / grid_shape[0]
    # final_grid = final_grid.permute(0, 2, 3, 4, 1)  # 转换回 (1, Z, Y, X, 3) 供 grid_sample 使用

    return final_grid, final_global_flow


def stage_reg(image_pairs: tuple, config=None):

    if config is not None:
        predictor = Pred(config['model_config'])
    else:
        predictor = Pred()

    # 分块进行配准
    flow, pixel_grid = patch_registration(image_pairs[0], image_pairs[1], predictor=predictor,
                                    grid_shape=image_pairs[0].shape,
                                    patch_shape= (64, 256, 256))
    pixel_grid = sitk.Cast(sitk.GetImageFromArray(np.array(pixel_grid.permute(0, 2, 3, 4, 1)[0]).astype(np.float32)),
                           sitk.sitkVectorFloat32)
    aligned_image, displacement_field = resample_sitk(image_pairs[1], pixel_grid,
                                  ref_image=[i for i in image_pairs[0].shape].reverse(), scale_factor=(1, 1, 1))
    print(pixel_grid.GetSize())

    return aligned_image, pixel_grid


def stage_patch(image_pairs: tuple, save_root, config=None, crop_shape=(64, 256, 256), idx=0):
    os.makedirs(save_root, exist_ok=True)
    # 1. 组合为 4维 图像
    combined_volume = np.stack((image_pairs[0],
                                image_pairs[1]), axis=-1)
    num_channels = combined_volume.shape[-1]

    # 2. 执行三维裁剪
    patches, core_coords, (pad_d, pad_h, pad_w) = crop_3d_volume_with_overlap(
        combined_volume, crop_size=crop_shape, overlap_ratio=0.1
    )
    for patch in patches:
        for c in range(num_channels):
            single_channel_data = patch[..., c]
            tif_filename = f"{idx:05d}_{c + 1}.tif"
            tif_filepath = os.path.join(save_root, tif_filename)
            tifffile.imwrite(tif_filepath, single_channel_data)
            idx += 1
    return idx




if __name__ == '__main__':
    root_patch_pairs = (
        r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\Raw\WT1\R1_4.0_downsample',
        r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\Raw\WT1\R2_4.0_downsample'
    )
    names_list = ['image_mapping.tif', 'image_0.0625.tif', 'image_0.125.tif', 'image_0.25.tif']
    save_root = r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Test\results'
    multi_stage_reg_after_sample(root_patch_pairs, save_root, names_list)