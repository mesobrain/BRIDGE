import os
import re
import csv
import numpy as np
import tifffile as tiff  # 用于读取 3D TIFF 图像 (pip install tifffile)
from skimage.metrics import structural_similarity as ssim


def calculate_ncc(array1, array2):
    """计算归一化互相关 (ZNCC)，支持 1D/2D/3D"""
    a = np.asarray(array1, dtype=np.float64).flatten()
    b = np.asarray(array2, dtype=np.float64).flatten()

    a_zero_mean = a - np.mean(a)
    b_zero_mean = b - np.mean(b)

    numerator = np.sum(a_zero_mean * b_zero_mean)
    denominator = np.sqrt(np.sum(a_zero_mean ** 2) * np.sum(b_zero_mean ** 2))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_mi(array1, array2, bins=256):
    """计算互信息 (Mutual Information)，支持 3D"""
    a = np.asarray(array1).flatten()
    b = np.asarray(array2).flatten()

    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    mi_value = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi_value


def calculate_folding_ratio_3d(displacement_field):
    """
    计算 3D 位移场的雅可比行列式折叠率 (|J| <= 0 的比例)
    假设 displacement_field 的 shape 为 (3, D, H, W)
    通道 0, 1, 2 分别代表 z, y, x 方向的位移
    """
    # np.gradient 会分别返回沿轴 0(z), 轴 1(y), 轴 2(x) 的偏导数
    dz_dz, dz_dy, dz_dx = np.gradient(displacement_field[0])
    dy_dz, dy_dy, dy_dx = np.gradient(displacement_field[1])
    dx_dz, dx_dy, dx_dx = np.gradient(displacement_field[2])

    # 构建 3x3 雅可比矩阵 J = I + \nabla u
    J11 = dz_dz + 1.0
    J12 = dz_dy
    J13 = dz_dx

    J21 = dy_dz
    J22 = dy_dy + 1.0
    J23 = dy_dx

    J31 = dx_dz
    J32 = dx_dy
    J33 = dx_dx + 1.0

    # 计算 3x3 矩阵的行列式: det(J)
    det_J = (J11 * (J22 * J33 - J23 * J32) -
             J12 * (J21 * J33 - J23 * J31) +
             J13 * (J21 * J32 - J22 * J31))

    # 统计折叠比例 (det_J <= 0)
    folding_voxels = np.sum(det_J <= 0)
    total_voxels = det_J.size
    folding_ratio = (folding_voxels / total_voxels) * 100.0

    return folding_ratio


def calc_metrics_dataset(image_root, flow_root, save_root):
    """
    遍历目录，配对计算 3D 图像的 NCC, MI, SSIM 以及形变场的 Folding Ratio 指标
    """
    os.makedirs(save_root, exist_ok=True)

    if not os.path.exists(image_root):
        print(f"错误: 找不到输入目录 '{image_root}'")
        return

    all_files = sorted(os.listdir(image_root))
    pattern = re.compile(r'^(.*)_1\.(tif|tiff)$', re.IGNORECASE)

    results = []

    for file in all_files:
        match = pattern.match(file)
        if match:
            prefix = match.group(1)
            ext = match.group(2)

            file1_name = file
            file2_name = f"{prefix}_2.{ext}"
            flow_name = f"{prefix}_2.npy"  # 假设 3D 形变场存为 .npy 格式

            file1_path = os.path.join(image_root, file1_name)
            file2_path = os.path.join(image_root, file2_name)
            flow_path = os.path.join(flow_root, flow_name) if flow_root else None

            if os.path.exists(file2_path):
                print(f"正在处理 3D 配对: {file1_name} <--> {file2_name}")
                try:
                    # 使用 tifffile 读取 3D 图像，返回的数组通常是 (D, H, W)
                    img1_arr = tiff.imread(file1_path)
                    img2_arr = tiff.imread(file2_path)

                    if img1_arr.shape != img2_arr.shape:
                        print(f"警告: 图像 {file1_name} 和 {file2_name} 尺寸不一致，跳过此对！")
                        continue

                    # --- 1. 计算图像相似度指标 ---
                    ncc_value = calculate_ncc(img1_arr, img2_arr)
                    mi_value = calculate_mi(img1_arr, img2_arr)

                    # skimage 的 SSIM 支持 3D 数组计算。
                    # 动态计算 data_range 防止 8-bit 和 16-bit 图像的计算误差
                    # d_range = img1_arr.max() - img1_arr.min()
                    # d_range = max(d_range, 1e-5)  # 防止除零
                    ssim_value = ssim(img1_arr, img2_arr, data_range=255)

                    # --- 2. 计算 3D 形变场折叠率 ---
                    # --- 2. 计算 3D 形变场折叠率 ---
                    folding_ratio = 0.0
                    if flow_root and os.path.exists(flow_path):
                        flow_arr = np.load(flow_path)

                        # 1. 处理 Batch 维度：如果存在 shape 为 (1, 3, 64, 256, 256) 的情况，去掉第0维
                        if flow_arr.ndim == 5 and flow_arr.shape[0] == 1:
                            flow_arr = flow_arr.squeeze(0)  # 变成 (3, 64, 256, 256)

                        # 2. 确保通道维度在最前面 (3, D, H, W)
                        if flow_arr.shape[-1] == 3:
                            # 应对 (D, H, W, 3) 的情况
                            flow_arr = flow_arr.transpose(3, 0, 1, 2)
                        elif flow_arr.shape[0] != 3:
                            print(f"  [警告] 形变场 {flow_name} 维度不符合 3D，当前 shape: {flow_arr.shape}")

                        # 3. 计算折叠率 (此时 flow_arr 保证是 (3, D, H, W))
                        folding_ratio = calculate_folding_ratio_3d(flow_arr)
                    else:
                        print(f"  [提示] 找不到对应的 3D 形变场文件: {flow_name}，折叠率记为 0.0")

                    results.append({
                        'image_id': prefix,
                        'image_1': file1_name,
                        'image_2': file2_name,
                        'ncc': round(ncc_value, 6),
                        'mi': round(mi_value, 6),
                        'ssim': round(ssim_value, 6),
                        'folding_ratio(%)': round(folding_ratio, 6)
                    })
                except Exception as e:
                    print(f"处理配对 {prefix} 时发生错误: {e}")
            else:
                print(f"警告: 未找到配对的 {file2_name}，已跳过。")

    csv_path = os.path.join(save_root, '{}_3d_metrics_results.csv'.format(os.path.split(image_root)[-1]))
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['image_id', 'image_1', 'image_2', 'ncc', 'mi', 'ssim', 'folding_ratio(%)']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)

        print(f"\n处理完成！共计算了 {len(results)} 对 3D 图像及其形变场。")
        print(f"结果保存至: {csv_path}")
    except Exception as e:
        print(f"保存 CSV 文件时发生错误: {e}")


if __name__ == '__main__':
    # 请确保安装了 tifffile (pip install tifffile)

    # 图像对所在的目录 (3D TIFF 格式)
    image_dir = r''
    # 结果保存目录
    save_dir = r''

    calc_metrics_dataset(image_dir, image_dir, save_dir)