import gc
import shutil

import numpy as np
import os
import tifffile


def normalize_3d_percentile(img, prec=[5, 95], min_thre=[96, 300], image_range=256):
    """
    3D图像 1-95%归一化 + 阈值约束
    img: numpy array (D, H, W)
    return: normalized image [0,1]
    """
    img = img.astype(np.float32)
    # 1️⃣ 计算百分位
    p1 = np.percentile(img, prec[0])
    p95 = np.percentile(img, prec[1])

    # 2️⃣ 加约束
    min_val = max(p1, min_thre[0])
    max_val = max(p95, min_thre[1])

    # 防止除0
    if max_val - min_val < 1e-6:
        return np.zeros_like(img, dtype=np.float32)

    # 3️⃣ 归一化
    img = (img - min_val) / (max_val - min_val)

    # 4️⃣ 截断到 [0,1]
    return np.clip(img, 0, 1) * image_range


if __name__ == '__main__':
    image_root = r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Dataset\Valiation\WT2_R123_256_v2'
    save_root = r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Dataset\Valiation\WT2_R123_256_v2_nor'
    os.makedirs(save_root, exist_ok=True)
    images_list = sorted([n for n in os.listdir(image_root) if '.tif' in n])
    for image_name in images_list:
        print(image_name)
        image = tifffile.imread(os.path.join(image_root, image_name))
        # enhance_image = np.clip(image, 96, 65535) - 96
        enhance_image = normalize_3d_percentile(image, prec=[1, 98], min_thre=[92, 256+92], image_range=255)
        tifffile.imwrite(os.path.join(save_root, image_name), enhance_image)
        del image, enhance_image
        gc.collect()
    """images_list = sorted([n for n in os.listdir(image_root) if '_2.tif' in n])
    for image_name in images_list:
        print(image_name)
        shutil.copy(os.path.join(image_root, image_name),
                    os.path.join(save_root, image_name))"""