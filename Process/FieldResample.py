import torch


def upscale_affine_matrix(T_small_3x4, orig_shape, small_shape):
    """
    将小图的 PyTorch 仿射矩阵 (B, 3, 4) 换算为原图的仿射矩阵
    参数:
        T_small_3x4: (B, 3, 4) 小尺寸图上预测的仿射矩阵
        orig_shape: (Z, Y, X) 原图尺寸
        small_shape: (Z, Y, X) 下采样图尺寸
    """
    device = T_small_3x4.device

    # 确保是 3 维张量 (B, 3, 4)
    if T_small_3x4.dim() == 2:
        T_small_3x4 = T_small_3x4.unsqueeze(0)
    b = T_small_3x4.shape[0]

    # 1. 补全为 4x4 齐次矩阵 (B, 4, 4)
    # 底部加上 [0, 0, 0, 1]
    T_4x4 = torch.zeros((b, 4, 4), dtype=torch.float32, device=device)
    T_4x4[:, :3, :4] = T_small_3x4
    T_4x4[:, 3, 3] = 1.0

    # 2. 计算放大倍数
    # 注意：PyTorch affine_grid 的坐标顺序是 (X, Y, Z)
    # 如果你的 shape 输入是 (Z, Y, X)，一定要像下面这样反着对应！
    S_z = orig_shape[0] / small_shape[0]
    S_y = orig_shape[1] / small_shape[1]
    S_x = orig_shape[2] / small_shape[2]

    # 构造缩放矩阵 S 及其逆矩阵 S_inv (支持 batch 运算)
    S = torch.tensor([
        [S_x, 0, 0, 0],
        [0, S_y, 0, 0],
        [0, 0, S_z, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1, -1)

    S_inv = torch.tensor([
        [1 / S_x, 0, 0, 0],
        [0, 1 / S_y, 0, 0],
        [0, 0, 1 / S_z, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1, -1)

    # 3. 矩阵乘法换算: T_orig = S * T_4x4 * S_inv
    # 使用 torch.bmm 进行 batch 矩阵相乘
    T_orig_4x4 = torch.bmm(S, torch.bmm(T_4x4, S_inv))

    # 4. 切割回 (B, 3, 4) 送给 PyTorch affine_grid
    T_orig_3x4 = T_orig_4x4[:, :3, :4]

    return T_orig_3x4