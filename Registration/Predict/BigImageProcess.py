import tifffile
import os
import numpy as np


def extract_patches_3d(volume, patch_size=(64, 64, 64), overlap=(16, 16, 16)):
    """
    Args:
        volume: numpy array (D, H, W)
        patch_size: tuple of 3 ints
        overlap: tuple of 3 ints

    Returns:
        patches: list of patches
        coords:  list of (z, y, x) start coordinates
        padded_shape: shape after padding
    """
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    oD, oH, oW = overlap

    sD = pD - oD
    sH = pH - oH
    sW = pW - oW

    # 计算需要padding多少
    pad_D = (sD - (D - pD) % sD) % sD
    pad_H = (sH - (H - pH) % sH) % sH
    pad_W = (sW - (W - pW) % sW) % sW

    volume = np.pad(volume,
                    ((0, pad_D), (0, pad_H), (0, pad_W)),
                    mode='constant')

    padded_shape = volume.shape
    Dp, Hp, Wp = padded_shape

    patches = []
    coords = []

    for z in range(0, Dp - pD + 1, sD):
        for y in range(0, Hp - pH + 1, sH):
            for x in range(0, Wp - pW + 1, sW):
                patch = volume[z:z+pD, y:y+pH, x:x+pW]
                patches.append(patch)
                coords.append((z, y, x))

    return patches, coords, padded_shape


def reconstruct_from_patches_3d(patches, coords, padded_shape,
                                patch_size=(64, 64, 64),
                                original_shape=None):
    """
    Args:
        patches: list of processed patches
        coords:  list of (z,y,x)
        padded_shape: shape after padding
        patch_size: tuple
        original_shape: original volume shape before padding

    Returns:
        reconstructed volume
    """

    pD, pH, pW = patch_size
    Dp, Hp, Wp = padded_shape

    volume = np.zeros(padded_shape, dtype=np.float32)
    weight = np.zeros(padded_shape, dtype=np.float32)

    for patch, (z, y, x) in zip(patches, coords):
        volume[z:z+pD, y:y+pH, x:x+pW] += patch
        weight[z:z+pD, y:y+pH, x:x+pW] += 1

    weight[weight == 0] = 1
    volume = volume / weight

    if original_shape is not None:
        D, H, W = original_shape
        volume = volume[:D, :H, :W]

    return volume
from tqdm import tqdm

def load_image(image_root):
    if os.path.isfile(image_root):
        image = tifffile.imread(image_root)
    else:
        names_list = sorted([n for n in os.listdir(image_root) if 'tif' in n])
        image_list = []
        for n in tqdm(names_list, desc='loading frame image '):
            image_list.append(tifffile.imread(os.path.join(image_root, n)))
        image = np.array(image_list)
    return image