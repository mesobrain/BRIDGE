import os
import SimpleITK as sitk
import tifffile
import numpy as np
from tqdm import tqdm


def load_brain(image_path, ref_channel='.tif'):
    image_name_list = sorted([n for n in os.listdir(image_path) if 'tif' in n and ref_channel in n])
    images_list = []
    for image_name in tqdm(image_name_list):
        images_list.append(tifffile.imread(os.path.join(image_path, image_name)))

    image = np.array(images_list)
    return image


# def brain_transformer()