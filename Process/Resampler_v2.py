import numpy as np
import SimpleITK as sitk


def image_resample(image_in, sample_scale, image_size):

    if isinstance(image_in, np.ndarray):
        image = sitk.GetImageFromArray(image_in)
    else:
        image = image_in
    # image_size = [int(image.GetSize()[i] * sample_scale[i]) for i in range(len(image.GetSize()))]
    origin = image.GetOrigin()
    new_img = sitk.Resample(image, image_size, sitk.Transform(), sitk.sitkLinear,
                            origin, [1 / sample_scale[i] for i in range(len(image.GetSize()))],
                            image.GetDirection(), 0.0, image.GetPixelID())
    if isinstance(image_in, np.ndarray):
        return sitk.GetArrayFromImage(new_img)
    else:
        return new_img