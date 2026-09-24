import os
import SimpleITK as sitk
import tifffile
import torch
import numpy as np



def resample_sitk(moving_image, displace_field, ref_image=None, scale_factor=(1, 1, 1)):
    moving_image = sitk.GetImageFromArray(np.array(moving_image))
    print(moving_image.GetSize())
    # displace_field = np.array(displace_field).astype(np.float64)
    if isinstance(displace_field, np.ndarray):
        displace_field = sitk.GetImageFromArray(displace_field)
        displace_field = sitk.Cast(displace_field, sitk.sitkVectorFloat64)
    elif isinstance(displace_field, torch.Tensor):
        displace_field = sitk.GetImageFromArray(np.array(displace_field))
        displace_field = sitk.Cast(displace_field, sitk.sitkVectorFloat64)

    displace_field = sitk.Compose(sitk.VectorIndexSelectionCast(displace_field, 2),
                                  sitk.VectorIndexSelectionCast(displace_field, 1),
                                  sitk.VectorIndexSelectionCast(displace_field, 0))

    displace_field.SetSpacing(scale_factor)
    print(displace_field.GetSize())
    # displace_field.SetSpacing(param_scale)
    tf2 = sitk.DisplacementFieldTransform(displace_field)
    print(displace_field.GetSize())
    if ref_image is None:
        ref_image = moving_image
    dst = sitk.Resample(moving_image, ref_image, tf2, sitk.sitkLinear)
    return sitk.GetArrayFromImage(dst), displace_field
    # sitk.WriteImage(dst, os.path.join(save_root, 'dst.tif'))

def resample_test(moving_image_path, displace_field_path, save_root):
    os.makedirs(save_root, exist_ok=True)
    moving_image = tifffile.imread(moving_image_path)
    moving_image = moving_image[:, 128:128+512, 128:128+512, ]
    moving_image = sitk.GetImageFromArray(moving_image)
    print(moving_image.GetSize())
    displace_field = tifffile.imread(displace_field_path).astype(np.float64)
    displace_field = sitk.GetImageFromArray(displace_field)
    print(displace_field.GetSize())
    displace_field = sitk.Compose(sitk.VectorIndexSelectionCast(displace_field, 0),
                                  sitk.VectorIndexSelectionCast(displace_field, 1),
                                  sitk.VectorIndexSelectionCast(displace_field, 2))

    # displace_field.SetSpacing((scale, scale, scale))
    # displace_field.SetSpacing(param_scale)
    tf2 = sitk.DisplacementFieldTransform(displace_field)
    dst = sitk.Resample(moving_image, moving_image, tf2, sitk.sitkLinear)
    sitk.WriteImage(dst, os.path.join(save_root, 'dst.tif'))



if __name__ == '__main__':
    moving_image_root = r'Q:\WangH_lab\BML_data\Pathology\Medulloblastoma\856985\R4\Reconstruction\SliceImage\multi_round_parameter\test\EXM_856985_R4_006_405nm_10X_[1024, 1536]_[1024, 1536].tif'
    displace_field_path = r'Q:\WangH_lab\BML_data\Pathology\Medulloblastoma\856985\R4\Reconstruction\SliceImage\multi_round_parameter\test\EXM_856985_R4_006_405nm_10X_[1024, 1536]_[1024, 1536]_field.tif'
    save_root = r'Q:\WangH_lab\BML_data\Pathology\Medulloblastoma\856985\R4\Reconstruction\SliceImage\multi_round_parameter\test'
    resample_test(moving_image_root, displace_field_path, save_root)