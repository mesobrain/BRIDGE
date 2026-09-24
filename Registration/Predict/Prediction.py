import gc
import os
import shutil
from tqdm import tqdm
import numpy as np
import tifffile
import torch
from Process.FieldResample import upscale_affine_matrix
from Registration.Baser.Baser import Baser
from Registration.Baser.Loss import NCC, Smooth


def get_pred_config():
    """集中管理预测参数"""
    name = 'mcanet_bspl_down1_cnn7'

    config = {
        'device': 'cuda:1',
        'model_epoch':100,
        'optimizer_param': {
            'max_epochs': 500,  # 预测时用不到，但保持字典结构完整
            'learning_rate': 0.0001
        },
        'dataset_param': {
            'batch_size': 2,
            'shuffle': False,  # 预测通常不需要shuffle
            'num_workers': 6,
        },
        'network_param': {
            'in_channels': 1,
        'enc_channels': 8,
        'dec_channels': 16,
            'downsample_stride': 1,
            'kernel_size': 7,
        },
        'downsample_stride': 4,'decoder_layer': 3,
        'downsample_stride': 0,
        'kernel_size': 5,
        'root': {
            'model_root': r'Registration.model',
            'dataset_root': r'',
            'pred_data_root': r'',
            'pred_save_root': r'',
        }
    }
    return config


def init_model(config):
    number = 17
    parameter_pred = get_pred_config()

    print("=== Prediction Parameters ===")
    print(parameter_pred)

    # 实例化引擎并执行预测
    tra = Baser(parameter_pred)
    tra.load_network(config['root']['model_root'], config['model_epoch'])
    return tra


class Pred:
    def __init__(self, config=get_pred_config()):
        self.config = config
        self.baser = init_model(config)

    def predict_image(self, fixed_image, moving_image):
        if np.max(fixed_image) <= 20:
            return moving_image, torch.zeros((1, 3, fixed_image.shape[-3], fixed_image.shape[-2], fixed_image.shape[-1]))
        else:
            with torch.no_grad():
                fixed_image = torch.from_numpy(fixed_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                    self.config['device'])
                moving_image = torch.from_numpy(moving_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                    self.config['device'])
                registrated_image, affine = self.baser.network(fixed_image, moving_image, )
                registrated_image = registrated_image[0].cpu().numpy()[0, 0]
            return registrated_image, affine[0].cpu()

    def predict_dataset(self):
        config = self.config
        os.makedirs(self.config['root']['pred_save_root'], exist_ok=True)
        image_name_list = sorted(
            [n for n in os.listdir(os.path.join(config['root']['pred_data_root'])) if '_1.tif' in n])
        for image_name in tqdm(image_name_list):
            fixed_image = tifffile.imread(os.path.join(config['root']['pred_data_root'], image_name))
            moving_image = tifffile.imread(os.path.join(config['root']['pred_data_root'], image_name[:5] + '_2.tif'))
            with torch.no_grad():
                ncc = NCC()

                fixed_image = torch.from_numpy(fixed_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                    self.config['device'])
                moving_image = torch.from_numpy(moving_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                    self.config['device'])
                ncc_1 = ncc.metric(fixed_image, moving_image)
                registrated_image, affine = self.baser.network(fixed_image, moving_image)
                ncc_2 = ncc.metric(fixed_image, registrated_image[0])
                print('{} before align:{}, after align:{}.'.format(image_name, ncc_1, ncc_2))
                registrated_image = registrated_image[0].cpu().numpy()[0, 0]
                tifffile.imwrite(os.path.join(config['root']['pred_save_root'], image_name[:5] + '_2.tif'),
                                 np.clip(registrated_image, 0, 65535).astype(np.uint16))
                shutil.copy(os.path.join(config['root']['pred_data_root'], image_name),
                            os.path.join(config['root']['pred_save_root'], image_name))
                np.save(os.path.join(config['root']['pred_save_root'], image_name[:5] + '_2.npy'), affine[0].cpu())

            del fixed_image, moving_image, registrated_image
            gc.collect()

    def pred_affine_field(self, fixed_image, moving_image, target_shape):
        _, small_affine_matrix = self.predict_image((np.clip(fixed_image,
                                                                  90, 65535) - 90).astype(np.float32),
                                                         (np.clip(moving_image,
                                                                  90, 65535) - 90).astype(np.float32),
                                                    )
        affine_matrix = upscale_affine_matrix(torch.from_numpy(small_affine_matrix),
                                              fixed_image.shape, target_shape)
        target_shape = tuple([1, 1] + list(target_shape))
        affine_grid = torch.nn.functional.affine_grid(affine_matrix, target_shape, align_corners=True)
        return affine_grid


if __name__ == '__main__':
    prediction = Pred()
    prediction.predict_dataset()
    pass
