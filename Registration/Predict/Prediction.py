import os
import gc
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import tifffile
import torch

from Process.FieldResample import upscale_affine_matrix
from Registration.Baser.Baser import Baser
from Registration.Baser.Loss import NCC, Smooth

def get_pred_config(args):
    """集中管理预测参数，动态接收命令行参数"""
    # 动态生成模型保存的根目录
    model_root = os.path.join(args.model_dir, args.name)

    config = {
        'device': args.device,
        'model_epoch': args.model_epoch,
        'optimizer_param': {
            'max_epochs': 500,  # 预测时用不到，但保持字典结构完整
            'learning_rate': 0.0001
        },
        'dataset_param': {
            'batch_size': args.batch_size,
            'shuffle': False,  # 预测通常不需要shuffle
            'num_workers': args.num_workers,
        },
        'network_param': {
            'in_channels': 1,
            'enc_channels': 8,
            'dec_channels': 16,
            'downsample_stride': args.downsample_stride,
            'kernel_size': args.kernel_size,
        },
        'decoder_layer': 3,
        'root': {
            'model_root': model_root,
            # 'dataset_root' 如果仅仅是Baser初始化需要，可以暂借 data_dir
            'dataset_root': args.data_dir, 
            'pred_data_root': args.data_dir,
            'pred_save_root': args.save_dir,
        }
    }
    return config

def init_model(config):
    print("=== Prediction Parameters ===")
    for k, v in config.items():
        print(f"{k}: {v}")

    # 实例化引擎并执行预测
    tra = Baser(config)
    tra.load_network(config['root']['model_root'], config['model_epoch'])
    return tra

class Pred:
    def __init__(self, config):
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
                registrated_image, affine = self.baser.network(fixed_image, moving_image)
                registrated_image = registrated_image[0].cpu().numpy()[0, 0]
            return registrated_image, affine[0].cpu()

    def predict_dataset(self):
        config = self.config
        os.makedirs(config['root']['pred_save_root'], exist_ok=True)
        
        image_name_list = sorted(
            [n for n in os.listdir(config['root']['pred_data_root']) if '_1.tif' in n])
            
        for image_name in tqdm(image_name_list):
            fixed_image = tifffile.imread(os.path.join(config['root']['pred_data_root'], image_name))
            moving_image = tifffile.imread(os.path.join(config['root']['pred_data_root'], image_name[:5] + '_2.tif'))
            
            with torch.no_grad():
                ncc = NCC()
                # 转换 tensor 变量名避免与 numpy array 冲突
                fixed_tensor = torch.from_numpy(fixed_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                    self.config['device'])
                moving_tensor = torch.from_numpy(moving_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                    self.config['device'])
                    
                ncc_1 = ncc.metric(fixed_tensor, moving_tensor)
                registrated_tensor, affine = self.baser.network(fixed_tensor, moving_tensor)
                ncc_2 = ncc.metric(fixed_tensor, registrated_tensor[0])
                
                # 使用 .item() 格式化输出张量标量
                print('{} before align: {:.4f}, after align: {:.4f}.'.format(image_name, ncc_1.item(), ncc_2.item()))
                registrated_image = registrated_tensor[0].cpu().numpy()[0, 0]
                
            tifffile.imwrite(os.path.join(config['root']['pred_save_root'], image_name[:5] + '_2.tif'),
                             np.clip(registrated_image, 0, 65535).astype(np.uint16))
            shutil.copy(os.path.join(config['root']['pred_data_root'], image_name),
                        os.path.join(config['root']['pred_save_root'], image_name))
            
            # 确保保存 numpy array 而非 tensor
            affine_np = affine[0].cpu().numpy() if isinstance(affine[0].cpu(), torch.Tensor) else affine[0].cpu()
            np.save(os.path.join(config['root']['pred_save_root'], image_name[:5] + '_2.npy'), affine_np)

            del fixed_image, moving_image, registrated_image, fixed_tensor, moving_tensor, registrated_tensor
            gc.collect()

    def pred_affine_field(self, fixed_image, moving_image, target_shape):
        _, small_affine_matrix = self.predict_image((np.clip(fixed_image, 90, 65535) - 90).astype(np.float32),
                                                    (np.clip(moving_image, 90, 65535) - 90).astype(np.float32))
        
        # 确保传入 upscale_affine_matrix 的是 Tensor
        if isinstance(small_affine_matrix, np.ndarray):
            small_affine_matrix = torch.from_numpy(small_affine_matrix)
            
        affine_matrix = upscale_affine_matrix(small_affine_matrix,
                                              fixed_image.shape, target_shape)
        target_shape_t = tuple([1, 1] + list(target_shape))
        affine_grid = torch.nn.functional.affine_grid(affine_matrix, target_shape_t, align_corners=True)
        return affine_grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BRIDGE (SwinB-Net) Inference Script")
    
    # 实验名称与设备
    parser.add_argument('--name', type=str, default='mcanet_bspl_down1_cnn7', help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda:1', help='Computing device')
    parser.add_argument('--model_epoch', type=int, default=100, help='Epoch of the model checkpoint to load')
    
    # 网络预测参数
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of dataloader workers')
    parser.add_argument('--downsample_stride', type=int, default=1, help='Network downsample stride')
    parser.add_argument('--kernel_size', type=int, default=7, help='Network kernel size')
    
    # 数据集与保存路径 (默认值已转为相对路径，适配开源)
    parser.add_argument('--model_dir', type=str, 
                        default='./results/models', 
                        help='Root directory where models are saved')
    parser.add_argument('--data_dir', type=str, 
                        default='./data/Validation/WT2_R123_256_v2_nor', 
                        help='Path to the prediction dataset')
    parser.add_argument('--save_dir', type=str, 
                        default='./results/Validation/predict_output', 
                        help='Directory to save prediction outputs')

    args = parser.parse_args()
    
    # 将 args 转为 config 传入类中
    config = get_pred_config(args)
    prediction = Pred(config)
    prediction.predict_dataset()
