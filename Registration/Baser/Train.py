import os
import argparse
import numpy as np
from Registration.E2ERegPipeline.Registration.Baser.Baser import Baser

# 固定随机种子确保复现性
def seed_worker(worker_id):
    np.random.seed(42 + worker_id)

def get_train_config(args):
    """集中管理训练参数，动态接收命令行参数"""
    # 动态生成模型保存的根目录
    model_root = os.path.join(args.model_dir, f"{args.name}_96_v2")

    config = {
        'device': args.device,
        'optimizer_param': {
            'max_epochs': args.epochs,
            'learning_rate': args.lr,
        },
        'dataset_param': {  
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers,
            'worker_init_fn': seed_worker,
            # 'target_shape': (128, 128, 128),  # 重采样尺寸
            # 'scale_factor': 0.5              # 或者使用倍数下采样
        },
        'network_param': {
            'in_channels': 1,
            'enc_channels': 8,
            'dec_channels': 16,
            'downsample_stride': 0,
            'kernel_size': 5,
        },
        'ndim': 3, 
        'in_channel': 2, 
        'out_channel': 3, 
        'image_size': (64, 256, 256),

        'root': {
            'model_root': model_root,
            'dataset_root': args.data_dir,
            # 预测所需的路径
            'pred_data_root': args.pred_data_dir,
            'pred_save_root': args.pred_save_dir
        }
    }
    return config


def training(args):
    parameter_train = get_train_config(args)
    os.makedirs(parameter_train['root']['model_root'], exist_ok=True)

    print("=== Training Parameters ===")
    for k, v in parameter_train.items():
        print(f"{k}: {v}")

    # 实例化引擎并开始训练
    tra = Baser(parameter_train)
    tra.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BRIDGE (SwinB-Net) Training Script")
    
    # 实验名称与设备
    parser.add_argument('--name', type=str, default='mcanet_bspl_v2_down421_cnn5', help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Computing device')
    
    # 核心训练超参数
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    
    # 数据集与保存路径 (默认值改为相对路径，适配开源)
    parser.add_argument('--model_dir', type=str, 
                        default='./results/models', 
                        help='Root directory to save trained models')
    parser.add_argument('--data_dir', type=str, 
                        default='./data/registration_test_2/data_test', 
                        help='Path to the training dataset')
    parser.add_argument('--pred_data_dir', type=str, 
                        default='./data/WT2_Seg/raw', 
                        help='Path to the prediction input data')
    parser.add_argument('--pred_save_dir', type=str, 
                        default='./results/WT2_Seg/predict', 
                        help='Directory to save prediction outputs')

    args = parser.parse_args()
    
    training(args)
