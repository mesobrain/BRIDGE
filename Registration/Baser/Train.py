import os
import numpy as np
from Registration.E2ERegPipeline.Registration.Baser.Baser import Baser


# 固定随机种子确保复现性
def seed_worker(worker_id):
    np.random.seed(42 + worker_id)


def get_train_config():
    """集中管理训练参数"""
    name = 'mcanet_bspl_v2_down421_cnn5'

    config = {
        'device': 'cuda:0',
        'optimizer_param': {
            'max_epochs': 500,
            'learning_rate': 0.0001,
        },
        'dataset_param': {  # 修复了你的代码中的全角下划线
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 0,
            'worker_init_fn': seed_worker,
            # 'target_shape': (128, 128, 128),  # 在这里传入我们在 Dataset.py 中添加的重采样尺寸！
            # 'scale_factor': 0.5            # 或者使用倍数下采样
        },
        'network_param': {'in_channels': 1,
        'enc_channels': 8,
        'dec_channels': 16,
        'downsample_stride': 0,
        'kernel_size': 5,
        },
        'ndim': 3, 'in_channel': 2, 'out_channel': 3, 'image_size': (64, 256, 256),

        'root': {
            'model_root': r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\model\{}_96_v2'.format(name),
            'dataset_root': r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Dataset\registration_test_2\data_test',
            # 预测所需的路径(由于Baser初始化需要，暂留)
            'pred_data_root': r'S:\Debin\RawData\Registration\WT2_Seg\raw',
            'pred_save_root': r'S:\Debin\RawData\Registration\WT2_Seg\predict'
        }
    }
    return config


def traning():
    parameter_train = get_train_config()
    os.makedirs(parameter_train['root']['model_root'], exist_ok=True)

    print("=== Training Parameters ===")
    print(parameter_train)

    # 实例化引擎并开始训练
    tra = Baser(parameter_train)
    tra.train_model()


if __name__ == '__main__':
    traning()