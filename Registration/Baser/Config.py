

def get_baser_config():
    """集中管理预测参数"""
    name = 'ParallelNet'

    config = {
        'device': 'cuda:1',
        'model_epoch':50,
        'optimizer_param': {
            'max_epochs': 500,  # 预测时用不到，但保持字典结构完整
            'learning_rate': 0.001
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
        },
        'root': {
            'model_root': r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\model\nice_conv3d_5_v3'.format(name),
            'dataset_root': r'R:\DebinXia\RawData\Registration\MultiRoundDataset\WT1\label_v2\R123_label_v2',
            'pred_data_root': r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Dataset\data_nor3',
            'pred_save_root': r'Q:\WangH_lab\User\Debin\RawData\Registration\E2E_Reg\E2E_Reg\Dataset\data_unet_affine',
        }
    }
    return config
