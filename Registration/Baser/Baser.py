import os
import time
import csv
import gc
import torch
import torch.optim as optim
from torch.utils import data
import tifffile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 请确保模型导入路径正确
from .Loss import NCC, Smooth
from Registration.model.SwinB_Net_bspl import NICE_Conv3D
# from model.NICE_Conv3D import NICE_Conv3D
# from model.UNet import UNet
# from model.NICE_Bspline import NICE_Conv3D
# from Registration.E2ERegPipeline.Registration.model.NICE_Conv3D_3 import NICE_Conv3D
# from Registration.E2ERegPipeline.Registration.model.NICE_Conv3D_Bspline import NICE_Conv3D
from Registration.Baser.Dataset import Dataset


class Baser:
    def __init__(self, param):
        """
        初始化核心管线，所有参数均通过 param 字典传入。
        """
        self.network_param = param['network_param']
        self.device = param['device']
        self.root = param['root']
        self.optimizer_param = param['optimizer_param']
        self.dataset_param = param['dataset_param']  # 已修复全角下划线

        self.network = None
        self.optimizer = None
        self.init_network()

    def init_network(self):
        self.network = NICE_Conv3D(**self.network_param).to(self.device)
        # self.network = TranMorphBSplineNet().to(self.device)
        # print(self.network)
        """
        self.optimizer = optim.SGD(
                    self.network.parameters(), lr=self.optimizer_param['learning_rate'], momentum=0.99)
        """
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.optimizer_param['learning_rate']
        )

    def save_network(self, save_root, epoch):
        os.makedirs(save_root, exist_ok=True)
        save_file_name = os.path.join(save_root, 'last_{:03d}.pth'.format(epoch))
        torch.save(self.network.state_dict(), save_file_name)

    def load_network(self, save_root, epoch=99):
        model_path = os.path.join(save_root, 'last_{:03d}.pth'.format(epoch))
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))

    def logging(self, save_root, tran_loss_list):
        os.makedirs(save_root, exist_ok=True)
        log_path = os.path.join(save_root, 'training_log.csv')
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for train_loss in tran_loss_list:
                writer.writerow(train_loss)
        return []

    def adjust_learning_rate(self, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

    def load_dataset(self):
        dataset_params = self.dataset_param.copy()

        # 提取数据重采样参数（如果存在）
        target_shape = dataset_params.pop('target_shape', None)
        scale_factor = dataset_params.pop('scale_factor', None)

        image_list = [m for m in os.listdir(self.root['dataset_root']) if 'tif' in m]
        filename = list(set([x.split('_')[0] for x in image_list]))
        print(f"找到 {len(filename)} 个有效配准对样本。")

        train_files, val_files = train_test_split(filename, test_size=0.10, random_state=42)

        training_set = Dataset(train_files, self.root['dataset_root'], target_shape, scale_factor)
        training_generator = data.DataLoader(training_set, **dataset_params)

        validation_set = Dataset(val_files, self.root['dataset_root'], target_shape, scale_factor)
        validation_generator = data.DataLoader(validation_set, **dataset_params)

        return training_set, validation_set, training_generator, validation_generator

    def grad_step(self, batch_fixed, batch_moving, train_model=True):
        batch_fixed = batch_fixed.to(self.device)
        batch_moving = batch_moving.to(self.device)

        if train_model:
            self.optimizer.zero_grad()
            registered_image, field = self.network(batch_fixed, batch_moving)

            ncc_loss_fn = NCC()
            smooth_loss_fn = Smooth()

            train_loss = ncc_loss_fn.loss(batch_fixed, registered_image[-1]) + smooth_loss_fn(field[-1])

            if len(field) >= 2:
                for i in range(0, len(field) - 1):
                    train_loss += 0.5 ** (i + 1) * ncc_loss_fn(batch_fixed, registered_image[i])
            # print(train_loss)
            train_loss.backward()
            self.optimizer.step()
            train_dice_score = NCC.metric(batch_fixed, registered_image[-1])
            return train_loss, train_dice_score
        else:
            with torch.no_grad():
                registered_image, field = self.network(batch_fixed, batch_moving)
                val_dice_score = NCC.metric(batch_fixed, registered_image[-1])
                val_loss = 1 - val_dice_score  # 简化版的 validation loss
                return val_loss, val_dice_score

    def train_model(self, init_epoch=0):
        training_set, validation_set, training_generator, validation_generator = self.load_dataset()
        train_loss_list = []

        for epoch in range(self.optimizer_param['max_epochs']):
            epoch = epoch + init_epoch
            start_time = time.time()
            train_loss, train_dice_score = 0, 0
            val_loss, val_dice_score = 0, 0

            self.network.train()
            t_num = 0
            for batch_fixed, batch_moving in tqdm(training_generator, desc=f"Epoch {epoch + 1} Train"):
                loss, dice = self.grad_step(batch_fixed, batch_moving, train_model=True)
                train_loss += loss.item()
                train_dice_score += dice.item()
                t_num += 1

            self.network.eval()
            for batch_fixed, batch_moving in tqdm(validation_generator, desc=f"Epoch {epoch + 1} Val"):
                loss, dice = self.grad_step(batch_fixed, batch_moving, train_model=False)
                val_loss += loss.item()
                val_dice_score += dice.item()

            # 记录并打印指标
            num_train_batches = len(training_generator)
            num_val_batches = len(validation_generator)

            print(f"[{((time.time() - start_time) / 60):.2f} mins] Epoch {epoch + 1}: "
                  f"Train Loss = {train_loss / num_train_batches:.4f}, Train NCC = {train_dice_score / num_train_batches:.4f} | "
                  f"Val Loss = {val_loss / num_val_batches:.4f}, Val NCC = {val_dice_score / num_val_batches:.4f}")

            train_loss_list.append([
                f'epoch {epoch:04d}',
                train_loss / num_train_batches, train_dice_score / num_train_batches,
                val_loss / num_val_batches, val_dice_score / num_val_batches
            ])

            self.save_network(self.root['model_root'], epoch)
            train_loss_list = self.logging(self.root['model_root'], train_loss_list)

    def pred(self, epoch=199):
        os.makedirs(self.root['pred_save_root'], exist_ok=True)
        self.load_network(self.root['model_root'], epoch=epoch)
        self.network.eval()

        fixed_list = sorted([f for f in os.listdir(self.root['pred_data_root']) if '_1.tif' in f])
        total_loss = []

        with torch.no_grad():
            for fixed_name in tqdm(fixed_list, desc="Predicting"):
                base_name = fixed_name[:-6]  # 去掉 '_1.tif'

                # 预测阶段也可以考虑复用 Dataset 里的读取逻辑，这里为了保持独立简化了流程
                fixed_img = torch.from_numpy(
                    tifffile.imread(os.path.join(self.root['pred_data_root'], fixed_name)).astype(
                        np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                moving_img = torch.from_numpy(
                    tifffile.imread(os.path.join(self.root['pred_data_root'], f"{base_name}_2.tif")).astype(
                        np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

                registered_image, field = self.network(fixed_img, moving_img)
                loss = NCC.metric(fixed_img, registered_image[-1]).item()
                total_loss.append(loss)

                # 保存结果
                res_img = np.clip(registered_image[-1].cpu().squeeze().numpy(), 0, 65535).astype(np.uint16)
                res_field = field[-1].cpu().squeeze().numpy()

                tifffile.imwrite(os.path.join(self.root['pred_save_root'], f"{base_name}_3.tif"), res_img)
                tifffile.imwrite(os.path.join(self.root['pred_save_root'], f"{base_name}_4.tif"), res_field)

                del fixed_img, moving_img, registered_image, field
                gc.collect()

        print(f"Mean Predict NCC Score: {np.mean(total_loss):.4f}")

    def predict_image(self, fixed_image, moving_image):
        with torch.no_grad():
            fixed_image = torch.from_numpy(fixed_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                self.config['device'])
            moving_image = torch.from_numpy(moving_image.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).to(
                self.config['device'])
            registrated_image, affine = self.baser.network(fixed_image, moving_image, )
            registrated_image = registrated_image[0].cpu().numpy()[0, 0]
        return registrated_image, affine[0].cpu()