import os

import numpy
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class AmosTrainDataset(Dataset):
    def __init__(self, data_root, modality_ct, modality_mr, n_slices=3,image_transform=None,label_transform=None):
        self.modality_ct = modality_ct.lower()  # ct图像或mri图像
        self.modality_mr = modality_mr.lower()  # ct图像或mri图像
        assert self.modality_ct in 'ct' and self.modality_mr in 'mr'
        self.n_slices = n_slices  # 一次要取的切片数量
        self.image_transform = image_transform
        self.label_transform = label_transform

        self.ct_data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.modality_ct), 'r')
        self.ct_image_nums = self.ct_data_file[self.modality_ct].shape[0]  # 病人数量
        self.ct_slice_nums = self.ct_data_file[self.modality_ct].shape[1] - n_slices + 1  # 每个病人的切片数量
        self.ct_length = self.ct_image_nums * self.ct_slice_nums  # 训练集长度(病人数量*每个病人的切片数量)

        self.mr_data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.modality_mr), 'r')
        self.mr_image_nums = self.mr_data_file[self.modality_mr].shape[0]  # 病人数量
        self.mr_slice_nums = self.mr_data_file[self.modality_mr].shape[1] - n_slices + 1  # 每个病人的切片数量
        self.mr_length = self.mr_image_nums * self.mr_slice_nums  # 训练集长度(病人数量*每个病人的切片数量)


    def __getitem__(self, index):
        ct_image_index = index // self.ct_slice_nums  # 第几个病人
        ct_slice_index = index % self.ct_slice_nums  # 第几份切片

        mr_image_index = index // self.mr_slice_nums  # 第几个病人
        mr_slice_index = index % self.mr_slice_nums  # 第几份切片

        ct_image = self.ct_data_file[self.modality_ct][ct_image_index, ct_slice_index:ct_slice_index + self.n_slices]
        # ct_image = np.pad(ct_image, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=-1)
        if self.image_transform is not None:
            # print(f"Before transform: {ct_image.shape}")
            # ct_image = ct_image.transpose((1, 2, 0))
            # print(f"transpose:{ct_image.shape}")
            ct_image = self.image_transform(ct_image.transpose((1, 2, 0)))
            # print(f"After transform: {ct_image.shape}")
            # ct_image = ct_image.permute(2, 0, 1)  # 调整维度顺序为 (3, 240, 320)
            # print(f"After permute: {ct_image.shape}")
        mr_image = self.mr_data_file[self.modality_mr][mr_image_index, mr_slice_index:mr_slice_index + self.n_slices]
        # mr_image = np.pad(mr_image, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=-1)
        if self.image_transform is not None:
            mr_image = self.image_transform(mr_image.transpose((1, 2, 0)))
        data_item = {self.modality_ct: ct_image, self.modality_mr: mr_image}

        if f'{self.modality_mr}_seg' in self.mr_data_file:
            label = self.mr_data_file[f'{self.modality_mr}_seg'][mr_image_index, mr_slice_index:mr_slice_index + self.n_slices]
            # label = np.pad(label, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=0)
            if self.label_transform is not None:
                label = self.label_transform(label.transpose((1, 2, 0)).astype(numpy.float32))
                data_item[f'{self.modality_mr}_seg'] = label.type(torch.LongTensor)
            else:
                data_item[f'{self.modality_mr}_seg'] = label.astype(np.int64)

        if f'{self.modality_ct}_seg' in self.ct_data_file:
            label = self.ct_data_file[f'{self.modality_ct}_seg'][ct_image_index,ct_slice_index:ct_slice_index+self.n_slices]
            # label = np.pad(label, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=0)
            if self.label_transform is not None:
                label = self.label_transform(label.transpose((1, 2, 0)).astype(numpy.float32))
                data_item[f'{self.modality_ct}_seg'] = label.type(torch.LongTensor)
            else:
                data_item[f"{self.modality_ct}_seg"] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.mr_length


class AmosTestDataset(Dataset):
    def __init__(self, data_root, modality1, n_slice = 3,image_transform=None):
        self.modality1 = modality1.lower()  # ct图像或mri图像
        self.n_slice = n_slice
        # assert self.modality1 in 'ct' or 'mri'
        self.image_transform = image_transform

        self.data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.modality1), 'r')
        self.image_nums = self.data_file[self.modality1].shape[0]  # 病人数量
        self.slice_nums = self.data_file[self.modality1].shape[1] - self.n_slice + 1  # 每个病人的切片数量
        self.length = self.image_nums * self.slice_nums  # 验证集或测试集长度(病人数量)

    def __getitem__(self, index):
        image_index = index // self.slice_nums  # 第几个病人
        slice_index = index % self.slice_nums  # 第几份切片

        image = self.data_file[self.modality1][image_index,slice_index:slice_index+self.n_slice]
        # image = np.pad(image, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=-1)
        if self.image_transform is not None:
            image = self.image_transform(image)
        data_item = {self.modality1: image}

        if f'{self.modality1}_seg' in self.data_file:
            label = self.data_file[f'{self.modality1}_seg'][image_index,slice_index:slice_index+self.n_slice]
            # label = np.pad(label, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=0)
            data_item['seg'] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.length