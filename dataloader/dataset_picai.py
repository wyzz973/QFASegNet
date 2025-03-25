import os
import numpy as np
import h5py
from torch.utils.data import Dataset

class PicaiTrainDataset(Dataset):
    def __init__(self, data_root, source, target, n_slices=3,image_transform=None):
        self.source = source.lower()  # ct图像或mri图像
        self.target = target.lower()  # ct图像或mri图像
        assert self.source in 't2w' and self.target in 'adc'
        self.n_slices = n_slices  # 一次要取的切片数量
        self.image_transform = image_transform

        self.source_data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.source), 'r')
        self.source_image_nums = self.source_data_file[self.source].shape[0]  # 病人数量
        self.source_slice_nums = self.source_data_file[self.source].shape[1] - n_slices + 1  # 每个病人的切片数量
        self.source_length = self.source_image_nums * self.source_slice_nums  # 训练集长度(病人数量*每个病人的切片数量)

        self.target_data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.target), 'r')
        self.target_image_nums = self.target_data_file[self.target].shape[0]  # 病人数量
        self.target_slice_nums = self.target_data_file[self.target].shape[1] - n_slices + 1  # 每个病人的切片数量
        self.target_length = self.target_image_nums * self.target_slice_nums  # 训练集长度(病人数量*每个病人的切片数量)


    def __getitem__(self, index):
        source_image_index = index // self.source_slice_nums  # 第几个病人
        source_slice_index = index % self.source_slice_nums  # 第几份切片

        target_image_index = index // self.target_slice_nums  # 第几个病人
        target_slice_index = index % self.target_slice_nums  # 第几份切片

        source_image = self.source_data_file[self.source][source_image_index, source_slice_index:source_slice_index + self.n_slices]
        # ct_image = np.pad(ct_image, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=-1)
        if self.image_transform is not None:
            source_image = self.image_transform(source_image)

        target_image = self.target_data_file[self.target][target_image_index, target_slice_index:target_slice_index + self.n_slices]
        # mr_image = np.pad(mr_image, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=-1)
        if self.image_transform is not None:
            target_image = self.image_transform(target_image)
        data_item = {self.source: source_image, self.target: target_image}

        if 'seg' in self.source_data_file:
            label = self.source_data_file['seg'][source_image_index, source_slice_index:source_slice_index + self.n_slices]
            # label = np.pad(label, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=0)
            data_item['seg'] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.source_length


class PicaiTestDataset(Dataset):
    def __init__(self, data_root, source,target, n_slice = 3,image_transform=None):
        self.target = target.lower()
        self.source = source.lower()
        self.n_slice = n_slice
        assert self.target in 'adc' and self.source in 't2w'
        self.image_transform = image_transform

        self.data_file = h5py.File(os.path.join(data_root, f"paired_{self.source}_{self.target}.h5" ), 'r')
        self.image_nums = self.data_file[self.target].shape[0]  # 病人数量
        self.slice_nums = self.data_file[self.target].shape[1] - self.n_slice + 1  # 每个病人的切片数量
        self.length = self.image_nums * self.slice_nums  # 验证集或测试集长度(病人数量)

    def __getitem__(self, index):
        image_index = index // self.slice_nums  # 第几个病人
        slice_index = index % self.slice_nums  # 第几份切片

        image = self.data_file[self.target][image_index,slice_index:slice_index+self.n_slice]
        # image = np.pad(image, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=-1)
        if self.image_transform is not None:
            image = self.image_transform(image)
        data_item = {self.target: image}

        if 'seg' in self.data_file:
            label = self.data_file['seg'][image_index,slice_index:slice_index+self.n_slice]
            # label = np.pad(label, ((0, 0), (2, 2), (1, 1)), mode='constant', constant_values=0)
            data_item['seg'] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.length