import os
import h5py
import numpy as np
import nibabel as nib

# 定义文件夹路径和输出文件路径
folder_path = '..\\Datas\\mmwhs\\process_data_version_one\\whole_dataset\\3.clip\\ct\\test'  # !!!!!!!!!!
output_file = '..\\Datas\\mmwhs\\process_data_version_one\\whole_dataset\\4.nii_h5\\test\\unpaired_ct.h5'  # !!!!!!!!!!

# 测试集
ct_test = ['1003', '1008', '1014', '1019']
mr_test = ['1007', '1009', '1018', '1019']

# 定义类别数和标签映射
# class_nums = 8
# label_map = {0.: 0, 205.: 1, 420.: 2, 500.: 3, 820.: 4, 550.: 0, 600.: 0, 850.: 0}
# class_nums = 9
label_map = {0.: 0, 205.: 1, 420.: 2, 500.: 3, 820.: 4, 550.: 0, 600.: 0, 850.: 0, 421.: 0}

# 获取文件夹中所有.nii.gz文件的路径
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.nii.gz')]
print('file_paths:', file_paths)

# 获取相关信息
print('----------相关信息----------')
patient_nums = len(file_paths) // 2
print('patient_nums:', patient_nums)
image0 = nib.load(file_paths[0] if 'image' in file_paths[0] else file_paths[1])
image0_data = image0.get_fdata()
image_shape = image0_data.shape
print('image0_data.shape:', image_shape)
print('image0_data.dtype:', image0_data.dtype)
print('image0_data.min():', image0_data.min())
print('image0_data.max():', image0_data.max())
label0 = nib.load(file_paths[0] if 'label' in file_paths[0] else file_paths[1])
label0_data = label0.get_fdata()
label_shape = label0_data.shape
print('label0_data.shape:', label_shape)
print('label0_data.dtype:', label0_data.dtype)

# 创建一个h5文件
with h5py.File(output_file, 'w') as hf:
    # 创建一个数据集
    print('----------HDF5数据集----------')
    image_dataset = hf.create_dataset(
        name='ct' if 'ct' in folder_path else 'mr',
        shape=(patient_nums, image_shape[0], image_shape[1], image_shape[2]),
        dtype=np.float32
    )
    print('image_dataset:', image_dataset)
    label_dataset = hf.create_dataset(
        name='ct_seg' if 'ct' in folder_path else 'mr_seg',
        shape=(patient_nums, label_shape[0], label_shape[1], label_shape[2]),
        dtype=np.uint16
    )
    print('label_dataset:', label_dataset)

    i, j = 0, 0
    # 遍历文件路径
    for file_path in file_paths:
        if 'image' in file_path:
            print('----------image----------')
            patient_id = file_path[-17:-13]
            print('patient_id:', patient_id)
            print('--------------------')
            # 读取.nii.gz文件
            image = nib.load(file_path)
            image_data = image.get_fdata()
            print('image_data.shape:', image_data.shape)
            print('image_data.dtype:', image_data.dtype)
            print('image_data.min():', image_data.min())
            print('image_data.max():', image_data.max())
            # 将数据保存到数据集中
            image_dataset[i] = image_data
            i += 1

        elif 'label' in file_path:
            print('----------label----------')
            patient_id = file_path[-17:-13]
            print('patient_id:', patient_id)
            print('--------------------')
            # 读取.nii.gz文件
            label = nib.load(file_path)
            label_data = label.get_fdata()
            print('label_data.shape:', label_data.shape)
            print('label_data.dtype:', label_data.dtype)
            label_values = np.unique(label_data)
            class_nums = len(label_values)
            print('label_values:', label_values)
            print('class_nums:', class_nums)
            print('--------------------')
            mapped_label_data = np.vectorize(label_map.get)(label_data.copy()).astype(np.uint16)
            print('mapped_label_data.shape:', mapped_label_data.shape)
            print('mapped_label_data.dtype:', mapped_label_data.dtype)
            new_label_values = np.unique(mapped_label_data)
            print('new_label_values:', new_label_values)
            print('new_class_nums:', len(new_label_values))

            # 将数据保存到数据集中
            label_dataset[j] = mapped_label_data
            j += 1

print('保存完成')
