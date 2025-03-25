import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def save_segmentation_images(nii_file_path, output_dir):
    # 定义颜色映射
    # label_colors = np.array([
    #     (0, 0, 0),        # 0: 背景
    #     (211, 44, 31),    # 1: d32c1f
    #     (205, 140, 149),  # 2: CD8C95
    #     (67, 107, 173),   # 3: 436bad
    #     (205, 173, 0),    # 4: CDAD00
    #     (4, 244, 137),    # 5: 04f489
    #     (254, 1, 154),    # 6: fe019a
    #     (6, 71, 12),      # 7: 06470c
    #     (97, 222, 42),    # 8: 61de2a
    #     (203, 248, 95),   # 9: cbf85f
    #     (255, 187, 255),  # 10: FFBBFF
    #     (127, 255, 212),  # 11: 7FFFD4
    #     (0, 0, 255),      # 12: 0000FF
    #     (2, 204, 254),    # 13: 02ccfe
    #     (153, 0, 250),    # 14: 9900fa
    #     (93, 20, 81)      # 15: 5d1451
    # ]) / 255.0  # 将RGB值转换为0到1的范围

    label_colors = np.array([
        (0, 0, 0),        # 0: 背景
        (67, 107, 173),   # 1: #f0807f
        (254, 1, 154),    # 2: #87cefa
        (67, 107, 173),   # 3: 436bad
        (205, 173, 0),    # 4: CDAD00
        (4, 244, 137),    # 5: 04f489
        (254, 1, 154),    # 6: fe019a
        (6, 71, 12),      # 7: 06470c
        (97, 222, 42),    # 8: 61de2a
        (203, 248, 95),   # 9: cbf85f
        (255, 187, 255),  # 10: FFBBFF
        (127, 255, 212),  # 11: 7FFFD4
        (0, 0, 255),      # 12: 0000FF
        (2, 204, 254),    # 13: 02ccfe
        (153, 0, 250),    # 14: 9900fa
        (93, 20, 81)      # 15: 5d1451
    ]) / 255.0  # 将RGB值转换为0到1的范围
    # 加载.nii文件
    nii_image = nib.load(nii_file_path)
    seg_images = nii_image.get_fdata()

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(seg_images.shape)
    # 获取图像尺寸
    num_slices,img_height,img_width = seg_images.shape

    # 遍历所有切片
    for i in range(num_slices):
        slice = seg_images[i, :, :]
        color_image = np.zeros((img_height, img_width, 3), dtype=np.float32)

        # 将每个标签映射到颜色
        for label in range(len(label_colors)):
            mask = slice == label
            color_image[mask] = label_colors[label]

        # 生成文件名并保存图像
        file_name = os.path.join(output_dir, f'slice_{i}.png')
        plt.imsave(file_name, color_image)


list = [2,34,109,111,134]
for i in list:
#     print(i)
#     adc_label = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\\vquda\\adc_label\\{i}.nii.gz'
#     output_adc_label = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\adc_label\\{i}'
#     save_segmentation_images(adc_label, output_adc_label)

    # arlgan_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\\arlgan\seg\\real_target_seg\\batch_{i}-patient_0.nii.gz'
    # arlgan_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\arlgan\\{i}'
    # save_segmentation_images(arlgan_seg, arlgan_output)
    #
    # synseg_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\\Synseg\\realctseg_batch_{i}-patient_0.nii.gz'
    # synseg_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\Synseg\\{i}'
    # save_segmentation_images(synseg_seg, synseg_output)
    #
    # PSIGAN_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\PSIGAN\seg_label\\{i}.nii.gz'
    # PSIGAN_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\PSIGAN\\{i}'
    # save_segmentation_images(PSIGAN_seg, PSIGAN_output)
    #
    # SIFA_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\SIFA\seg_label\\{i}.nii.gz'
    # SIFA_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\SIFA\\{i}'
    # save_segmentation_images(SIFA_seg, SIFA_output)
    #
    # bmcan_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\\bmcan\seg_label\\{i}.nii.gz'
    # bmcan_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\bmcan\\{i}'
    # save_segmentation_images(bmcan_seg, bmcan_output)
    #
    # dada_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\dada\outputs\Seg_images\\real_CT_Seg\\{i}.nii.gz'
    # dada_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\dada\\{i}'
    # save_segmentation_images(dada_seg, dada_output)
    #
    # vquda_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\\vquda\seg_label\\{i}.nii.gz'
    # vquda_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\vquda\\{i}'
    # save_segmentation_images(vquda_seg, vquda_output)

    # c3r_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\c3r\seg_label\\{i}.nii.gz'
    # c3r_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\c3r\\{i}'
    # save_segmentation_images(c3r_seg, c3r_output)

    fpl_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\picai\\fpl+\seg_label\\{i}.nii.gz'
    fpl_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\picai\png\\fpl+\\{i}'
    save_segmentation_images(fpl_seg, fpl_output)


# c3r_seg = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\amos\\fpl+\seg_label\\57.nii.gz'
# c3r_output = f'C:\\Users\\14322\Desktop\共享文件夹\segmentation images\\amos\png\\fpl+'
# save_segmentation_images(c3r_seg, c3r_output)