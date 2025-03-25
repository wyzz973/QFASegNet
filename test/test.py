import argparse
import ast
from datetime import datetime
import itertools
import os.path

import nibabel
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from VQUDA.dataset_amos import AmosTestDataset
from VQUDA.metrics import calc_multi_dice, calc_multi_assd, calc_multi_asd
from VQUDA.version_2.config_amos import Config_amos
from VQUDA.version_2.network import (
    VQ,
    NLayerDiscriminator,
    Decoder,
    Encoder,
    EncoderSharedLayers,
    EncoderShare,
    DecoderSharedLayers,
    DecoderShare,
)
from config import Config





def run():
    opt_dict_decoder = opt.filter_config_for_network(decoder=True)
    opt_dict = opt.filter_config_for_network(decoder=False)
    # encoder_s = Encoder(**opt_dict).to(device=opt.device)
    # encoder_t = Encoder(**opt_dict).to(device=opt.device)
    # decoder_s = Decoder(**opt_dict_decoder).to(device=opt.device)
    # decoder_t = Decoder(**opt_dict_decoder).to(device=opt.device)

    encoder_shared_layers_module = EncoderSharedLayers(
        in_channel=opt.in_channel,
        ch_mult=opt.ch_mult,
        norm_type=opt.norm_type,
        act_type=opt.act_type,
        independent_layer_count=opt.independent_layer_count,
        quant_nums=opt.quant_nums,
        e_dim=opt.e_dim,
    ).cuda()
    encoder_s = EncoderShare(
        slice_nums=opt.slice_nums,
        in_channel=opt.in_channel,
        ch_mult=opt.ch_mult,
        shared_layers_module=encoder_shared_layers_module,
        independent_layer_count=opt.independent_layer_count,
    ).cuda()
    encoder_t = EncoderShare(
        slice_nums=opt.slice_nums,
        in_channel=opt.in_channel,
        ch_mult=opt.ch_mult,
        shared_layers_module=encoder_shared_layers_module,
        independent_layer_count=opt.independent_layer_count,
    ).cuda()

    decoder_shared_layers_module = DecoderSharedLayers(
        in_channel=opt.in_channel,
        ch_mult=opt.ch_mult,
        independent_layer_count=opt.independent_layer_count,
        e_dim=opt.e_dim,
        quant_nums=opt.quant_nums,
        norm_type=opt.norm_type,
        act_type=opt.act_type,
    ).cuda()
    decoder_s = DecoderShare(
        ch_mult=opt.ch_mult,
        slice_nums=opt.slice_nums,
        class_nums=opt.class_nums,
        in_channel=opt.in_channel,
        shared_layers_module=decoder_shared_layers_module,
        independent_layer_count=opt.independent_layer_count,
    ).cuda()
    decoder_t = DecoderShare(
        ch_mult=opt.ch_mult,
        slice_nums=opt.slice_nums,
        class_nums=opt.class_nums,
        in_channel=opt.in_channel,
        shared_layers_module=decoder_shared_layers_module,
        independent_layer_count=opt.independent_layer_count,
    ).cuda()
    segmentor = Decoder(**opt_dict_decoder).to(device=opt.device)

    d_s = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device=opt.device)
    d_t = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device=opt.device)
    vq = VQ(n_e=opt.n_e, e_dim=opt.e_dim, beta=opt.beta,quant_nums=opt.quant_nums).to(device=opt.device)

    test_dataset = AmosTestDataset(
        "/home/chenxu/wangyang/dataset/amos/test",
        modality1="ct",
        n_slice=opt.slice_nums,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    encoder_s.load_state_dict(
        torch.load(f"checkpoint/amos/version_{opt.version}/best_Encoder_s.pth")
    )
    encoder_t.load_state_dict(
        torch.load(f"checkpoint/amos/version_{opt.version}/best_Encoder_t.pth")
    )
    segmentor.load_state_dict(
        torch.load(f"checkpoint/amos/version_{opt.version}/best_Segmentor.pth")
    )
    vq.load_state_dict(torch.load(f"checkpoint/amos/version_{opt.version}/best_VQ.pth"))
    decoder_s.load_state_dict(
        torch.load(f"checkpoint/amos/version_{opt.version}/best_Decoder_s.pth")
    )
    decoder_t.load_state_dict(
        torch.load(f"checkpoint/amos/version_{opt.version}/best_Decoder_t.pth")
    )
    d_s.load_state_dict(torch.load(f"checkpoint/amos/version_{opt.version}/best_D_s.pth"))
    d_t.load_state_dict(torch.load(f"checkpoint/amos/version_{opt.version}/best_D_t.pth"))

    if not os.path.exists("test_results"):
        os.mkdir("test_results")

    if not os.path.exists("test_results/amos"):
        os.makedirs("test_results/amos")

    results_file = f"test_results/amos/{opt.version}_test_results.txt"


    encoder_s.eval()
    encoder_t.eval()
    decoder_s.eval()
    decoder_t.eval()
    segmentor.eval()
    d_s.eval()
    d_t.eval()
    vq.eval()

    test_indicator = tqdm(test_loader, ncols=80)
    test_indicator.set_description(f"test")
    all_batch_ct_dice = []
    all_batch_assd = []
    all_batch_asd = []
    xt_label_image = torch.zeros(216, 240, 320, device=opt.device)  # label
    xt_image = torch.zeros(216, 240, 320, device=opt.device)  # image
    output_image = torch.zeros(14, 216, 240, 320, device=opt.device)  # seg

    count = torch.zeros(216, device=opt.device)

    with torch.no_grad():
        for i, (indicator, data) in enumerate(zip(test_indicator, test_loader)):
            j = i % 214

            if j == 0:
                xt_label_image[:] = 0
                xt_image[:] = 0
                output_image[:] = 0
                count[:] = 0

            xt, xt_label = (
                data["ct"].float().to(opt.device),
                data["seg"].float().to(opt.device),
            )

            output = encoder_t(xt)
            output, _, _ = vq(output)
            output = segmentor(output, seg=True)

            output = torch.reshape(output, (1, 14, 3, 240, 320))
            output = F.softmax(output, dim=1)
            # print("output.shape",output.shape)
            # print("output_image.shape",output_image.shape)
            # print(j)
            output_image[:, j : j + 3, :, :] += output[0, :, :, :, :]

            xt_image[j : j + 3] += xt[0]
            # print(xt_label.shape)
            xt_label_image[j : j + 3] += xt_label.squeeze().long()
            count[j : j + 3] += 1

            if (j + 1) == 214:
                print('----------------------------------------')
                for k in range(216):
                    xt_label_image[k] /= count[k]

                    output_image[:, k] /= count[k]

                    xt_image[k] /= count[k]

                output_images = torch.argmax(output_image, dim=0)
                ct_dsc = calc_multi_dice(output_images, xt_label_image, 14)
                # print(ct_dsc)
                all_batch_ct_dice.append(ct_dsc)
                ct_assd = calc_multi_assd(output_images.cpu().numpy(), xt_label_image.cpu().numpy(), 14)
                # ct_asd = calc_multi_asd(output_images.cpu().numpy(), xt_label_image.cpu().numpy(), 14)
                # all_batch_asd.append(ct_asd)
                all_batch_assd.append(ct_assd)
                if opt.save_images:
                    nibabel.save(
                        nibabel.Nifti1Image(output_images.cpu().numpy().astype(np.int16), np.eye(4)),
                        "/home/chenxu/wangyang/VQUDA/version_2/outputs/amos/seg_label/%d.nii.gz"
                        % (i // 214),
                    )
                    nibabel.save(
                        nibabel.Nifti1Image(xt_image.cpu().numpy(), np.eye(4)),
                        "/home/chenxu/wangyang/VQUDA/version_2/outputs/amos/ct_image/%d.nii.gz"
                        % (i // 214),
                    )
                    nibabel.save(
                        nibabel.Nifti1Image(xt_label_image.cpu().numpy().astype(np.int16), np.eye(4)),
                        "/home/chenxu/wangyang/VQUDA/version_2/outputs/amos/ct_label/%d.nii.gz"
                        % (i // 214),
                    )

        all_batch_dice = torch.tensor(
            all_batch_ct_dice, device=opt.device
        )  # 转换为 GPU 上的 Tensor
        all_batch_dice_numpy = all_batch_dice.cpu().numpy()  # 转换为 NumPy 数组
        # 保存到文件

        np.save(f"/home/chenxu/wangyang/VQUDA/version_2/p_value/vquda_{opt.version}_all_batch_dice.npy", all_batch_dice_numpy)

        mean_dice = torch.mean(all_batch_dice, axis=0)
        std_dice = torch.std(all_batch_dice, axis=0)

        all_batch_assd_numpy = all_batch_assd  # 转换为 NumPy 数组
        # 保存到文件
        np.save(
            f"/home/chenxu/wangyang/VQUDA/version_2/p_value/vquda_{opt.version}_all_batch_assd.npy",
            all_batch_assd_numpy,
        )
        all_batch_assd = np.array(all_batch_assd)
        mean_assd = np.mean(all_batch_assd, axis=0)
        std_assd = np.std(all_batch_assd, axis=0)


        # np.save(
        #     "/home/chenxu/wangyang/VQUDA/version_2/p_value/vquda_all_batch_asd.npy",
        #     all_batch_asd,
        # )
        with open(results_file, "a") as file:
            file.write(f"-----------test_results----time:{datetime.now()}----------\n")
            file.write("Dice mean:{}\n".format(mean_dice.cpu().numpy()))
            file.write("Dice std:{}\n".format(std_dice.cpu().numpy()))
            file.write(
                "total mean dice: {}\n".format(torch.mean(mean_dice).cpu().numpy())
            )
            file.write("-----------\n")

            # ASSD 计算和写入结果的部分已省略
            file.write("-----------\n")
            file.write("ASSD mean:{}\n".format(mean_assd))
            file.write("ASSD std:{}\n".format(std_assd))
            file.write("total mean assd: {}\n".format(np.mean(mean_assd)))
            file.write("-----------\n")


if __name__ == "__main__":
    param_df = pd.read_excel("parameters_combination_16_16.xlsx")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--desired_value", type=str, default="sd112", help="choose a parameter"
    )
    parser.add_argument("--version", type=str, default="sd112", help="choose a version")
    args = parser.parse_args()
    # 设定的值，用来确定要选择的行
    desired_value = f"{args.desired_value}"

    # 找到第一列值与 desired_value 匹配的行
    selected_row = param_df[param_df[param_df.columns[0]] == desired_value]

    # 如果找到了匹配的行，则继续处理
    if not selected_row.empty:
        # 从匹配的行更新配置参数
        # 如果最后一列不是配置的一部分，则排除最后一列
        config_params = selected_row.iloc[0].to_dict()
        config_params["displace_scale"] = ast.literal_eval(
            config_params["displace_scale"]
        )
        print(config_params)

        # 使用参数初始化 Config 对象
        opt = Config_amos(**config_params)
        opt.version = args.version
        opt.save_images = False
        opt.quant_nums = 1
        opt.e_dim = 16
        # opt.temperature = 0.07
        opt.batch_size = 1
        opt.independent_layer_count = 1
        opt.displace_scale = [1,2,3]
        opt.align_type = 'js'
        opt.version = 'sd112_quant_nums=1'
        opt.class_nums = 14
        run()
