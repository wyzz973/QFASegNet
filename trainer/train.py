import argparse
import sys
sys.path.append('/home/chenxu/wangyang')
print(sys.path)
from datetime import datetime
import os
import ast
import itertools
import os.path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import torchvision.utils as vutils
from dataset_picai import PicaiTrainDataset, PicaiTestDataset
from version_2.config_amos import Config_amos
from dataset_amos import AmosTrainDataset, AmosTestDataset
from metrics import calc_multi_dice, psnr
from cross_align import (
    displacement,
    distance_to_similarity,
    compute_joint_distribution,
    compute_align_loss,
)
from network import (
    Encoder,
    Decoder,
    NLayerDiscriminator,
    VQ,
    EncoderShare,
    EncoderSharedLayers,
    DecoderSharedLayers,
    DecoderShare,
)
from save_images import (
    save_medical_images,
    save_segmentation_images,
    plot_vq_indices_kde,
)

from segmentor import *

import torch.distributed as dist



def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def valid(
    epoch, encoder_s, encoder_t, decoder_s, decoder_t, segmentor, d_s, d_t, vq, modality
):
    if not os.path.exists("results/amos"):
        os.makedirs("results/amos")

    if not os.path.exists(f"checkpoint/amos/version_{opt.version}"):
        os.makedirs(f"checkpoint/amos/version_{opt.version}")
    if opt.dataset == 'amos':
        valid_dataset = AmosTestDataset(
            f"{opt.dataroot}/valid", modality1=modality, n_slice=opt.slice_nums
        )
    else:
        valid_dataset = PicaiTestDataset(
          f"{opt.dataroot}/valid", source=opt.source_modality,target=opt.target_modality, n_slice=opt.slice_nums
      )

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    results_file = f"results/amos/{opt.version}_evaluation_results.txt"
    encoder_s.eval()
    encoder_t.eval()
    decoder_s.eval()
    decoder_t.eval()
    segmentor.eval()
    d_s.eval()
    d_t.eval()
    vq.eval()
    valid_indicator = tqdm(valid_dataloader, ncols=120)
    valid_indicator.set_description(
        f"{modality}_valid_{opt.version} Training Epoch {epoch + 1:03d}"
    )
    all_batch_ct_dice = []

    xt_label_image = torch.zeros(14, 216, 240, 320, device=opt.device)
    xt_image = torch.zeros(216, 240, 320, device=opt.device)
    count = torch.zeros(216, device=opt.device)
    with torch.no_grad():
        for i, (indicator, data) in enumerate(zip(valid_indicator, valid_dataloader)):
            j = i % 214

            if j == 0:
                xt_label_image[:] = 0
                xt_image[:] = 0
                count[:] = 0
            if modality == "ct":
                image, target = (
                    data["ct"].float().to(opt.device),
                    data["seg"].float().to(opt.device),
                )
                image_data = image
                target_data = target
                output = encoder_t(image_data)
                output, _, _ = vq(output)
            else:
                image, target = (
                    data["mr"].float().to(opt.device),
                    data["seg"].float().to(opt.device),
                )
                image_data = image

                output = encoder_s(image_data)
                output, _, _ = vq(output)

            output = segmentor(output,seg = True)
            # output = segmentor(output)


            # output_image = torch.reshape(output, (1, 14, 3, 240, 320))
            output_image = F.softmax(output.data, dim=1)
            # save_segmented_image(epoch, output_image[0], f'{modality}_seg', opt.slice_nums, j)

            xt_label_image[:, j : j + 3, :, :] += output_image[0, :, :, :, :]
            xt_image[j : j + 3] += target[0]

            count[j : j + 3] += 1

            if (j + 1) == 214:
                for k in range(216):
                    xt_label_image[:, k] /= count[k]
                    xt_image[k] /= count[k]

                xt_label_image_data = torch.argmax(xt_label_image, dim=0)

                ct_dsc = calc_multi_dice(xt_label_image_data, xt_image, 14)
                all_batch_ct_dice.append(ct_dsc)

        all_batch_dice = torch.tensor(
            all_batch_ct_dice, device=opt.device
        )  # 转换为 GPU 上的 Tensor
        mean_dice = torch.mean(all_batch_dice, axis=0)
        std_dice = torch.std(all_batch_dice, axis=0)

        if torch.mean(mean_dice) > opt.best_dsc and modality == "ct":
            opt.best_dsc = torch.mean(mean_dice)
            torch.save(
                encoder_s.state_dict(),
                f"checkpoint/amos/version_{opt.version}/best_Encoder_s.pth",
            )
            torch.save(
                encoder_t.state_dict(),
                f"checkpoint/amos/version_{opt.version}/best_Encoder_t.pth",
            )
            torch.save(
                decoder_s.state_dict(),
                f"checkpoint/amos/version_{opt.version}/best_Decoder_s.pth",
            )
            torch.save(
                decoder_t.state_dict(),
                f"checkpoint/amos/version_{opt.version}/best_Decoder_t.pth",
            )
            torch.save(
                segmentor.state_dict(),
                f"checkpoint/amos/version_{opt.version}/best_Segmentor.pth",
            )
            torch.save(
                d_s.state_dict(), f"checkpoint/amos/version_{opt.version}/best_D_s.pth"
            )

            torch.save(
                d_t.state_dict(), f"checkpoint/amos/version_{opt.version}/best_D_t.pth"
            )
            torch.save(vq.state_dict(), f"checkpoint/amos/version_{opt.version}/best_VQ.pth")

        with open(results_file, "a") as file:
            file.write(f'-----------time:{datetime.now()}---------------------\n')
            file.write(f"-----------{modality}_valid_{epoch + 1}--------------\n")
            file.write("Dice mean:{}\n".format(mean_dice.cpu().numpy()))
            file.write("Dice std:{}\n".format(std_dice.cpu().numpy()))
            file.write(
                "total mean dice: {}\n".format(torch.mean(mean_dice).cpu().numpy())
            )
            file.write("-----------\n")
    return opt.best_dsc, torch.mean(mean_dice)


def train():
    # setup(rank, world_size)
    # torch.cuda.set_device(rank)
    # device = torch.device("cuda", rank)
    displacement_map_list = displacement(
        displace_scale=opt.displace_scale, displacement=opt.displacement
    )

    # 设置随机种子
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 确保CUDA的行为确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = opt.to_dict()
    wandb.init(
        project=f"VQUDA7_share", entity="wyzz973", name=f"{opt.version}", config=config
    )

    transforms_options = [
        transforms.ToTensor(),
    ]
    label_transforms_options = [
        transforms.ToTensor(),
    ]
    
    if opt.data_augment:
        transforms_options.extend(
            [
                transforms.RandomRotation(
                    5, fill=-1, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomResizedCrop(
                    (240, 320),
                    scale=(0.8, 1.0),
                    antialias=None,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )
        label_transforms_options.extend(
            [
                transforms.RandomRotation(
                    5, fill=0, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.RandomResizedCrop(
                    (240, 320),
                    scale=(0.8, 1.0),
                    antialias=None,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )


    if opt.dataset == "amos":
        train_dataset = AmosTrainDataset(
            f"{opt.dataroot}/train",
            modality_ct=opt.target_modality,
            modality_mr=opt.source_modality,
            n_slices=opt.slice_nums,
            image_transform=transforms.Compose(transforms_options),
            label_transform=transforms.Compose(label_transforms_options)
        )
    else:
        train_dataset = PicaiTrainDataset(
            f"{opt.dataroot}/train",
            target=opt.target_modality,
            source=opt.source_modality,
            n_slices=opt.slice_nums,
        )


    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True
    )

    valid_label = torch.ones(opt.batch_size, 1, 30, 40).to(
        device=opt.device
    )  ## 定义真实的图片label为1
    fake_label = torch.zeros(opt.batch_size, 1, 30, 40).to(
        device=opt.device
    )  ## 定义假的图片的label为0

    # valid_label = torch.ones(opt.batch_size, 1).to(
    #     device=opt.device
    # )  ## 定义真实的图片label为1
    # fake_label = torch.zeros(opt.batch_size, 1).to(
    #     device=opt.device
    # )  ## 定义假的图片的label为0

    L1 = torch.nn.L1Loss().to(device=opt.device)
    # wandb.config(opt.to_dict())
    opt_dict = opt.filter_config_for_network(decoder=False)
    opt_dict_decoder = opt.filter_config_for_network(decoder=True)
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
    # encoder_s = Encoder_unet(slice_nums=opt.slice_nums, embed_dim=opt.e_dim, in_channel=opt.in_channel,
    #                          ch_mult=opt.ch_mult, norm=opt.norm).to(device=opt.device)
    # decoder_s = Decoder_unet(slice_nums=opt.slice_nums, class_nums=opt.class_nums, embed_dim=opt.e_dim,
    #                          in_channel=opt.in_channel, ch_mult=opt.ch_mult, norm=opt.norm).to(device=opt.device)
    # encoder_t = Encoder_unet(slice_nums=opt.slice_nums, embed_dim=opt.e_dim, in_channel=opt.in_channel,
    #                          ch_mult=opt.ch_mult, norm=opt.norm).to(device=opt.device)
    # decoder_t = Decoder_unet(slice_nums=opt.slice_nums, class_nums=opt.class_nums, embed_dim=opt.e_dim,
    #                          in_channel=opt.in_channel, ch_mult=opt.ch_mult, norm=opt.norm).to(device=opt.device)
    # segmentor = DecoderShare(
    #     ch_mult=[1, 2, 4, 8],
    #     slice_nums=3,
    #     class_nums=14,
    #     in_channel=64,
    #     shared_layers_module=decoder_shared_layers_module,
    #     independent_layer_count=1,
    #     seg=True
    # ).cuda()
    segmentor = Decoder(**opt_dict_decoder).to(device=opt.device)
    # d_s = Image_Discriminator(height=opt.img_height, width=opt.img_width).to(
    #     device=opt.device
    # )
    # d_t = Image_Discriminator(height=opt.img_height, width=opt.img_width).to(
    #     device=opt.device
    # )
    d_s = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device=opt.device)
    d_t = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device=opt.device)
    # VQ_S = VectorQuantizer2(n_e=opt.n_e, e_dim=opt.e_dim, beta=opt.beta,sane_index_shape=True).to(device=opt.device)
    vq = VQ(n_e=opt.n_e, e_dim=opt.e_dim, beta=opt.beta,quant_nums=opt.quant_nums).to(device=opt.device)

    # encoder_s = DDP(encoder_s, device_ids=[rank])
    # encoder_t = DDP(encoder_t, device_ids=[rank])
    # decoder_s = DDP(decoder_s, device_ids=[rank])
    # decoder_t = DDP(decoder_t, device_ids=[rank])
    # segmentor = DDP(segmentor, device_ids=[rank])
    # d_s = DDP(d_s, device_ids=[rank])
    # d_t = DDP(d_t, device_ids=[rank])
    # vq = DDP(vq, device_ids=[rank])
    if opt.epoch_start != 0:
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
        d_s.load_state_dict(
            torch.load(f"checkpoint/amos/version_{opt.version}/best_D_s.pth")
        )
        d_t.load_state_dict(
            torch.load(f"checkpoint/amos/version_{opt.version}/best_D_t.pth")
        )

    MSE = torch.nn.MSELoss().to(device=opt.device)
    kl_div = torch.nn.KLDivLoss(reduction="batchmean").to(device=opt.device)
    CE = torch.nn.CrossEntropyLoss().to(device=opt.device)

    # 假设所有模型部件都已初始化并设置到正确的设备上
    model_parts = [encoder_s, decoder_s, encoder_t, decoder_t]
    
    # 收集所有参数，确保没有重复
    all_parameters = set()
    for part in model_parts:
        for param in part.parameters():
            if param not in all_parameters:
                all_parameters.add(param)

    optimizer_G = torch.optim.AdamW(
        # itertools.chain(
        #     encoder_s.parameters(),encoder_t.parameters(),decoder_s.parameters(),decoder_t.parameters(),
        # ),
        itertools.chain(
            all_parameters,
            vq.parameters()
        ),
        lr=opt.learning_rate,
    )
    optimizer_S = torch.optim.AdamW(
        itertools.chain(segmentor.parameters()), lr=opt.learning_rate
    )
    optimizer_D = torch.optim.AdamW(
        itertools.chain(d_s.parameters(), d_t.parameters()), lr=opt.learning_rate
    )
    if not os.path.exists("training_log/amos"):
        os.makedirs("training_log/amos")
    loss_file = f"training_log/amos/{opt.version}_training_loss.txt"

    with open(loss_file, "a") as file:
        file.write(str(opt.to_dict())+'\n')

    epochs_no_improve = 0
    n_epochs_stop = 5
    for epoch in range(opt.epoch_start, opt.n_epochs):
        # train_sampler.set_epoch(epoch)
        psnr_S = []
        psnr_T = []
        batch_indicator = tqdm(train_dataloader, ncols=120)
        # batch_indicator = tqdm(range(1,10), ncols=120)

        batch_indicator.set_description(f"{opt.version} Training Epoch {epoch + 1:03d}")
        for i, (indicator, batch) in enumerate(zip(batch_indicator, train_dataloader)):
            encoder_s.train()
            encoder_t.train()
            decoder_s.train()
            decoder_t.train()

            segmentor.train()
            d_s.train()
            d_t.train()
            vq.train()
            source, S_label = (
                batch[opt.source_modality].float().to(opt.device),
                batch[f"{opt.source_modality}_seg"].float().to(opt.device),
            )
            target = batch[opt.target_modality].float().to(opt.device)

            """train D"""
            with torch.no_grad():
                source_feature = encoder_s(source)
                target_feature = encoder_t(target)

                source_quant, _, _ = vq(source_feature)
                target_quant, _, _ = vq(target_feature)


                s_to_t = decoder_t(source_quant)
                t_to_s = decoder_s(target_quant)

            D_s_fake = MSE(d_s(t_to_s), fake_label)
            D_s_real = MSE(d_s(source), valid_label)
            D_s_loss = D_s_fake + D_s_real

            D_t_fake = MSE(d_t(s_to_t), fake_label)
            D_t_real = MSE(d_t(target), valid_label)
            D_t_loss = D_t_fake + D_t_real

            loss_D = (D_s_loss + D_t_loss) * opt.lambda_D

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            """train G"""
            source_feature = encoder_s(source)
            target_feature = encoder_t(target)


            source_quant, loss_s_vq, d_s_vq = vq(source_feature)
            target_quant, loss_t_vq, d_t_vq = vq(target_feature)

            vq_loss = loss_s_vq + loss_t_vq

            re_source = decoder_s(source_quant)
            re_target = decoder_t(target_quant)

            re_loss_s = L1(re_source, source)
            re_loss_t = L1(re_target, target)
            re_loss = re_loss_s + re_loss_t

            s_to_t = decoder_t(source_quant)
            t_to_s = decoder_s(target_quant)

            s_to_t_feature = encoder_t(s_to_t)
            t_to_s_feature = encoder_s(t_to_s)


            s_to_t_quant, _, _ = vq(s_to_t_feature)
            t_to_s_quant, _, _ = vq(t_to_s_feature)

            # js散度
            # js_div_loss = js(s_vq_indices, t_vq_indices)
            align_loss_vq = []
            # for rs in range(opt.multi_scale):

            cluster_s = distance_to_similarity(d_s_vq, opt.temperature)
            cluster_t = distance_to_similarity(d_t_vq, opt.temperature)

            align_loss = compute_align_loss(
                cluster_s=cluster_s,
                cluster_t=cluster_t,
                displacement_map_list=displacement_map_list,
                align_type=opt.align_type,
            )

            align_loss_vq.append(align_loss)

            # feature_loss =MSE(s_to_t_feature, source_feature) +MSE(t_to_s_feature, target_feature)
            align_loss = sum(align_loss_vq) / len(align_loss_vq)
            feature_loss = MSE(s_to_t_quant, source_quant) + MSE(
                t_to_s_quant, target_quant
            )

            # feature_loss = L1(t_to_s_feature, target_quant) + L1(s_to_t_feature, source_quant)
            # with torch.no_grad():
            gan_s_loss = MSE(d_s(t_to_s), valid_label)
            gan_t_loss = MSE(d_t(s_to_t), valid_label)

            loss_GAN = gan_s_loss + gan_t_loss

            loss_G = (
                opt.lambda_re * re_loss
                + opt.lambda_vq_feature * feature_loss
                + opt.lambda_vq * vq_loss
                + opt.lambda_align_loss * align_loss
                + opt.lambda_gan * loss_GAN
            )

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            """ seg loss """
            with torch.no_grad():
                source_feature = encoder_s(source)
                target_feature = encoder_t(target)

                source_quant, _, _ = vq(source_feature)
                target_quant, _, _ = vq(target_feature)

                s_to_t = decoder_t(source_quant)

                s_to_t_feature = encoder_t(s_to_t)

                s_to_t_quant, _, _ = vq(s_to_t_feature)

            seg_S = segmentor(source_quant,seg = True)
            seg_s_to_t = segmentor(s_to_t_quant,seg = True)
            #
            # seg_S = segmentor(source_quant)
            # seg_s_to_t = segmentor(s_to_t_quant)

            loss_seg_S = CE(seg_S, S_label.long())
            loss_seg_s_to_t = CE(seg_s_to_t, S_label.long())

            loss_seg = loss_seg_S + opt.lambda_target * loss_seg_s_to_t
            # loss_seg = opt.lambda_target * loss_seg_s_to_t

            loss_S = loss_seg

            optimizer_S.zero_grad()
            loss_S.backward()
            optimizer_S.step()

            psnr_S.append(psnr(source, re_source))
            psnr_T.append(psnr(target, re_target))

            batch_indicator.set_postfix(
                {
                    "loss_D": loss_D.item(),
                    "loss_G": loss_G.item(),
                    "loss_S": loss_S.item(),
                    "psnr_S": sum(psnr_S) / len(psnr_S),
                    "psnr_T": sum(psnr_T) / len(psnr_T),
                }
            )
            wandb.log(
                {
                    "loss_D_s": D_s_loss.item(),
                    "loss_D_t": D_t_loss.item(),
                    "loss_D": loss_D.item(),
                    "feature_loss": feature_loss.item(),
                    # "kl_div_loss": kl_div_loss.item(),
                    "vq_loss": vq_loss.item(),
                    "D_s_fake": D_s_fake.item(),
                    "D_t_fake": D_t_fake.item(),
                    "D_s_real": D_s_real.item(),
                    "D_t_real": D_t_real.item(),
                    "gan_s_loss": gan_s_loss.item(),
                    "gan_t_loss": gan_t_loss.item(),
                    "loss_s_vq": loss_s_vq.item(),
                    "loss_t_vq": loss_t_vq.item(),
                    "re_loss_s": re_loss_s.item(),
                    "re_loss_t": re_loss_t.item(),
                    "re_loss": re_loss.item(),
                    "loss_seg_S": loss_seg_S.item(),
                    "loss_seg_s_to_t": loss_seg_s_to_t.item(),
                    "loss_S": loss_S.item(),
                    "psnr_S": psnr(source, re_source),
                    "psnr_T": psnr(target, re_target),
                    "align_loss": align_loss.item(),
                }
            )
            """save images"""
            if (i + 1) % 214 == 0 and opt.save_images:
                save_medical_images(
                    epoch,
                    source,
                    "source",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )
                save_medical_images(
                    epoch,
                    target,
                    "target",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )
                save_medical_images(
                    epoch,
                    re_source,
                    "re_source",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )
                save_medical_images(
                    epoch,
                    re_target,
                    "re_target",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )
                save_medical_images(
                    epoch,
                    s_to_t,
                    "s_to_t",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )
                save_medical_images(
                    epoch,
                    t_to_s,
                    "t_to_s",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )

                save_segmentation_images(
                    epoch,
                    seg_S,
                    "seg_S",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )
                save_segmentation_images(
                    epoch,
                    seg_s_to_t,
                    "seg_s_to_t",
                    version=opt.version,
                    n_slice=opt.slice_nums,
                    interval=2140 // (i + 1),
                )
                # save_segmentation_images(epoch, seg_T, 'seg_T', version=opt.version, n_slice=opt.slice_nums,
                #                          interval=2140 // (i + 1))

                # plot_vq_indices_kde(epoch, s_vq_indices=s_vq_indices, s_to_t_vq_indices=s_to_t_vq_indices,
                #                     t_vq_indices=t_vq_indices, t_to_s_vq_indices=t_to_s_vq_indices, prefix='js_KDE',
                #                     version=opt.version, interval=2140 // (i + 1))

        """valid"""
        ct_best_dsc, ct_mean_dsc = valid(
            epoch=epoch,
            encoder_s=encoder_s,
            encoder_t=encoder_t,
            decoder_s=decoder_s,
            decoder_t=decoder_t,
            segmentor=segmentor,
            d_s=d_s,
            d_t=d_t,
            vq=vq,
            modality="ct",
        )
        # _, mr_mean_dsc = valid(
        #     epoch=epoch,
        #     encoder_s=encoder_s,
        #     encoder_t=encoder_t,
        #     decoder_s=decoder_s,
        #     decoder_t=decoder_t,
        #     segmentor=segmentor,
        #     d_s=d_s,
        #     d_t=d_t,
        #     vq=vq,
        #     modality="mr",
        # )
        if ct_best_dsc <= ct_mean_dsc:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(
            f"best_dsc:{ct_best_dsc}\tmean_dsc:{ct_mean_dsc}\tmr_mean_dsc"
        )
        psnr_S_avg = sum(psnr_S) / len(psnr_S)
        psnr_T_avg = sum(psnr_T) / len(psnr_T)
        with open(loss_file, "a") as file:
            file.write(
                "---------------epoch:" + str(epoch + 1) + "-------------------\n"
            )
            file.write(
                f"loss_D_s:{D_s_loss.item():.6f}\t"
                f"loss_D_t:{D_t_loss.item():.6f}\t"
                f"loss_D:{loss_D.item():.6f}\n"
            )
            file.write(
                f"feature_loss:{feature_loss.item()}\t"
                # f"js_div_loss:{js_div_loss.item():.6f}\n"
                f"loss_s_vq:{loss_s_vq.item():.6f}\t"
                f"loss_t_vq:{loss_t_vq.item():.6f}\t"
                f"VQ_loss:{vq_loss.item():.6f}\n"
                f" loss_GAN:{loss_GAN.item():.6f}\t"
                f"align_loss:{align_loss.item():.6f}"
                f"loss_G:{loss_G.item():.6f}\n"
            )
            file.write(
                f"loss_seg_S:{loss_seg_S.item():.6f}\t"
                f"loss_seg_s_to_t:{loss_seg_s_to_t.item():.6f}\t"
                f"loss_S:{loss_S.item():.6f}\n"
            )
            file.write(
                f"best_dsc:{ct_best_dsc}\t"
                f"ct_mean_dsc:{ct_mean_dsc}\t"
                # f"mr_mean_dsc{mr_mean_dsc}\n"
            )
            file.write(f"psnr_S_avg:{psnr_S_avg}\t" f"psnr_T_avg:{psnr_T_avg}\n")

        wandb.log(
            {
                "ct_best_dsc": ct_best_dsc,
                "ct_mean_dsc": ct_mean_dsc,
                # "mr_mean_dsc": mr_mean_dsc,
            }
        )

        if epochs_no_improve >= n_epochs_stop and opt.early_stop:
            print(f"Early stopping after {epoch+1} epochs!")
            wandb.finish()
            break

    wandb.finish()


if __name__ == "__main__":
    start_time = datetime.now()
    param_df = pd.read_excel("Ablation_amos.xlsx")
    # param_df = param_df.sample(frac=1).reset_index(drop=True)
    param_df = param_df.reset_index(drop=True)
    # 遍历 DataFrame 中的每一行
    for index, row in param_df.iterrows():
    # 跳过第一行
        if index != 0:
            continue

        # 从当前行更新配置参数
        # 如果最后一列不是配置的一部分，则排除最后一列
        config_params = row.to_dict()
        config_params["displace_scale"] = ast.literal_eval(config_params["displace_scale"])
        config_params["independent_layer_count"] = int(config_params["independent_layer_count"])
        config_params["e_dim"] = int(config_params["e_dim"])
        config_params["quant_nums"] = int(config_params["quant_nums"])
        print(config_params)

        # 使用参数初始化 Config 对象
        opt = Config_amos(**config_params)
        opt.n_epochs = 100
        opt.epoch_start = 0
        opt.best_dsc = 0
        opt.early_stop = False
        os.environ["WANDB_MODE"] = "offline"
        opt.version = 'sd112_973_ccc_tem=0.07'
        opt.seed = 973
        opt.data_augment = False
        opt.beta = 0.25
        # opt.temperature = 0.07
        # opt.independent_layer_count = 1
        # opt.e_dim = 16
        opt.quant_nums = 1
        opt.version = 'sd112_quant_nums=1'
        # 打印当前配置以进行跟踪
        print(opt.print_attributes())
        print(opt.align_type)
        # 调用训练函数
        train()
    end_time = datetime.now()
    print("Start time:", start_time)
    print("End time:", end_time)
    print("Total execution time:", end_time - start_time)
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--desired_value", type=str, default="sd112", help="choose a parameter"
    # )
    # parser.add_argument(
    #     "--learning_rate", type=float, default=0.0001, help="choose a parameter"
    # )
    # param_df = pd.read_excel("parameters_combination_16_16.xlsx")
    # args = parser.parse_args()
    # # 设定的值，用来确定要选择的行
    # desired_value = f"{args.desired_value}"
    #
    # # 找到第一列值与 desired_value 匹配的行
    # selected_row = param_df[param_df[param_df.columns[0]] == desired_value]
    #
    # # 如果找到了匹配的行，则继续处理
    # if not selected_row.empty:
    #     # 从匹配的行更新配置参数
    #     # 如果最后一列不是配置的一部分，则排除最后一列
    #     config_params = selected_row.iloc[0].to_dict()
    #     config_params["displace_scale"] = ast.literal_eval(
    #         config_params["displace_scale"]
    #     )
    #     print(config_params)
    #
    #     # 使用参数初始化 Config 对象
    #     opt = Config(**config_params)
    #
    #     opt.n_epochs = 30
    #     opt.epoch_start = 0
    #     opt.batch_size = 1
    #
    #     opt.early_stop = True
    #     opt.save_images = True
    #     opt.displace_scale = [1,2]
    #     # opt.lambda_gan = 0.1
    #     # opt.lambda_vq_feature = 0.1
    #     # opt.lambda_target = 1
    #     # opt.lambda_align_loss = 1
    #     opt.quant_nums = 16
    #     opt.e_dim = 16
    #     opt.temperature = 1
    #     opt.learning_rate = args.learning_rate
    #     opt.class_nums = 14
    #     opt.independent_layer_count = 1
    #     opt.version += f"displace_scale = [1,2]"
    #     # 打印当前配置以进行跟踪
    #     print(opt.print_opt())
    #
    #     print(torch.cuda.is_available())
    #     os.environ["WANDB_MODE"] = "offline"
    #     train()
    # else:
    #     print(f"No row found for the value {desired_value}")
    # opt = Config()
    # # opt.version = f'prevq_align_{opt.e_dim}_{opt.quant_nums}_{opt.temperature}'
    # print(opt.print_opt())
    #
    # train()
