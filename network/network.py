import copy
import functools
import warnings

import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
# from mmcv.cnn import ConvModule
# from mmseg.models.decode_heads import UPerHead
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead

# from library.nn import SynchronizedBatchNorm2d, PrRoIPool2D
from VQUDA.version_2.config import Config



class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_channels,
        dim=2,
        drop_path=0.0,
        kernel_size=3,
        padding=1,
        norm_type="group",
        act_type="relu",
    ):
        super(ResidualBlock, self).__init__()

        self.drop_path = (
            timm.models.layers.DropPath(drop_path)
            if drop_path > 0.0
            else torch.nn.Identity()
        )

        if dim == 1:
            conv = torch.nn.Conv1d
        elif dim == 2:
            conv = torch.nn.Conv2d
        elif dim == 3:
            conv = torch.nn.Conv3d
        else:
            assert 0

        self.basic = torch.nn.Sequential(
            conv(num_channels, num_channels, kernel_size=kernel_size, padding=padding),
            native_norm(norm_type, num_channels, dim=dim),
            native_act(act_type),
            conv(num_channels, num_channels, kernel_size=kernel_size, padding=padding),
            native_norm(norm_type, num_channels, dim=dim),
        )
        self.act = native_act(act_type)  # 激活

    def forward(self, x):
        out = self.basic(x)
        return self.act(x + self.drop_path(out))


def native_norm(norm_type, num_features, dim=2):
    if norm_type == "batch":
        if dim == 1:
            return torch.nn.BatchNorm1d(num_features)
        elif dim == 2:
            return torch.nn.BatchNorm2d(num_features)
        elif dim == 3:
            return torch.nn.BatchNorm3d(num_features)
    elif norm_type == "group":
        return torch.nn.GroupNorm(
            32 if num_features > 32 else num_features // 2, num_features
        )
    elif norm_type == "layer":
        return torch.nn.LayerNorm(num_features)
    elif norm_type == "instance":
        if dim == 1:
            return torch.nn.InstanceNorm1d(num_features)
        elif dim == 2:
            return torch.nn.InstanceNorm2d(num_features)
        elif dim == 3:
            return torch.nn.InstanceNorm3d(num_features)
    elif norm_type == "adain":
        assert dim == 2
        return AdaptiveInstanceNorm2D(num_features)
    elif norm_type == "lin":
        assert dim == 2
        return LIN(num_features)
    else:
        print("Unknown norm type:%s" % norm_type)
        assert 0


def native_act(act_type, inplace=False):
    if act_type == "prelu":
        return torch.nn.PReLU()
    elif act_type == "lrelu":
        return torch.nn.LeakyReLU(0.2, inplace=inplace)
    elif act_type == "relu":
        return torch.nn.ReLU(inplace=inplace)
    elif act_type == "gelu":
        return torch.nn.GELU()
    elif act_type == "silu":
        return torch.nn.SiLU()
    elif act_type == "elu":
        return torch.nn.ELU(inplace=inplace)
    elif act_type == "mish":
        return torch.nn.Mish(inplace)
    elif act_type == "swish":
        return timm.layers.activations.Swish(inplace)
    elif act_type == "":
        return input
    else:
        print("Unknown act type:%s" % act_type)
        assert 0



class Decoder(nn.Module):
    def __init__(
        self,
        slice_nums,
        class_nums,
        in_channel,
        ch_mult,
        e_dim,
        quant_nums=1,
        nums_res_block=2,
        norm_type="group",
        act_type="swish",
    ):
        super(Decoder, self).__init__()
        self.in_ch = slice_nums
        layers = [
            torch.nn.Conv2d(e_dim * quant_nums, in_channel * ch_mult[-1], 1),
            native_norm(norm_type, in_channel * ch_mult[-1]),
            native_act(act_type),
        ]
        for i in range(len(ch_mult) - 1, 0, -1):
            cur_ch = in_channel * ch_mult[i]
            for j in range(nums_res_block):
                layers.append(
                    ResidualBlock(cur_ch, norm_type=norm_type, act_type=act_type)
                )

            layers.extend(
                [
                    torch.nn.ConvTranspose2d(
                        cur_ch,
                        in_channel * ch_mult[i - 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    native_norm(norm_type, in_channel * ch_mult[i - 1]),
                    native_act(act_type),
                ]
            )

        for i in range(nums_res_block):
            layers.append(
                ResidualBlock(
                    in_channel * ch_mult[0], norm_type=norm_type, act_type=act_type
                )
            )

        # layers.append(torch.nn.Conv2d(in_channel * ch_mult[0], slice_nums, 1))
        self.out = torch.nn.Conv2d(in_channel * ch_mult[0], slice_nums, 1)
        self.seg = torch.nn.Conv2d(in_channel * ch_mult[0], slice_nums * class_nums, 1)
        self.model = nn.Sequential(*layers)

    def forward(self, x, seg=False):
        x = self.model(x)
        if seg:
            x = self.seg(x)
            x = torch.stack(torch.split(x, self.in_ch, 1), 1)
            # print(x.shape)
            # x = torch.reshape(x, (1, 14, 3, 240, 320))   #amos
            # x = torch.reshape(x, (1, 2, 3, 184, 176))  #cmf
        else:
            x = self.out(x)
            x = x.clamp(-1.0, 1.0)
        return x




"""
share_encoder_decoder
"""


class EncoderShare(nn.Module):
    def __init__(
        self,
        slice_nums,
        in_channel,
        ch_mult,
        shared_layers_module,
        independent_layer_count=1,
        norm_type="group",
        act_type="swish",
    ):
        super(EncoderShare, self).__init__()
        # 初始化独立层
        self.independent_layers = nn.Sequential()
        for i in range(independent_layer_count):
            self.independent_layers.add_module(
                f"ind_layer_{i}",
                nn.Sequential(
                    nn.Conv2d(
                        slice_nums if i == 0 else in_channel * ch_mult[i - 1],
                        in_channel * ch_mult[i],
                        stride=1 if i == 0 else 2,
                        kernel_size=3,
                        padding=1,
                    ),
                    native_norm(norm_type, in_channel * ch_mult[i]),
                    native_act(act_type),
                    ResidualBlock(
                        in_channel * ch_mult[i], norm_type=norm_type, act_type=act_type
                    ),
                    ResidualBlock(
                        in_channel * ch_mult[i], norm_type=norm_type, act_type=act_type
                    ),
                ),
            )

        # 使用共享层
        self.shared_layers_module = shared_layers_module

    def forward(self, x):
        x = self.independent_layers(x)
        # print("encoder_independent_layers",x.shape)
        x = self.shared_layers_module(x)
        # print("encoder_shared_layers_module",x.shape)
        return x


class EncoderSharedLayers(nn.Module):
    def __init__(
        self,
        in_channel,
        ch_mult,
        independent_layer_count,
        e_dim,
        quant_nums,
        norm_type,
        act_type,
    ):
        super(EncoderSharedLayers, self).__init__()
        self.layers = nn.Sequential()
        for i in range(independent_layer_count - 1, len(ch_mult) - 1):
            self.layers.add_module(
                f"shared_layer_{i}",
                nn.Sequential(
                    nn.Conv2d(
                        in_channel * ch_mult[i],
                        in_channel * ch_mult[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    native_norm(norm_type, in_channel * ch_mult[i + 1]),
                    native_act(act_type),
                    ResidualBlock(
                        in_channel * ch_mult[i + 1],
                        norm_type=norm_type,
                        act_type=act_type,
                    ),
                    ResidualBlock(
                        in_channel * ch_mult[i + 1],
                        norm_type=norm_type,
                        act_type=act_type,
                    ),
                ),
            )

        self.final_conv = nn.Conv2d(
            in_channel * ch_mult[-1], e_dim * quant_nums, kernel_size=1
        )

        ##notgq
        # self.final_conv = nn.Conv2d(in_channel * ch_mult[-1], e_dim, kernel_size=1)
        ##notgq
    def forward(self, x):
        x = self.layers(x)
        x = self.final_conv(x)

        return x


class DecoderShare(nn.Module):
    def __init__(
        self,
        slice_nums,
        class_nums,
        in_channel,
        ch_mult,
        shared_layers_module,
        independent_layer_count=1,
        seg=False,
        norm_type="group",
        act_type="swish",
    ):
        super(DecoderShare, self).__init__()

        # 使用共享层
        self.shared_layers_module = shared_layers_module

        self.Seg = seg
        self.in_ch = in_channel
        # 初始化独立层
        self.independent_layers = nn.Sequential()
        ch_mult = ch_mult[:independent_layer_count]
        # print(ch_mult)
        for i in range(len(ch_mult) - 1, 0, -1):
            self.independent_layers.add_module(
                f"ind_layer_{i}",
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channel * ch_mult[i],
                        in_channel * ch_mult[i - 1],
                        stride=2,
                        kernel_size=3,
                        padding=1,
                        output_padding=1,
                    ),
                    native_norm(norm_type, in_channel * ch_mult[i - 1]),
                    native_act(act_type),
                    ResidualBlock(
                        in_channel * ch_mult[i - 1],
                        norm_type=norm_type,
                        act_type=act_type,
                    ),
                    ResidualBlock(
                        in_channel * ch_mult[i - 1],
                        norm_type=norm_type,
                        act_type=act_type,
                    ),
                ),
            )

        # 定义输出卷积层
        self.out = torch.nn.Conv2d(in_channel * ch_mult[0], slice_nums, 1)
        self.seg = torch.nn.Conv2d(in_channel * ch_mult[0], slice_nums * class_nums, 1)

    def forward(self, x):
        x = self.shared_layers_module(x)
        # print(x.shape)
        x = self.independent_layers(x)
        # print(x.shape)
        if self.Seg:
            x = self.seg(x)
            x = torch.stack(torch.split(x, self.in_ch, 1), 1)
        else:
            x = self.out(x)
            x = torch.clamp(x, -1.0, 1.0)

        return x


class DecoderSharedLayers(nn.Module):
    def __init__(
        self,
        in_channel,
        ch_mult,
        independent_layer_count,
        e_dim,
        quant_nums,
        norm_type,
        act_type,
    ):
        super(DecoderSharedLayers, self).__init__()

        self.first_conv = nn.Conv2d(
            e_dim * quant_nums, in_channel * ch_mult[-1], kernel_size=1
        )
        ##notgq
        # self.first_conv = nn.Conv2d(e_dim, in_channel * ch_mult[-1], kernel_size=1)
        ##notgq
        self.layers = nn.Sequential()
        for i in range(len(ch_mult) - 2, independent_layer_count - 2, -1):
            self.layers.add_module(
                f"shared_layer_{i}",
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channel * ch_mult[i + 1],
                        in_channel * ch_mult[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    native_norm(norm_type, in_channel * ch_mult[i]),
                    native_act(act_type),
                    ResidualBlock(
                        in_channel * ch_mult[i], norm_type=norm_type, act_type=act_type
                    ),
                    ResidualBlock(
                        in_channel * ch_mult[i], norm_type=norm_type, act_type=act_type
                    ),
                ),
            )

    def forward(self, x):
        x = self.first_conv(x)
        # print("decoder_first_conv",x.shape)
        x = self.layers(x)
        # print("decoder_layers",x.shape)
        return x

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return (
            z_q,
            loss,
            min_encoding_indices,
            d.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3], self.n_e),
        )


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q



class VQ(nn.Module):
    def __init__(self, n_e, e_dim, beta, quant_nums):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantizer2(n_e=n_e, e_dim=e_dim, beta=beta) for _ in range(quant_nums)])
        self.e_dim = e_dim
        self.quant_nums = quant_nums

    def forward(self, x):
        h_list = torch.split(x, self.e_dim, 1)
        quant_list = []
        emb_loss_list = []
        d_list = []

        for idx, h_ in enumerate(h_list):
            quantizer = self.quantizers[idx]
            quant, emb_loss, _, d = quantizer(h_)

            quant_list.append(quant)
            emb_loss_list.append(emb_loss)
            d_list.append(d)

        quant = torch.cat(quant_list, dim=1)
        emb_loss = sum(emb_loss_list) / len(emb_loss_list)
        d = torch.stack(d_list, dim=1)

        return quant, emb_loss,d

def straight_through_estimator(x, c):
    # this method replaces x by c without stopping the gradient flowing through x
    return x + (c - x).detach()


def compute_perplexity(encodings):
    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity






def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 3
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, image_channels, num_filters_last=32, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(image_channels, num_filters_last, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    4,
                    2 if i < n_layers else 1,
                    1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True),
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def compute_joint_distribution(x_out, displacement_map: (int, int, int)):
    n, c, d, h, w = x_out.shape

    print(displacement_map[0], displacement_map[1], displacement_map[2])
    after_displacement = x_out.roll(
        shifts=[displacement_map[0], displacement_map[1], displacement_map[2]],
        dims=[2, 3, 4],
    )
    print(after_displacement.shape)
    x_out = x_out.reshape(n, c, d * h * w)
    after_displacement = after_displacement.reshape(n, c, d * h * w).transpose(2, 1)
    p_i_j = (x_out @ after_displacement).mean(0).unsqueeze(0).unsqueeze(0)
    p_i_j += 1e-8
    p_i_j /= p_i_j.sum(dim=3, keepdim=True)  # norm
    return p_i_j.contiguous()


def distance_to_similarity(distance, temperature=1.0, dim=-1, eps=1e-8):
    exp_distances = torch.exp(-distance / temperature) + eps
    similarity = exp_distances / torch.sum(exp_distances, dim, keepdim=True)
    return similarity


