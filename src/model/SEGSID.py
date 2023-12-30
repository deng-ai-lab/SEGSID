import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as P
import time
import torchvision.models as models

from . import regist_model
from .RFS import RF_scale
from .pixel_shuffle import pixel_shuffle_up_sampling, pixel_shuffle_down_sampling


@regist_model
class SEGSID(nn.Module):
    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, max_epoch=128, sematic_type=None,
                 in_ch=3, bsn_base_ch=128, bsn_num_module=9, is_refine=False):
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        self.max_epoch = max_epoch
        self.is_refine = is_refine
        self.bsn = SEGSID_Model(in_ch=in_ch, out_ch=in_ch, base_ch=bsn_base_ch, num_module=bsn_num_module, sematic_type=sematic_type)

    def forward(self, img, pd=None):
        if pd is None:
            pd = self.pd_a

        img_denoised = self.bsn(img, pd, pad=self.pd_pad)
        return img_denoised

    def denoise(self, x):  # Denoise process for inference.
        b, c, h, w = x.shape

        assert h % self.pd_b == 0, f'but w is {h}'
        assert w % self.pd_b == 0, f'but w is {w}'

        temp_pd_b = self.pd_b

        """forward process with inference pd factor = 2"""
        img_denoised = self.forward(img=x, pd=temp_pd_b)

        if not self.R3:
            return img_denoised
        else:
            """ with R3 strategy """
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p
                tmp_input = torch.clone(img_denoised).detach()
                tmp_input[mask] = x[mask]
                denoised[..., t] = self.bsn(tmp_input, pd=temp_pd_b, is_refine=self.is_refine, pad=self.pd_pad)

            denoised = torch.mean(denoised, dim=-1)
            return denoised


class SEGSID_Model(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9, head_ch=24, sematic_type=None):
        super().__init__()
        assert base_ch % 2 == 0, "base channel should be divided with 2"

        self.sematic_type = sematic_type

        ly = []
        ly += [nn.Conv2d(in_ch, head_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.in_ch = in_ch
        self.head = nn.Sequential(*ly)

        self.branch_11 = Branch(branch_type='11', dilated_factor=2, head_ch=head_ch, out_ch=base_ch, num_module=num_module)
        self.branch_21 = Branch(branch_type='21', dilated_factor=3, head_ch=head_ch, out_ch=base_ch, num_module=num_module)

        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

        if sematic_type == 'ResNet':
            print(f'[{True}]    Init ResNet backbone as sematic encoder')
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.semantic_encoder = torch.nn.Sequential(*(list(resnet18.children())[:-1]),
                                                        )  # in_channel=3 out_channel=512
            print('Warning Please check the norm mode of sematic branch')

        else:
            for _ in range(50):
                print('Warning sematic_type', False)

    def forward(self, x, pd=5, is_refine=False, pad=2):
        if is_refine is True:
            pd = 1

        if self.sematic_type == 'ResNet':
            x_norm = 2 * (x / 255 - 0.5)
            if is_refine is False:
                x_norm = pixel_shuffle_down_sampling(x_norm, f=pd, pad=pad)
            else:
                x_norm = F.pad(x_norm, (pad, pad, pad, pad))

            if self.in_ch == 1:
                f_semantic = self.semantic_encoder(x_norm.repeat(1, 3, 1, 1))
            elif self.in_ch == 3:
                f_semantic = self.semantic_encoder(x_norm)
            else:
                print('Wrong in_ch: hope 1 or 3, but get ', self.in_ch)
                f_semantic = self.semantic_encoder(x_norm.repeat(1, 3, 1, 1))
        else:
            f_semantic = None

        x = self.head(x)
        br1 = self.branch_11(x, pd=pd, f_semantic=f_semantic, is_refine=is_refine, pad=pad)
        br2 = self.branch_21(x, pd=pd, f_semantic=f_semantic, is_refine=is_refine, pad=pad)
        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)


class Branch(nn.Module):
    def __init__(self, branch_type, dilated_factor, head_ch, out_ch, num_module, is_bias=True):
        super().__init__()

        padding_mode = 'zeros'

        self.branch_type = branch_type
        if branch_type == '11':
            self.rfa = RFA_11(head_ch, out_ch, dilated_factor=dilated_factor, padding_mode=padding_mode, is_bias=is_bias)
            assert dilated_factor == 2

        elif branch_type == '21':
            self.rfa = RFA_21(head_ch, out_ch, dilated_factor=dilated_factor, padding_mode=padding_mode, is_bias=is_bias)
            assert dilated_factor == 3

        in_ch = out_ch
        body1 = []
        body1 += [nn.ReLU(inplace=True)]
        body1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body1 += [nn.ReLU(inplace=True)]
        body1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body1 += [nn.ReLU(inplace=True)]
        self.body1 = nn.Sequential(*body1)

        si_blocks = []
        si_blocks += [Semantic_Injection_block(dilated_factor, in_ch, padding_mode=padding_mode) for _ in range(num_module)]
        self.si_blocks = nn.Sequential(*si_blocks)
        print('Semantic is used')

        body2 = []
        body2 += [nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=is_bias)]
        body2 += [nn.ReLU(inplace=True)]
        self.body2 = nn.Sequential(*body2)

    def forward(self, x: torch.Tensor, pd, f_semantic=None, is_refine=False, pad=2):

        # RFA Module
        x = self.rfa(x, is_refine=is_refine, f=pd, pad=pad)

        x = self.body1(x)
        for si_block in self.si_blocks:  # Stacked SI blocks in SEI Modul
            x = si_block(x, f_semantic)  
        x = self.body2(x)
        
        # Inverse PD
        if is_refine is False:
            x = pixel_shuffle_up_sampling(x, f=pd, pad=pad)
        else:
            if pad != 0:
                x = x[:, :, pad:-pad, pad:-pad]

        return x


class Semantic_Injection_block(nn.Module):

    def __init__(self, stride, in_ch, padding_mode='reflect'):
        super().__init__()

        self.semantic_affine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(512, in_ch, kernel_size=1),  # semantic function
        )

        self.spatial_mix = nn.Sequential(
            # F-Conv + ReLu + 1x1 Conv
            Dilation_fork_3x3(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1)
        )

    def forward(self, x, f_semantic: torch.Tensor):
        if f_semantic is None:
            y = x + self.spatial_mix(x)
            # print('No semantic')
        else:
            y = x + self.spatial_mix(self.semantic_injection_function(x, self.semantic_affine(f_semantic)))
        return y

    @staticmethod
    def semantic_injection_function(spatial_feature, semantic_feature):
        return spatial_feature * semantic_feature


# D-Conv in Paper, represents 'Diamond-shaped Masked Convolution
class DSM_Convolution(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()

        self.mask.fill_(1)
        center = kH // 2
        for i in range(kH):
            for j in range(kW):
                if abs(i - center) + abs(j - center) <= 4:
                    self.mask[:, :, i, j] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# P-Conv in Paper
class Dilation_plus_3x3(nn.Conv2d):
    """
    Masked Kernel Shape:
    0, 1, 0,
    1, 0, 1,
    0, 1, 0,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()

        self.mask.fill_(0)
        self.mask[:, :, 0, 1] = 1
        self.mask[:, :, 1, 0] = 1
        self.mask[:, :, 1, 2] = 1
        self.mask[:, :, 2, 1] = 1

        # print('Dilation_plus_3x3\n',self.mask[0,0,:,:].data)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# F-Conv in paper
class Dilation_fork_3x3(nn.Conv2d):
    """
    Masked Kernel Shape:
    1, 0, 1,
    0, 1, 0,
    1, 0, 1,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()

        self.mask.fill_(1)
        self.mask[:, :, 0, 1] = 0
        self.mask[:, :, 1, 0] = 0
        self.mask[:, :, 1, 2] = 0
        self.mask[:, :, 2, 1] = 0

        # print('Dilation_fork_3x3\n', self.mask[0, 0, :, :].data)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# RFA modul
class RFA_11(nn.Module):
    """ RFA module with kernel size is 11x11 """

    def __init__(self, head_ch, out_ch, dilated_factor, padding_mode='reflect', is_bias=True):
        super().__init__()
        self.ratio = 2 / 5
        self.pd1_flag = False
        self.padding = 5
        self.padding_mode = padding_mode
        self.shave = 5 + 3
        self.kernel_size = ks = 11

        self.d_conv = DSM_Convolution(head_ch, out_ch, kernel_size=ks, stride=1, padding=ks // 2, padding_mode=padding_mode, bias=is_bias)
        self.normal_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks), stride=(1, 1), padding=ks // 2, padding_mode=padding_mode, bias=is_bias)
        self.is_diamond_mask = True if hasattr(self.d_conv, 'mask') else False

        if self.is_diamond_mask:
            print('11 d_conv get mask')
        else:
            print('11 d_conv no mask')

        self.test_d_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks), stride=(ks, ks), padding=0, bias=is_bias)
        self.test_normal_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks), stride=(ks, ks), padding=0, bias=is_bias)

        self.f_conv = Dilation_fork_3x3(out_ch, out_ch, kernel_size=3, stride=1, dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode, bias=is_bias)
        self.p_conv = Dilation_plus_3x3(out_ch, out_ch, kernel_size=3, stride=1, dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode, bias=is_bias)

        self.rf_scale = RF_scale(head_ch, out_ch, kernel_size=ks, padding=ks // 2, padding_mode=padding_mode, ratio=self.ratio)

    def forward(self, x: torch.Tensor, is_refine=False, f=5, pad=2):
        # Training
        if self.training:
            d_conv_x = self.d_conv(x)
            normal_conv_x = self.normal_conv(x)
            d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
            normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)
            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
            return out_x

        # Testing
        else:
            # testing w/o refine
            if is_refine is False:
                self.rf_scale.ratio = 2 / 5
                d_conv_x, normal_conv_x = self.forward_refine(x)
                d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
                normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)

            # testing + refine
            else:
                self.rf_scale.ratio = 1 / 5
                d_conv_x, normal_conv_x = self.forward_refine(x)
                p = pad
                d_conv_x = F.pad(d_conv_x, (p, p, p, p))
                normal_conv_x = F.pad(normal_conv_x, (p, p, p, p))

            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
            return out_x

    def forward_refine(self, x):
        n_GPUs_for_deformable = 1
        n_GPUs = min(n_GPUs_for_deformable, 4)

        b, c, h, w = x.shape

        top = slice(0, h // 2 + self.shave)
        bottom = slice(h - h // 2 - self.shave, h)
        left = slice(0, w // 2 + self.shave)
        right = slice(w - w // 2 - self.shave, w)
        x_chop = torch.cat([
            x[..., top, left],
            x[..., top, right],
            x[..., bottom, left],
            x[..., bottom, right]
        ])

        hole_chop = []
        full_chop = []
        assert h * w < 4 * 200000
        for i in range(0, 4, n_GPUs):
            x = x_chop[i:(i + n_GPUs)]
            x_offset = P.data_parallel(self.rf_scale, x, range(n_GPUs))

            self.test_d_conv.weight.data = self.d_conv.weight.data * self.d_conv.mask
            self.test_d_conv.bias = self.d_conv.bias

            self.test_normal_conv.weight.data = self.normal_conv.weight.data
            self.test_normal_conv.bias = self.normal_conv.bias

            temp_hole_chop = P.data_parallel(self.test_d_conv, x_offset, range(n_GPUs))
            temp_full_chop = P.data_parallel(self.test_normal_conv, x_offset, range(n_GPUs))
            del x_offset

            if not hole_chop:
                hole_chop = [c for c in temp_hole_chop.chunk(n_GPUs, dim=0)]
            else:
                hole_chop.extend(temp_hole_chop.chunk(n_GPUs, dim=0))

            if not full_chop:
                full_chop = [c for c in temp_full_chop.chunk(n_GPUs, dim=0)]
            else:
                full_chop.extend(temp_full_chop.chunk(n_GPUs, dim=0))

        """ merge """
        assert h % 2 == 0 and w % 2 == 0
        top = slice(0, h // 2)
        bottom = slice(h - h // 2, h)
        left = slice(0, w // 2)
        right = slice(w - w // 2, w)

        right_r = slice(w // 2 - w, None)
        bottom_r = slice(h // 2 - h, None)

        b, c = hole_chop[0].size()[:-2]
        d_conv_x = hole_chop[0].new(b, c, h, w)
        normal_conv_x = full_chop[0].new(b, c, h, w)

        d_conv_x[:, :, top, left] = hole_chop[0][:, :, top, left]
        d_conv_x[..., top, right] = hole_chop[1][..., top, right_r]
        d_conv_x[..., bottom, left] = hole_chop[2][..., bottom_r, left]
        d_conv_x[..., bottom, right] = hole_chop[3][..., bottom_r, right_r]

        normal_conv_x[..., top, left] = full_chop[0][..., top, left]
        normal_conv_x[..., top, right] = full_chop[1][..., top, right_r]
        normal_conv_x[..., bottom, left] = full_chop[2][..., bottom_r, left]
        normal_conv_x[..., bottom, right] = full_chop[3][..., bottom_r, right_r]

        return d_conv_x, normal_conv_x


# RFA modul
class RFA_21(nn.Module):
    """ RFA module with kernel size is 11x11 """

    def __init__(self, head_ch, out_ch, dilated_factor, padding_mode='reflect', is_bias=True):
        super().__init__()
        self.ratio = 4 / 10
        self.padding = 10
        self.padding_mode = padding_mode
        self.shave = 10 + 2
        self.kernel_size = ks = 21

        self.d_conv = DSM_Convolution(head_ch, out_ch, kernel_size=ks, stride=1, padding=ks // 2, padding_mode=padding_mode, bias=is_bias)
        self.normal_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks), stride=(1, 1), padding=ks // 2, padding_mode=padding_mode, bias=is_bias)
        self.is_diamond_mask = True if hasattr(self.d_conv, 'mask') else False

        if self.is_diamond_mask:
            print('21 d_conv get mask')
        else:
            print('21 d_conv no mask')

        self.test_d_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks), stride=(ks, ks), padding=0, bias=is_bias)
        self.test_normal_conv = nn.Conv2d(head_ch, out_ch, kernel_size=(ks, ks), stride=(ks, ks), padding=0, bias=is_bias)

        self.f_conv = Dilation_fork_3x3(out_ch, out_ch, kernel_size=3, stride=1, dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode, bias=is_bias)
        self.p_conv = Dilation_plus_3x3(out_ch, out_ch, kernel_size=3, stride=1, dilation=dilated_factor, padding=dilated_factor, padding_mode=padding_mode, bias=is_bias)

        self.rf_scale = RF_scale(head_ch, out_ch, kernel_size=ks, padding=ks // 2, padding_mode=padding_mode, ratio=self.ratio)

    def forward(self, x: torch.Tensor, is_refine=False, f=5, pad=2):

        # Training
        if self.training :
            d_conv_x = self.d_conv(x)
            normal_conv_x = self.normal_conv(x)
            d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
            normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)
            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
            return out_x

        # Testing
        else:
            # testing w/o refine
            if is_refine is False:
                self.rf_scale.ratio = 2 / 5
                d_conv_x, normal_conv_x = self.forward_refine(x)
                d_conv_x = pixel_shuffle_down_sampling(d_conv_x, f, pad=pad)
                normal_conv_x = pixel_shuffle_down_sampling(normal_conv_x, f, pad=pad)

            # testing + refine
            else:
                self.rf_scale.ratio = 1 / 5
                d_conv_x, normal_conv_x = self.forward_refine(x)
                p = pad
                d_conv_x = F.pad(d_conv_x, (p, p, p, p))
                normal_conv_x = F.pad(normal_conv_x, (p, p, p, p))

            out_x = self.f_conv(d_conv_x) + self.p_conv(normal_conv_x)
            return out_x


    def forward_refine(self, x):
        n_GPUs_for_deformable = 1
        n_GPUs = min(n_GPUs_for_deformable, 4)

        b, c, h, w = x.shape

        top = slice(0, h // 2 + self.shave)
        bottom = slice(h - h // 2 - self.shave, h)
        left = slice(0, w // 2 + self.shave)
        right = slice(w - w // 2 - self.shave, w)
        x_chop = torch.cat([
            x[..., top, left],
            x[..., top, right],
            x[..., bottom, left],
            x[..., bottom, right]
        ])

        hole_chop = []
        full_chop = []
        assert h * w < 4 * 200000
        for i in range(0, 4, n_GPUs):
            x = x_chop[i:(i + n_GPUs)]
            x_offset = P.data_parallel(self.rf_scale, x, range(n_GPUs))

            self.test_d_conv.weight.data = self.d_conv.weight.data * self.d_conv.mask
            self.test_d_conv.bias = self.d_conv.bias

            self.test_normal_conv.weight.data = self.normal_conv.weight.data
            self.test_normal_conv.bias = self.normal_conv.bias

            temp_hole_chop = P.data_parallel(self.test_d_conv, x_offset, range(n_GPUs))
            temp_full_chop = P.data_parallel(self.test_normal_conv, x_offset, range(n_GPUs))
            del x_offset

            if not hole_chop:
                hole_chop = [c for c in temp_hole_chop.chunk(n_GPUs, dim=0)]
            else:
                hole_chop.extend(temp_hole_chop.chunk(n_GPUs, dim=0))

            if not full_chop:
                full_chop = [c for c in temp_full_chop.chunk(n_GPUs, dim=0)]
            else:
                full_chop.extend(temp_full_chop.chunk(n_GPUs, dim=0))

        """ merge """
        assert h % 2 == 0 and w % 2 == 0
        top = slice(0, h // 2)
        bottom = slice(h - h // 2, h)
        left = slice(0, w // 2)
        right = slice(w - w // 2, w)

        right_r = slice(w // 2 - w, None)
        bottom_r = slice(h // 2 - h, None)

        b, c = hole_chop[0].size()[:-2]
        d_conv_x = hole_chop[0].new(b, c, h, w)
        normal_conv_x = full_chop[0].new(b, c, h, w)

        d_conv_x[..., top, left] = hole_chop[0][..., top, left]
        d_conv_x[..., top, right] = hole_chop[1][..., top, right_r]
        d_conv_x[..., bottom, left] = hole_chop[2][..., bottom_r, left]
        d_conv_x[..., bottom, right] = hole_chop[3][..., bottom_r, right_r]

        normal_conv_x[..., top, left] = full_chop[0][..., top, left]
        normal_conv_x[..., top, right] = full_chop[1][..., top, right_r]
        normal_conv_x[..., bottom, left] = full_chop[2][..., bottom_r, left]
        normal_conv_x[..., bottom, right] = full_chop[3][..., bottom_r, right_r]

        return d_conv_x, normal_conv_x
