import torch
import torch.nn as nn
import torch.nn.functional as F
import time

""" Receptive field scaling """
class RF_scale(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, padding_mode='reflect', ratio=1):
        super(RF_scale, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.ratio = ratio
        assert stride == 1

        if padding_mode == 'reflect':
            self.module_padding = nn.ReflectionPad2d(padding)
        else:
            self.module_padding = nn.ZeroPad2d(padding)

    def forward(self, x:torch.Tensor):
        # print(x.shape)
        b, c, h, w = x.shape
        ks = self.kernel_size
        N = self.kernel_size**2

        if self.padding:
            x = self.module_padding(x)

        # a = time.time()
        # (b, 2N, h, w) where 2N represents convolution for each position, compared to the center of the convolution kernel,
        # the new relative position (float) of the rest of the points
        p = self._get_p(N,h,w)
        p = p.to(x.device)

        # Generate absolute coordinates for each point relative to the center of the convolution kernel
        # b = time.time()

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()  # Obtain the coordinates of the upper left corner after obtaining the absolute coordinates
        q_rb = q_lt + 1  # Obtain the coordinates of the lower right corner

        # c = time.time()
        # Clamp the coordinates of the upper left and lower right corners within the range of the feature map
        # q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lt = torch.cat([q_lt[..., :N], q_lt[..., N:]], dim=-1)  # Because only scaling operations are performed, there is no overflow
        q_rb = torch.cat([q_rb[..., :N], q_rb[..., N:]], dim=-1).long()
        # Obtain the coordinates of the upper right and lower left corners based on the upper left and lower right corners
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # d = time.time()
        # Clip p Adjust the range of p ——> Actually, there is no need to control overflow
        # p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))  # 1+x0-x * 1+y0-y
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # N is x coordinate; N: is y coordinate
        # e = time.time()

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # f = time.time()


        # (b, c, h, w, N) ——>
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        # After offset, face the new input for each convolution kernel

        # g = time.time()
        x_offset = self._reshape_x_offset(x_offset, ks)  # B-C-H-W-N
        # out = self.conv(x_offset)
        # print(time.time()-g,g-f,f-e,e-d,d-c,c-b,b-a)

        # return out
        return x_offset

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),indexing='ij')

        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        # The size of each convolution kernel is N, generating relative displacement of size kernel_size * kernel_size, in both x and y directions.
        # --> Here is a vector, where the first half is the flattened x coordinates row by row and column by column,
        # and then the corresponding y coordinates. Therefore, if you take the first half and the second half of p_n,
        # the corresponding positions will form a set of coordinate points.

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h, self.stride), # Note: the stride is 1 here
            torch.arange(0, w, self.stride),indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        # The central position of each convolution kernel. It may not be the same size as the input x (when there is dilation, p0 appears at intervals).
        p_0 = p_0 + self.kernel_size // 2  # -- Note: an offset is introduced here

        return p_0

    def _get_p(self, N,h,w, dtype=torch.float32):

        # (1, 2N, 1, 1) Obtain pixel offset coordinates for the convolution kernel
        p_n = self._get_p_n(N, dtype)

        # (1, 2N, h, w) Obtain the central pixel coordinates for each location
        p_0 = self._get_p_0(h, w, N, dtype)

        # (1, 2N, h, w) New pixel positions corresponding to the scaled convolution kernel
        p = p_0 + p_n * self.ratio

        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # print(x.device,index.device)
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

