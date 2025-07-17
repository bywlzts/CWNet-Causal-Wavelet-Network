import torch.nn
from models.archs.arch_util import *
from models.archs.SS2D_arch import SS2D6
from models.archs.ffc import *
from models.archs.wtconv.wtconv2d import *

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LightBlock(nn.Module):
    def __init__(self, dim):
        super(LightBlock, self).__init__()
        self.channel = dim
        self.SIM = nn.Sequential(
            LayerNorm2d(dim),
            FFCResnetBlock(dim),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, bias=True),
        )
        self.CIM = nn.Sequential(
            LayerNorm2d(dim),
            FFCResnetBlock(dim),
            nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        y = self.SIM(x) + x
        y = self.CIM(y) + y
        return y    


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class ProcessBlock(nn.Module):
    def __init__(self, dims, d_state=16, n_l_block=1, n_h_block=1, LayerNorm_type='WithBias'):
        super(ProcessBlock,self).__init__()
        self.dim = dims
        self.dwt = DWT(fuseh=False)
        self.idwt = IDWT()
        self.lnum = n_l_block
        self.hnum = n_h_block
        self.hhenhance = Depth_conv(self.dim, self.dim)
        self.llenhance = nn.ModuleList()
        for layer in range(2):
            self.llenhance.append(
                WTConv2d(dims, dims, kernel_size=5, wt_levels=3))

        self.hhmamba = nn.ModuleList()
        self.norm2 = LayerNorm(self.dim, LayerNorm_type)

        for layer in range(self.hnum):
            self.hhmamba.append(nn.ModuleList([
                SS2D6(d_model=dims, dropout=0, d_state=d_state, scan_type='lh'),
                PreNorm(dims, FeedForward(dim=dims))
            ]))

        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self.create_wave_conv()

        self.posenhance = nn.ModuleList()
        for layer in range(self.lnum):
            self.posenhance.append(
                LightBlock(self.dim))

        self.conv_fusechannel = nn.Conv2d(self.dim*2, self.dim, 1, stride=1, bias=False)

    def create_conv_layer(self, kernel):
        conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1, bias=False)
        conv.weight.data = kernel.repeat(self.dim, self.dim, 1, 1)  
        return conv

    def create_wave_conv(self):
        horizontal_kernel = torch.tensor([[1, 0, -1],
                                          [1, 0, -1],
                                          [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        vertical_kernel = torch.tensor([[1, 1, 1],
                                        [0, 0, 0],
                                        [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        diagonal_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        horizontal_conv = self.create_conv_layer(horizontal_kernel)
        vertical_conv = self.create_conv_layer(vertical_kernel)
        diagonal_conv = self.create_conv_layer(diagonal_kernel)
        return horizontal_conv, vertical_conv, diagonal_conv

    def forward(self, x):
        b, c, h, w = x.shape
        xori = x
        ll, hl, lh, hh = self.dwt(x)
        for layer in self.llenhance:
            ll = layer(ll)
        hh = torch.cat((hl, lh, hh), dim=0)
        hh = self.hhenhance(hh)
        e_hl, e_lh, e_hh = hh[:b, ...], hh[b:2 * b, ...], hh[2 * b:, ...]

        ll_hl = self.horizontal_conv(ll)
        ll_lh = self.vertical_conv(ll)
        ll_hh = self.diagonal_conv(ll)

        e_hl = torch.cat((e_hl, ll_hl), dim=1)
        e_lh = torch.cat((e_lh, ll_lh), dim=1)
        e_hh = torch.cat((e_hh, ll_hh), dim=1)

        e_high = torch.cat((e_hl, e_lh, e_hh), dim=0)
        e_high = self.conv_fusechannel(e_high)
        e_high = self.norm2(e_high)

        for (ss2d, ff) in self.hhmamba:
            y = e_high.permute(0, 2, 3, 1)
            e_high = ss2d(y) + e_high.permute(0, 2, 3, 1)
            e_high = ff(e_high) + e_high
            e_high = e_high.permute(0, 3, 1, 2)
        x_out = self.idwt(torch.cat((ll, e_high), dim=0)) + xori

        for layer in self.posenhance:
            x_out = layer(x_out)

        return x_out

class CWNet(nn.Module):
    def __init__(self, nc, n_l_blocks, n_h_blocks):
        super(CWNet,self).__init__()
        self.conv0 = nn.Conv2d(3,nc,1,1,0)
        self.conv1 = ProcessBlock(nc,d_state=16,  n_l_block=n_l_blocks[0], n_h_block=n_h_blocks[0])
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
        self.conv2 = ProcessBlock(nc*2,d_state=16,   n_l_block=n_l_blocks[1],n_h_block= n_h_blocks[1])
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
        self.conv3 = ProcessBlock(nc*3, d_state=16, n_l_block= n_l_blocks[2], n_h_block=n_h_blocks[2])
        self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
        self.conv4 = ProcessBlock(nc*2, d_state=16,  n_l_block=n_l_blocks[3], n_h_block=n_h_blocks[3])
        self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
        self.conv5 = ProcessBlock(nc,d_state=16,  n_l_block=n_l_blocks[4], n_h_block=n_h_blocks[4])
        self.convout = nn.Conv2d(nc,3,1,1,0)


    def forward(self, x):
        x_ori = x
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3,size=(x12.size()[2],x12.size()[3]),mode='bilinear'),x12],1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4,size=(x01.size()[2],x01.size()[3]),mode='bilinear'),x01],1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)
        xout = x_ori + xout

        return xout