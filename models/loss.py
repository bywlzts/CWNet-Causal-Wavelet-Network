import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.transforms import Resize
#import models.archs.clip.open_clip
from models.archs.clip.CLIP_model.clip import tokenize, load
from torchvision.transforms import ToTensor

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, inpput, gt):
        style_loss = self.l1(gram_matrix(inpput),
                             gram_matrix(gt))
        return style_loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class WGANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(WGANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            self.loss = self.wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def wgan_loss(self, input, target):
        if target:
            return -input.mean()   # minimize this for real samples
        else:
            return input.mean()    # maximize this for fake samples

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class FourierAmplitudeLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super(FourierAmplitudeLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Invalid loss_type. Use 'l1' or 'l2'.")

    def forward(self, input, target):
        fft_input = torch.fft.fftn(input, dim=(-2, -1))
        fft_target = torch.fft.fftn(target, dim=(-2, -1))

        mag_input = torch.abs(fft_input)
        mag_target = torch.abs(fft_target)
        loss = self.loss(mag_input, mag_target)
        return loss


class FourierPhaseLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super(FourierPhaseLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Invalid loss_type. Use 'l1' or 'l2'.")

    def forward(self, input, target):
        fft_input = torch.fft.fftn(input, dim=(-2, -1))
        fft_target = torch.fft.fftn(target, dim=(-2, -1))
        phase_input = torch.angle(fft_input)
        phase_target = torch.angle(fft_target)
        loss = self.loss(phase_input, phase_target)
        return loss

class CombinedFourierLoss(nn.Module):
    def __init__(self, amplitude_weight=1.0, phase_weight=1.0, loss_type='l1'):
        super(CombinedFourierLoss, self).__init__()
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight
        self.amplitude_loss = FourierAmplitudeLoss(loss_type)
        self.phase_loss = FourierPhaseLoss(loss_type)

    def forward(self, input, target):
        amplitude_loss = self.amplitude_loss(input, target)
        phase_loss = self.phase_loss(input, target)
        total_loss = self.amplitude_weight * amplitude_loss + self.phase_weight * phase_loss
        return total_loss

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.cri_pix = nn.MSELoss(reduction='sum')
        self.model = vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '15': "relu3_3"
        }

    def gen_features(self, x):
        output = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output
    def forward(self, inp, pos, neg, out):
        inp_t = inp
        inp_x0 = self.gen_features(inp_t)
        pos_t = pos
        pos_x0 = self.gen_features(pos_t)
        out_t = out
        out_x0 = self.gen_features(out_t)
        neg_t, neg_x0 = [],[]
        for i in range(neg.shape[1]):
            neg_i = neg[:,i,:,:]
            neg_t.append(neg_i)
            neg_x0_i = self.gen_features(neg_i)
            neg_x0.append(neg_x0_i)
        loss = 0
        for i in range(len(pos_x0)):
            pos_term = self.l1(out_x0[i], pos_x0[i].detach())
            inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
            neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
            loss = loss + pos_term / (inp_term+neg_term+1e-7)
        return loss / len(pos_x0)

class CLIPLOSS(nn.Module):
    def __init__(self):
        super(CLIPLOSS, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_resize = Resize([224, 224])
        self.text = tokenize(["high light image", "low light image"]).to("cuda" if torch.cuda.is_available() else "cpu")
        self.real_T = torch.Tensor([1., 0.]).to("cuda" if torch.cuda.is_available() else "cpu")
        self.CLIP, self.t = load("ViT-B/32", device='cuda')
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, seg_pred, input_1, input_2, type='CLIP', device='cuda'):
        N, C, H, W = seg_pred.shape
        seg_pred_cls = seg_pred.reshape(N, C, -1).argmax(dim=1)  # (N, H*W)
        input_1 = input_1.reshape(N, 3, -1)  # (N, 3, H*W)
        input_2 = input_2.reshape(N, 3, -1)  # (N, 3, H*W)

        image_list = []

        for n in range(N):
            cls = seg_pred_cls[n]
            img2 = input_2[n]
            for c in range(C):
                cls_index = torch.nonzero(cls == c).squeeze()

                if cls_index.numel() == 0:
                    continue

                segmented_part = input_1[n][:, cls_index]
                combined_image = img2.clone()

                combined_image[:, cls_index] = segmented_part
                combined_image = combined_image.permute(1, 0)
                combined_image = combined_image.reshape(3, H, W)
                image_list.append(combined_image)

        if type == 'CLIP':
            output_clip = torch.stack([self.torch_resize(img) for img in image_list]).to(self.device)
            output_clip = output_clip.permute(0, 2, 3, 1)
            output_clip = output_clip.permute(0, 3, 1, 2)

            target = torch.tensor([0] * len(image_list)).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                logits_per_image, logits_per_text = self.CLIP(output_clip, self.text)
                loss_CLIP = self.criterion(logits_per_image, target)
                return loss_CLIP

