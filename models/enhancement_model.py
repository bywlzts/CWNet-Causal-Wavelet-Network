import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
from models.archs.segment.hrseg_model import create_hrnet
from models.loss import CharbonnierLoss, VGGLoss, SSIM, ContrastLoss, CLIPLOSS

logger = logging.getLogger('base')

class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        ####  segment
        if opt['seg']:
            print(" ********************  load segment model *********************")
            self.seg_model = create_hrnet().cuda()
            self.seg_model.eval()
        else:
            self.seg_model = None

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']

            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))

            self.is_vgg_loss = train_opt['vgg_loss']
            self.l_pix_w = train_opt['pixel_weight']
            self.cri_pix_ill = nn.MSELoss(reduction='sum').to(self.device)
            self.cri_pix_ill2 = nn.MSELoss(reduction='sum').to(self.device)
            self.con_loss = ContrastLoss().to(self.device)
            self.cri_vgg = VGGLoss().to(self.device)
            self.ssim_loss = SSIM().to(self.device)
            self.l1_loss = torch.nn.L1Loss().to(self.device)
            self.clip_loss = CLIPLOSS().to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()


            self.log_dict = OrderedDict()

    def combine_elements(self, arr1, arr2):
        combine_dict = {}
        for i in range(len(arr1)):
            combine_dict[i] = {arr1[i], arr2[i]}
        combine_list = list(combine_dict.values())
        return combine_list

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.neg_H = data['NEG'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        if self.seg_model is not None:
            self.seg_map, self.seg_feature = self.seg_model(self.real_H)
        else:
            self.seg_map, self.seg_feature = None, None

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H  = self.netG(self.var_L)
        _, _, H, W = self.real_H.shape
        c_loss = self.con_loss(self.var_L, self.real_H, self.neg_H, self.fake_H) * 0.01
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_ssim = (1 - self.ssim_loss(self.fake_H, self.real_H))* 0.3

        vgg_loss_state = False
        if self.is_vgg_loss:
            l_vgg = self.l_pix_w * self.cri_vgg(self.fake_H, self.real_H) * 0.2
            vgg_loss_state = True

        clip_loss_state = False
        if self.seg_map is not None and step % 200 == 0:
            l_clip = self.clip_loss(self.seg_map, self.fake_H, self.real_H) * 0.01
            if vgg_loss_state:
                l_final = l_pix + l_ssim + l_clip + c_loss + l_vgg
            else:
                l_final = l_pix + l_ssim + l_clip + c_loss
            clip_loss_state = True
        else:
            if vgg_loss_state:
                l_final = l_pix + l_ssim + c_loss + l_vgg
            else:
                l_final = l_pix + l_ssim + c_loss


        l_final.backward()

        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.01)
        self.optimizer_G.step()
        if clip_loss_state:
            self.log_dict['l_clip'] = l_clip.item()
        self.log_dict['c_loss'] = c_loss.item()
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_ssim'] = l_ssim.item()
        if vgg_loss_state:
            self.log_dict['l_vgg'] = l_vgg.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del self.real_H
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
