import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools

def read_img(env, path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("img = no")
        if size is not None:
            img = cv2.resize(img, (size[0], size[1]))
            # img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def read_img_fr(env, path, size=None):
    """read image by infrared   return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("img = no")
        if size is not None:
            img = cv2.resize(img, (size[0], size[1]))
    img = img.astype(np.float32) / 255.
    return img

def read_img_infrared_seq(path, size=None):
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))

    img_l = [read_img_fr(None, v, size) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

def read_img_seq(path, size=None):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """
    # print(path)
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))

    img_l = [read_img(None, v, size) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    try:
        imgs = imgs[:, :, :, [2, 1, 0]]
    except Exception:
        import ipdb; ipdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def augment_torch(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    # rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = flip(img, 2)
        if vflip:
            img = flip(img, 1)
        # if rot90:
        #     # import pdb; pdb.set_trace()
        #     img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def cmp(x, y):
    x_index = x.split('/')[-1]
    y_index = y.split('/')[-1]
    x_index = int(x_index)
    y_index = int(y_index)
    if x_index > y_index:
        return 1
    else:
        return -1

class ll_dataset(data.Dataset):
    def __init__(self, opt):
        super(ll_dataset, self).__init__()
        self.opt = opt
        self.GT_root, self.LQ_root, self.FR_root = opt['dataroot_GT'], opt['dataroot_LQ'], opt['dataroot_FR']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'path_FR': [], 'folder': [], 'idx': [], 'border': []}
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT, self.imgs_FR = {}, {}, {}

        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        subfolders_FR = util.glob_file_list(self.FR_root)

        count = 0
        for subfolder_LQ, subfolder_GT, subfolder_FR in zip(subfolders_LQ, subfolders_GT, subfolders_FR):
            #print(subfolder_LQ, subfolder_GT, subfolder_FR)
            subfolder_name = osp.basename(subfolder_LQ)

            img_paths_LQ = [subfolder_LQ]
            img_paths_GT = [subfolder_GT]
            img_paths_FR = [subfolder_FR]

            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['path_FR'].extend(img_paths_FR)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            self.data_info['idx'].append('{}/{}'.format(count, len(subfolder_LQ)))

            self.imgs_LQ[subfolder_name] = img_paths_LQ
            self.imgs_GT[subfolder_name] = img_paths_GT
            self.imgs_FR[subfolder_name] = img_paths_FR

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        img_LQ_path = self.imgs_LQ[folder][0]
        img_GT_path = self.imgs_GT[folder][0]
        img_FR_path = self.imgs_FR[folder][0]
        img_LQ_path = [img_LQ_path]
        img_GT_path = [img_GT_path]
        img_FR_path = [img_FR_path]

        if self.opt['phase'] == 'train':
            img_LQ, img_LQ_YCrBr = util.read_img_seq(img_LQ_path)
            img_GT, img_GT_YCrBr = util.read_img_seq(img_GT_path)
            img_FR = read_img_seq(img_FR_path)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_FR = img_FR[0]
            img_LQ_YCrBr = img_LQ_YCrBr[0]
            img_GT_YCrBr = img_GT_YCrBr[0]

            LQ_size = self.opt['LQ_size']
            GT_size = self.opt['GT_size']
            FR_size = self.opt['FR_size']
            _, H, W = img_GT.shape  # real img size

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_FR = img_FR[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_LQ_YCrBr = img_LQ_YCrBr[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
            img_GT_YCrBr = img_GT_YCrBr[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            img_LQ_l.append(img_FR)
            img_LQ_l.append(img_LQ_YCrBr)
            img_LQ_l.append(img_GT_YCrBr)
            rlt = augment_torch(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]
            img_FR = rlt[2]
            img_LQ_YCrBr = rlt[2]
            img_GT_YCrBr = rlt[3]

        elif self.opt['phase'] == 'test':
            img_LQ, img_LQ_YCrBr = util.read_img_seq(img_LQ_path)
            img_GT, img_GT_YCrBr = util.read_img_seq(img_GT_path)
            img_FR = read_img_seq(img_FR_path)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_FR = img_FR[0]
            img_LQ_YCrBr = img_LQ_YCrBr[0]
            img_GT_YCrBr = img_GT_YCrBr[0]

        else:
            img_LQ, img_LQ_YCrBr = util.read_img_seq(img_LQ_path)
            img_GT, img_GT_YCrBr = util.read_img_seq(img_GT_path)
            img_FR = read_img_seq(img_FR_path)
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_FR = img_FR[0]
            img_LQ_YCrBr = img_LQ_YCrBr[0]
            img_GT_YCrBr = img_GT_YCrBr[0]

        img_nf = img_LQ.permute(1, 2, 0).numpy() * 255.0
        img_nf = cv2.blur(img_nf, (5, 5))
        img_nf = img_nf * 1.0 / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

        return {
            'LQs': img_LQ,
            'GT': img_GT,
            'FR': img_FR,
            'nf': img_nf,
            'LQs_YCrBr': img_LQ_YCrBr,
            'GT_YCrBr': img_GT_YCrBr,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': 0
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
