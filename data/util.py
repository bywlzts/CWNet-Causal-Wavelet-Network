import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2
import albumentations as A
from PIL import Image, ImageFile

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*' )))


###################### read images ######################
def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img

def guideFilter(I, p, winSize, eps):
    mean_I = cv2.blur(I, winSize)
    mean_p = cv2.blur(p, winSize)
    mean_II = cv2.blur(I * I, winSize)
    mean_Ip = cv2.blur(I * p, winSize)
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)
    q = mean_a * I + mean_b
    return q

def syn_low(img, light, img_gray, light_max=5, light_min=2, noise_max=0.08, noise_min=0.03):
    light = guideFilter(light, img_gray, (3, 3), 0.01)[:, :, np.newaxis]
    n = np.random.uniform(noise_min, noise_max)
    R = img / (light + 1e-7)
    L = (light + 1e-7) ** np.random.uniform(light_min, light_max)
    return np.clip(R * L + np.random.normal(0, n, img.shape), 0, 1)

def random_color_distortion_with_noise_large(image):
    transform = A.Compose([
        # A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=(0.3, 0.7), p=1.0), 
        A.HueSaturationValue(hue_shift_limit=(-60, 60), sat_shift_limit=(-80, 80), val_shift_limit=0, p=1.0),
        A.RGBShift(r_shift_limit=(-80, 80), g_shift_limit=(-160, 160), b_shift_limit=(-160, 160), p=1.0),
        # A.ChannelShuffle(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(20, 60), p=0.5)
    ])
    augmented = transform(image=image)
    return augmented['image']

def color_invention(img_path):
    with Image.open(img_path) as pil_img:
        image_rgb = np.array(pil_img)

    transformed_image = random_color_distortion_with_noise_large(image_rgb)
    transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    return transformed_image_bgr

def light_invention(hq_file, light_file):
    with Image.open(hq_file) as pil_img:
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

    w, h, _ = img.shape
    with Image.open(light_file) as light_img:
        light = np.array(light_img)
        light = cv2.cvtColor(light, cv2.COLOR_RGB2BGR)
        light = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY) / 255.0

    lq = img.copy() / 255.0
    lq = syn_low(lq, light, img_gray)
    out = lq * 255.0
    return out

def read_img_invention(env, path, lightmap_path, size=None):
    img_color_invention = color_invention(path)
    img_light_invention = light_invention(path, lightmap_path)

    img_color_invention = img_color_invention.astype(np.float32) / 255.
    img_light_invention = img_light_invention.astype(np.float32) / 255.

    return img_color_invention, img_light_invention

def read_img_seq_invention(path, lightmap_path, size=None):
    if type(path) is list:
        img_path_l = path
        light_map_path = lightmap_path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
        light_map_path = sorted(glob.glob(os.path.join(lightmap_path, '*')))

    img_invention = [read_img_invention(None, v, g, size)
                                                for v, g in zip(img_path_l, light_map_path)]
    img_color_invention, img_light_invention = img_invention[0]

    # stack to Torch tensor
    img_color_invention = np.stack([img_color_invention], axis=0)
    img_light_invention = np.stack([img_light_invention], axis=0)

    try:
        img_color_invention = img_color_invention[:, :, :, [2, 1, 0]]
        img_light_invention = img_light_invention[:, :, :, [2, 1, 0]]
    except Exception:
        import ipdb; ipdb.set_trace()
    img_color_invention = torch.from_numpy(np.ascontiguousarray(np.transpose(img_color_invention, (0, 3, 1, 2)))).float()
    img_light_invention = torch.from_numpy(np.ascontiguousarray(np.transpose(img_light_invention, (0, 3, 1, 2)))).float()
    return img_color_invention, img_light_invention

def read_img(env, path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('img no')
        if size is not None:
            img = cv2.resize(img, (size[0], size[1]))
    else:
        img = _read_img_lmdb(env, path, size)

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def read_img_seq(path, size=None):
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


def read_img_pil(env, path, size=None):
    if env is None:
        try:
            with Image.open(path) as pil_img:
                if pil_img.mode != 'RGB':
                    if pil_img.mode == 'RGBA':
                        pil_img = pil_img.convert('RGB')
                    elif pil_img.mode == 'L':
                        pil_img = pil_img.convert('RGB')
                    else:
                        pil_img = pil_img.convert('RGB')

                if size is not None:
                    pil_img = pil_img.resize((size[0], size[1]), Image.LANCZOS)

                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"[ERROR] Failed to read image with PIL: {path}")
            print(f"[ERROR] Exception: {e}")
            try:
                print(f"[FALLBACK] Trying OpenCV for: {path}")
                img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    print(f"[ERROR] OpenCV also failed: {path}")
                    return None
                if size is not None:
                    img_bgr = cv2.resize(img_bgr, (size[0], size[1]))
            except Exception as e2:
                print(f"[ERROR] OpenCV fallback failed: {e2}")
                return None
    else:
        img_bgr = _read_img_lmdb(env, path, size)

    img_bgr = img_bgr.astype(np.float32) / 255.0

    if img_bgr.ndim == 2:
        img_bgr = np.expand_dims(img_bgr, axis=2)
    if img_bgr.shape[2] > 3:
        img_bgr = img_bgr[:, :, :3]

    return img_bgr

def read_img_seq_pil(path, size=None):
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
    img_l = []

    for i, img_path in enumerate(img_path_l):
        try:
            img_bgr = read_img_pil(None, img_path, size)
            if img_bgr is None:
                print(f"[WARNING] Skipping failed image: {img_path}")
                continue
            img_l.append(img_bgr)
        except Exception as e:
            print(f"[ERROR] Failed to process image {img_path}: {e}")
            continue

    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l


####################
# image processing
# process on numpy image
####################

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


def channel_convert(in_c, tar_type, img_list):
    """conversion among BGR, gray and y"""
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    """same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    """img_in: Numpy, HWC or HW"""
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

