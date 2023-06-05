import cv2
import numpy as np
import math
from scipy import signal
import os


def modify_mask_threshold_and_shape(mask, img_shape):
    mask[mask >= 128] = 255
    mask[mask < 128] = 0
    mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
    return mask


def plot_mosaic(img, mask, kernel_size=10):
    mask = modify_mask_threshold_and_shape(mask, img.shape)
    mask = mask // 255

    blur_img = cv2.blur(img, (kernel_size, kernel_size))
    mask_img = np.ones(img.shape, np.int8)
    mask_img[mask == 1] = 0
    mask_img_reverse = np.ones(img.shape, np.int8) - mask_img
    mosaic_img = mask_img * img + mask_img_reverse * blur_img
    return mosaic_img


def calculate_psnr(img_by_mask, mosaic_by_pre):
    mse = np.mean((img_by_mask / 255. - mosaic_by_pre / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calculate_ssim(img_by_mask, mosaic_by_pre):
    img_by_mask = img_by_mask.astype(np.float32)
    mosaic_by_pre = mosaic_by_pre.astype(np.float32)
    img_by_mask = cv2.cvtColor(img_by_mask, cv2.COLOR_BGR2GRAY)
    mosaic_by_pre = cv2.cvtColor(mosaic_by_pre, cv2.COLOR_BGR2GRAY)
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    # M, N = np.shape(img_by_mask)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img_by_mask = np.float64(img_by_mask)
    mosaic_by_pre = np.float64(mosaic_by_pre)

    mu1 = signal.convolve2d(img_by_mask, window, 'valid')
    mu2 = signal.convolve2d(mosaic_by_pre, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img_by_mask * img_by_mask, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(mosaic_by_pre * mosaic_by_pre, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img_by_mask * mosaic_by_pre, window, 'valid') - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim

def detect_face(img):
    img = img.astype('uint8')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier('callbacks/haarcascade_frontalface_default.xml')

    face = face_detect.detectMultiScale(gray, 1.01, 50, 0, (5, 5))
    return face


def calculate_tdir(img_by_mask, mosaic_by_pre):
    if len(detect_face(img_by_mask)) > 0:
        if len(detect_face(mosaic_by_pre)) > 0:
            return [1, 1]
        else:
            return [1, 0]
    else:
        return [0, 0]


def get_person_by_mask(img, mask):
    # 根据mask扣出img对应区域的图
    mask = modify_mask_threshold_and_shape(mask, img.shape)
    img[mask == 0] = 0
    return img


def tpcount(imgp,imgl):
    n = 0
    WIDTH = imgl.shape[0]
    HIGTH = imgl.shape[1]
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgp[i,j] == 255 and imgl[i,j] == 255:
                n = n+1
    return n


def fncount (imgp,imgl):
    n = 0
    WIDTH = imgl.shape[0]
    HIGTH = imgl.shape[1]
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 255 and imgp[i,j] == 0:
                n = n+1
    return n


def fpcount(imgp,imgl):
    n = 0
    WIDTH = imgl.shape[0]
    HIGTH = imgl.shape[1]
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 0 and imgp[i,j] == 255:
                n+=1
    return n


def tncount(imgp,imgl):
    n=0
    WIDTH = imgl.shape[0]
    HIGTH = imgl.shape[1]
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i,j] == 0 and imgp[i,j] == 0:
                n += 1
    return n


def cal_privacy_metric(img, mask, pre):
    mosaic = plot_mosaic(img, pre)

    mosaic_by_pre = get_person_by_mask(mosaic, pre)
    img_by_mask = get_person_by_mask(img, mask)

    res_tdir = calculate_tdir(img_by_mask, mosaic_by_pre)
    ssim = calculate_ssim(mosaic_by_pre, img_by_mask)
    psnr = calculate_psnr(mosaic_by_pre, img_by_mask)

    return res_tdir, ssim, psnr, tpcount(pre, mask), fncount(pre, mask), fpcount(pre, mask), tncount(pre, mask)