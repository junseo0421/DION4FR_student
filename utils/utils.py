import torch
from torch.autograd import Variable
import math
import torch
import numpy as np
import torch.nn.functional as F

import os
import glob as _glob

def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches


def cos_function_weight(batchSize, imgSize, device):
    weight = torch.ones((imgSize, imgSize)) 
    for i in range(imgSize):
        weight[:, i] = (1. + math.cos(math.pi * i / float(imgSize-1))) * 0.5
    weight = weight.view(1,1,imgSize,imgSize).repeat(batchSize,1,1,1)
    return Variable(weight).cuda(device)

def gaussian_weight(batchSize, imgSize, device=0):
    weight = torch.ones((imgSize, imgSize)) 
    var = (imgSize/4)**2
    for i in range(imgSize):
        weight[:, i] = math.exp(-(float(i))**2/(2*var))
    weight = weight.view(1,1,imgSize,imgSize).repeat(batchSize,1,1,1)
    return Variable(weight).cuda(device)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x
    
def compute_fsp_matrix(feature1, feature2):
    """
    feature1: LSTM의 입력 특징 맵 (B, C, H, W)
    feature2: LSTM의 출력 특징 맵 (B, C, H, W)
    두 특징 맵 간의 FSP 매트릭스를 계산하는 함수
    """
    # 입력과 출력을 (B, C, -1) 크기로 변경
    B, C, H, W = feature1.size()
    feature1 = feature1.view(B, C, -1)
    feature2 = feature2.view(B, C, -1)

    # FSP 매트릭스를 계산 (특징 간의 행렬 내적)
    fsp_matrix = torch.bmm(feature1, feature2.transpose(1, 2)) / (H * W)

    return fsp_matrix
