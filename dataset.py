from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import random
import os
from os.path import join, splitext, basename
from glob import glob
import pandas as pd
import math
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop

def rand_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and random crop to target size

    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = random.randint(0, width - target_width)
        starting_y = random.randint(0, height - target_height)
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = random.randint(0, new_width - target_width)
        starting_y = random.randint(0, new_height - target_height)
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

## 2023 11 14 홍진성 사용
def center_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and center crop to target size
    
    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = (width - target_width) / 2
        starting_y = (height - target_height) / 2
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = (new_width - target_width) / 2
        starting_y = (new_height - target_height) / 2
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

class dataset_norm(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, imglist1=[], imglist2=[], imglist3=[]):  # 24.09.20 수정
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        #self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.inputsize = inputsize

        self.img_list1 = imglist1
        self.img_list2 = imglist2
        self.img_list3 = imglist3

        # for name in file_list:
        #     img = Image.open(name)
        #     if (img.size[0] >= (self.imgSize)) and (img.size[1] >= self.imgSize):
        #         self.img_list += [name]

        self.size = len(self.img_list1)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region
        index = index % self.size

        # 각 경로에서 이미지를 그레이스케일로 로드
        img = Image.open(self.img_list1[index]).convert("RGB")  # GT

        img1 = Image.open(self.img_list1[index]).convert("L")
        img2 = Image.open(self.img_list2[index]).convert("L")
        img3 = Image.open(self.img_list3[index]).convert("L")

        # 이미지를 numpy 배열로 변환
        img1 = np.array(img1)
        img2 = np.array(img2)
        img3 = np.array(img3)

        # 세 이미지 채널을 결합하여 RGB 이미지를 만듭니다.
        img_cat = np.stack([img1, img2, img3], axis=-1)  # (H, W, C) 형식으로 결합

        # 변환을 적용하기 위해 이미지를 PIL 이미지로 변환
        img_cat = Image.fromarray(img_cat)

        # 변환 적용
        if self.transforms:
            img = self.transforms(img)
            img_cat = self.transforms(img_cat)  # 자동으로 C이 index 0으로 옮겨짐

        i = (self.imgSize - self.inputsize)//2  # (192 - 128) / 2 = 32

        ## 2023 11 14 홍진성 마스킹 수정 (너비 방향으로만 처리)
        iner_img = img_cat[:, :, i:i+self.inputsize]
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:, :, i:i + self.inputsize] = iner_img

        # # 텐서로 변환하여 저장
        # mask_img_tensor = torch.tensor(mask_img, dtype=torch.float32)
        #
        # # 저장할 경로 설정
        # save_dir = r'D:\DION4FR\output\HKdb-2\mask_images'  # 저장할 디렉토리 경로를 설정
        # os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
        #
        # # 파일 이름 구성 및 저장
        # file_name = f'mask_image_{index}.png'
        # file_path = os.path.join(save_dir, file_name)
        # save_image(mask_img_tensor, file_path)

        return img, mask_img

    def __len__(self):
        return self.size



## 2023 11 14 홍진성 사용
class dataset_test4(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1, imglist=[]):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        self.inputsize2 = inputsize + 64 * (pred_step - 1)

        self.img_list = imglist

        # for name in file_list:
        #     img = Image.open(name)
        #     #if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
        #     if (img.size[0] >= 0) and (img.size[1] >= 0):
        #         self.img_list += [name]

        self.size = len(self.img_list)
    ## 128 x 128 real 이고 나머지는 마스킹 영역
    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')

        i = (self.imgSize - self.inputsize) // 2
        # j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img

        ## 2023 11 14 홍진성 마스킹 수정 (너비 방향으로만 처리)
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
        #mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i:i + self.inputsize2, i:i+self.inputsize2] = img
        else:
            mask_img[:, :, i:i + self.inputsize2] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0], name.replace('\\', '/').split('/')[-2]
        # return torch.rand((3,192,192)), torch.rand((3,192,192)), torch.rand((3,192,192)), '', ''

    def __len__(self):
        return self.size
        # return 10000

class dataset_test3(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, crop='center', rand_pair=False, imgSize=192, inputsize=128):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.rand_pair = rand_pair
        self.inputsize = inputsize
        if crop == 'rand':
            self.cropFunc = rand_crop
        elif crop == 'center':
            self.cropFunc = center_crop
        else:
            self.cropFunc = None

        file_list = sorted(glob(join(root, '*.jpg')))

        for name in file_list:
            img = Image.open(name)
            if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
                self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2

        if self.cropFunc is not None:
            img = self.cropFunc(img, self.imgSize, self.imgSize)

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i+self.inputsize]
        mask_img = np.zeros((3, self.imgSize, self.imgSize))
        mask_img[:, i:i + self.inputsize, i:i + self.inputsize] = iner_img
        #mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        #mask_img = img * mask
        #mask_img = np.ones((3, self.imgSize, self.imgSize))
        #mask_img[:, i:i + self.inputsize, i:i + self.inputsize] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize

        file_list = sorted(glob(join(root, '*.png')))

        for name in file_list:
            img = Image.open(name)
            # if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
            if (img.size[0] >= 0) and (img.size[1] >= 0):
                self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2
        j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        #iner_img = img[:, i:i + self.inputsize, :]
        #iner_img = iner_img[:, :, i:i + self.inputsize]
        # mask_img = np.zeros((3, self.imgSize, self.imgSize))
        # mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        # mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
            # mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i:i + self.imgSize, i:i + self.imgSize] = img
        else:
            mask_img[:, i:i + self.inputsize2, i:i + self.inputsize2] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi2(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        #self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        #self.cropsize = int(imgSize//(1.5**pred_step))

        self.img_list = sorted(glob(join(root, '*.jpg')))

        # for name in file_list:
        #     img = Image.open(name)
        #     # if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
        #     if (img.size[0] >= 0) and (img.size[1] >= 0):
        #         self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2
        #j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        #crop = img[:,]

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i + self.inputsize]
        crop_img = Resize(128)(iner_img)
        # mask_img = np.zeros((3, self.imgSize, self.imgSize))
        # mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        # mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = crop_img

        return img, crop_img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi3(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.file_list = sorted(glob(join(root, '*.png')))

        self.size = len(self.file_list)

    def __getitem__(self, index):

        index = index % self.size
        name = self.file_list[index]
        img = Image.open(name).convert('RGB')


        if self.transforms is not None:
            img = self.transforms(img)

        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi4(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.file_list = sorted(glob(join(root, '*.jpg')))

        self.size = len(self.file_list)

    def __getitem__(self, index):

        index = index % self.size
        name = self.file_list[index]
        img = Image.open(name)


        if self.transforms is not None:
            img = self.transforms(img)

        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size
