import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize
import os
from os.path import join
from models.build4 import build_model
from dataset import dataset_test4
import argparse
import skimage
from skimage import io
from scipy.ndimage import distance_transform_edt
import skimage.transform
import time
from torchvision.utils import save_image

from utils.utils import *

from models.unet.sep_unet_model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Evaluate function
def evaluate(gen, eval_loader, rand_pair, save_dir):
    gen.eval()

    com_total = 0
    for batch_idx, (gt, iner_img, mask_img, name, fol) in enumerate(eval_loader):

        os.makedirs(join(save_dir, fol[0]), exist_ok=True)
        imgSize = gt.shape[2]

        gt, iner_img, mask_img = Variable(gt).cuda(), Variable(iner_img).cuda(), Variable(
            mask_img.type(torch.FloatTensor)).cuda()

        with torch.no_grad():
            t_start = time.time()
            I_pred, _ = gen(mask_img)
            t_end = time.time()
            comsum = t_end - t_start
            com_total += comsum

        for i in range(gt.size(0)):
            pre_img = np.transpose(I_pred[i].data.cpu().numpy(), (1, 2, 0))
            std_ = np.expand_dims(np.expand_dims(np.array(std), 0), 0)
            mean_ = np.expand_dims(np.expand_dims(np.array(mean), 0), 0)
            real = np.transpose(gt[i].data.cpu().numpy(), (1, 2, 0))
            real = real * std_ + mean_
            real = np.clip(real, 0, 1)

            iner = np.transpose(iner_img[i].data.cpu().numpy(), (1, 2, 0))

            iner = iner * std_ + mean_
            iner = np.clip(iner, 0, 1)

            pre_img = pre_img * std_ + mean_
            pre_img = np.clip(pre_img, 0, 1)

            io.imsave(join(save_dir, fol[0], '%s.bmp' % (name[i])), skimage.img_as_ubyte(pre_img))

    avg = com_total / len(eval_loader)
    print(f'Average processing time: {avg:.4f} seconds')


if __name__ == '__main__':

    TEST_DATA_DIR = '/content/datasets/HKPU_A_CROP_W25P_V2' 
    SAVE_DIR = '/content/drive/MyDrive/dq_unet_sep4_sobel_weight_10/output/HKdb-2/test_result' 

    # TEST_DATA_DIR = '/content/datasets/HKPU_B_CROP_W25P_V2' 
    # SAVE_DIR = '/content/drive/MyDrive/dq_unet_sep4_sobel_weight_10/output/HKdb-1/test_result'  

    # TEST_DATA_DIR = '/content/datasets/SDU_A_original_CROP_W25P_V2'  
    # SAVE_DIR = '/content/drive/MyDrive/dq_unet_sep4_sobel_weight_10/output/SDdb-2/test_result' 

    # TEST_DATA_DIR = '/content/datasets/SDU_B_original_CROP_W25P_V2' 
    # SAVE_DIR = '/content/drive/MyDrive/dq_unet_sep4_sobel_weight_10/output/SDdb-1/test_result'  


    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--rand_pair', type=bool, help='pair testing data randomly', default=True)
        parser.add_argument('--test_data_dir', type=str, help='directory of testing data', default=TEST_DATA_DIR)
        opts = parser.parse_args()
        return opts


    args = get_args()

    # Experiment settings
    pred_step = 1
    times = 1
    input_size = [128, 86, 56, 38]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # List of epochs to test
    # epoch_list = list(range(211, 301, 1))  # 24.10.10 HKdb-2 test에 맞춰 변경함
    # epoch_list = list(range(210, 250, 10)) + list(range(260, 300, 10)) + list(range(310, 350, 10))  # 24.09.24 SDDB-2 test에 맞춰 변경함
    epoch_list = list(range(200, 550, 50))

    # epoch_list = []
    # for j in range(200, 510, 10):
    #     if j % 50 != 0:
    #         epoch_list.append(j)

    # Load data
    print('Loading data...')
    transformations = transforms.Compose([Resize((192, 128)), ToTensor(), Normalize(mean, std)])
    tds = glob(args.test_data_dir, '*/*.bmp', True)
    eval_data = dataset_test4(root=args.test_data_dir, transforms=transformations, imgSize=192,
                              inputsize=input_size[times - 1], pred_step=pred_step, imglist=tds)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)

    for epoch in epoch_list:
        # Construct weight path and save directory based on the epoch
        load_weight_dir = f'/content/drive/MyDrive/dq_unet_sep4_sobel_weight_10/output/HKdb-2/checkpoints/Gen_former_{epoch}.pt'  # 24.10.10 HKdb-2 test에 맞춰 변경함
        save_dir_epoch = join(SAVE_DIR, f'epoch_{epoch}')

        # Create save directory if not exists
        os.makedirs(save_dir_epoch, exist_ok=True)

        # Initialize the model
        print(f'Initializing model for epoch {epoch}...')
        gen = DQ_Thin_Sep_UNet_4(n_channels=3, n_classes=3).cuda()  # U-net

        # Load pre-trained weight
        print(f'Loading model weight for epoch {epoch}...')
        gen.load_state_dict(torch.load(load_weight_dir))

        # Evaluate
        print(f'Evaluating model for epoch {epoch}...')
        evaluate(gen, eval_loader, args.rand_pair, save_dir_epoch)

    print('All experiments completed.')
