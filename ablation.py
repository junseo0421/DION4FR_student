import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop
import os
from os.path import join, basename, splitext
from utils.loss import *
from models.Discriminator_ml import MsImageDis
from tensorboardX import SummaryWriter
from dataset import dataset_norm, dataset_norm_ablation
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from loss import *
import timm

from utils.utils import *

from models.unet.sep_unet_model import *

from tqdm import tqdm
import time

# this version is with normlized input with mean and std, all layers are normalized,
# change the order of the 'make_layer' with norm-activate-conv,and use the multi-scal D
# use two kind feature, horizon and vertical

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

## 24.10.11 model parameter 측정 위함
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Swin-Transformer와 TSP module (LSTM_small2) 파라미터 수를 출력하는 함수
def print_model_parameters(gen):
    print("generator : ", gen.__class__.__name__)

    print("Calculating total model parameters...")

    # 전체 모델 파라미터 수
    total_params = count_parameters(gen)
    print(f"Total parameters in the Student model: {total_params}")

# Training
def train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer, teacher_gen, is_afa_loss, is_sobel=True, afa_layer=3):  # 24.09.19 recognizer
    gen.train()
    dis.train()

    mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)  # 평균 절대 오차(MAE)를 사용하여 픽셀 간의 차이 계산
    mrf = IDMRFLoss(device=0)  # 텍스처 일관성 평가
    ssim_loss = SSIM_loss().cuda(0)  # 구조적 유사성
    sobel_loss = Sobel_loss().cuda(0)
    ofd_loss = OFD().cuda(0)

    acc_pixel_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    acc_ssim_loss = 0
    acc_total_sobel_loss = 0

    acc_original_kd_loss = 0  # 24.10.14 original KD loss
    acc_feature_kd_loss = 0
    acc_afa_kd_loss = 0

    total_gen_loss = 0

    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}") as pbar:
        for batch_idx, (gt, mask_img) in enumerate(train_loader):  # 24.09.20 수정
            batchSize = mask_img.shape[0]
            imgSize = mask_img.shape[2]

            gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)

            iner_img = gt[:, :, :, 32:32 + 128]  # 가로로 32~160 pixel

            ## Generate Image
            I_pred, features_s = gen(mask_img)
            f1_s, f2_s, f3_s, f4_s = features_s["x1"], features_s["x2"], features_s["x3"], features_s["x4"]

            if is_afa_loss:
                f1_s_afa, f2_s_afa, f3_s_afa = features_s["x1_out"], features_s["x2_out"], features_s["x3_out"]
            else:
                pass

            mask_pred = I_pred[:, :, :, 32:32 + 128]  # 생성된 image의 일부분 선택

            ## Compute losses
            ## Update Discriminator
            opt_dis.zero_grad()
            dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)  # 생성된 image와 gt와의 구별 능력 학습
            dis_loss = dis_adv_loss
            dis_loss.backward()
            opt_dis.step()  # 가중치 update

            # Pixel Reconstruction Loss
            pixel_rec_loss = mae(I_pred, gt) * 20  # pixel 재구성 손실

            # Texture Consistency Loss (IDMRF Loss)
            mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize  # 텍스처 일관성 손실

            ### SSIM loss
            left_loss = ssim_loss(I_pred[:, :, :, 0:32], I_pred[:, :, :, 32:64])
            right_loss = ssim_loss(I_pred[:, :, :, 160:192], I_pred[:, :, :, 128:160])
            total_ssim_loss = left_loss + right_loss

            ### Sobel loss
            if is_sobel:
                sobel_left_loss = sobel_loss(I_pred[:, :, :, 0:32], gt[:, :, :, 0:32])
                sobel_right_loss = sobel_loss(I_pred[:, :, :, 160:192], gt[:, :, :, 160:192])
                total_sobel_loss = sobel_left_loss + sobel_right_loss

                sobel_loss_weight = 20.0

                total_sobel_loss = total_sobel_loss * sobel_loss_weight
            else:
                pass

            ### original KD loss
            with torch.no_grad():
                teacher_pred, features_t = teacher_gen(mask_img)  # (B, C, H, W) 형태의 Teacher 출력
                f1_t, f2_t, f3_t, f4_t = features_t["x1"], features_t["x2"], features_t["x3"], features_t["x4"]  # model forward 수정해야함!!!!

                if is_afa_loss:
                    f1_t_afa, f2_t_afa, f3_t_afa = features_t["x1_out"], features_t["x2_out"], features_t["x3_out"]
                else:
                    pass

            original_kd_loss = mae(teacher_pred, I_pred) * 20  # 가중치는 pxiel_rec_loss와 똑같이 설정. 나중에 조절 필요할 수도

            ### feature KD loss
            feature_kd_loss = ofd_loss(f1_s, f1_t) + ofd_loss(f2_s, f2_t) + ofd_loss(f3_s, f3_t) + ofd_loss(f4_s, f4_t)

            if is_afa_loss:
                if afa_layer == 1:
                    afa_kd_loss = mse(f3_s_afa, f3_t_afa)
                elif afa_layer == 2:
                    afa_kd_loss = mse(f2_s_afa, f2_t_afa) + mse(f3_s_afa, f3_t_afa)
                else:
                    afa_kd_loss = mse(f1_s_afa, f1_t_afa) + mse(f2_s_afa, f2_t_afa) + mse(f3_s_afa, f3_t_afa)

                afa_kd_loss_weight = 1.0
                afa_kd_loss = afa_kd_loss_weight * afa_kd_loss
            else:
                pass

            # ## Update Generator
            gen_adv_loss = dis.calc_gen_loss(I_pred, gt)  # generator에 대한 적대적 손실

            # 24.10.14 original kd loss
            if is_sobel and is_afa_loss:
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + total_sobel_loss + original_kd_loss + feature_kd_loss + afa_kd_loss
            elif is_sobel and (not is_afa_loss):
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + total_sobel_loss + original_kd_loss + feature_kd_loss
            elif (not is_sobel) and is_afa_loss:
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + original_kd_loss + feature_kd_loss + afa_kd_loss
            else:
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + original_kd_loss + feature_kd_loss

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            acc_pixel_rec_loss += pixel_rec_loss.data
            acc_gen_adv_loss += gen_adv_loss.data
            acc_mrf_loss += mrf_loss.data
            acc_dis_adv_loss += dis_adv_loss.data
            acc_ssim_loss += total_ssim_loss

            if is_sobel:
                acc_total_sobel_loss += total_sobel_loss.data
            else:
                pass

            acc_original_kd_loss += original_kd_loss.data  # 24.10.14 Original kd Loss
            acc_feature_kd_loss += feature_kd_loss.data

            if is_afa_loss:
                acc_afa_kd_loss += afa_kd_loss.data
            else:
                pass

            total_gen_loss += gen_loss.data

            # tqdm의 상태 업데이트
            pbar.update(1)
            pbar.set_postfix({'gen_loss': gen_loss.item(),
                              'dis_loss': dis_loss.item(),
                              'feature_kd_loss': feature_kd_loss.item()
                              })

    ## Tensor board
    writer.add_scalars('train/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(train_loader.dataset)},
                       epoch)

    if is_sobel:
        writer.add_scalars('train/Sobel_loss',
                           {'sobel_loss': acc_total_sobel_loss / len(train_loader.dataset)}, epoch)
    else:
        pass

    writer.add_scalars('train/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/generator_loss', {'Original KD Loss': acc_original_kd_loss / len(train_loader.dataset)},
                       epoch)  # 24.10.14 Original kd Loss
    writer.add_scalars('train/feature_kd_loss', {'Feature KD Loss': acc_feature_kd_loss / len(train_loader.dataset)},
                       epoch)  # 24.10.14 Feature kd Loss

    if is_afa_loss:
        writer.add_scalars('train/feature_kd_loss', {'AFA KD Loss': acc_afa_kd_loss / len(train_loader.dataset)},
                           epoch)
    else:
        pass

    writer.add_scalars('train/SSIM_loss', {'total gen Loss': acc_ssim_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/total_gen_loss', {'total gen Loss': total_gen_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(train_loader.dataset)},
                       epoch)


# Training
def valid(gen, dis, opt_gen, opt_dis, epoch, valid_loader, writer, teacher_gen, is_afa_loss, is_sobel=True, afa_layer=3):
    gen.eval()
    dis.eval()

    mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)
    mrf = IDMRFLoss(device=0)
    ssim_loss = SSIM_loss().cuda(0)
    sobel_loss = Sobel_loss().cuda(0)
    ofd_loss = OFD().cuda(0)

    acc_pixel_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    acc_ssim_loss = 0
    acc_total_sobel_loss = 0

    acc_original_kd_loss = 0  # 24.10.14 original KD loss
    acc_feature_kd_loss = 0
    acc_afa_kd_loss = 0

    total_gen_loss = 0

    with tqdm(total=len(valid_loader), desc=f"Validation Epoch {epoch}") as pbar:
        for batch_idx, (gt, mask_img) in enumerate(valid_loader):  # 24.09.19 labels 추가 / 관련 matching 수정 필요
            batchSize = mask_img.shape[0]
            imgSize = mask_img.shape[2]

            # gt, mask_img, iner_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0), Variable(iner_img).cuda(0)
            gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)

            iner_img = gt[:, :, :, 32:32 + 128]

            ## Generate Image
            with torch.no_grad():
                I_pred, features_s = gen(mask_img)
                f1_s, f2_s, f3_s, f4_s = features_s["x1"], features_s["x2"], features_s["x3"], features_s["x4"]

                if is_afa_loss:
                    f1_s_afa, f2_s_afa, f3_s_afa = features_s["x1_out"], features_s["x2_out"], features_s["x3_out"]
                else:
                    pass

            mask_pred = I_pred[:, :, :, 32:32 + 128]

            ## Compute losses
            ## Update Discriminator
            opt_dis.zero_grad()
            dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
            dis_loss = dis_adv_loss

            # Pixel Reconstruction Loss
            pixel_rec_loss = mae(I_pred, gt) * 20

            # Texture Consistency Loss (IDMRF Loss)
            mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize

            # ## Update Generator

            ### SSIM loss
            left_loss = ssim_loss(I_pred[:, :, :, 0:32], I_pred[:, :, :, 32:64])
            right_loss = ssim_loss(I_pred[:, :, :, 160:192], I_pred[:, :, :, 128:160])
            total_ssim_loss = left_loss + right_loss

            ### Sobel loss
            if is_sobel:
                sobel_left_loss = sobel_loss(I_pred[:, :, :, 0:32], gt[:, :, :, 0:32])
                sobel_right_loss = sobel_loss(I_pred[:, :, :, 160:192], gt[:, :, :, 160:192])
                total_sobel_loss = sobel_left_loss + sobel_right_loss

                sobel_loss_weight = 20.0

                total_sobel_loss = total_sobel_loss * sobel_loss_weight
            else:
                pass

            ### original KD loss
            with torch.no_grad():
                teacher_pred, features_t = teacher_gen(mask_img)  # (B, C, H, W) 형태의 Teacher 출력
                f1_t, f2_t, f3_t, f4_t = features_t["x1"], features_t["x2"], features_t["x3"], features_t["x4"]

                if is_afa_loss:
                    f1_t_afa, f2_t_afa, f3_t_afa = features_t["x1_out"], features_t["x2_out"], features_t["x3_out"]
                else:
                    pass

            original_kd_loss = mae(teacher_pred, I_pred) * 20

            ### feature KD loss
            feature_kd_loss = ofd_loss(f1_s, f1_t) + ofd_loss(f2_s, f2_t) + ofd_loss(f3_s, f3_t) + ofd_loss(f4_s, f4_t)

            if is_afa_loss:
                if afa_layer == 1:
                    afa_kd_loss = mse(f3_s_afa, f3_t_afa)
                elif afa_layer == 2:
                    afa_kd_loss = mse(f2_s_afa, f2_t_afa) + mse(f3_s_afa, f3_t_afa)
                else:
                    afa_kd_loss = mse(f1_s_afa, f1_t_afa) + mse(f2_s_afa, f2_t_afa) + mse(f3_s_afa, f3_t_afa)

                afa_kd_loss_weight = 1.0
                afa_kd_loss = afa_kd_loss_weight * afa_kd_loss
            else:
                pass

            gen_adv_loss = dis.calc_gen_loss(I_pred, gt)

            # 24.10.14 original kd loss
            if is_sobel and is_afa_loss:
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + total_sobel_loss + original_kd_loss + feature_kd_loss + afa_kd_loss
            elif is_sobel and (not is_afa_loss):
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + total_sobel_loss + original_kd_loss + feature_kd_loss
            elif (not is_sobel) and is_afa_loss:
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + original_kd_loss + feature_kd_loss + afa_kd_loss
            else:
                gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + total_ssim_loss + original_kd_loss + feature_kd_loss

            opt_gen.zero_grad()

            acc_pixel_rec_loss += pixel_rec_loss.data
            acc_gen_adv_loss += gen_adv_loss.data
            acc_mrf_loss += mrf_loss.data
            acc_dis_adv_loss += dis_adv_loss.data
            acc_ssim_loss += total_ssim_loss.data

            if is_sobel:
                acc_total_sobel_loss += total_sobel_loss.data
            else:
                pass

            acc_original_kd_loss += original_kd_loss.data
            acc_feature_kd_loss += feature_kd_loss.data

            if is_afa_loss:
                acc_afa_kd_loss += afa_kd_loss.data
            else:
                pass

            total_gen_loss += gen_loss.data

            # tqdm의 상태 업데이트
            pbar.update(1)
            pbar.set_postfix({'gen_loss': gen_loss.item(),
                              'dis_loss': dis_adv_loss.item(),
                              'feature_kd_loss': feature_kd_loss.item()
                              })

    ## Tensor board
    writer.add_scalars('valid/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(valid_loader.dataset)}, epoch)
    writer.add_scalars('valid/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(valid_loader.dataset)},
                       epoch)

    if is_sobel:
        writer.add_scalars('valid/Sobel_loss',
                           {'sobel_loss': acc_total_sobel_loss / len(valid_loader.dataset)}, epoch)
    else:
        pass

    writer.add_scalars('valid/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/generator_loss', {'Original KD Loss': acc_original_kd_loss / len(valid_loader.dataset)},
                       epoch)  # 24.10.14 Original kd Loss
    writer.add_scalars('valid/feature_kd_loss', {'Feature KD Loss': acc_feature_kd_loss / len(valid_loader.dataset)},
                       epoch)  # 24.12.09 Feature kd Loss

    if is_afa_loss:
        writer.add_scalars('valid/feature_kd_loss', {'AFA KD Loss': acc_afa_kd_loss / len(valid_loader.dataset)},
                           epoch)  # 24.12.09 Feature kd Loss
    else:
        pass

    writer.add_scalars('valid/SSIM_loss', {'total gen Loss': acc_ssim_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/total_gen_loss', {'total gen Loss': total_gen_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(valid_loader.dataset)},
                       epoch)

if __name__ == '__main__':
    # ablation은 SDdb 만 !!!!!!!
    NAME_DATASET = 'SDdb-1'
    SAVE_BASE_DIR = '/content/drive/MyDrive/ab2_no_afa_sep/output'

    is_afa_hp = False
    is_afa_lp = False
    is_afa_both = True
    is_afa_sep = False  # True : 분리 O, False : 분리 X
    afa_layer = None  # None or 1 or 2 or 3

    is_input_change = True
    is_sobel = False

    if is_afa_hp or is_afa_lp or is_afa_both:
        is_afa_loss = True
    else:
        is_afa_loss = False

    SAVE_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET, 'checkpoints')
    SAVE_LOG_DIR = join(SAVE_BASE_DIR, NAME_DATASET, 'logs_all')
    LOAD_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET, 'checkpoints')

    if NAME_DATASET == 'HKdb-1':
        best_epoch = 400
    elif NAME_DATASET == 'HKdb-2':
        best_epoch = 450
    elif NAME_DATASET == 'SDdb-1':
        best_epoch = 460
    elif NAME_DATASET == 'SDdb-2':
        best_epoch = 440
    else:
        raise Exception("에러 메시지 : 잘못된 NAME_DATASET이 입력되었습니다.")

    LOAD_TEACHER_WEIGHT_DIR = join('/content/drive/MyDrive/dq_u_net_sep4_sobel_afa_best', NAME_DATASET, f'Gen_former_{best_epoch}.pt')
    TRAIN_DATA_DIR = ''

    seed_everything(2024)  # Seed 고정


    def get_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('--train_batch_size', type=int, help='batch size of training data', default=8)
        parser.add_argument('--test_batch_size', type=int, help='batch size of testing data', default=16)
        parser.add_argument('--epochs', type=int, help='number of epoches', default=600)
        parser.add_argument('--lr', type=float, help='learning rate', default=0.0004)
        parser.add_argument('--alpha', type=float, help='learning rate decay for discriminator', default=0.1)
        parser.add_argument('--load_pretrain', type=bool, help='load pretrain weight', default=False)  # pretrain !!
        parser.add_argument('--test_flag', type=bool, help='testing while training', default=False)
        parser.add_argument('--adjoint', type=bool, help='if use adjoint in odenet', default=True)

        # parser.add_argument('--skip_connection', type=int,help='layers with skip connection', nargs='+', default=[0,1,2,3,4])
        # parser.add_argument('--attention', type=int,help='layers with attention mechanism applied on skip connection', nargs='+', default=[1])

        parser.add_argument('--load_weight_dir', type=str, help='directory of pretrain model weights',
                            default=LOAD_WEIGHT_DIR)
        parser.add_argument('--save_weight_dir', type=str, help='directory of saving model weights',
                            default=SAVE_WEIGHT_DIR)
        parser.add_argument('--log_dir', type=str, help='directory of saving logs', default=SAVE_LOG_DIR)
        parser.add_argument('--train_data_dir', type=str, help='directory of training data', default=TRAIN_DATA_DIR)
        # parser.add_argument('--test_data_dir', type=str, help='directory of testing data', default=TEST_DATA_DIR)
        # parser.add_argument('--gpu', type=str, help='gpu device', default='0')
        parser.add_argument('--load_teacher_weight_dir', type=str, help='directory of teacher model weight',
                            default=LOAD_TEACHER_WEIGHT_DIR)  # 24.09.19 Teacher weight dir

        opts = parser.parse_args()
        return opts

    args = get_args()

    # 24.10.14 Teacher 모델 로드
    # teacher_gen = Sep_UNet_4(n_channels=3, n_classes=3).cuda()
    teacher_gen = DQ_Thin_Sep_UNet_4_Feature(n_channels=3, n_classes=3).cuda()

    teacher_gen.load_state_dict(torch.load(args.load_teacher_weight_dir))
    print(f'Loading Teacher model weight...at epoch {best_epoch}')

    # Freeze: 가중치 업데이트를 막음
    for param in teacher_gen.parameters():
        param.requires_grad = False

    teacher_gen.eval()  # 학습 모드가 아닌 평가 모드로 설정

    ## 2023 11 08 class-wise하게 8:2로 나눠줌
    base_dir = '/content'

    if NAME_DATASET == 'HKdb-1' or NAME_DATASET == 'HKdb-2':
        modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
        db_dir = join('HK-db', modified_NAME_DATASET)
    elif NAME_DATASET == 'SDdb-1' or NAME_DATASET == 'SDdb-2':
        modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
        db_dir = join('SD-db', modified_NAME_DATASET)
    else:
        raise Exception("에러 메시지 : 잘못된 db_dir이 입력되었습니다.")

    # 각 서브 폴더의 경로를 설정
    original_dir = join(base_dir, 'original_images_split', db_dir)
    mask_dir = join(base_dir, 'mask_images_split_con', db_dir)
    clahe_dir = join(base_dir, 'clahe_images_split', db_dir)

    # 각 디렉토리가 존재하는지 확인
    assert os.path.isdir(original_dir), f"Original directory does not exist: {original_dir}"
    assert os.path.isdir(mask_dir), f"Mask directory does not exist: {mask_dir}"
    assert os.path.isdir(clahe_dir), f"CLAHE directory does not exist: {clahe_dir}"

    # 이미지 파일 리스트를 가져옴
    original_list = glob(original_dir, '*', True)
    mask_list = glob(mask_dir, '*', True)
    clahe_list = glob(clahe_dir, '*', True)

    # 각 리스트의 길이가 동일한지 확인
    assert len(original_list) == len(mask_list) == len(clahe_list)

    # 리스트 길이 출력
    print('Original list:', len(original_list))
    print('Mask list:', len(mask_list))
    print('CLAHE list:', len(clahe_list))

    # 데이터셋을 학습용과 검증용으로 분할
    train_ls_original, train_ls_mask, train_ls_clahe = [], [], []
    valid_ls_original, valid_ls_mask, valid_ls_clahe = [], [], []

    train_ls_original_list = original_list[:int(len(original_list) * 0.8)]
    train_ls_mask_list = mask_list[:int(len(mask_list) * 0.8)]
    train_ls_clahe_list = clahe_list[:int(len(clahe_list) * 0.8)]

    valid_ls_original_list = original_list[int(len(original_list) * 0.8):]
    valid_ls_mask_list = mask_list[int(len(mask_list) * 0.8):]
    valid_ls_clahe_list = clahe_list[int(len(clahe_list) * 0.8):]

    for path in train_ls_original_list:
        train_ls_original += glob(path, '*', True)

    for path in train_ls_mask_list:
        train_ls_mask += glob(path, '*', True)

    for path in train_ls_clahe_list:
        train_ls_clahe += glob(path, '*', True)

    for path in valid_ls_original_list:
        valid_ls_original += glob(path, '*', True)

    for path in valid_ls_mask_list:
        valid_ls_mask += glob(path, '*', True)

    for path in valid_ls_clahe_list:
        valid_ls_clahe += glob(path, '*', True)

    # 학습 및 검증 데이터셋 길이 출력
    print('Training Original list:', len(train_ls_original))
    print('Training Mask list:', len(train_ls_mask))
    print('Training CLAHE list:', len(train_ls_clahe))

    print('Validation Original list:', len(valid_ls_original))
    print('Validation Mask list:', len(valid_ls_mask))
    print('Validation CLAHE list:', len(valid_ls_clahe))

    pred_step = 1
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    os.makedirs(args.save_weight_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(join(args.log_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Initialize the model
    print('Initializing model...')
    # gen = DQ_Thin_Sep_UNet_4_AFA(n_channels=3, n_classes=3).cuda()  # student model

    if is_afa_both:
        if is_afa_sep:
            if afa_layer:
                gen = V_Thin_Sep_UNet_4_Feature_Ablation(n_channels=3, n_classes=3, layer=afa_layer).cuda()  # student model
            else:
                gen = V_Thin_Sep_UNet_4_Feature(n_channels=3, n_classes=3).cuda()  # student model
        else:
            gen = V_Thin_Sep_UNet_4_Feature_Ablation_sep(n_channels=3, n_classes=3).cuda()  # student model
    elif is_afa_hp:
        gen = V_Thin_Sep_UNet_4_Ablation_HP(n_channels=3, n_classes=3).cuda()  # student model
    elif is_afa_lp:
        gen = V_Thin_Sep_UNet_4_Ablation_LP(n_channels=3, n_classes=3).cuda()  # student model
    else:
        gen = V_Thin_Sep_UNet_4(n_channels=3, n_classes=3).cuda()  # student model

    # 24.10.11 모델 파라미터 수 출력
    print_model_parameters(gen)

    dis = MsImageDis().cuda()
    # fake_pool = ImagePool(500)
    # real_pool = ImagePool(500)

    opt_gen = optim.Adam(
        list(gen.parameters()), lr=args.lr / 2, betas=(0, 0.9), weight_decay=1e-4
    )

    opt_dis = optim.Adam(dis.parameters(), lr=args.lr * 2, betas=(0, 0.9), weight_decay=1e-4)

    # Load pre-trained weight
    if args.load_pretrain:
        start_epoch = 500
        print(f'Loading model weight...at epoch {start_epoch}')
        gen.load_state_dict(torch.load(join(args.load_weight_dir, f'Gen_former_{start_epoch}.pt')))
        dis.load_state_dict(torch.load(join(args.load_weight_dir, f'Dis_former_{start_epoch}.pt')))
    else:
        start_epoch = 0

    # Load data
    print('Loading data...')
    transformations = transforms.Compose(
        [torchvision.transforms.RandomResizedCrop((192, 192), scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333), ),
         CenterCrop(192), ToTensor(), Normalize(mean, std)])  # augmentation

    if is_input_change:
        train_data = dataset_norm(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=128,
                                  imglist1=train_ls_original,  # First list of images
                                  imglist2=train_ls_mask,  # Second list of images
                                  imglist3=train_ls_clahe)   # 24.09.20 수정
    else:
        train_data = dataset_norm_ablation(root=args.train_data_dir, transforms=transformations, imgSize=192,
                                           inputsize=128, imglist1=train_ls_original)

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print('train data: %d images' % (len(train_loader.dataset)))

    if is_input_change:
        valid_data = dataset_norm(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=128,
                                  imglist1=valid_ls_original,  # First list of images
                                  imglist2=valid_ls_mask,  # Second list of images
                                  imglist3=valid_ls_clahe)  # Third list of images)
    else:
        valid_data = dataset_norm_ablation(root=args.train_data_dir, transforms=transformations, imgSize=192,
                                           inputsize=128, imglist1=valid_ls_original)

    valid_loader = DataLoader(valid_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print('valid data: %d images' % (len(valid_loader.dataset)))

    # Train & test the model
    for epoch in range(start_epoch + 1, 1 + args.epochs):
        print("----Start training[%d / %d]----" % (epoch, args.epochs))

        train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer, teacher_gen, is_afa_loss, is_sobel, afa_layer)  # 24.09.25 recognizier

        # Update the valid function to iterate over the tqdm-wrapped loader
        valid(gen, dis, opt_gen, opt_dis, epoch, valid_loader, writer, teacher_gen, is_afa_loss, is_sobel, afa_layer)  # 24.09.25 recognizier

        # Save the model weight every 10 epochs
        if (epoch % 10) == 0:
            torch.save(gen.state_dict(), join(args.save_weight_dir, 'Gen_former_%d.pt' % epoch))
            torch.save(dis.state_dict(), join(args.save_weight_dir, 'Dis_former_%d.pt' % epoch))

    writer.close()
