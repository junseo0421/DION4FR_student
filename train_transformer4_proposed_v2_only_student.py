#### only student ###

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
from models.build4 import build_model, ImagePool
# from models.Generator_former import Generator_former
from utils.loss import *
from models.Discriminator_ml import MsImageDis
# from utils.utils import gaussian_weight
from tensorboardX import SummaryWriter
from dataset import dataset_norm
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from loss import *
import timm
from PIL import Image, ImageDraw, ImageFont

from utils.utils import *

from torchvision.utils import save_image

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
    print("Calculating total model parameters...")

    # 전체 모델 파라미터 수
    total_params = count_parameters(gen)
    print(f"Total parameters in the Student model: {total_params}")

    # simple TSP module 파라미터 수
    if hasattr(gen, 'simple_MRB'):
        tsp_params = count_parameters(gen.simple_MRB)
        print(f"simple TSP Module parameters: {tsp_params}")
    else:
        print("simple TSP Module not found in the model")

# Training
def train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer):  #24.09.19 recognizer
    gen.train()
    dis.train()

    # mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)  # 평균 절대 오차(MAE)를 사용하여 픽셀 간의 차이 계산
    mrf = IDMRFLoss(device=0)  # 텍스처 일관성 평가
    ssim_loss = SSIM_loss().cuda(0)  # 구조적 유사성

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_feat_cons_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    acc_ssim_loss = 0
    # acc_original_kd_loss = 0 # 24.10.14 original KD loss

    total_gen_loss = 0

    for batch_idx, (gt, mask_img) in enumerate(train_loader):  # 24.09.20 수정
        batchSize = mask_img.shape[0]
        imgSize = mask_img.shape[2]
        # print(batch_idx, ":", random_matching_list)

        # gt, mask_img, iner_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0), Variable(iner_img).cuda(0)
        gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)

        iner_img = gt[:, :, :, 32:32 + 128]  # 가로로 32~160 pixel
        # I_groundtruth = torch.cat((I_l, I_r), 3)  # shape: B,C,H,W

        ## feature size 맞춰주기 f_de 와 f_en
        ## Generate Image
        I_pred, f_de = gen(mask_img)  # 생성된 image(2, 3, 192, 192), 중간 feature map
        # I_pred = gen(mask_img)
        f_en = gen(iner_img, only_encode=True)  #iner_img(GT)를 encoding하여 feature map을 얻음

        # i_mask = torch.ones_like(gt)
        # i_mask[:, :, 32:32 + 128, 32:32 + 128] = 0
        # mask_pred = I_pred * i_mask
        mask_pred = I_pred[:, :, :, 32:32 + 128]  # 생성된 image의 일부분 선택

        ## Compute losses
        ## Update Discriminator
        opt_dis.zero_grad()
        dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)  # 생성된 image와 gt와의 구별 능력 학습
        dis_loss = dis_adv_loss
        dis_loss.backward()
        opt_dis.step()  # 가중치 update

        # Pixel Reconstruction Loss
        # pixel_rec_loss = mse(I_pred, gt) * 10
        # pixel_rec_loss = mse(mask_pred, iner_img) * 10
        pixel_rec_loss = mae(I_pred, gt) * 20  # pixel 재구성 손실

        # Texture Consistency Loss (IDMRF Loss)
        # mrf_loss = mrf(((I_pred * img_mask).cuda(0) + 1) / 2.0, ((gt * img_mask).cuda(0) + 1) / 2.0) * 0.01 / batchSize
        mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize  # 텍스처 일관성 손실
        # mrf_loss = mrf((I_pred_split[1].cuda(1)+1)/2.0, (I_m.cuda(1)+1)/2.0) * 0.01

        # Feature Reconstruction Loss
        # feat_rec_loss = mse(f_all, f_all_gt.detach()).mean() * batchSize
        # feat_rec_loss = mse(f_all, f_all_gt.detach()) * 5

        ### dimension 맞춰주기 진행
        feat_rec_loss = mae(f_de, f_en.detach())   # [2, 6, 4, 768]
        # 생성된 imgae의 feature map과 gt의 feature map 간의 L1 손실

        ### SSIM loss
        left_loss = ssim_loss(I_pred[:, :, :, 0:32], I_pred[:, :, :, 32:64])
        right_loss = ssim_loss(I_pred[:, :, :, 160:192], I_pred[:, :, :, 128:160])
        total_ssim_loss = left_loss+right_loss

        # ### original KD loss
        # with torch.no_grad():
        #     teacher_pred, _ = teacher_gen(mask_img)  # (B, C, H, W) 형태의 Teacher 출력
        #
        # original_kd_loss = mae(teacher_pred, I_pred) * 20  # 가중치는 pxiel_rec_loss와 똑같이 설정. 나중에 조절 필요할 수도

        # ## Update Generator
        gen_adv_loss = dis.calc_gen_loss(I_pred, gt)  # generator에 대한 적대적 손실

        # 24.09.19 Recognition Loss, 24.10.14 original kd loss
        # gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0) + total_ssim_loss + original_kd_loss
        gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0) + total_ssim_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        acc_pixel_rec_loss += pixel_rec_loss.data
        acc_gen_adv_loss += gen_adv_loss.data
        acc_mrf_loss += mrf_loss.data
        acc_feat_rec_loss += feat_rec_loss.data
        # acc_feat_cons_loss += feat_cons_loss.data
        acc_dis_adv_loss += dis_adv_loss.data
        acc_ssim_loss += total_ssim_loss
        # acc_original_kd_loss += original_kd_loss.data  # 24.10.14 Original kd Loss

        total_gen_loss += gen_loss.data

        if batch_idx % 10 == 0:
            print("train iter %d" % batch_idx)
            print('generate_loss:', gen_loss.item())
            print('dis_loss:', dis_loss.item())

    ## Tensor board
    writer.add_scalars('train/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/generator_loss',
                       {'Feature Reconstruction Loss': acc_feat_rec_loss / len(train_loader.dataset)}, epoch)
    # writer.add_scalars('train/generator_loss', {'Feature Consistency Loss': acc_feat_cons_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(train_loader.dataset)},
                       epoch)
    # writer.add_scalars('train/generator_loss', {'Original KD Loss': acc_original_kd_loss / len(train_loader.dataset)},
    #                    epoch)  # 24.10.14 Original kd Loss
    writer.add_scalars('train/SSIM_loss', {'total gen Loss': acc_ssim_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/total_gen_loss', {'total gen Loss': total_gen_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(train_loader.dataset)},
                       epoch)

# Training
def valid(gen, dis, opt_gen, opt_dis, epoch, valid_loader, writer):
    gen.eval()
    dis.eval()

    # mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)
    mrf = IDMRFLoss(device=0)
    ssim_loss = SSIM_loss().cuda(0)

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_feat_cons_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0
    acc_ssim_loss = 0
    # acc_original_kd_loss = 0  # 24.10.14 original KD loss
    # acc_fsp_loss = 0  # 24.10.15 fsp loss

    total_gen_loss = 0

    for batch_idx, (gt, mask_img) in enumerate(valid_loader):  # 24.09.19 labels 추가 / 관련 matching 수정 필요
        batchSize = mask_img.shape[0]
        imgSize = mask_img.shape[2]

        # gt, mask_img, iner_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0), Variable(iner_img).cuda(0)
        gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)

        iner_img = gt[:, :, :, 32:32 + 128]
        # I_groundtruth = torch.cat((I_l, I_r), 3)  # shape: B,C,H,W

        ## feature size match f_de and f_en
        ## Generate Image
        with torch.no_grad():
            I_pred, f_de = gen(mask_img)
        # I_pred = gen(mask_img)
        with torch.no_grad():
            f_en = gen(iner_img, only_encode=True)

        # i_mask = torch.ones_like(gt)
        # i_mask[:, :, 32:32 + 128, 32:32 + 128] = 0
        # mask_pred = I_pred * i_mask
        mask_pred = I_pred[:, :, :, 32:32 + 128]

        ## Compute losses
        ## Update Discriminator
        opt_dis.zero_grad()
        dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
        dis_loss = dis_adv_loss

        # Pixel Reconstruction Loss
        # pixel_rec_loss = mse(I_pred, gt) * 10
        # pixel_rec_loss = mse(mask_pred, iner_img) * 10
        pixel_rec_loss = mae(I_pred, gt) * 20

        # Texture Consistency Loss (IDMRF Loss)
        # mrf_loss = mrf(((I_pred * img_mask).cuda(0) + 1) / 2.0, ((gt * img_mask).cuda(0) + 1) / 2.0) * 0.01 / batchSize
        mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize
        # mrf_loss = mrf((I_pred_split[1].cuda(1)+1)/2.0, (I_m.cuda(1)+1)/2.0) * 0.01

        # Feature Reconstruction Loss
        # feat_rec_loss = mse(f_all, f_all_gt.detach()).mean() * batchSize
        # feat_rec_loss = mse(f_all, f_all_gt.detach()) * 5

        ### dimension 맞춰주기 진행
        feat_rec_loss = mae(f_de, f_en.detach())

        # ## Update Generator

        ### SSIM loss
        left_loss = ssim_loss(I_pred[:, :, :, 0:32], I_pred[:, :, :, 32:64])
        right_loss = ssim_loss(I_pred[:, :, :, 160:192], I_pred[:, :, :, 128:160])
        total_ssim_loss = left_loss + right_loss

        # ### original KD loss
        # with torch.no_grad():
        #     teacher_pred, _ = teacher_gen(mask_img)  # (B, C, H, W) 형태의 Teacher 출력
        #
        # original_kd_loss = mae(teacher_pred, I_pred) * 20

        gen_adv_loss = dis.calc_gen_loss(I_pred, gt)

        # 24.09.19 Recognition Loss, 24.10.14 original kd loss
        # gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0) + total_ssim_loss + original_kd_loss
        gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0) + total_ssim_loss
        opt_gen.zero_grad()

        acc_pixel_rec_loss += pixel_rec_loss.data
        acc_gen_adv_loss += gen_adv_loss.data
        acc_mrf_loss += mrf_loss.data
        acc_feat_rec_loss += feat_rec_loss.data
        # acc_feat_cons_loss += feat_cons_loss.data
        acc_dis_adv_loss += dis_adv_loss.data
        acc_ssim_loss += total_ssim_loss.data
        # acc_original_kd_loss += original_kd_loss.data  # 24.10.14 Original KD Loss

        total_gen_loss += gen_loss.data


    ## Tensor board
    writer.add_scalars('valid/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(valid_loader.dataset)}, epoch)
    writer.add_scalars('valid/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/generator_loss',
                       {'Feature Reconstruction Loss': acc_feat_rec_loss / len(valid_loader.dataset)}, epoch)
    # writer.add_scalars('train/generator_loss', {'Feature Consistency Loss': acc_feat_cons_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('valid/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(valid_loader.dataset)},
                       epoch)
    # writer.add_scalars('valid/generator_loss', {'Original KD Loss': acc_original_kd_loss / len(valid_loader.dataset)},
    #                    epoch)  # 24.10.14 Original kd Loss
    writer.add_scalars('valid/SSIM_loss', {'total gen Loss': acc_ssim_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/total_gen_loss', {'total gen Loss': total_gen_loss / len(valid_loader.dataset)},
                       epoch)

    writer.add_scalars('valid/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(valid_loader.dataset)},
                       epoch)

if __name__ == '__main__':
    NAME_DATASET = 'SDdb-2'
    SAVE_BASE_DIR = '/content/drive/MyDrive/only_student_2d/output'

    SAVE_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET , 'checkpoints')
    SAVE_LOG_DIR = join(SAVE_BASE_DIR, NAME_DATASET , 'logs_all')
    LOAD_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET , 'checkpoints')

    # if NAME_DATASET == 'HKdb-1':
    #     best_epoch = 350
    # elif NAME_DATASET == 'HKdb-2':
    #     best_epoch = 378
    # elif NAME_DATASET == 'SDdb-1':
    #     best_epoch = 340
    # elif NAME_DATASET == 'SDdb-2':
    #     best_epoch = 500
    # else:
    #     raise Exception("에러 메시지 : 잘못된 NAME_DATASET이 입력되었습니다.")
    #
    # LOAD_TEACHER_WEIGHT_DIR = join('D:\DION4FR\mask_change\mask_change_best_epoch', NAME_DATASET, f'Gen_former_{best_epoch}.pt')

    TRAIN_DATA_DIR = ''

    seed_everything(2024)  # Seed 고정


    def get_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('--train_batch_size', type=int, help='batch size of training data', default=2)
        parser.add_argument('--test_batch_size', type=int, help='batch size of testing data', default=16)
        parser.add_argument('--epochs', type=int, help='number of epoches', default=500)
        parser.add_argument('--lr', type=float, help='learning rate', default=0.0004)
        parser.add_argument('--alpha', type=float, help='learning rate decay for discriminator', default=0.1)
        parser.add_argument('--load_pretrain', type=bool, help='load pretrain weight', default=True)  # pretrain !!
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

        # # teacher model weight
        # parser.add_argument('--load_teacher_weight_dir', type=str, help='directory of teacher model weight',
        #                     default=LOAD_TEACHER_WEIGHT_DIR)  # 24.09.19 Teacher weight dir

        opts = parser.parse_args()
        return opts

    args = get_args()

    # # 24.10.14 teacher model
    # config_t = {}
    # config_t['pre_step'] = 1
    # config_t['TYPE'] = 'swin_cross_attn_ResB_v2'
    # config_t['IMG_SIZE'] = 224
    # config_t['SWIN.PATCH_SIZE'] = 4
    # config_t['SWIN.IN_CHANS'] = 3
    # config_t['SWIN.EMBED_DIM'] = 96
    # config_t['SWIN.DEPTHS'] = [2, 2, 6, 2]
    # config_t['SWIN.NUM_HEADS'] = [3, 6, 12, 24]
    # config_t['SWIN.WINDOW_SIZE'] = 7
    # config_t['SWIN.MLP_RATIO'] = 4.
    # config_t['SWIN.QKV_BIAS'] = True
    # config_t['SWIN.QK_SCALE'] = None
    # config_t['DROP_RATE'] = 0.0
    # config_t['DROP_PATH_RATE'] = 0.2
    # config_t['SWIN.PATCH_NORM'] = True
    # config_t['TRAIN.USE_CHECKPOINT'] = False

    # 24.10.14 student model
    config = {}
    config['pre_step'] = 1
    config['TYPE'] = 'swin_cross_attn_ResB_v2_student'  # 24.11.20 only_student_2d
    config['IMG_SIZE'] = 224
    config['SWIN.PATCH_SIZE'] = 4
    config['SWIN.IN_CHANS'] = 3
    config['SWIN.EMBED_DIM'] = 96
    config['SWIN.DEPTHS'] = [2, 2, 6, 2]
    config['SWIN.NUM_HEADS'] = [3, 6, 12, 24]
    config['SWIN.WINDOW_SIZE'] = 7
    config['SWIN.MLP_RATIO'] = 4.
    config['SWIN.QKV_BIAS'] = True
    config['SWIN.QK_SCALE'] = None
    config['DROP_RATE'] = 0.0
    config['DROP_PATH_RATE'] = 0.2
    config['SWIN.PATCH_NORM'] = True
    config['TRAIN.USE_CHECKPOINT'] = False

    # # 24.10.14 Teacher 모델 로드
    # teacher_gen = build_model(config_t).cuda()
    # teacher_gen.load_state_dict(torch.load(args.load_teacher_weight_dir))
    #
    # # Freeze: 가중치 업데이트를 막음
    # for param in teacher_gen.parameters():
    #     param.requires_grad = False
    #
    # teacher_gen.eval()  # 학습 모드가 아닌 평가 모드로 설정

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
    gen = build_model(config).cuda()  # student model
    # gen = Generator7(pred_step, device=0).cuda(0)

    # 24.10.11 모델 파라미터 수 출력
    print_model_parameters(gen)

    dis = MsImageDis().cuda()
    # fake_pool = ImagePool(500)
    # real_pool = ImagePool(500)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr / 2, betas=(0, 0.9), weight_decay=1e-4)
    opt_dis = optim.Adam(dis.parameters(), lr=args.lr * 2, betas=(0, 0.9), weight_decay=1e-4)

    # Load pre-trained weight
    if args.load_pretrain:
        start_epoch = 330
        print(f'Loading model weight...at epoch {start_epoch}')
        gen.load_state_dict(torch.load(join(args.load_weight_dir, f'Gen_former_{start_epoch}.pt')))
        dis.load_state_dict(torch.load(join(args.load_weight_dir, f'Dis_former_{start_epoch}.pt')))
    else:
        start_epoch = 0

    # Load data
    print('Loading data...')
    # transformations = transforms.Compose([ToTensor(), Normalize(mean, std)])
    transformations = transforms.Compose(
        [torchvision.transforms.RandomResizedCrop((192, 192), scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333), ),
         CenterCrop(192), ToTensor(), Normalize(mean, std)])  # augmentation
    # train_data = torch.load(args.train_data_dir)
    # train_dataset = TensorDataset(train_data['gt'], train_data['mask'], train_data['iner'])
    # train_dataset = TensorDataset(train_data['gt'], train_data['mask'])
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_data = dataset_norm(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=128,
                              imglist1=train_ls_original,  # First list of images
                              imglist2=train_ls_mask,  # Second list of images
                              imglist3=train_ls_clahe)   # 24.09.20 수정
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print('train data: %d images' % (len(train_loader.dataset)))

    valid_data = dataset_norm(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=128,
                              imglist1=valid_ls_original,  # First list of images
                              imglist2=valid_ls_mask,  # Second list of images
                              imglist3=valid_ls_clahe)  # Third list of images)
    valid_loader = DataLoader(valid_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print('valid data: %d images' % (len(valid_loader.dataset)))
    # if args.test_flag:
    #     test_data = dataset_around(root=args.test_data_dir, transforms=transformations, crop='center', imgSize=128)
    #     test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    #     print('test data: %d images' % (len(test_loader.dataset)))

    # Train & test the model
    for epoch in range(start_epoch + 1, 1 + args.epochs):
        print("----Start training[%d / %d]----" % (epoch, args.epochs))
        # train(gen, dis, fake_pool, real_pool, opt_gen, opt_dis, epoch, train_loader, writer)

        train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer)  # 24.09.25 teacher

        # if args.test_flag:
        #     print("----Start testing[%d]----" % epoch)
        #     test(gen, dis, epoch, test_loader, writer)

        # Update the valid function to iterate over the tqdm-wrapped loader
        valid(gen, dis, opt_gen, opt_dis, epoch, valid_loader, writer)  # 24.09.25 teacher

        # Save the model weight every 10 epochs
        if (epoch % 10) == 0:
            torch.save(gen.state_dict(), join(args.save_weight_dir, 'Gen_former_%d.pt' % epoch))
            torch.save(dis.state_dict(), join(args.save_weight_dir, 'Dis_former_%d.pt' % epoch))

    writer.close()
