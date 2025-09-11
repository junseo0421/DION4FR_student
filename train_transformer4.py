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
from utils.loss import IDMRFLoss
from models.Discriminator_ml import MsImageDis
# from utils.utils import gaussian_weight
from tensorboardX import SummaryWriter
from dataset import dataset_norm_mmcbnu_ori
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from loss import *
from tqdm import tqdm

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
def print_swin_lstm_parameters(gen):
    print("Calculating parameters...")

    # 전체 모델 파라미터 수
    total_params = count_parameters(gen)
    print(f"Total parameters in the U-Transformer model: {total_params}")

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def roi_align_token_columns_from_fde(
    f_de,
    img_w: int = 192,
    crop_left: int = 50,
    crop_w: int = 92,
    target_w: int = 3,   # 보통 f_en.shape[2]를 넣어주세요
    target_h: int | None = None,  # 기본은 f_de의 H 유지(대개 6)
    padding_mode: str = "border",  # 경계 안전
    align_corners: bool = True
):
    """
    f_de : (B,H,W,C)  또는 (B,S,N,C)  또는 (B,N,C)/(B,6,6,C)
           내부 토큰 격자가 HxW라고 가정(대개 6x6; 코드가 자동 추정)
    반환 : (B, Ht, target_w, C)  # 가로를 정확히 crop 위치/폭에 맞춰 샘플링
    """
    # 1) (B,H,W,C)로 정규화
    if f_de.dim() == 4:
        B, a, b, c = f_de.shape
        # (B,S,N,C) 케이스인지 (B,H,W,C) 케이스인지 구분
        # (B,S,N,C)라면 b가 N=H*W여야 함
        if a > 1 and b != a and (int((b)**0.5)**2 == b):  # heuristic
            # (B,S,N,C) -> (B,S,H,W,C) -> S축 풀어 (B,H,W,C)로 합치기보단 S=1로 취급 불가
            B, S, N, C = f_de.shape
            H = W = int(N**0.5)
            f = f_de.view(B, S, H, W, C)
            # S>1이면 일반적으로 stage 축이거나 스택인데, 여기서는 한 덩어리로 다룰 수 없으니
            # 보통 S==1이거나, S차원 평균/선택을 해야 합니다. 여기서는 첫 S만 사용 (필요시 바꾸세요).
            f = f[:, 0]  # (B,H,W,C)
        else:
            # (B,H,W,C)
            f = f_de
            B, H, W, C = f.shape
    elif f_de.dim() == 3:
        # (B,N,C) -> (B,H,W,C)
        B, N, C = f_de.shape
        H = W = int(N**0.5)
        assert H * W == N, "N은 완전제곱이어야 합니다."
        f = f_de.view(B, H, W, C)
    elif f_de.dim() == 5:
        # (B,6,6,C) 같은 형식이면 (B,H,W,C)로
        B, H, W, C, *rest = f_de.shape + (None,)
        if rest[0] is not None:
            raise ValueError("지원하지 않는 5D 형식입니다.")
        f = f_de
    else:
        raise ValueError("지원 shape: (B,H,W,C), (B,S,N,C), (B,N,C), (B,H,W,C-like)")

    # target_h 기본은 기존 H 유지
    if target_h is None:
        target_h = H

    # 2) grid_sample 입력: (N,C,H,W)
    f_nchw = f.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

    # 3) crop 좌우 경계를 토큰 좌표(0..W-1)로 매핑
    x0_tok = (crop_left / img_w) * (W - 1)
    x1_tok = ((crop_left + crop_w) / img_w) * (W - 1)

    # 4) 목표 가로 해상도(target_w)만큼 균등 샘플 포인트
    xs = torch.linspace(x0_tok, x1_tok, steps=target_w, device=f_nchw.device, dtype=f_nchw.dtype)
    ys = torch.linspace(0, H - 1, steps=target_h, device=f_nchw.device, dtype=f_nchw.dtype)

    # 5) 정규화 좌표(-1..1)로 변환 (align_corners=True 기준)
    xs_n = 2.0 * xs / (W - 1) - 1.0
    ys_n = 2.0 * ys / (H - 1) - 1.0

    # 6) 샘플링 그리드 (B, target_h, target_w, 2)
    # torch>=1.10: indexing='ij' 지원. 낮은 버전이면 기본 meshgrid로 대체
    try:
        grid_y, grid_x = torch.meshgrid(ys_n, xs_n, indexing='ij')
    except TypeError:
        grid_y, grid_x = torch.meshgrid(ys_n, xs_n)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    # 7) grid_sample: (B,C,target_h,target_w)
    f_crop = F.grid_sample(
        f_nchw, grid, mode='bilinear', align_corners=align_corners, padding_mode=padding_mode
    )

    # 8) (B,target_h,target_w,C)로 복원
    f_crop = f_crop.permute(0, 2, 3, 1).contiguous()
    return f_crop



# Training
def train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer):
    gen.train()
    dis.train()

    # mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)  # 평균 절대 오차(MAE)를 사용하여 픽셀 간의 차이 계산
    mrf = IDMRFLoss(device=0)  # 텍스처 일관성 평가

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0

    total_gen_loss = 0

    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}") as pbar:
        for batch_idx, (gt, mask_img) in enumerate(train_loader):

            batchSize = mask_img.shape[0]
            imgSize = mask_img.shape[2]

            # gt, mask_img, iner_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0), Variable(iner_img).cuda(0)
            gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)
            iner_img = gt[:, :, :, 50:50 + 92]  # 가로로 32~160 pixel

            ## Generate Image
            I_pred, f_de = gen(mask_img)  # 생성된 image, 중간 feature map
            f_en = gen(iner_img, only_encode=True)  #iner_img(GT)를 encoding하여 feature map을 얻음

            mask_pred = I_pred[:, :, :, 50:50 + 92]  # 생성된 image의 일부분 선택

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

            # Feature Reconstruction Loss
            f_de_aligned = roi_align_token_columns_from_fde(f_de)
            feat_rec_loss = mae(f_de_aligned, f_en.detach())  # 생성된 imgae의 feature map과 gt의 feature map 간의 L1 손실

            # ## Update Generator
            gen_adv_loss = dis.calc_gen_loss(I_pred, gt)  # generator에 대한 적대적 손실

            gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0)
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            acc_pixel_rec_loss += pixel_rec_loss.data
            acc_gen_adv_loss += gen_adv_loss.data
            acc_mrf_loss += mrf_loss.data
            acc_feat_rec_loss += feat_rec_loss.data
            acc_dis_adv_loss += dis_adv_loss.data

            total_gen_loss += gen_loss.data

            # tqdm의 상태 업데이트
            pbar.update(1)
            pbar.set_postfix({'gen_loss': gen_loss.item(), 'dis_loss': dis_loss.item()})

    ## Tensor board
    writer.add_scalars('train/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/generator_loss',
                       {'Feature Reconstruction Loss': acc_feat_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(train_loader.dataset)},
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

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0

    total_gen_loss = 0

    with tqdm(total=len(valid_loader), desc=f"Validation Epoch {epoch}") as pbar:
        for batch_idx, (gt, mask_img) in enumerate(valid_loader):
            batchSize = mask_img.shape[0]
            imgSize = mask_img.shape[2]

            # gt, mask_img, iner_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0), Variable(iner_img).cuda(0)
            gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)
            iner_img = gt[:, :, :, 50:50 + 92]
            # I_groundtruth = torch.cat((I_l, I_r), 3)  # shape: B,C,H,W

            ## feature size match f_de and f_en
            ## Generate Image
            with torch.no_grad():
                I_pred, f_de = gen(mask_img)

            with torch.no_grad():
                f_en = gen(iner_img, only_encode=True)

            mask_pred = I_pred[:, :, :, 50:50 + 92]

            ## Compute losses
            ## Update Discriminator
            opt_dis.zero_grad()
            dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
            dis_loss = dis_adv_loss

            # Pixel Reconstruction Loss
            pixel_rec_loss = mae(I_pred, gt) * 20

            # Texture Consistency Loss (IDMRF Loss)
            mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize

            # Feature Reconstruction Loss
            f_de_aligned = roi_align_token_columns_from_fde(f_de)
            feat_rec_loss = mae(f_de_aligned, f_en.detach())

            # ## Update Generator
            gen_adv_loss = dis.calc_gen_loss(I_pred, gt)

            gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0)
            opt_gen.zero_grad()

            acc_pixel_rec_loss += pixel_rec_loss.data
            acc_gen_adv_loss += gen_adv_loss.data
            acc_mrf_loss += mrf_loss.data
            acc_feat_rec_loss += feat_rec_loss.data
            acc_dis_adv_loss += dis_adv_loss.data

            total_gen_loss += gen_loss.data

            # tqdm의 상태 업데이트
            pbar.update(1)
            pbar.set_postfix({'gen_loss': gen_loss.item(), 'dis_loss': dis_adv_loss.item()})

    ## Tensor board
    writer.add_scalars('valid/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(valid_loader.dataset)}, epoch)
    writer.add_scalars('valid/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/generator_loss',
                       {'Feature Reconstruction Loss': acc_feat_rec_loss / len(valid_loader.dataset)}, epoch)
    writer.add_scalars('valid/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(valid_loader.dataset)},
                       epoch)
    writer.add_scalars('valid/total_gen_loss', {'total gen Loss': total_gen_loss / len(valid_loader.dataset)},
                       epoch)

    writer.add_scalars('valid/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(valid_loader.dataset)},
                       epoch)

if __name__ == '__main__':
    NAME_DATASET = 'mmcbnu-2'
    SAVE_BASE_DIR = '/content/drive/MyDrive/comparison/utrans/output'

    SAVE_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET , 'checkpoints')
    SAVE_LOG_DIR = join(SAVE_BASE_DIR, NAME_DATASET , 'logs_all')
    LOAD_WEIGHT_DIR = join(SAVE_BASE_DIR, NAME_DATASET , 'checkpoints')
    

    TRAIN_DATA_DIR = ''

    seed_everything(2024)  # Seed 고정

    def get_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('--train_batch_size', type=int, help='batch size of training data', default=8)
        parser.add_argument('--test_batch_size', type=int, help='batch size of testing data', default=16)
        parser.add_argument('--epochs', type=int, help='number of epoches', default=500)
        parser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
        parser.add_argument('--alpha', type=float, help='learning rate decay for discriminator', default=0.1)
        parser.add_argument('--load_pretrain', type=bool, help='load pretrain weight', default=False)
        parser.add_argument('--test_flag', type=bool, help='testing while training', default=False)
        parser.add_argument('--adjoint', type=bool, help='if use adjoint in odenet', default=True)

        parser.add_argument('--load_weight_dir', type=str, help='directory of pretrain model weights',
                            default=LOAD_WEIGHT_DIR)
        parser.add_argument('--save_weight_dir', type=str, help='directory of saving model weights',
                            default=SAVE_WEIGHT_DIR)
        parser.add_argument('--log_dir', type=str, help='directory of saving logs', default=SAVE_LOG_DIR)
        parser.add_argument('--train_data_dir', type=str, help='directory of training data', default=TRAIN_DATA_DIR)
        
        opts = parser.parse_args()
        return opts


    args = get_args()
    config = {}
    config['pre_step'] = 1
    config['TYPE'] = 'swin'
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

    ## 2023 11 08 class-wise하게 8:2로 나눠줌
    base_dir = '/content'

    if NAME_DATASET == 'HKdb-1' or NAME_DATASET == 'HKdb-2':
        modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
        db_dir = join('HK-db', modified_NAME_DATASET)
    elif NAME_DATASET == 'SDdb-1' or NAME_DATASET == 'SDdb-2':
        modified_NAME_DATASET = NAME_DATASET.replace('-', '_')
        db_dir = join('SD-db', modified_NAME_DATASET)
    elif NAME_DATASET == 'mmcbnu-1' or NAME_DATASET == 'mmcbnu-2':
        db_dir = NAME_DATASET
    else:
        raise Exception("에러 메시지 : 잘못된 db_dir이 입력되었습니다.")

    # 각 서브 폴더의 경로를 설정
    original_dir = join(base_dir, 'original_images_split', db_dir)
    # mask_dir = join(base_dir, 'mask_images_split_con', db_dir)
    # clahe_dir = join(base_dir, 'clahe_images_split', db_dir)

    # 각 디렉토리가 존재하는지 확인
    assert os.path.isdir(original_dir), f"Original directory does not exist: {original_dir}"
    # assert os.path.isdir(mask_dir), f"Mask directory does not exist: {mask_dir}"
    # assert os.path.isdir(clahe_dir), f"CLAHE directory does not exist: {clahe_dir}"


    # 이미지 파일 리스트를 가져옴
    original_list = glob(original_dir, '*', True)
    # mask_list = glob(mask_dir, '*', True)
    # clahe_list = glob(clahe_dir, '*', True)

    # 각 리스트의 길이가 동일한지 확인
    # assert len(original_list) == len(mask_list) == len(clahe_list)

    # 리스트 길이 출력
    print('Original list:', len(original_list))
    # print('Mask list:', len(mask_list))
    # print('CLAHE list:', len(clahe_list))

    # 데이터셋을 학습용과 검증용으로 분할
    train_ls_original, train_ls_mask, train_ls_clahe = [], [], []
    valid_ls_original, valid_ls_mask, valid_ls_clahe = [], [], []

    train_ls_original_list = original_list[:int(len(original_list) * 0.8)]
    # train_ls_mask_list = mask_list[:int(len(mask_list) * 0.8)]
    # train_ls_clahe_list = clahe_list[:int(len(clahe_list) * 0.8)]

    valid_ls_original_list = original_list[int(len(original_list) * 0.8):]
    # valid_ls_mask_list = mask_list[int(len(mask_list) * 0.8):]
    # valid_ls_clahe_list = clahe_list[int(len(clahe_list) * 0.8):]

    for path in train_ls_original_list:
        train_ls_original += glob(path, '*', True)

    # for path in train_ls_mask_list:
    #     train_ls_mask += glob(path, '*', True)

    # for path in train_ls_clahe_list:
    #     train_ls_clahe += glob(path, '*', True)

    for path in valid_ls_original_list:
        valid_ls_original += glob(path, '*', True)

    # for path in valid_ls_mask_list:
    #     valid_ls_mask += glob(path, '*', True)

    # for path in valid_ls_clahe_list:
    #     valid_ls_clahe += glob(path, '*', True)


    # 학습 및 검증 데이터셋 길이 출력
    print('Training Original list:', len(train_ls_original))
    # print('Training Mask list:', len(train_ls_mask))
    # print('Training CLAHE list:', len(train_ls_clahe))

    print('Validation Original list:', len(valid_ls_original))
    # print('Validation Mask list:', len(valid_ls_mask))
    # print('Validation CLAHE list:', len(valid_ls_clahe))

    pred_step = 1
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    os.makedirs(args.save_weight_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(join(args.log_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Initialize the model
    print('Initializing model...')
    gen = build_model(config).cuda()

    # 24.10.11 Swin-Transformer와 TSP(LSTM_small2) 모듈 파라미터 수 출력
    print_swin_lstm_parameters(gen)

    # gen = Generator7(pred_step, device=0).cuda(0)
    dis = MsImageDis().cuda()

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr / 2, betas=(0.0, 0.9), weight_decay=1e-4)
    opt_dis = optim.Adam(dis.parameters(), lr=args.lr * 2, betas=(0.0, 0.9), weight_decay=1e-4)

    # Load pre-trained weight
    if args.load_pretrain:
        start_epoch = 500
        print(f'Loading model weight...at epoch {start_epoch}')
        gen.load_state_dict(torch.load(join(args.load_weight_dir, 'Gen_former_360.pt')))
        dis.load_state_dict(torch.load(join(args.load_weight_dir, 'Dis_former_360.pt')))
    else:
        start_epoch = 0

    # Load data
    print('Loading data...')
    transformations = transforms.Compose(
        [Resize(192), CenterCrop(192), ToTensor(), Normalize(mean, std)])  # augmentation
    train_data = dataset_norm_mmcbnu_ori(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=92,
                              imglist1=train_ls_original) 
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print('train data: %d images' % (len(train_loader.dataset)))

    valid_data = dataset_norm_mmcbnu_ori(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=92,
                              imglist1=valid_ls_original)  
    valid_loader = DataLoader(valid_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    print('valid data: %d images' % (len(valid_loader.dataset)))
    
    # # Train & test the model
    for epoch in range(start_epoch + 1, 1 + args.epochs):
        print("----Start training[%d / %d]----" % (epoch, args.epochs))

        train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer)  

        # Update the valid function to iterate over the tqdm-wrapped loader
        valid(gen, dis, opt_gen, opt_dis, epoch, valid_loader, writer)  

        # Save the model weight every 10 epochs
        if (epoch % 50) == 0:
            torch.save(gen.state_dict(), join(args.save_weight_dir, 'Gen_former_%d.pt' % epoch))
            torch.save(dis.state_dict(), join(args.save_weight_dir, 'Dis_former_%d.pt' % epoch))

    writer.close()
