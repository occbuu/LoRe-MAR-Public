# -*- coding: utf-8 -*-
"""
==============================================================================
 train_dental_mar.py  -  GIAM NHIEU KIM LOAI (GLARE / METAL ARTEFACT) CBCT RANG
==============================================================================
 Gop 3 notebook cu thanh 1 file duy nhat, chay bang:

        python train_dental_mar.py

 (khong can them tham so nao - moi cai dat nam trong class CFG ngay ben duoi)

 3 giai doan chay lien tiep, tu dong:
   [1] DU LIEU     (notebook preprocess_data.ipynb)
        - ToothFairy_Dataset (*.npy 3D)  ->  Dataset/{Glare,Normal}/{Train,Val,Test}/slice_*.png
          (dung to chuc thu muc giong CBCT_MAR). Neu Dataset_TF da co san thi bo qua.
        - Thong ke, xuat anh mau de kiem tra, ham trich ROI (Otsu 3 lop) dung lai duoc.
   [2] HUAN LUYEN  (notebook code (1).ipynb)
        - CycleGAN + UNet generator co Attention Mask + TV loss + TensorBoard.
        - Luu checkpoint MOI EPOCH (last.pth: model + optimizer + scheduler + history + RNG).
          Ngat giua chung (Ctrl+C, mat dien, crash) -> chay lai dung lenh cu, tu tiep tuc.
   [3] HAU XU LY   (notebook poss_processing.ipynb)
        - Mask vung kim loai (threshold+dilate+blur) -> bilateral tren anh GAN -> blending
          -> (tuy chon) lam min nen theo ROI. Xuat anh + so sanh + bang so lieu.
        - V7: THEM BUOC KET HOP TACH VUNG - "khung rang" (ROI Otsu, vung ham/rang) lay ket qua
          GAN-da-hau-xu-ly (final/final_smooth); vung BEN NGOAI khung rang (nen/khong khi, GAN
          khong duoc huan luyen de sua vung nay -> de "bia" chi tiet sai) bi thay hoan toan bang
          anh GOC da lam mo bang thuat toan co dien (Gaussian/Bilateral/NLM, KHONG dung GAN), roi
          "dem" khung rang (GAN) len tren, vien ghep duoc lam min (ROI_MASK_BLUR) de khong lo seam.
          -> ket qua: results/final_hybrid/<split>/. Xem CFG.HYBRID_COMPOSE.
        - V7: them chi so danh gia (edge-preservation/gradient-correlation vung khong kim loai),
          bang + bieu do so sanh 3 phien ban (final / final_smooth / final_hybrid) tren Train/Val/Test.

 V8 - CHAN DOAN TU KET QUA CHAY THAT (run V7 21-57-50_22-09-2026) + SUA GAN DE THUC SU GIAM NHIEU:
   Doc pixel truc tiep (khong chi nhin bieu do) tren anh that cho thay: GAN CO giam duoc vung kim loai
   BAO HOA CUNG (>=240, "core"): trung binh 253 -> 222 (-12%). NHUNG vung "streak/halo" toa ra tu kim
   loai (sang vua, ~150-220, chinh la vet soc/star-burst kinh dien cua nhieu kim loai CT/CBCT) HAU NHU
   KHONG doi (174 -> 170.7, -2%). Nguyen nhan GOC RE (khong phai do train qua it epoch, tuy 10 epoch
   cung qua it so voi ban goc train 600 epoch):
     (a) loss_mask (LAMBDA_MASK) ep mask m PHAI BANG CHINH XAC vung core (nguong 240, gian no 4px) ->
         cong thuc tron m*img+(1-m)*x khien OUTPUT = INPUT Y HET o moi noi m~0, tuc la KIEN TRUC TU
         CHAN generator khong bao gio duoc sua vung streak/halo, du train bao nhieu epoch cung vay.
     (b) hau xu ly (postprocess_one) TU TINH LAI 1 mask co dien rieng tu nguong 240, VUT BO hoan toan
         mask m ma generator da hoc -> dau ra final/final_smooth/hybrid bi "cap" lai lan 2 vao vung
         core, xoa sach moi thay doi generator co the da tao ra ben ngoai vung do.
     (c) tieu chi chon "best.pth" (val_score) chi dua vao sat_reduction/SSIM giu nguyen - de dat diem
         cao ngay ca khi generator gan nhu khong doi anh (khong co so hang nao bat buoc phai THUC SU
         giam duoc vung streak) -> "best epoch=2" bi chon som, chua kip hoc sua streak.
   V8 sua ca 3 diem tren:
     (a) loss_mask doi thanh 1 CHIEU: bat buoc PHU het vung core (khong duoc < target), nhung KHONG
         bi phat khi mask > target min la con trong 1 vung "halo" rong hon (nguong thap hon, gian no
         xa hon, xem CFG.HALO_THRESH_U8/HALO_DILATE_PX/LAMBDA_MASK_HALO) -> go bo "tran" kien truc,
         generator duoc TU DO hoc sua ca vung streak neu adversarial/cycle loss thay co loi.
     (b) postprocess_one nhan them mask cua generator (net_mask), HOP (OR) voi mask co dien thay vi
         thay the -> vung generator that su sua duoc khong bi "cap" lai boi hau xu ly nua.
     (c) them chi so val_halo_reduction (% giam sang trong vung halo, KHONG tinh vung core) vao
         val_score -> "best" khong the con la 1 epoch gan-nhu-khong-doi-anh.
   Ngoai ra: EPOCHS mac dinh tang 10 -> 600 (giong ban goc) + EARLY STOPPING (patience, xem
   CFG.EARLY_STOP_PATIENCE) de tu dung neu khong con cai thien, khoi phai cho du 600 epoch; LR va
   LAMBDA_CYCLE/LAMBDA_IDENTITY dua ve gan gia tri ban goc (2e-4, 10.0, 5.0) vi EMA/grad-clip/label-
   smoothing (V4) da du on dinh de chiu duoc cac gia tri manh hon nay qua nhieu epoch hon.
   THEM PHUONG PHAP SO SANH: CFG.METHODS chay TUAN TU trong 1 lan `python train_dental_mar_V8.py`:
   "gan" (CycleGAN da sua o tren) roi den "classical" (3 bien the xu ly anh co dien - khong mang no-
   ron, port tu dental_mar_classical.py, cung bo chi so preservation-region de so sanh cong bang) ->
   cuoi cung tu dong gop ket qua 2 phuong phap vao 1 bang xep hang + bieu do so sanh chung, nam trong
   <OUTPUT_ROOT>/ALL_METHODS_comparison_.../ - dung thang de viet paper.

 V8.3 - CHAN DOAN TU RUN THAT tren SERVER (gan_17-46-47_23-09-2026, 133 epoch, early-stop tai epoch
   93 do 40 epoch khong cai thien, val_score~0.39): GAN van CHUA giam nhieu du manh (val_halo_reduction
   dinh ~9% roi CHUNG LAI). Doc lai code + log/history.csv THAT (khong doan) cho thay 2 nguyen nhan:
     (d) LR KHONG BAO GIO giam: lich giam LR (LambdaLR) chi bat dau tu DECAY_START_EPOCH (mac dinh
         EPOCHS//2 = 300), nhung early-stop luon kich hoat truoc do RAT nhieu (epoch ~93-133) vi
         EARLY_STOP_PATIENCE=40 << 300 -> ca qua trinh train CHUA BAO GIO co giai doan "fine-tune"
         voi LR thap, luc mo hinh moi thuc su hoi tu sau vao chi tiet nho (streak/halo).
     (e) loss_mask (V8) chi ep MASK (m) phu vung kim loai, KHONG ep GIA TRI PIXEL dau ra (fake_n)
         tien ve gia tri "sach" trong vung do - duoi CycleGAN khong ghep cap (unpaired), khong co
         cap pixel that de so sanh truc tiep trong vung metal/halo, nen tin hieu duy nhat de generator
         "biet" phai sua thanh gi trong vung do la adversarial+cycle (rat gian tiep, khap ca anh).
   V8.3 sua ca 2 diem tren (+ don gian hoa duong dan de chay duoc ca local Windows lan server Linux
   tu 1 file duy nhat, xem CFG.DATASET_DIR/OUTPUT_ROOT/BATCH_SIZE/NUM_WORKERS - doc tu bien moi truong
   MAR_* neu co, khong thi tu chon theo he dieu hanh):
     (d) LambdaLR (lich co dinh theo epoch) -> ReduceLROnPlateau (LR_G/LR_D) theo doi thang val_score,
         tu dong giam LR (nhan LR_DECAY_FACTOR) sau LR_DECAY_PATIENCE epoch lien tiep khong cai thien
         (mac dinh = EARLY_STOP_PATIENCE//2) - dam bao LUON co it nhat 1 lan giam LR truoc khi early-
         stop kich hoat, bat ke qua trinh train dai/ngan the nao (khong con phu thuoc doan truoc EPOCHS
         se dung o dau nhu DECAY_START_EPOCH cu).
     (e) THEM loss_pseudo (LAMBDA_PSEUDO): trong VUNG HALO (rim/rìa quanh kim loai, KHONG dung vung
         core bao hoa hoan toan - o do du lieu that su mat, ep gia tri se la "bia"), so sanh fake_n
         (dau ra GAN) voi 1 "pseudo-target" = anh dau vao da lam mo (box-blur, torch thuan, nhanh,
         tren GPU) - lam mo o day dong vai tro "gia lap ban sach", cho generator 1 tin hieu pixel-level
         TRUC TIEP de biet HUONG can sua (giam sang, lam min) ngay ca khi adversarial/cycle chua du
         manh - dung y tuong nguoi dung de xuat ("lam mo phan rìa ben ngoai roi ket hop tinh loss").
         Day la SUPERVISION YEU/HEURISTIC (khong phai ground-truth that), trong so LAMBDA_PSEUDO vua
         phai de khong lam mat chi tiet giai phau that trong vung khong bi kim loai lan toi.

 V9 - THEO YEU CAU NGUOI DUNG: "bo classical di (khong con la 1 phuong phap doc lap so sanh voi GAN),
   classical chi de lam mo cac vet sang ben ngoai thoi, GAN de len anh da xu ly mo vien ngoai". Doi
   CFG.METHODS mac dinh chi con ["gan"] (bo "classical" khoi bang xep hang doc lap - van con giu ham
   stage_classical() neu sau nay muon bat lai de doi chieu). QUAN TRONG HON: postprocess_one() (buoc
   hau xu ly cua CHINH GAN) gio la 3 vung thay vi 2:
     - vung "core" (kim loai chac chan + mask GAN tu hoc, bien M nhu cu)   -> GAN (+bilateral), "de len".
     - vung "halo/streak" (sang vua, RONG hon core - xem make_halo_band_mask, dung CHUNG nguong voi
       loss_pseudo luc train) NHUNG NGOAI vung GAN o tren                  -> inpaint co dien (method
       da kiem chung giam sang MANH NHAT trong 3 bien the co dien, ~0.27, manh hon GAN nhieu du GAN
       da co loss_pseudo - xem so sanh run V8.3b/V8.4).
     - phan con lai (ngoai ca 2 vung tren)                                 -> giu nguyen anh goc.
   Tat ca dung mask LIEN TUC (Gaussian-blur) nen ghep muot, khong seam. BAT/TAT bang
   CFG.STREAK_CLASSICAL_SMOOTH (mac dinh True; False = quay lai hanh vi V8 - vung ngoai M la anh goc
   CHUA qua xu ly). Anh minh hoa (comparison_montage) them 2 panel "halo band"/"streak inpainted" de
   thay ro dau la phan GAN sua, dau la phan co dien sua.

 V9.1 - CHAN DOAN TU KET QUA THAT (run train_dental_mar_V8_3_gan_00-59-05_24-09-2026, best epoch 104,
   ma nguon chay run nay la ban TRUOC V8.4/V9 - config_snapshot.json cua no van con LAMBDA_PSEUDO=2.0/
   LAMBDA_MASK_HALO=1.0 va METHODS=["gan","classical"], mask_area_pct van con 33.8-38.3% nhu da chan
   doan o V8.4) + 2 phan anh moi cua nguoi dung: "anh sau GAN bi mo qua" va "phan lam min ben ngoai
   qua nhat so voi anh normal". So sanh truc tiep figures/qualitative_Val.png (cot "Raw GAN output")
   voi data_check/samples_Normal_train.png xac nhan CA HAI:
     (a) MO: cot "Raw GAN output" mat chi tiet/ket cau CA O NHUNG VUNG KHONG HE CO GLARE (vd rang/xuong
         binh thuong o hang (b)(c)(d)) - goc re la total_variation_loss(fake_n) truoc day tinh TREN
         TOAN BO anh sinh ra, phat ca vung khong lien quan gi den nhieu kim loai -> generator "hoc"
         cach lam mo/mat chi tiet CA ANH de giam loss nay. SUA: total_variation_loss() them tham so
         mask tuy chon, ap dung cho fake_n voi mask=halo_target (CHI vung nhieu + phu can, xem sua o
         vong lap train) - can HUAN LUYEN LAI moi co hieu luc. Bu them (khong can train lai, hieu luc
         NGAY o lan hau xu ly tiep theo): giam SIGMA_COLOR/SIGMA_SPACE cua bilateral 75->35 (đang lam
         min qua manh vung GAN) + them unsharp_mask() (CFG.SHARPEN_AMOUNT/SHARPEN_SIGMA) tra lai chi
         tiet SAU bilateral.
     (b) NHAT: nen/mo mem xung quanh trong ket qua qua PHANG/DEU mau xam so voi anh Normal that (von
         co lom dom/nhieu TU NHIEN o mo/xuong, xem samples_Normal_train.png) - do BG_NLM_H=15 +
         BG_GAUSS_KSIZE=7 (bg_smooth_roi) va HYBRID_BG_NLM_H=12 + HYBRID_BG_GAUSS_KSIZE=9 +
         HYBRID_BG_BILAT_SIGMA=60 (hybrid_compose, ap dung THEM 1 lan nua len nen) qua manh, xoa sach
         ca ket cau that. SUA: giam con BG_NLM_H=7/BG_GAUSS_KSIZE=3 va HYBRID_BG_NLM_H=6/
         HYBRID_BG_GAUSS_KSIZE=3/HYBRID_BG_BILAT_SIGMA=30 - hieu luc NGAY, khong can train lai (chi la
         tham so hau xu ly). Muon thu nhanh CAC THAM SO HAU XU LY moi ma KHONG train lai: dat
         CFG.RESUME = r"<duong dan thu muc run GAN cu>", CFG.RUN_TRAIN=False, CFG.RUN_POSTPROCESS=True,
         CFG.POST_OVERWRITE=True roi chay lai - script se dung lai best.pth co san, chi chay lai
         stage_post() voi CFG hau xu ly moi (vai phut thay vi vai gio).

 Moi lan chay tao 1 thu muc rieng CHO TUNG PHUONG PHAP: <OUTPUT_ROOT>/train_dental_mar_V8_<method>_HH-MM-SS_DD-MM-YYYY/
        checkpoints/  progress_images/  results/  data_check/  tb/  history.csv  log.txt ...

 Cai thu vien (1 lan):
   E:\\Project\\.venv\\Scripts\\activate
   pip install numpy pandas opencv-python scikit-image matplotlib tensorboard tqdm
   pip install torch torchvision      (ban CUDA: xem https://pytorch.org/get-started/locally )
==============================================================================
"""
from __future__ import annotations

import os
import sys
import re
import json
import time
import random
import shutil
import copy
import itertools
import traceback
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
except ImportError as _e:  # pragma: no cover
    sys.exit(
        f"\n[LOI] Thieu thu vien: {_e}\n"
        "Hay activate venv roi chay:\n"
        "   pip install numpy pandas opencv-python scikit-image matplotlib tensorboard tqdm\n"
        "   pip install torch torchvision\n"
    )

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard chua cai -> bo qua, khong loi
    SummaryWriter = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ==============================================================================
#  CFG  -  TAT CA CAI DAT O DAY
# ==============================================================================
class CFG:
    # ------------------------------ giai doan chay ---------------------------
    RUN_BUILD_DATA = True        # [1] tao Dataset_TF tu ToothFairy (tu bo qua neu da co)
    RUN_TRAIN = True             # [2] huan luyen
    RUN_POSTPROCESS = True       # [3] hau xu ly + xuat anh ket qua

    # ------------------------------ duong dan --------------------------------
    # V8.3: file nay gio chay ca tren may local Windows LAN ca tren server Linux (cung 1 storage,
    # chi khac cach mount: o Windows la o dia E:\Project\Dental, ben server la /workspace/projects/
    # CBCT - da xac nhan qua run thuc te "classical_20-44-18" co CA HAI duong dan trong log.txt cua
    # no tuy noi doc). THAY VI sua tay 2 ban file khac nhau cho tung may (de gay lech cau hinh nhu
    # BATCH_SIZE 4 vs 16 da xay ra), duong dan/batch size/so worker gio TU CHON theo bien moi truong
    # MAR_* neu co, khong thi theo he dieu hanh dang chay (os.name == "nt" -> Windows). Vi du chay
    # tren server voi duong dan khac: MAR_DATASET_DIR=/data/Dataset_TF python train_dental_mar_V8.py
    _IS_WIN = (os.name == "nt")
    TOOTHFAIRY_DIR = os.environ.get(
        "MAR_TOOTHFAIRY_DIR",
        r"E:\Project\Dental\ToothFairy_dataset\ToothFairy_Dataset" if _IS_WIN
        else "/workspace/projects/CBCT/ToothFairy_dataset/ToothFairy_Dataset")   # co splits.json + Dataset/P*/data.npy
    # Thu muc dataset dang Glare/Normal x Train/Val/Test. Neu chua co -> tu tao tu ToothFairy.
    DATASET_DIR = os.environ.get(
        "MAR_DATASET_DIR",
        r"E:\Project\Dental\Dataset_TF" if _IS_WIN else "/workspace/projects/CBCT/Dataset_TF")
    OUTPUT_ROOT = os.environ.get("MAR_OUTPUT_ROOT") or None   # None = <thu muc chua file nay>/runs
    FORCE_REBUILD = False        # True: xoa & tao lai DATASET_DIR (chi xoa neu do script nay tao ra)

    # Tiep tuc sau khi bi ngat:
    #   "auto" : tim run chua xong moi nhat trong OUTPUT_ROOT va tiep tuc, neu khong co thi tao run moi
    #   "new"  : luon tao run moi (bo qua run dang do)
    #   duong dan thu muc run cu (vd r"E:\...\runs\train_dental_mar_14-02-11_21-09-2026") de tiep tuc dung run do
    # V8.3: giu "new" - ham loss vua doi THEM (loss_pseudo) + doi ca lich giam LR (xem LR_DECAY_*
    # ben duoi), nen so sanh cong bang can 1 run MOI hoan toan; cac run GAN truoc (01-09-37, 12-17-41,
    # server 17-46-47) coi nhu baseline/tham khao, KHONG resume tiep bang ham loss cu.
    RESUME = os.environ.get("MAR_RESUME", "new")

    # ------------------------------ [1] DU LIEU: ToothFairy -> PNG ------------
    INCLUDE_SYNTHETIC_TRAIN = False   # splits.json co nhom 'synthetic' (290 ca sinh nhan tao) -> mac dinh KHONG dung
    SLICE_AXIS = None                 # None = tu chon truc co so lat NHO NHAT (thuong la truc axial). Hoac 0/1/2
    SLICE_TRANSPOSE = False           # xoay/lat anh cho dung chieu (xem data_check/orientation_preview.png)
    SLICE_FLIP_UD = False
    SLICE_FLIP_LR = False
    SLICE_STEP = 2                    # xet moi SLICE_STEP lat
    SUBSAMPLE = 4                     # lay mau 1/4 moi chieu de tinh percentile (nhanh)
    P_LO, P_HI = 0.5, 99.5            # cua so cuong do -> uint8
    MIN_FG_FRAC = 0.15                # bo lat gan nhu rong (dau/cuoi khoi)
    ONLY_SLICES_WITH_LABEL = False    # True: chi lay lat co nhan gt_sparse (vung ong tuy ham duoi)

    # Phan loai Glare / Normal (heuristic - ToothFairy KHONG co nhan "co kim loai")
    #   kim loai = voxel > bone_ref + METAL_MARGIN*(vmax-bone_ref), bone_ref = percentile BONE_PERCENTILE cua vung co mo
    #   Xem cot gap_ratio trong <DATASET_DIR>/volumes_summary.csv de chinh: khoi sach thuong co gap_ratio thap.
    FG_FRAC = 0.10                    # "co mo" = cuong do > p1 + FG_FRAC*(p99-p1)
    BONE_PERCENTILE = 98
    VOLUME_METAL_GAP = 1.0            # khoi "co kim loai" neu gap_ratio=(vmax-bone_ref)/(bone_ref-median) > VOLUME_METAL_GAP. Giam -> nhay hon
    METAL_MARGIN = 0.5
    GLARE_MIN_FRAC = 0.002            # lat la Glare neu dien tich kim loai >= 0.2% anh
    NORMAL_MAX_FRAC = 0.0             # lat la Normal neu dien tich kim loai <= gia tri nay ...
    NORMAL_MIN_DIST_SLICES = 20       # ... VA cach moi lat Glare >= 20 lat (vet sang lan sang lat ke ben)
    # so lat toi da lay tren MOI benh nhan:
    CAP_PER_VOLUME = {
        "Glare":  {"Train": 30, "Val": 30, "Test": 30},
        "Normal": {"Train": 12, "Val": 30, "Test": 30},
    }

    # ------------------------------ [2] HUAN LUYEN ---------------------------
    SEED = 42
    IMG_SIZE = 256                    # anh duoc resize ve IMG_SIZE x IMG_SIZE (giong ban goc)
    # V8.3: BATCH_SIZE tu chon theo may (GPU laptop Windows thuong yeu hon server) - dat MAR_BATCH_SIZE
    # de ghi de neu can (vd GPU server nhieu VRAM co the tang them).
    BATCH_SIZE = int(os.environ.get("MAR_BATCH_SIZE", "4" if _IS_WIN else "16"))
    # Windows: moi worker phai nap lai torch+CUDA (rat ton RAM/pagefile) -> de 0 cho an toan (anh nho, GPU moi la nut that).
    # Neu muon nhanh hon: tang pagefile Windows roi thu 2. (Val loader luon dung 0 worker.)
    NUM_WORKERS = int(os.environ.get("MAR_NUM_WORKERS", "0" if _IS_WIN else "4"))
    # V8: EPOCHS tang 10 -> 600 (giong dung ban goc/notebook cua paper_1). Ly do doi lai: doc pixel
    # tren anh that (run V7) cho thay 10 epoch la QUA IT - generator moi hoc giam duoc vung kim loai
    # bao hoa cung (>=240), con vung streak/halo toa ra (~150-220, chinh la vet soc kinh dien) hau nhu
    # chua duoc dung toi. Dung EARLY_STOP_* ben duoi de tu dung neu qua nhieu epoch khong con cai
    # thien nua, khong bat buoc phai chay du 600. best.pth luon la epoch diem cao nhat (val_score moi,
    # xem SCORE_WEIGHTS) bat ke dung o epoch nao.
    EPOCHS = 600
    EARLY_STOP_PATIENCE = 40          # V8: dung neu val_score khong cai thien (> +EARLY_STOP_MIN_DELTA)
                                       # trong EARLY_STOP_PATIENCE epoch lien tiep. 0 = tat (luon chay du EPOCHS).
    EARLY_STOP_MIN_DELTA = 0.001
    # V8: dua LR ve gan ban goc (2e-4, khong tach TTUR) - ly do doi lai: V4 giam LR + tach TTUR de
    # chong dao dong khi chi train vai chuc epoch; nay train lau hon nhieu (EPOCHS=600) va da co
    # EMA/grad-clip/label-smoothing (van giu duoi day) du de hap thu dao dong, nen dung LR manh hon
    # nhu ban goc de hoc nhanh/sau hon trong cung so buoc cap nhat.
    LR_G = 2e-4                       # V8: LR cho generator (G+F) - ve gan ban goc (truoc V4: 1e-4)
    LR_D = 2e-4                       # V8: LR cho discriminator (DX+DY) - ve gan ban goc (truoc V4: 1e-4)
    BETAS = (0.5, 0.999)
    # V8.3: BO lich giam LR co dinh theo epoch (LambdaLR + DECAY_START_EPOCH=EPOCHS//2=300) - early-
    # stop (patience 40) hau nhu LUON kich hoat truoc epoch 300 rat xa (xem run that: dung o epoch
    # 93-133) nen LR KHONG BAO GIO thuc su giam, mo hinh khong co giai doan fine-tune LR thap. Thay
    # bang ReduceLROnPlateau: tu giam LR (nhan LR_DECAY_FACTOR) khi val_score khong cai thien qua
    # LR_DECAY_PATIENCE epoch lien tiep - LUON co it nhat 1-2 lan giam LR truoc khi early-stop kich
    # hoat, bat ke train dai/ngan bao nhieu (khong con phu thuoc doan truoc EPOCHS se dung o dau).
    LR_DECAY_FACTOR = 0.5             # moi lan "plateau": LR *= LR_DECAY_FACTOR
    LR_DECAY_PATIENCE = max(1, EARLY_STOP_PATIENCE // 2)   # EARLY_STOP_PATIENCE da dinh nghia o tren
    LR_MIN = 1e-6                     # san LR toi thieu (ReduceLROnPlateau se khong giam duoi muc nay)
    LABEL_SMOOTH_REAL = 0.9           # V4: label smoothing 1 chieu - nhan "real" cho discriminator la 0.9 thay vi 1.0
                                       # (nhan "fake" van la 0.0). Giup discriminator khong qua tu tin -> gian tiep
                                       # on dinh generator. 1.0 = tat (giong ban goc).
    EMA_DECAY = 0.999                 # V4: Exponential Moving Average cua trong so gen_G, dung de danh gia/luu
                                       # best.pth/xuat anh/hau xu ly (KHONG dung trong duong lan truyen nguoc luc train).
                                       # Muc tieu: giam nhieu tu viec chon "epoch tot nhat" theo 1 epoch don le dang dao dong.
                                       # 0 = tat (dung thang gen_G nhu ban goc).
    GRAD_CLIP_NORM = 5.0              # V4: gioi han gradient-norm truoc moi opt.step() cho ca G+F va DX+DY,
                                       # chan cac buoc cap nhat qua lon gay dao dong/NaN. 0 = tat.
    GAN_LOSS = "lsgan"                # "lsgan" (on dinh hon) hoac "bce" (nhu ban goc)
    # V8: dua LAMBDA_CYCLE/LAMBDA_IDENTITY ve dung gia tri ban goc (10.0/5.0). Ly do doi lai: nghi
    # ngo ban dau (V3) la 2 he so nay qua manh gay dao dong khong con dung nua - can nguyen goc thuc
    # su cua viec generator "khong dam sua" la KIEN TRUC mask-gate (xem LAMBDA_MASK_HALO ben duoi),
    # khong phai do 2 he so nay. Dua ve ban goc de gan voi ket qua da cong bo cua paper_1 hon.
    LAMBDA_CYCLE = 10.0
    LAMBDA_IDENTITY = 5.0
    LAMBDA_TV = 0.1                   # Phuong phap 2 - TV loss
    LAMBDA_MASK = 2.0                 # Ep attention mask PHAI phu vung kim loai that (core, nguong
                                       # METAL_THRESH_U8, gian no MASK_DILATE_PX) - xem loss_mask 1-chieu ben duoi.
    MASK_DILATE_PX = 4                # V5: ban kinh gian no vung "core" chac chan la kim loai (khong doi o V8 -
                                       # day van la vung PHAI duoc sua, chi la khong con la GIOI HAN TREN nua).
    # V8: THEM vung "halo" rong hon vung core - day la sua chinh cho van de "GAN khong that su giam
    # nhieu" phat hien tu anh that. Truoc V8, loss_mask ep mask m = CHINH XAC vung core (2 chieu, L1),
    # khien cong thuc blend m*img+(1-m)*x buoc OUTPUT=INPUT o MOI noi ngoai vung core rat nho -> generator
    # KHONG THE sua vung streak/halo (sang vua ~150-220) du hoc bao nhieu epoch. V8 doi loss_mask thanh
    # 1 CHIEU: bat buoc phu het core (relu(core_target - m)), nhung CHI bi phat neu vuot ra NGOAI ca
    # vung halo rong hon (relu(m - halo_target)) - trong khoang [core..halo] generator duoc TU DO
    # quyet dinh co sua hay khong (do adversarial/cycle loss dan dat), khong con bi ep = 0.
    HALO_THRESH_U8 = 150               # nguong "sang vua" - vung streak/halo dien hinh (< METAL_THRESH_U8)
    HALO_DILATE_PX = 20                # gian no vung halo - RONG hon core nhieu (chi la GIOI HAN TREN, khong phai muc tieu bat buoc)
    # V8.4: TANG LAMBDA_MASK_HALO (1.0 -> 1.5) - run V8.3b (pseudo-target = inpaint THAT, xem
    # LAMBDA_PSEUDO ben duoi) cho thay mask_area_pct nhay tu ~22% (box-blur, loss gan nhu vo hieu)
    # len 34-38% (>> sat_orig ~20-26%, tuc mask dang LAN RA ngoai vung kim loai that) - halo_reduction
    # tang manh (0.09 -> 0.13-0.26) NHUNG doi lai preservation (SSIM/PSNR/MAE vung khong kim loai)
    # giam ro (0.85->0.82 Test, te hon nua o Val). Tang phat mask "lan ra ngoai" de keo vung sua lai
    # gan dung ranh gioi halo hon, giam tac dung phu nay.
    LAMBDA_MASK_HALO = 1.5             # trong so phat khi mask vuot ra NGOAI ca vung halo (giu correction trong pham vi giai phau hop ly)
    # V8.3: loss_pseudo - THEM tin hieu pixel-level TRUC TIEP (unpaired CycleGAN khong co san) trong
    # vung halo/rim quanh kim loai: so sanh dau ra GAN (fake_n) voi 1 pseudo-target = anh da INPAINT
    # (xem build_pseudo_clean_targets) - CHI trong vung halo (KHONG dung vung core bao hoa hoan toan,
    # o do khong con du lieu that de "doan sach"). Y tuong nay tu nguoi dung de xuat ("lam mo phan
    # rìa roi ket hop tinh loss"), sau nang cap tu box-blur (qua yeu, gan nhu vo hieu) len inpaint
    # (V8.3b). V8.4: GIAM trong so (2.0 -> 1.0) - voi target la inpaint THAT (chenh lech lon hon
    # nhieu so voi box-blur truoc do), trong so 2.0 cu (tinh luc target con yeu) hoa ra qua manh,
    # keo mask lan ra ngoai vung kim loai that (xem LAMBDA_MASK_HALO o tren) va lam giam preservation.
    LAMBDA_PSEUDO = 1.0                # 0 = tat hoan toan (quay lai hanh vi V8/V8.2 khong co loss nay)
    PSEUDO_BLUR_KSIZE = 21             # le; fallback box-blur (chi dung khi thieu file pseudo-target da precompute)
    IMAGE_POOL_SIZE = 50              # bo dem anh gia cho discriminator (CycleGAN chuan). 0 = tat
    AUG_ROTATE_DEG = 10               # augmentation: lat ngang + xoay +-10 do
    AUG_SCALE = 0.05
    TRAIN_ON_ROI = False              # True: cat ROI (Otsu 3 lop) tren moi anh truoc khi dua vao mang

    # Kiem dinh (validation) moi epoch
    VAL_MAX_IMAGES = 200              # toi da so anh Glare/Normal dung cho validation
    METAL_THRESH_U8 = 240             # nguong "sang bao hoa" (giong THRESH cua hau xu ly)
    # V8: diem chon best.pth them so hang "halo" (% giam sang vung streak/halo, KHONG tinh vung core)
    # - truoc V8 chi co sat/pres/id, ca 3 deu de dat diem cao ngay ca khi generator gan nhu khong doi
    # anh (chinh la ly do "best epoch=2" bi chon qua som). "halo" bat buoc phai co giam sang THAT o
    # vung ngoai core thi diem moi cao, khong the "lach" bang cach khong doi gi.
    SCORE_WEIGHTS = {"sat": 0.30, "pres": 0.20, "id": 0.15, "halo": 0.35}

    # Luu / hien thi
    SAVE_EVERY = 5                    # luu them gen_G/gen_F rieng moi N epoch
    KEEP_LAST_N_EPOCH_FILES = 4       # chi giu N file epoch gan nhat (tiet kiem o cung); 0 = giu het
    PREVIEW_EVERY = 1                 # xuat anh tien trinh moi N epoch
    PREVIEW_N_GLARE = 4
    PREVIEW_N_NORMAL = 2
    PLOT_EVERY = 5                    # ve loss_curves.png moi N epoch
    USE_TENSORBOARD = True
    TB_IMAGE_EVERY = 5

    # ------------------------------ [3] HAU XU LY ----------------------------
    POST_WEIGHTS = "best"             # "best" | "last"
    POST_SPLITS = ["Val", "Test"]     # them "Train" neu muon
    POST_MAX_IMAGES_PER_SPLIT = None  # None = het
    POST_COMPARISON_MAX = 40          # so anh so sanh 8 panel xuat moi split
    POST_OVERWRITE = False
    # Giai doan 1: mask
    THRESH = 240
    KERNEL_SIZE = 7
    DILATE_ITERS = 3
    BLUR_KSIZE = 15                   # le
    # V8: nguong (tren anh mask 0..255 do generator xuat ra, xem run_gan()) de coi la "generator THAT
    # SU muon sua o day" -> mask nay duoc HOP (OR) voi mask co dien (THRESH) khi hau xu ly, thay vi
    # hau xu ly tu tinh lai 1 mask rieng roi VUT BO mask cua generator nhu V6/V7 (xem make_artifact_mask).
    NET_MASK_THRESH_U8 = 60
    # Giai doan 2: bilateral
    # V9.1: giam SIGMA_COLOR/SIGMA_SPACE 75->35 (D9 giu nguyen) - 75 la muc lam min RAT manh (gan nhu
    # "son phang" vung GAN), la 1 phan nguyen nhan gay "anh sau GAN mo qua" nguoi dung phan anh (cung
    # voi TV loss toan anh da sua o total_variation_loss). SHARPEN_* ben duoi bu lai chi tiet mat do
    # bilateral (unsharp mask SAU khi bilateral, khong thay the bilateral - van can bilateral de xoa
    # rang cua/artifact tu GAN, chi giam do "phang" cua no).
    BILAT_D = 9
    SIGMA_COLOR = 35
    SIGMA_SPACE = 35
    # V9.1: unsharp mask ap dung SAU bilateral (chi trong vung GAN, gan_s) de "tra lai" mot phan chi
    # tiet/canh sac da mat do bilateral + generator lam mo - lam theo cong thuc unsharp mask co dien:
    # sharp = gan_s + SHARPEN_AMOUNT * (gan_s - gaussian_blur(gan_s, SHARPEN_SIGMA)). AMOUNT=0 -> tat.
    SHARPEN_AMOUNT = 0.6
    SHARPEN_SIGMA = 1.0
    # Buoc lam min nen theo ROI (cell 7 cua notebook)
    POST_BG_SMOOTH = True
    OTSU_OFFSET = 0
    ROI_CLOSE_KERNEL = 9
    ROI_DILATE_ITERS = 1
    ROI_MIN_AREA_RATIO = 0.001
    BACKGROUND_VALUE = 0
    ROI_MASK_BLUR = 11
    # V9.1: giam BG_NLM_H 15->7 va BG_GAUSS_KSIZE 7->3 - anh CT that (xem data_check/samples_Normal_*.png)
    # co nhieu/kem mo lom dom TU NHIEN o vung mo/xuong, NLM h=15 + Gauss 7x7 xoa gan het no, bien nen
    # thanh mang mau xam PHANG DEU - day chinh la "phan lam min ben ngoai dang bi qua nhat so voi anh
    # normal" nguoi dung phan anh (nhat = mat tuong phan/lom dom tu nhien, khong phai sang hon ve gia
    # tri pixel trung binh). Gia tri moi giu du de triet noise ro (dot sang do CT-artifact) nhung khong
    # xoa sach ket cau mo mem/xuong nhu truoc.
    BG_NLM_H = 7
    BG_GAUSS_KSIZE = 3

    # ------------------------------ [4] XUAT BIEU DO / SO LIEU CHO PAPER -----
    EXPORT_PAPER_ASSETS = True        # V6: xuat them bieu do rieng tung nhom (do phan giai cao), bang thong ke,
                                       # metric theo tung anh (preservation-region SSIM/PSNR/MAE), thoi gian suy luan,
                                       # so tham so mo hinh, anh so sanh dinh dang "paper" (matplotlib, nhan a/b/c..)
    PAPER_DPI = 300                   # do phan giai xuat anh (chuan toi thieu cho paper/luan van)
    PAPER_QUALITATIVE_N = 6           # so vi du trong anh so sanh dinh tinh (qualitative) moi split

    # ------------------------------ [5] V7: KET HOP "KHUNG RANG" (GAN) + NEN LAM MO CO DIEN --------
    # Y tuong: GAN chi duoc huan luyen/dang tin cay o VUNG RANG (noi co du lieu Glare/Normal de hoc).
    # O VUNG NGOAI KHUNG RANG (nen, khong khi, vien anh...) GAN co the "bia" (hallucinate) chi tiet
    # khong co that. V7: tach rieng 2 vung bang ROI Otsu 3 lop (extract_roi_mask, da dung san cho
    # POST_BG_SMOOTH) - trong khung rang GIU nguyen ket qua GAN da hau xu ly (final/final_smooth),
    # NGOAI khung rang THAY bang anh goc da lam mo THUAN CO DIEN (khong qua GAN), roi "dem" khung
    # rang GAN len tren nen do (vien mask duoc lam min de ghep tu nhien, khong bi seam cung).
    HYBRID_COMPOSE = True             # V7: BAT/TAT xuat them results/final_hybrid/
    HYBRID_SOURCE = "final_smooth"    # nguon anh GAN dung trong khung rang: "final_smooth" (neu co) hoac "final"
    HYBRID_BG_METHOD = "nlm_gauss"    # lam mo vung NGOAI khung rang: "gaussian" | "bilateral" | "nlm_gauss"
    # V9.1: giam HYBRID_BG_GAUSS_KSIZE 9->3, HYBRID_BG_BILAT_SIGMA 60->30, HYBRID_BG_NLM_H 12->6 - cung
    # ly do nhu BG_NLM_H/BG_GAUSS_KSIZE o tren (day la buoc lam mo nen THU 2, ap dung cho ca.the anh o
    # ket qua "hybrid"): gia tri cu xoa het lom dom mo/xuong tu nhien -> nen "qua nhat" (mat tuong phan)
    # so voi anh Normal that. Neu van con nhat sau khi doi, thu tiep HYBRID_BG_METHOD="bilateral" (giu
    # canh xuong ro hon nlm_gauss) hoac giam them nua.
    HYBRID_BG_GAUSS_KSIZE = 3          # le; dung khi HYBRID_BG_METHOD co buoc Gaussian
    HYBRID_BG_BILAT_D = 9
    HYBRID_BG_BILAT_SIGMA = 30
    HYBRID_BG_NLM_H = 6                # cuong do khu nhieu NLM (fastNlMeansDenoising)
    # Tham so ROI "khung rang" (giong extract_roi_mask dung cho POST_BG_SMOOTH, tach rieng de chinh doc lap)
    HYBRID_ROI_OTSU_OFFSET = 0
    HYBRID_ROI_CLOSE_KERNEL = 9
    HYBRID_ROI_DILATE_ITERS = 1
    HYBRID_ROI_MIN_AREA_RATIO = 0.001
    HYBRID_ROI_MASK_BLUR = 15         # le; blur bien ROI de "dem" khung rang vao nen khong bi vien cung

    # ------------------------------ [6] V8: NHIEU PHUONG PHAP, CHAY 1 LAN ----
    # Chay TUAN TU (khong cung luc - 1 GPU laptop khong du VRAM de train 2 mang cung luc) tung phuong
    # phap trong danh sach nay, MOI phuong phap ra 1 thu muc run rieng (ten co hau to _<method>), roi
    # CUOI CUNG tu dong gop ket qua TAT CA phuong phap da xong vao 1 bang xep hang + bieu do so sanh
    # chung (xem aggregate_all_methods) - chay 1 lenh duy nhat la co du so lieu de viet paper.
    #   "gan"       : CycleGAN (co sua o V8, xem ghi chu dau file) - phai train, lau nhat.
    #   "classical" : 3 bien the xu ly anh co dien (KHONG mang noron, port tu dental_mar_classical.py)
    #                 - khong can train, chay xong trong vai phut, dung lam baseline so sanh.
    # V9: THEO YEU CAU - bo "classical" khoi danh sach PHUONG PHAP DOC LAP (khong con la 1 hang doi
    # dau voi GAN trong bang xep hang nua). Co dien gio la 1 THANH PHAN o BEN TRONG chinh pipeline GAN
    # (xem postprocess_one/make_halo_band_mask): GAN chi con lo vung "core" (kim loai chac chan + mask
    # generator tu hoc), CON vung "vet sang ben ngoai" (streak/halo, KHONG bao hoa) duoc inpaint co
    # dien (method da kiem chung halo_reduction manh nhat, ~0.27, trong 3 bien the co dien) TRUOC, roi
    # GAN moi "de len" (M*gan_smoothed) phia tren nen da duoc lam sach do - dung y tuong nguoi dung de
    # xuat ("classical chi de lam mo vet sang ben ngoai, GAN de len anh da xu ly mo vien ngoai").
    # Muon so sanh lai voi "classical" doc lap (baseline cu) thi them "classical" vao list nay.
    METHODS = ["gan"]
    # --- [6a] tham so rieng cho phuong phap "classical" (giu nguyen y nghia nhu dental_mar_classical.py) ---
    CLASSICAL_INPAINT_RADIUS = 7
    CLASSICAL_INPAINT_DILATE_EXTRA = 2
    # V9: BAT/TAT buoc inpaint co dien cho vung streak/halo BEN TRONG postprocess_one cua GAN (xem
    # ghi chu o tren). Dung CHUNG nguong/gian no voi loss_pseudo luc train (HALO_THRESH_U8/
    # HALO_DILATE_PX) va ban kinh inpaint cua "classical" (CLASSICAL_INPAINT_RADIUS) o tren - tat ca
    # deu la "co dien", chi khac la gio chay NGAY TRONG buoc hau xu ly cua GAN thay vi la 1 phuong
    # phap doc lap. False = quay lai hanh vi V8 (vung ngoai M la anh GOC, chua qua co dien).
    STREAK_CLASSICAL_SMOOTH = True


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GROUPS = ("Glare", "Normal")
SPLITS = ("Train", "Val", "Test")
SCRIPT_PATH = Path(__file__).resolve()
RUN_PREFIX = SCRIPT_PATH.stem


# ==============================================================================
#  TIEN ICH CHUNG
# ==============================================================================
class Logger:
    def __init__(self):
        self.file = None

    def open(self, path: Path):
        self.file = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode(), flush=True)
        if self.file:
            self.file.write(line + "\n")


log = Logger()


def read_gray_uint8(path) -> np.ndarray:
    """Doc anh -> gray uint8 (an toan voi duong dan Unicode, RGBA, 16-bit)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img


def write_image(path, img) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"Khong ghi duoc {path}")
    buf.tofile(str(path))


def list_images(folder) -> list:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def even_pick(items: list, cap: int) -> list:
    """Chon toi da `cap` phan tu cach deu nhau (co dinh, tai lap duoc)."""
    if cap is None or len(items) <= cap:
        return list(items)
    idx = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def cfg_snapshot() -> dict:
    out = {}
    for k, v in vars(CFG).items():
        if k.startswith("_") or callable(v):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


# ==============================================================================
#  ROI (Otsu 3 lop)  -  tu preprocess_data.ipynb + poss_processing.ipynb
# ==============================================================================
def multi_otsu_3(values_u8: np.ndarray):
    """Otsu 3 lop, vector hoa (nhanh hon ban vong lap)."""
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    cp = np.cumsum(p)
    cm = np.cumsum(p * np.arange(256))
    mt = cm[-1]
    t1 = np.arange(256)[:, None]
    t2 = np.arange(256)[None, :]
    w0 = cp[t1]
    w1 = cp[t2] - cp[t1]
    w2 = 1.0 - cp[t2]
    valid = (t1 < t2) & (w0 > 1e-12) & (w1 > 1e-12) & (w2 > 1e-12)
    m0 = cm[t1] / np.where(w0 > 1e-12, w0, 1)
    m1 = (cm[t2] - cm[t1]) / np.where(w1 > 1e-12, w1, 1)
    m2 = (mt - cm[t2]) / np.where(w2 > 1e-12, w2, 1)
    var = w0 * (m0 - mt) ** 2 + w1 * (m1 - mt) ** 2 + w2 * (m2 - mt) ** 2
    var = np.where(valid, var, -1.0)
    a, b = np.unravel_index(np.argmax(var), var.shape)
    if var[a, b] < 0:
        return 85, 170
    return int(a), int(b)


def extract_roi_mask(gray: np.ndarray, otsu_offset=0, close_kernel=9, dilate_iters=1,
                     min_area_ratio=0.001, background_value=0, blur_ksize=11):
    """Tra ve (mask_min 0..255, nguong t_roi, mask_dac)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    fg = blur > background_value
    if not np.any(fg):
        raise ValueError("Khong tim thay vung foreground (FOV).")
    _, t2 = multi_otsu_3(blur[fg])
    t_roi = int(np.clip(t2 + otsu_offset, 0, 254))
    _, raw = cv2.threshold(blur, t_roi, 255, cv2.THRESH_BINARY)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    expanded = cv2.dilate(closed, kernel, iterations=dilate_iters)
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Khong tim thay contour.")
    min_area = max(64.0, gray.size * min_area_ratio)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area] or [max(contours, key=cv2.contourArea)]
    solid = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(solid, valid, -1, 255, thickness=-1)
    smooth = cv2.GaussianBlur(solid, (blur_ksize, blur_ksize), 0) if blur_ksize > 1 else solid
    return smooth, t_roi, solid


def extract_roi(gray: np.ndarray, dilate_iters=1, min_area_ratio=0.001):
    """Cat vung xuong ham (dung cho TRAIN_ON_ROI). Loi -> tra ve anh goc."""
    try:
        _, _, solid = extract_roi_mask(gray, 0, 9, dilate_iters, min_area_ratio, 0, 1)
        return cv2.bitwise_and(gray, gray, mask=solid)
    except Exception:
        return gray


# ==============================================================================
#  [1] DU LIEU: ToothFairy npy  ->  Glare/Normal x Train/Val/Test PNG
# ==============================================================================
def dataset_counts(root: Path) -> dict:
    return {g: {s: len(list_images(root / g / s)) for s in SPLITS} for g in GROUPS}


def dataset_is_ready(root: Path) -> bool:
    """Co du Glare/Normal x Train/Val va (neu do script tao) da build xong."""
    if (root / "build_state.json").exists() and not (root / "BUILD_COMPLETE.json").exists():
        return False
    c = dataset_counts(root)
    return all(c[g][s] > 0 for g in GROUPS for s in ("Train", "Val"))


def load_splits(cfg) -> dict:
    raw = json.load(open(Path(cfg.TOOTHFAIRY_DIR) / "splits.json", encoding="utf-8"))
    out = {s: [] for s in SPLITS}
    for k, v in raw.items():
        kl = k.lower()
        if kl.startswith("tr"):
            out["Train"] += v
        elif kl.startswith("va"):
            out["Val"] += v
        elif kl.startswith("te"):
            out["Test"] += v
        elif kl.startswith("syn"):
            if cfg.INCLUDE_SYNTHETIC_TRAIN:
                out["Train"] += v
        else:
            log(f"  (bo qua nhom '{k}' trong splits.json)")
    return out


def take_slice(vol, axis: int, z: int) -> np.ndarray:
    """Lat thu z theo truc axis (chi doc 1 lat tu memmap)."""
    idx = [slice(None)] * vol.ndim
    idx[axis] = z
    return np.asarray(vol[tuple(idx)])


def _orient(sl: np.ndarray, cfg) -> np.ndarray:
    if cfg.SLICE_TRANSPOSE:
        sl = sl.T
    if cfg.SLICE_FLIP_UD:
        sl = sl[::-1]
    if cfg.SLICE_FLIP_LR:
        sl = sl[:, ::-1]
    return np.ascontiguousarray(sl)


def _to_u8(sl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def process_volume(pid: str, split: str, vol_dir: Path, out_root: Path, cfg) -> dict:
    """Doc 1 khoi, chon lat Glare/Normal, ghi PNG. Tra ve thong ke."""
    vol = np.load(vol_dir / "data.npy", mmap_mode="r")
    if vol.ndim != 3:
        raise ValueError(f"{pid}: data.npy phai 3D, nhan shape={vol.shape}")
    axis = cfg.SLICE_AXIS if cfg.SLICE_AXIS is not None else int(np.argmin(vol.shape))
    s = max(1, cfg.SUBSAMPLE)
    sub = np.asarray(vol[::s, ::s, ::s], dtype=np.float32)
    lo, hi = (float(x) for x in np.percentile(sub, [cfg.P_LO, cfg.P_HI]))
    p1, p99 = (float(x) for x in np.percentile(sub, [1, 99]))
    fg = sub > (p1 + cfg.FG_FRAC * (p99 - p1))          # vung co mo (loai khong khi), khong bi kim loai keo lech
    if fg.sum() < 100:
        raise ValueError(f"{pid}: khoi gan nhu rong")
    fgv = sub[fg]
    bone_ref = float(np.percentile(fgv, cfg.BONE_PERCENTILE))
    med = float(np.median(fgv))
    vmax = float(vol.max())
    gap_ratio = (vmax - bone_ref) / max(bone_ref - med, 1e-6)
    has_metal = gap_ratio > cfg.VOLUME_METAL_GAP
    metal_thr = bone_ref + cfg.METAL_MARGIN * (vmax - bone_ref)
    if has_metal:   # cua so khong bi kim loai keo gian -> kim loai bao hoa (255) giong anh Glare cua CBCT_MAR
        hi = min(hi, bone_ref + 0.10 * (vmax - bone_ref))

    label = None
    if cfg.ONLY_SLICES_WITH_LABEL and (vol_dir / "gt_sparse.npy").exists():
        label = np.load(vol_dir / "gt_sparse.npy", mmap_mode="r")

    n_z = vol.shape[axis]
    metal_frac, ok_z = {}, []
    for z in range(0, n_z, cfg.SLICE_STEP):
        sl = _orient(np.asarray(take_slice(vol, axis, z), dtype=np.float32), cfg)
        if (_to_u8(sl, lo, hi) > 20).mean() < cfg.MIN_FG_FRAC:
            continue
        if label is not None and not take_slice(label, axis, z).any():
            continue
        ok_z.append(z)
        metal_frac[z] = float((sl >= metal_thr).mean()) if has_metal else 0.0

    glare_z = [z for z in ok_z if metal_frac[z] >= cfg.GLARE_MIN_FRAC]
    all_glare_z = glare_z  # khoang cach tinh theo cac lat glare da xet
    normal_z = []
    for z in ok_z:
        if metal_frac[z] <= cfg.NORMAL_MAX_FRAC:
            if not all_glare_z or min(abs(z - g) for g in all_glare_z) >= cfg.NORMAL_MIN_DIST_SLICES:
                normal_z.append(z)

    pnum = int(re.sub(r"\D", "", pid) or 0)
    rows = []
    for grp, zs in (("Glare", glare_z), ("Normal", normal_z)):
        for z in even_pick(zs, cfg.CAP_PER_VOLUME[grp][split]):
            sl = _orient(np.asarray(take_slice(vol, axis, z), dtype=np.float32), cfg)
            name = f"slice_{pnum:03d}_{z}.png"
            write_image(out_root / grp / split / name, _to_u8(sl, lo, hi))
            rows.append({"group": grp, "split": split, "patient": pid, "z": z, "file": name,
                         "metal_frac": round(metal_frac[z], 5)})
    return {
        "patient": pid, "split": split, "shape": list(vol.shape), "axis": axis,
        "has_metal": bool(has_metal), "gap_ratio": round(gap_ratio, 3), "bone_ref": bone_ref, "vmax": vmax,
        "n_glare_candidates": len(glare_z), "n_normal_candidates": len(normal_z),
        "n_glare_saved": sum(r["group"] == "Glare" for r in rows),
        "n_normal_saved": sum(r["group"] == "Normal" for r in rows),
        "rows": rows,
    }


def build_dataset_from_toothfairy(cfg, check_dir: Path) -> None:
    src = Path(cfg.TOOTHFAIRY_DIR)
    vol_root = src / "Dataset"
    out_root = Path(cfg.DATASET_DIR)
    if not (src / "splits.json").exists() or not vol_root.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay {src}\\splits.json hoac {vol_root}. Kiem tra CFG.TOOTHFAIRY_DIR.")
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "build_state.json"
    state = json.load(open(state_path, encoding="utf-8")) if state_path.exists() else {"done": {}}

    splits = load_splits(cfg)
    log("Tao dataset tu ToothFairy ->", out_root)
    log("  so benh nhan:", {k: len(v) for k, v in splits.items()})
    jobs = [(pid, sp) for sp in SPLITS for pid in splits[sp]]
    it = tqdm(jobs, desc="Build dataset", unit="ca") if tqdm else jobs
    first_preview_done = (check_dir / "orientation_preview.png").exists()
    for pid, sp in it:
        if pid in state["done"]:
            continue
        vdir = vol_root / pid
        if not (vdir / "data.npy").exists():
            log(f"  [bo qua] {pid}: khong co data.npy")
            state["done"][pid] = {"patient": pid, "split": sp, "skipped": True, "rows": []}
            continue
        try:
            if not first_preview_done:
                save_orientation_preview(vdir / "data.npy", check_dir / "orientation_preview.png", cfg)
                first_preview_done = True
            info = process_volume(pid, sp, vdir, out_root, cfg)
        except Exception as e:
            log(f"  [LOI] {pid}: {e}")
            state["done"][pid] = {"patient": pid, "split": sp, "skipped": True, "error": str(e), "rows": []}
        else:
            state["done"][pid] = info
        atomic_json(state_path, state)          # -> ngat giua chung van tiep tuc duoc

    rows = [r for v in state["done"].values() for r in v.get("rows", [])]
    pd.DataFrame(rows).to_csv(out_root / "manifest.csv", index=False)
    vinfo = [{k: v for k, v in d.items() if k != "rows"} for d in state["done"].values()]
    pd.DataFrame(vinfo).to_csv(out_root / "volumes_summary.csv", index=False)
    atomic_json(out_root / "BUILD_COMPLETE.json",
                {"finished": datetime.now().isoformat(timespec="seconds"),
                 "counts": dataset_counts(out_root), "config": cfg_snapshot()})
    log("  xong. So anh:", json.dumps(dataset_counts(out_root)))
    n_metal = sum(1 for v in vinfo if v.get("has_metal"))
    log(f"  {n_metal}/{len(vinfo)} khoi duoc xem la CO kim loai (xem volumes_summary.csv)")


def save_orientation_preview(npy_path: Path, out_path: Path, cfg) -> None:
    """Anh 3 chieu cat giua khoi: de kiem tra SLICE_AXIS co dung truc axial khong."""
    vol = np.load(npy_path, mmap_mode="r")
    lo, hi = (float(x) for x in np.percentile(np.asarray(vol[::4, ::4, ::4], dtype=np.float32),
                                               [cfg.P_LO, cfg.P_HI]))
    auto_axis = int(np.argmin(vol.shape))
    panels = []
    for ax in range(3):
        sl = np.asarray(take_slice(vol, ax, vol.shape[ax] // 2), dtype=np.float32)
        u8 = cv2.cvtColor(_to_u8(sl, lo, hi), cv2.COLOR_GRAY2BGR)
        u8 = cv2.resize(u8, (300, int(300 * u8.shape[0] / u8.shape[1])))
        tag = f"axis={ax} shape={vol.shape}" + ("  <- SELECTED" if ax == (cfg.SLICE_AXIS if cfg.SLICE_AXIS is not None else auto_axis) else "")
        cv2.putText(u8, tag, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(u8)
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
    write_image(out_path, np.hstack(panels))


def save_sample_montage(files: list, out_path: Path, n=12, size=200, title="") -> None:
    files = even_pick(files, n)
    if not files:
        return
    tiles = []
    for p in files:
        im = cv2.resize(read_gray_uint8(p), (size, size))
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, p.stem[-14:], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    cols = min(6, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    grid = np.vstack(rows)
    if title:
        cv2.putText(grid, title, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    write_image(out_path, grid)


def stage_data(cfg, run_dir: Path) -> Path:
    check_dir = run_dir / "data_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.DATASET_DIR)

    if cfg.FORCE_REBUILD and root.exists():
        if (root / "BUILD_COMPLETE.json").exists() or (root / "build_state.json").exists():
            log("FORCE_REBUILD: xoa", root)
            shutil.rmtree(root)
        else:
            log("FORCE_REBUILD bi bo qua: thu muc nay khong do script tao ra (khong xoa de an toan).")

    if cfg.RUN_BUILD_DATA and not dataset_is_ready(root):
        build_dataset_from_toothfairy(cfg, check_dir)
    elif dataset_is_ready(root):
        log("Dataset san sang:", root)
    if not dataset_is_ready(root):
        raise RuntimeError(
            f"Dataset {root} chua day du (can Glare & Normal, Train & Val co anh). "
            "Bat CFG.RUN_BUILD_DATA hoac chinh CFG.DATASET_DIR.")

    try:   # anh kiem tra chieu cat (ke ca khi dataset da build tu truoc)
        if not (check_dir / "orientation_preview.png").exists() and (root / "BUILD_COMPLETE.json").exists():
            first = next(iter(sorted((Path(cfg.TOOTHFAIRY_DIR) / "Dataset").glob("P*/data.npy"))), None)
            if first is not None:
                save_orientation_preview(first, check_dir / "orientation_preview.png", cfg)
    except Exception as e:
        log("  (khong xuat duoc orientation_preview:", e, ")")
    counts = dataset_counts(root)
    log("So anh:", json.dumps(counts))
    (check_dir / "dataset_summary.txt").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    for g in GROUPS:
        save_sample_montage(list_images(root / g / "Train"), check_dir / f"samples_{g}_train.png", title=g)
    # vi du ROI
    try:
        ex = even_pick(list_images(root / "Glare" / "Train"), 4)
        tiles = []
        for p in ex:
            g = read_gray_uint8(p)
            r = extract_roi(g)
            a = cv2.resize(g, (220, 220)); b = cv2.resize(r, (220, 220))
            tiles.append(np.vstack([a, b]))
        if tiles:
            write_image(check_dir / "roi_examples.png", np.hstack(tiles))
    except Exception as e:
        log("  (khong xuat duoc roi_examples:", e, ")")
    if min(counts["Glare"]["Train"], counts["Normal"]["Train"]) < 50:
        log("  [CANH BAO] Train co qua it anh (<50) o 1 nhom -> xem lai tham so phan loai Glare/Normal.")
    return root


# ==============================================================================
#  DATASET / DATALOADER
# ==============================================================================
class SliceDataset(Dataset):
    def __init__(self, folder, size, train, rot_deg, scale_aug, use_roi, pseudo_dir=None):
        self.paths = list_images(folder)
        if not self.paths:
            raise RuntimeError(f"Khong co anh trong {folder}")
        self.size, self.train, self.rot, self.scl, self.roi = size, train, rot_deg, scale_aug, use_roi
        # V8.3: pseudo_dir (tuy chon) - thu muc pseudo-target da inpaint vung halo (xem
        # build_pseudo_clean_targets), CUNG TEN FILE voi anh nguon. Khi dat, __getitem__ tra ve THEM
        # 1 tensor pseudo-target, ap dung DUNG CUNG 1 phep bien doi hinh hoc (flip/xoay/scale) nhu anh
        # nguon de 2 tensor luon khop pixel-to-pixel (dung cho loss_pseudo trong vong lap train).
        # (Gia dinh TRAIN_ON_ROI=False - neu bat ROI, crop Otsu co the lech nhe giua 2 anh vi anh
        # pseudo da bi inpaint khac di chut it; khong anh huong hanh vi mac dinh.)
        self.pseudo_dir = Path(pseudo_dir) if pseudo_dir is not None else None

    def __len__(self):
        return len(self.paths)

    def _prep(self, img, M=None, flip=False):
        if self.roi:
            img = extract_roi(img)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA if max(img.shape) > self.size else cv2.INTER_LINEAR)
        if flip:
            img = img[:, ::-1]
        if M is not None:
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (self.size, self.size),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return torch.from_numpy(np.ascontiguousarray(img)).float().div_(127.5).sub_(1.0).unsqueeze(0)

    def __getitem__(self, i):
        img = read_gray_uint8(self.paths[i])
        flip, M = False, None
        if self.train:
            flip = random.random() < 0.5
            ang = random.uniform(-self.rot, self.rot)
            sc = 1.0 + random.uniform(-self.scl, self.scl)
            if abs(ang) > 1e-3 or abs(sc - 1) > 1e-3:
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, sc)
        t = self._prep(img, M, flip)
        if self.pseudo_dir is None:
            return t
        pseudo_path = self.pseudo_dir / self.paths[i].name
        # an toan: neu thieu file pseudo (vd cache cu chua kip tao) -> dung anh goc, khong crash.
        p_img = read_gray_uint8(pseudo_path) if pseudo_path.exists() else img
        t_pseudo = self._prep(p_img, M, flip)
        return t, t_pseudo


def make_loader(folder, cfg, train, batch=None, shuffle=None, max_items=None, pseudo_dir=None):
    ds = SliceDataset(folder, cfg.IMG_SIZE, train, cfg.AUG_ROTATE_DEG, cfg.AUG_SCALE, cfg.TRAIN_ON_ROI, pseudo_dir=pseudo_dir)
    if max_items and len(ds) > max_items:
        ds.paths = even_pick(ds.paths, max_items)
    bs = batch or cfg.BATCH_SIZE
    shuffle = train if shuffle is None else shuffle
    nw = cfg.NUM_WORKERS if train else 0
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      drop_last=bool(train and len(ds) >= 2 * bs),
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0)


def pseudo_clean_mask_np(orig_u8: "np.ndarray", cfg) -> "np.ndarray":
    """V8.3: mask nhi phan vung halo (GIONG HET halo_target dung trong loss_pseudo luc train, xem
    CFG.HALO_THRESH_U8/HALO_DILATE_PX) - tinh bang numpy/cv2 de dung 1 lan luc precompute pseudo-target."""
    _, halo_bin = cv2.threshold(orig_u8, cfg.HALO_THRESH_U8, 255, cv2.THRESH_BINARY)
    k = 2 * cfg.HALO_DILATE_PX + 1
    return cv2.dilate(halo_bin, np.ones((k, k), np.uint8), iterations=1)


def build_pseudo_clean_image(orig_u8: "np.ndarray", cfg) -> "np.ndarray":
    """V8.3: 'anh sach gia lap' cho loss_pseudo, tao bang cv2.inpaint - VA vung halo bang thong tin
    lan truyen tu VIEN NGOAI vao trong (nhu classical inpaint, da kiem chung halo_reduction~0.27 tren
    du lieu that - cao hon nhieu so voi box-blur don gian tu CHINH anh nhiem: box-blur trung binh cac
    pixel TRONG cung vung sang, gia tri trung binh van con cao; inpaint lay gia tri tu vung THAT SU
    sach ben ngoai vung halo, nen keo duoc gia tri xuong dung muc "sach" hon)."""
    mask = pseudo_clean_mask_np(orig_u8, cfg)
    if not mask.any():
        return orig_u8.copy()
    return cv2.inpaint(orig_u8, mask, cfg.CLASSICAL_INPAINT_RADIUS, cv2.INPAINT_TELEA)


def build_pseudo_clean_targets(cfg, glare_train_dir: Path) -> Path:
    """V8.3: precompute (1 lan, CACHE lai) pseudo-target cho MOI anh Glare/Train - goi 1 lan dau
    stage_train() truoc khi tao train_g loader (xem stage_train). Cache tai <DATASET_DIR>/
    _pseudo_clean_train/ kem _meta.json (thong so halo/inpaint) - neu thong so doi (vd tune lai
    HALO_THRESH_U8) se TU DONG xoa cache cu va tao lai, tranh dung nham pseudo-target cu/sai."""
    src_paths = list_images(glare_train_dir)
    out_dir = glare_train_dir.parent.parent / "_pseudo_clean_train"
    meta_path = out_dir / "_meta.json"
    meta = {"halo_thresh": cfg.HALO_THRESH_U8, "halo_dilate": cfg.HALO_DILATE_PX,
            "inpaint_radius": cfg.CLASSICAL_INPAINT_RADIUS, "count": len(src_paths)}
    stale = True
    if meta_path.exists():
        try:
            stale = (json.loads(meta_path.read_text(encoding="utf-8")) != meta)
        except Exception:
            stale = True
    if stale:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"  Dang tao pseudo-target (inpaint vung halo) cho loss_pseudo: {len(src_paths)} anh "
            f"(cache moi hoac tham so halo da doi) -> {out_dir}")
        it = tqdm(src_paths, desc="pseudo-target", unit="img") if tqdm else src_paths
        for p in it:
            img = read_gray_uint8(p)
            clean = build_pseudo_clean_image(img, cfg)
            cv2.imwrite(str(out_dir / p.name), clean)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        log(f"  Xong pseudo-target: {len(src_paths)} anh.")
    else:
        log(f"  Pseudo-target cho loss_pseudo da co san ({len(src_paths)} anh, cache hop le) -> bo qua tao lai.")
    return out_dir


def infinite(loader):
    while True:
        for b in loader:
            yield b


# ==============================================================================
#  MO HINH  (giu kien truc notebook: UNet 5 tang + 2 nhanh img/mask ; PatchGAN 3 lop)
# ==============================================================================
class UNetGenerator10(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.enc1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1);   self.norm_e2 = nn.InstanceNorm2d(128)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1);  self.norm_e3 = nn.InstanceNorm2d(256)
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1);  self.norm_e4 = nn.InstanceNorm2d(512)
        self.enc5 = nn.Conv2d(512, 512, 4, 2, 1);  self.norm_e5 = nn.InstanceNorm2d(512)
        # V5: doi ConvTranspose2d -> Upsample(nearest) + Conv2d. ConvTranspose2d (kernel 4, stride 2)
        # la nguyen nhan kinh dien gay vet caro/luoi o vuong (checkerboard artifact) trong |diff| - overlap
        # khong deu giua cac buoc nhay cua kernel. Upsample+Conv khong co van de nay (kich thuoc dau ra giu nguyen).
        def _up(cin, cout):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, 1, 1))
        self.dec1 = _up(512, 512);   self.norm_d1 = nn.InstanceNorm2d(512)
        self.dec2 = _up(1024, 256);  self.norm_d2 = nn.InstanceNorm2d(256)
        self.dec3 = _up(512, 128);   self.norm_d3 = nn.InstanceNorm2d(128)
        self.dec4 = _up(256, 64);    self.norm_d4 = nn.InstanceNorm2d(64)
        self.dec5 = _up(128, 64);    self.norm_d5 = nn.InstanceNorm2d(64)
        # Phuong phap 2 / Giai doan 2: tach nhanh cuoi -> anh + attention mask
        self.img_branch = nn.Sequential(nn.Conv2d(64, in_channels, 3, 1, 1), nn.Tanh())
        self.mask_branch = nn.Sequential(nn.Conv2d(64, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.norm_e2(self.enc2(e1)))
        e3 = self.relu(self.norm_e3(self.enc3(e2)))
        e4 = self.relu(self.norm_e4(self.enc4(e3)))
        e5 = self.relu(self.norm_e5(self.enc5(e4)))
        d1 = self.relu(self.norm_d1(self.dec1(e5)))
        d2 = self.relu(self.norm_d2(self.dec2(torch.cat([d1, e4], 1))))
        d3 = self.relu(self.norm_d3(self.dec3(torch.cat([d2, e3], 1))))
        d4 = self.relu(self.norm_d4(self.dec4(torch.cat([d3, e2], 1))))
        d5 = self.relu(self.norm_d5(self.dec5(torch.cat([d4, e1], 1))))
        img = self.img_branch(d5)
        m = self.mask_branch(d5)
        return m * img + (1 - m) * x, m       # tron theo attention mask


class Discriminator3(nn.Module):
    """PatchGAN; xuat LOGIT (khong sigmoid) -> dung duoc cho lsgan / bce-with-logits."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 4, 1, 1))

    def forward(self, x):
        return self.model(x)


def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ImagePool:
    def __init__(self, size):
        self.size, self.buf = size, []

    def query(self, imgs):
        if self.size <= 0:
            return imgs
        out = []
        for im in imgs.detach():
            im = im.unsqueeze(0)
            if len(self.buf) < self.size:
                self.buf.append(im); out.append(im)
            elif random.random() < 0.5:
                i = random.randrange(self.size)
                tmp = self.buf[i].clone(); self.buf[i] = im; out.append(tmp)
            else:
                out.append(im)
        return torch.cat(out, 0)


def total_variation_loss(img, mask=None):
    """V9.1: them tham so `mask` tuy chon (0..1, cung kich thuoc voi img, vd halo_target). Neu co ->
    CHI tinh TV loss o CANH nam HOAN TOAN trong mask (lay min cua 2 pixel ke nhau lam trong so cua
    canh do, tranh dem canh o rin mask), va chuan hoa theo SO CANH trong mask (khong phai H*W toan
    anh). Ly do them: truoc day ham nay luon tinh TREN TOAN BO anh sinh ra - phat ca vung rang/xuong
    BINH THUONG khong lien quan gi den nhieu kim loai, khien generator hoc cach lam mo/mat chi tiet
    TOAN ANH de giam loss nay -> day la 1 nguyen nhan chinh gay "anh sau GAN bi mo qua" (nguoi dung
    phan anh, xem qualitative_*.png cac run V8.3/V8.3b: raw GAN output mat texture ca vung xuong ham
    khong he bi glare). mask=None -> hanh vi CU (toan anh), giu tuong thich nguoc."""
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2)
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2)
    if mask is None:
        return (tv_h.sum() + tv_w.sum()) / (b * c * h * w)
    m_h = torch.minimum(mask[:, :, 1:, :], mask[:, :, :-1, :])
    m_w = torch.minimum(mask[:, :, :, 1:], mask[:, :, :, :-1])
    denom = (m_h.sum() + m_w.sum()).clamp_min(1.0)
    return ((tv_h * m_h).sum() + (tv_w * m_w).sum()) / denom


def gan_criterion(kind, real_label=1.0):
    """V4: real_label < 1.0 -> label smoothing 1 chieu (nhan 'fake' luon la 0.0, khong doi)."""
    if kind == "bce":
        bce = nn.BCEWithLogitsLoss()
        return lambda logit, real: bce(logit, torch.full_like(logit, real_label if real else 0.0))
    mse = nn.MSELoss()
    return lambda logit, real: mse(logit, torch.full_like(logit, real_label if real else 0.0))


def ema_update(ema_model, model, decay):
    """V4: cap nhat truong so EMA cua ema_model tien ve model, muot dan qua nhieu buoc (khong dao ngat quang)."""
    with torch.no_grad():
        for pe, p in zip(ema_model.parameters(), model.parameters()):
            pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


# ==============================================================================
#  KIEM DINH (VALIDATION)
# ==============================================================================
def box_blur(x: "torch.Tensor", ksize: int) -> "torch.Tensor":
    """V8.3: box-blur thuan torch (avg_pool2d + reflect pad), chay thang tren GPU, khong round-trip
    qua cv2/numpy -> du nhanh de goi MOI batch trong vong lap train (dung tao 'pseudo-target' cho
    loss_pseudo). ksize le; ksize<=1 -> tra ve nguyen ban (khong lam mo)."""
    if ksize <= 1:
        return x
    pad = ksize // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), ksize, stride=1)


def t2u8(t):
    """(B,1,H,W) [-1,1] -> list[np.uint8 HxW]"""
    a = ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5 + 0.5).numpy().astype(np.uint8)
    return [x[0] for x in a]


def metal_region(u8, thr):
    core = (u8 >= thr).astype(np.uint8) * 255
    return cv2.dilate(core, np.ones((7, 7), np.uint8), iterations=3) > 0


def _psnr_safe(a, b):
    """PSNR tinh tay, tranh RuntimeWarning divide-by-zero cua skimage khi mse=0 (anh giong het nhau)."""
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse))


def edge_corr(a, b, mask=None):
    """V7: he so tuong quan Pearson giua gradient Sobel (bien/canh rang) cua 2 anh, tinh trong `mask`
    neu co (thuong la vung KHONG kim loai). Danh gia "edge-preservation" - MAR ly tuong phai giam
    nhieu kim loai nhung KHONG lam mo/lech duong vien giai phau that (rang, xuong ham) o vung con lai.
    1.0 = giu bien hoan hao, 0 = khong con lien he, am = bien bi dao nguoc/hong."""
    ga = cv2.Sobel(a.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
    gb = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
    if mask is not None:
        ga = ga[mask]; gb = gb[mask]
    else:
        ga = ga.ravel(); gb = gb.ravel()
    if ga.size < 32 or ga.std() < 1e-6 or gb.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(ga, gb)[0, 1])


@torch.no_grad()
def evaluate(gen_G, gen_F, val_glare, val_normal, cfg, dev):
    gen_G.eval(); gen_F.eval()
    # V6: them pres_mae/pres_rmse (loi tuyet doi/binh phuong vung khong kim loai) va id_mae
    # - cac chi so MAE/RMSE thuong duoc yeu cau trong paper MAR ben canh SSIM/PSNR.
    # V7: them pres_edge_corr (tuong quan gradient Sobel vung khong kim loai - do "edge-preservation").
    # V8: them halo_red (% giam sang vung "streak/halo" - sang vua, KHONG tinh vung core bao hoa cung) -
    # day la chi so KHONG THE "lach" bang cach gan nhu khong doi anh, thay cho viec chi dua vao
    # sat_reduction (de dat diem cao chi bang cach dim nhe vai pixel sat nguong, xem ghi chu dau file).
    (cyc, id_ssim, id_psnr, id_mae, pres_ssim, pres_psnr, pres_mae, pres_rmse, pres_edge,
     sat_red, sat_int_red, halo_red, att_g, att_n) = ([] for _ in range(14))
    thr = cfg.METAL_THRESH_U8
    halo_lo = cfg.HALO_THRESH_U8

    for rg in val_glare:
        rg = rg.to(dev)
        fake_n, m = gen_G(rg)
        rec_g, _ = gen_F(fake_n)
        cyc.append(F.l1_loss(rec_g, rg).item())
        for a, b, mm in zip(t2u8(rg), t2u8(fake_n), m.cpu().numpy()):
            att_g.append(float(mm.mean()))
            reg = metal_region(a, thr)
            keep = ~reg
            # V8.1 FIX (phat hien tu log chay thuc te): "sat_red" (ty le SO PIXEL vuot nguong 240
            # truoc/sau) rat NHIEU (std~0.25 tren toan run, nhay 0.15<->0.99 giua 2 epoch lien tiep)
            # vi mau so sb (so pixel bao hoa trong 1 anh) thuong RAT NHO -> ty le luong tu hoa manh.
            # Day chinh la nguyen nhan "val nhay loan" nguoi dung thay - va vi no la 1 thanh phan
            # cua val_score (trong so SCORE_WEIGHTS['sat']), no lam ca diem tong hop nhay loan theo,
            # khien "best" cu (chon luc con o mot epoch may man/gan-nhu-identity) rat kho vuot qua
            # du model dang THUC SU hoc tot hon (xem val_halo_reduction). GIU LAI "val_sat_reduction"
            # (cu) chi de doi chieu/xem, nhung DIEM SO tu nay dung "val_sat_intensity_reduction" moi -
            # do % giam CUONG DO TRUNG BINH trong vung core (da gian no, mau lon) truoc/sau, cung cong
            # thuc lien tuc nhu val_halo_reduction (da chung minh on dinh hon rat nhieu, std~0.04).
            sb = float((a >= thr).mean()); sa = float((b >= thr).mean())
            if sb > 0:
                sat_red.append(max(0.0, 1.0 - sa / sb))
            if reg.sum() > 64:
                ma_core = float(a[reg].mean()); mb_core = float(b[reg].mean())
                if ma_core > 1e-6:
                    sat_int_red.append(max(0.0, 1.0 - mb_core / ma_core))
            halo = (a >= halo_lo) & (a < thr)          # V8: vung streak/halo - sang vua, KHONG bao hoa
            if halo.sum() > 64:
                ma = float(a[halo].mean()); mb = float(b[halo].mean())
                if ma > 1e-6:
                    halo_red.append(max(0.0, 1.0 - mb / ma))
            if keep.sum() > 64:
                _, smap = ssim_fn(a, b, data_range=255, full=True)
                pres_ssim.append(float(smap[keep].mean()))
                diff = (a.astype(np.float32) - b.astype(np.float32))[keep]
                mse = float((diff ** 2).mean())
                pres_psnr.append(99.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse))
                pres_mae.append(float(np.abs(diff).mean()))
                pres_rmse.append(float(np.sqrt(mse)))
                ec = edge_corr(a, b, keep)
                if not np.isnan(ec):
                    pres_edge.append(ec)

    for rn in val_normal:
        rn = rn.to(dev)
        idn, m = gen_G(rn)
        fake_g, _ = gen_F(rn)
        rec_n, _ = gen_G(fake_g)
        cyc.append(F.l1_loss(rec_n, rn).item())
        for a, b, mm in zip(t2u8(rn), t2u8(idn), m.cpu().numpy()):
            att_n.append(float(mm.mean()))
            id_ssim.append(float(ssim_fn(a, b, data_range=255)))
            id_psnr.append(_psnr_safe(a, b))
            id_mae.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()))

    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    out = {
        "val_cycle": mean(cyc), "val_id_ssim": mean(id_ssim), "val_id_psnr": mean(id_psnr), "val_id_mae": mean(id_mae),
        "val_pres_ssim": mean(pres_ssim), "val_pres_psnr": mean(pres_psnr),
        "val_pres_mae": mean(pres_mae), "val_pres_rmse": mean(pres_rmse), "val_pres_edge_corr": mean(pres_edge),
        "val_sat_reduction": mean(sat_red),                       # cu - giu de doi chieu, KHONG dung tinh score
        "val_sat_intensity_reduction": mean(sat_int_red),         # V8.1: moi - on dinh hon, dung tinh score
        "val_halo_reduction": mean(halo_red),
        "val_att_glare": mean(att_g), "val_att_normal": mean(att_n),
    }
    w = cfg.SCORE_WEIGHTS
    parts = [(w["sat"], out["val_sat_intensity_reduction"]), (w["pres"], out["val_pres_ssim"]), (w["id"], out["val_id_ssim"]),
             (w.get("halo", 0.0), out["val_halo_reduction"])]
    out["val_score"] = float(sum(a * (0.0 if np.isnan(b) else b) for a, b in parts))
    return out


# ==============================================================================
#  ANH TIEN TRINH + DO THI
# ==============================================================================
def _label(img_bgr, text):
    cv2.putText(img_bgr, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


@torch.no_grad()
def save_progress_image(gen_G, batch, tags, epoch, path: Path, dev):
    gen_G.eval()
    x = batch.to(dev)
    out, m = gen_G(x)
    rows = []
    for i, (o, c) in enumerate(zip(t2u8(x), t2u8(out))):
        mk = (m[i, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        heat = cv2.applyColorMap(mk, cv2.COLORMAP_JET)
        diff = np.clip(cv2.absdiff(o, c).astype(np.float32) * 4, 0, 255).astype(np.uint8)
        cols = [_label(cv2.cvtColor(o, cv2.COLOR_GRAY2BGR), f"{tags[i]} input"),
                _label(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR), f"output ep{epoch}"),
                _label(heat, "attention mask"),
                _label(cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), "|diff| x4")]
        rows.append(np.hstack(cols))
    write_image(path, np.vstack(rows))


def _mark_metric_fix(ax, df: pd.DataFrame, col: str = "val_sat_intensity_reduction") -> None:
    """V8.1: ve 1 duong doc net dut tai epoch DAU TIEN co gia tri o cot `col` (mac dinh
    val_sat_intensity_reduction) - danh dau ranh gioi "truoc/sau khi sua cong thuc diem" tren
    cac bieu do Validation. Neu khong co, day KHONG phai 1 su kien huan luyen bat thuong (mo hinh
    dot nhien dao dong) ma chi la tu epoch do, val_score bat dau dung cong thuc moi (it nhieu hon,
    xem doc dau file). Bo qua neu ca history deu truoc hoac deu sau moc sua (khong co gi de danh dau)."""
    if col not in df or df[col].notna().sum() == 0:
        return
    valid_epochs = df.loc[df[col].notna(), "epoch"]
    if len(valid_epochs) == len(df):
        return   # toan bo history da o "sau khi sua" -> khong co ranh gioi de ve
    ep0 = int(valid_epochs.min())
    ax.axvline(ep0, color="crimson", linestyle="--", linewidth=1, alpha=0.7, zorder=0)
    ax.text(ep0, 1.0, f" sua cong thuc diem (ep {ep0})", color="crimson", fontsize=6.5,
            va="top", ha="left", transform=ax.get_xaxis_transform())


def plot_history(history: list, path: Path):
    if not history:
        return
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    def draw(a, cols, title):
        for c in cols:
            if c in df:
                a.plot(df["epoch"], df[c], label=c)
        a.set_title(title); a.grid(alpha=.3); a.legend(fontsize=7)
    draw(ax[0, 0], ["G_total", "G_adv"], "Generator")
    draw(ax[0, 1], ["D_X", "D_Y"], "Discriminator")
    draw(ax[0, 2], ["cycle", "identity"], "Cycle / Identity")
    draw(ax[1, 0], ["tv", "mask_loss", "pseudo_loss"], "TV / Mask loss")
    draw(ax[1, 1], ["val_sat_reduction", "val_sat_intensity_reduction", "val_halo_reduction", "val_pres_ssim", "val_id_ssim", "val_score"], "Validation (higher=better)")
    _mark_metric_fix(ax[1, 1], df)
    draw(ax[1, 2], ["val_att_glare", "val_att_normal", "val_cycle"], "Attention area / val cycle")
    for a in ax.ravel():
        a.set_xlabel("epoch")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ==============================================================================
#  V6: BIEU DO + SO LIEU RIENG, DO PHAN GIAI CAO - DE DAN VAO PAPER/LUAN VAN
# ==============================================================================
def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def plot_metric_group(df: pd.DataFrame, cols, title, path: Path, dpi=300, ylabel=None, mark_score_fix=False):
    cols = [c for c in cols if c in df]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in cols:
        ax.plot(df["epoch"], df[c], marker="o", markersize=3, linewidth=1.3, label=c)
    if mark_score_fix:
        _mark_metric_fix(ax, df)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or title)
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def export_training_figures(history: list, run_dir: Path, cfg) -> None:
    """V6: xuat rieng tung nhom bieu do o do phan giai cao (CFG.PAPER_DPI) thay vi 1 anh loss_curves.png
    gop chung kho doc khi in vao paper/luan van. Cung xuat bang thong ke (min/max/mean/std) tung cot."""
    if not history or not cfg.EXPORT_PAPER_ASSETS:
        return
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(history)
    dpi = cfg.PAPER_DPI
    plot_metric_group(df, ["G_total", "G_adv"], "Generator loss", fig_dir / "fig_loss_generator.png", dpi, "Loss")
    plot_metric_group(df, ["D_X", "D_Y"], "Discriminator loss", fig_dir / "fig_loss_discriminator.png", dpi, "Loss")
    plot_metric_group(df, ["cycle", "identity"], "Cycle-consistency & Identity loss",
                      fig_dir / "fig_cycle_identity.png", dpi, "Loss")
    plot_metric_group(df, ["tv", "mask_loss", "pseudo_loss"], "TV loss & Attention-mask supervision loss",
                      fig_dir / "fig_tv_mask.png", dpi, "Loss")
    plot_metric_group(df, ["val_sat_reduction", "val_sat_intensity_reduction", "val_halo_reduction", "val_pres_ssim", "val_id_ssim", "val_score"],
                      "Validation metrics (higher=better)", fig_dir / "fig_validation_scores.png", dpi, "Value", mark_score_fix=True)
    plot_metric_group(df, ["val_pres_psnr", "val_id_psnr"], "Validation PSNR",
                      fig_dir / "fig_validation_psnr.png", dpi, "PSNR (dB)")
    plot_metric_group(df, ["val_pres_mae", "val_pres_rmse", "val_id_mae"], "Validation MAE / RMSE",
                      fig_dir / "fig_validation_mae_rmse.png", dpi, "Gray level (0-255)")
    # V7: duong cong edge-preservation (tuong quan gradient Sobel vung khong kim loai) qua cac epoch.
    plot_metric_group(df, ["val_pres_edge_corr"], "Edge-preservation (non-metal region)",
                      fig_dir / "fig_validation_edge_corr.png", dpi, "Correlation (higher=better)")
    # V8: rieng duong halo-reduction - chi so chinh de kiem tra generator co THUC SU giam duoc vung
    # streak/halo hay khong (khong chi vung core bao hoa cung, xem ghi chu dau file).
    plot_metric_group(df, ["val_halo_reduction"], "Halo/streak brightness reduction (non-saturated)",
                      fig_dir / "fig_validation_halo_reduction.png", dpi, "Relative reduction (higher=better)")
    plot_metric_group(df, ["val_att_glare", "val_att_normal"], "Mean attention-mask area",
                      fig_dir / "fig_attention_area.png", dpi, "Mean mask value")
    plot_metric_group(df, ["lr"], "Learning rate schedule", fig_dir / "fig_learning_rate.png", dpi, "LR")
    plot_metric_group(df, ["epoch_sec"], "Epoch time", fig_dir / "fig_epoch_time.png", dpi, "Seconds")
    df.describe().T.to_csv(run_dir / "history_summary_stats.csv")
    # V7: 1 bang tong hop (dashboard) do phan giai cao, gop nhieu nhom chi so vao 1 hinh duy nhat -
    # tien dan nhanh vao paper khi khong du cho cho tung hinh rieng.
    try:
        groups = [
            (["G_total", "G_adv"], "Generator loss"),
            (["D_X", "D_Y"], "Discriminator loss"),
            (["cycle", "identity"], "Cycle / Identity loss"),
            (["tv", "mask_loss", "pseudo_loss"], "TV / Mask loss"),
            (["val_sat_reduction", "val_sat_intensity_reduction", "val_pres_ssim", "val_id_ssim", "val_score"], "Validation (higher=better)"),
            (["val_halo_reduction"], "Halo/streak reduction (key metric)"),
            (["val_pres_psnr", "val_id_psnr"], "Validation PSNR"),
            (["val_pres_mae", "val_pres_rmse"], "Validation MAE/RMSE"),
            (["val_pres_edge_corr"], "Edge-preservation"),
            (["val_att_glare", "val_att_normal"], "Attention mask area"),
            (["lr"], "Learning rate"),
            (["epoch_sec"], "Epoch time (s)"),
        ]
        fig, axes = plt.subplots(4, 3, figsize=(15, 14.5))
        for ax, (cols, title) in zip(axes.ravel(), groups):
            cols = [c for c in cols if c in df]
            for c in cols:
                ax.plot(df["epoch"], df[c], marker="o", markersize=2, linewidth=1.1, label=c)
            if title == "Validation (higher=better)":
                _mark_metric_fix(ax, df)
            ax.set_title(title, fontsize=9); ax.set_xlabel("Epoch", fontsize=8); ax.grid(alpha=.3)
            ax.legend(fontsize=6.5)
        for ax in axes.ravel()[len(groups):]:
            ax.axis("off")
        fig.suptitle("Training overview (V8)", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_overview_dashboard.png", dpi=dpi)
        plt.close(fig)
    except Exception as e:
        log("  (khong xuat duoc fig_overview_dashboard:", e, ")")
    log("  Da xuat", len(list(fig_dir.glob("fig_*.png"))), "bieu do huan luyen (", dpi, "dpi) vao", fig_dir)


def export_model_info(gen_G, gen_F, disc_X, disc_Y, cfg, run_dir: Path, dev, extra: dict) -> None:
    """V6: luu so tham so mo hinh + kien truc dang text + cac so lieu tong hop (best epoch, thoi gian train,...)
    - can cho phan 'Methodology'/'Experimental setup' cua paper."""
    if not cfg.EXPORT_PAPER_ASSETS:
        return
    info = {
        "params_generator_G": count_params(gen_G),
        "params_generator_F": count_params(gen_F),
        "params_discriminator_X": count_params(disc_X),
        "params_discriminator_Y": count_params(disc_Y),
        "params_total": count_params(gen_G) + count_params(gen_F) + count_params(disc_X) + count_params(disc_Y),
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    info.update(extra)
    atomic_json(run_dir / "model_info.json", info)
    (run_dir / "model_architecture.txt").write_text(
        "===== Generator G / F (UNetGenerator10, kien truc giong nhau) =====\n" + str(gen_G) +
        f"\n\nSo tham so moi generator: {count_params(gen_G):,}\n\n" +
        "===== Discriminator DX / DY (Discriminator3 - PatchGAN, kien truc giong nhau) =====\n" + str(disc_X) +
        f"\n\nSo tham so moi discriminator: {count_params(disc_X):,}\n\n" +
        f"Tong so tham so (G+F+DX+DY): {info['params_total']:,}\n",
        encoding="utf-8")
    log("  Da xuat model_info.json + model_architecture.txt (tong", f"{info['params_total']:,}", "tham so)")


def export_qualitative_figure(cfg, root: Path, run_dir: Path, split: str) -> None:
    """V6: lap rap vai vi du thanh 1 hinh dinh dang 'paper' (matplotlib, nhan (a)(b)(c)..), thay cho
    cv2.putText mau xanh - de dan thang vao bai bao/luan van. Doc lai cac anh da luu trong results/."""
    if not cfg.EXPORT_PAPER_ASSETS:
        return
    out_root = run_dir / "results"
    files = list_images(root / "Glare" / split)
    if not files:
        return
    final_dir = "final_smooth" if (out_root / "final_smooth" / split).exists() else "final"
    ex = even_pick(files, cfg.PAPER_QUALITATIVE_N)
    cols = [("Input (Glare)", root / "Glare" / split),
            ("Raw GAN output", out_root / "gan_raw" / split),
            ("Attention mask", out_root / "attention_mask" / split),
            ("Post-processed", out_root / final_dir / split)]
    if cfg.HYBRID_COMPOSE and (out_root / "final_hybrid" / split).exists():
        # V7: them cot ket qua hybrid (rang=GAN, nen=blur co dien) vao anh minh hoa dinh dang paper
        cols.append(("Hybrid (GAN teeth + blurred bg)", out_root / "final_hybrid" / split))
    n = len(ex)
    if n == 0:
        return
    fig, axes = plt.subplots(n, len(cols), figsize=(3.0 * len(cols), 3.0 * n), squeeze=False)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, p in enumerate(ex):
        for j, (title, folder) in enumerate(cols):
            ax = axes[i, j]
            fp = folder / p.name
            if fp.exists():
                im = read_gray_uint8(fp)
                ax.imshow(im, cmap="jet" if "mask" in title.lower() else "gray")
            else:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"({letters[i % 26]})", fontsize=10, rotation=0, labelpad=16)
    fig.suptitle(f"Qualitative results - {split}", fontsize=12)
    fig.tight_layout()
    fig_dir = run_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / f"qualitative_{split}.png", dpi=cfg.PAPER_DPI)
    plt.close(fig)
    log(f"  Da xuat anh minh hoa dinh dang paper: figures/qualitative_{split}.png ({n} vi du)")


def export_distribution_figures(df_new: pd.DataFrame, run_dir: Path, cfg) -> None:
    """V6: boxplot phan phoi (khong chi trung binh) cua cac chi so quan trong theo tung split - paper thuong
    yeu cau the hien do bien thien, khong chi 1 con so trung binh."""
    if not cfg.EXPORT_PAPER_ASSETS or df_new.empty:
        return
    fig_dir = run_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    metrics_to_plot = [c for c in ["ssim_preservation", "psnr_preservation", "mae_preservation",
                                    "edge_corr_preservation", "halo_reduction", "ssim_orig_vs_final",
                                    "psnr_orig_vs_final", "inference_ms", "ssim_preservation_hybrid",
                                    "psnr_preservation_hybrid", "mae_preservation_hybrid",
                                    "edge_corr_preservation_hybrid", "halo_reduction_hybrid"] if c in df_new]
    splits_here = list(df_new["split"].unique())
    for m in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        data = [df_new[df_new["split"] == s][m].dropna().values for s in splits_here]
        try:
            ax.boxplot(data, tick_labels=splits_here, showmeans=True)   # matplotlib >= 3.9
        except TypeError:
            ax.boxplot(data, labels=splits_here, showmeans=True)        # matplotlib cu hon
        ax.set_ylabel(m); ax.set_title(f"{m} distribution by split"); ax.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(fig_dir / f"box_{m}.png", dpi=cfg.PAPER_DPI)
        plt.close(fig)
    if metrics_to_plot:
        log("  Da xuat", len(metrics_to_plot), "bieu do phan phoi (boxplot) vao", fig_dir)


def export_method_comparison_figure(df_new: pd.DataFrame, run_dir: Path, cfg) -> None:
    """V7: bieu do cot nhom (grouped bar, co error bar = std) so sanh 2 phien ban dau ra - GAN da hau
    xu ly (final/final_smooth, toan anh qua GAN) so voi HYBRID (rang=GAN, nen=blur co dien) - tren
    tung split (Train/Val/Test tuy CFG.POST_SPLITS) va tung chi so quan trong. Giup thay ro hybrid co
    danh doi gi (thuong: giam nguy co 'bia' chi tiet o nen, doi lai co the it thay doi vung rang)."""
    if not cfg.EXPORT_PAPER_ASSETS or not cfg.HYBRID_COMPOSE or df_new.empty:
        return
    if "ssim_preservation_hybrid" not in df_new:
        return
    fig_dir = run_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    pairs = [
        ("ssim_preservation", "ssim_preservation_hybrid", "SSIM (non-metal region)"),
        ("psnr_preservation", "psnr_preservation_hybrid", "PSNR (non-metal region, dB)"),
        ("mae_preservation", "mae_preservation_hybrid", "MAE (non-metal region, gray level)"),
        ("edge_corr_preservation", "edge_corr_preservation_hybrid", "Edge-preservation (gradient corr.)"),
        ("halo_reduction", "halo_reduction_hybrid", "Halo/streak reduction (key metric)"),
    ]
    splits_here = list(df_new["split"].unique())
    for final_col, hybrid_col, title in pairs:
        if final_col not in df_new or hybrid_col not in df_new:
            continue
        means_f = [df_new[df_new["split"] == s][final_col].mean() for s in splits_here]
        std_f = [df_new[df_new["split"] == s][final_col].std() for s in splits_here]
        means_h = [df_new[df_new["split"] == s][hybrid_col].mean() for s in splits_here]
        std_h = [df_new[df_new["split"] == s][hybrid_col].std() for s in splits_here]
        x = np.arange(len(splits_here)); w = 0.32
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.bar(x - w / 2, means_f, w, yerr=std_f, capsize=3, label="GAN (post-processed)")
        ax.bar(x + w / 2, means_h, w, yerr=std_h, capsize=3, label="Hybrid (teeth=GAN, bg=blur)")
        ax.set_xticks(x); ax.set_xticklabels(splits_here)
        ax.set_ylabel(title); ax.set_title(f"Method comparison - {title}")
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"compare_{final_col}.png", dpi=cfg.PAPER_DPI)
        plt.close(fig)
    # V7: bang so lieu tong hop (mean +- std) theo split, xuat rieng ca 2 phuong phap de dan thang vao paper
    cols_final = [c for c, _, _ in pairs if c in df_new] + (["sat_final"] if "sat_final" in df_new else [])
    cols_hybrid = [h for _, h, _ in pairs if h in df_new] + (["sat_hybrid"] if "sat_hybrid" in df_new else [])
    summ = pd.concat({
        "GAN_final": df_new.groupby("split")[cols_final].agg(["mean", "std"]),
        "Hybrid": df_new.groupby("split")[cols_hybrid].agg(["mean", "std"]),
    }, axis=1)
    summ.to_csv(run_dir / "results" / "method_comparison_summary.csv")
    log("  Da xuat", len(pairs), "bieu do so sanh phuong phap (compare_*.png) + method_comparison_summary.csv")


# ==============================================================================
#  CHECKPOINT
# ==============================================================================
def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_last(path: Path, state: dict):
    prev = path.with_name("last_prev.pth")
    tmp = path.with_suffix(".pth.tmp")
    torch.save(state, tmp)
    if path.exists():
        os.replace(path, prev)
    os.replace(tmp, path)


def load_last(ck_dir: Path, dev):
    for name in ("last.pth", "last_prev.pth"):
        p = ck_dir / name
        if p.exists():
            try:
                return torch.load(p, map_location=dev, weights_only=False), p
            except Exception as e:
                log(f"  Khong doc duoc {name} ({e}) -> thu file du phong")
    return None, None


def rng_state():
    st = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    try:
        random.setstate(st["py"]); np.random.set_state(st["np"])
        torch.set_rng_state(st["torch"].cpu())
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
    except Exception as e:
        log("  (khong khoi phuc duoc RNG:", e, ")")


def prune_epoch_files(ck_dir: Path, keep: int):
    if keep <= 0:
        return
    files = sorted(ck_dir.glob("gen_G_epoch_*.pth"))
    for f in files[:-keep]:
        for pre in ("gen_G", "gen_F"):
            (ck_dir / f.name.replace("gen_G", pre)).unlink(missing_ok=True)


# ==============================================================================
#  [2] HUAN LUYEN
# ==============================================================================
def stage_train(cfg, root: Path, run_dir: Path, state: dict) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        log("[CANH BAO] Khong thay GPU CUDA -> chay bang CPU se RAT cham.")
    else:
        log("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    ck_dir = run_dir / "checkpoints"; ck_dir.mkdir(exist_ok=True)
    prog_dir = run_dir / "progress_images"; prog_dir.mkdir(exist_ok=True)

    # ---- du lieu
    # V8.3: pseudo-target THAT (inpaint vung halo, xem build_pseudo_clean_targets) cho loss_pseudo -
    # thay vi box-blur tu CHINH anh nhiem (van con giu lam fallback trong vong lap train neu vi ly do
    # gi do thieu file). Chi can cho Glare/Train (huong duy nhat dung loss_pseudo, xem stage_train).
    pseudo_dir = build_pseudo_clean_targets(cfg, root / "Glare" / "Train") if cfg.LAMBDA_PSEUDO > 0 else None
    train_g = make_loader(root / "Glare" / "Train", cfg, True, pseudo_dir=pseudo_dir)
    train_n = make_loader(root / "Normal" / "Train", cfg, True)
    val_g = make_loader(root / "Glare" / "Val", cfg, False, batch=8, max_items=cfg.VAL_MAX_IMAGES)
    val_n = make_loader(root / "Normal" / "Val", cfg, False, batch=8, max_items=cfg.VAL_MAX_IMAGES)
    log(f"Train Glare={len(train_g.dataset)} | Train Normal={len(train_n.dataset)} | "
        f"Val Glare={len(val_g.dataset)} | Val Normal={len(val_n.dataset)} | 1 epoch = {len(train_g)} iter (batch {cfg.BATCH_SIZE})")

    # anh co dinh cho anh tien trinh
    pg = even_pick(val_g.dataset.paths, cfg.PREVIEW_N_GLARE)
    pn = even_pick(val_n.dataset.paths, cfg.PREVIEW_N_NORMAL)
    tmp_ds = SliceDataset(root / "Glare" / "Val", cfg.IMG_SIZE, False, 0, 0, cfg.TRAIN_ON_ROI)
    prev_x, prev_tags = [], []
    for tag, paths in (("glare", pg), ("normal", pn)):
        for p in paths:
            tmp_ds.paths = [p]
            prev_x.append(tmp_ds[0]); prev_tags.append(tag)
    prev_batch = torch.stack(prev_x) if prev_x else None

    # ---- mo hinh
    gen_G, gen_F = UNetGenerator10(1).to(dev), UNetGenerator10(1).to(dev)      # G: glare->normal ; F: normal->glare
    disc_X, disc_Y = Discriminator3(1).to(dev), Discriminator3(1).to(dev)      # X: thuc Normal ; Y: thuc Glare
    for n in (gen_G, gen_F, disc_X, disc_Y):
        init_weights(n)
    # V4: EMA cua gen_G - ban "trung binh truot" chi dung de danh gia/luu/xuat anh, KHONG tham gia backward.
    # EMA_DECAY=0 -> tat, ema_G tro thang vao gen_G (0 chi phi them, giu hanh vi cu).
    ema_G = UNetGenerator10(1).to(dev) if cfg.EMA_DECAY > 0 else gen_G
    if cfg.EMA_DECAY > 0:
        ema_G.load_state_dict(gen_G.state_dict())
        for p in ema_G.parameters():
            p.requires_grad_(False)
        log(f"  EMA (decay={cfg.EMA_DECAY}) BAT: best/preview/hau xu ly se dung trong so EMA cua gen_G.")
    ema_step = 0
    opt_G = torch.optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()), lr=cfg.LR_G, betas=cfg.BETAS)
    opt_D = torch.optim.Adam(itertools.chain(disc_X.parameters(), disc_Y.parameters()), lr=cfg.LR_D, betas=cfg.BETAS)
    # V8.3: ReduceLROnPlateau thay LambdaLR (xem ghi chu CFG.LR_DECAY_* dau file) - theo doi val_score
    # (mode="max"), tu giam LR khi khong cai thien qua LR_DECAY_PATIENCE epoch, khong phu thuoc EPOCHS.
    sch_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_G, mode="max", factor=cfg.LR_DECAY_FACTOR, patience=cfg.LR_DECAY_PATIENCE, min_lr=cfg.LR_MIN)
    sch_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_D, mode="max", factor=cfg.LR_DECAY_FACTOR, patience=cfg.LR_DECAY_PATIENCE, min_lr=cfg.LR_MIN)

    history, best = [], {"score": -1e9, "epoch": 0}
    no_improve = 0            # V8: so epoch lien tiep khong cai thien val_score (xem EARLY_STOP_PATIENCE)
    start_epoch = 1
    ck, ck_path = load_last(ck_dir, dev)
    if ck is not None:
        gen_G.load_state_dict(ck["G"]); gen_F.load_state_dict(ck["F"])
        disc_X.load_state_dict(ck["DX"]); disc_Y.load_state_dict(ck["DY"])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        # V8.3: checkpoint cu (truoc V8.3) luu state_dict cua LambdaLR - khac dinh dang voi
        # ReduceLROnPlateau moi. Bat loi de khong crash luc resume; chi mat vi tri trong lich su
        # "kien nhan" cua scheduler (se tu xay dung lai tu cac epoch tiep theo), KHONG mat trong so.
        try:
            sch_G.load_state_dict(ck["sch_G"]); sch_D.load_state_dict(ck["sch_D"])
        except Exception as e:
            log(f"  (khong khoi phuc duoc lich giam LR tu checkpoint cu - dung lich moi tu dau: {e})")
        history, best = ck["history"], ck["best"]
        no_improve = ck.get("no_improve", 0)     # V8: khoi phuc dem early-stop khi resume
        start_epoch = ck["epoch"] + 1
        set_rng_state(ck["rng"])
        if cfg.EMA_DECAY > 0:
            if ck.get("G_ema") is not None:
                ema_G.load_state_dict(ck["G_ema"])
            else:
                ema_G.load_state_dict(gen_G.state_dict())      # checkpoint cu (truoc V4) -> khoi tao lai tu gen_G
            ema_step = ck.get("ema_step", 0)
        log(f"==> TIEP TUC tu {ck_path.name}: da xong epoch {ck['epoch']}, se chay tu epoch {start_epoch}/{cfg.EPOCHS}")
    else:
        log("Bat dau huan luyen tu dau.")
    if start_epoch > cfg.EPOCHS:
        log("Da huan luyen du so epoch -> bo qua.")
        state["train_done"] = True
        return
    # V8: neu lan chay truoc da EARLY STOP (no_improve >= patience luu trong checkpoint), khong resume
    # train tiep - coi nhu da xong (tranh chay lai tu dau bo dem sau khi da chu dong dung som).
    if cfg.EARLY_STOP_PATIENCE and no_improve >= cfg.EARLY_STOP_PATIENCE:
        log(f"Lan chay truoc da EARLY STOP (no_improve={no_improve}) -> bo qua, dung ket qua da co.")
        state["train_done"] = True
        return

    adv = gan_criterion(cfg.GAN_LOSS, cfg.LABEL_SMOOTH_REAL)
    l1 = nn.L1Loss()
    pool_n, pool_g = ImagePool(cfg.IMAGE_POOL_SIZE), ImagePool(cfg.IMAGE_POOL_SIZE)
    thr_t = 2.0 * cfg.METAL_THRESH_U8 / 255.0 - 1.0
    halo_thr_t = 2.0 * cfg.HALO_THRESH_U8 / 255.0 - 1.0     # V8: nguong vung halo (sang vua) trong khong gian [-1,1]
    writer = SummaryWriter(str(run_dir / "tb")) if (cfg.USE_TENSORBOARD and SummaryWriter) else None
    if cfg.USE_TENSORBOARD and writer is None:
        log("  (tensorboard chua cai -> bo qua ghi log TB)")
    normal_iter = infinite(train_n)
    t_run = time.time(); epoch_times = []
    epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, cfg.EPOCHS + 1):
            t0 = time.time()
            for n in (gen_G, gen_F, disc_X, disc_Y):
                n.train()
            acc = {k: 0.0 for k in ["D_X", "D_Y", "G_total", "G_adv", "cycle", "identity", "tv", "mask_loss", "pseudo_loss", "att_glare"]}
            nb = 0
            loop = tqdm(train_g, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False, unit="it") if tqdm else train_g
            for batch in loop:
                # V8.3: train_g tra ve (rg, pseudo_batch) khi co pseudo_dir (LAMBDA_PSEUDO>0, xem
                # stage_train) - pseudo_batch la anh da inpaint san (build_pseudo_clean_targets), dung
                # THAT lam pseudo-target thay vi box_blur(rg) (xem loss_pseudo ben duoi).
                if isinstance(batch, (list, tuple)):
                    rg, pseudo_batch = batch
                    pseudo_batch = pseudo_batch.to(dev, non_blocking=True)
                else:
                    rg, pseudo_batch = batch, None
                rg = rg.to(dev, non_blocking=True)
                rn = next(normal_iter).to(dev, non_blocking=True)

                # ---------- Generators ----------
                for d in (disc_X, disc_Y):
                    for p in d.parameters():
                        p.requires_grad_(False)
                opt_G.zero_grad(set_to_none=True)
                fake_n, mask_n = gen_G(rg)
                fake_g, _ = gen_F(rn)
                rec_n, _ = gen_G(fake_g)
                rec_g, _ = gen_F(fake_n)
                id_n, mask_id = gen_G(rn)
                id_g, _ = gen_F(rg)
                loss_adv = adv(disc_X(fake_n), True) + adv(disc_Y(fake_g), True)
                loss_cyc = l1(rec_n, rn) + l1(rec_g, rg)
                loss_id = l1(id_n, rn) + l1(id_g, rg)
                # V9.1: core_target/halo_target chuyen len TRUOC (khong con nam trong khoi if
                # LAMBDA_MASK/LAMBDA_PSEUDO nua) vi loss_tv ben duoi can halo_target de CHI phat TV
                # trong vung nhieu, thay vi tinh lai 2 lan.
                dpx = cfg.MASK_DILATE_PX
                core_target = F.max_pool2d((rg >= thr_t).float(), 2 * dpx + 1, 1, dpx)
                hpx = cfg.HALO_DILATE_PX
                halo_target = F.max_pool2d((rg >= halo_thr_t).float(), 2 * hpx + 1, 1, hpx)
                # V9.1: loss_tv cho fake_n (chieu Glare->sach, chieu nguoi dung THUC SU nhin ket qua) gio
                # CHI tinh trong halo_target (vung glare/vet sang + phu can) thay vi TOAN BO anh - truoc
                # day TV phat luon ca vung rang/xuong BINH THUONG khong lien quan gi den nhieu kim loai,
                # khien generator lam mo/mat chi tiet CA ANH de giam loss nay (nguyen nhan chinh gay
                # "anh sau GAN bi mo qua" nguoi dung phan anh). fake_g (Normal->gia-Glare, chieu nguoc,
                # khong dung de hien thi ket qua) khong co khai niem "vung nhieu" nen van giu TV toan anh.
                loss_tv = total_variation_loss(fake_n, halo_target) + total_variation_loss(fake_g)
                loss_mask = torch.zeros((), device=dev)
                loss_pseudo = torch.zeros((), device=dev)
                if cfg.LAMBDA_MASK > 0 or cfg.LAMBDA_PSEUDO > 0:
                    # V8: loss_mask 1 CHIEU thay vi L1 2 chieu (xem ghi chu dau file). core_target = vung
                    # kim loai chac chan (nguong METAL_THRESH_U8, gian no MASK_DILATE_PX) - mask BAT BUOC
                    # phai phu het vung nay (under_core, phat khi mask < target). halo_target = vung rong
                    # hon nhieu (nguong HALO_THRESH_U8 thap hon, gian no HALO_DILATE_PX xa hon) - CHI la
                    # GIOI HAN TREN, mask duoc TU DO trong khoang [core..halo] ma khong bi phat (over_halo
                    # chi phat khi mask vuot ra NGOAI ca halo, tuc la lan vao vung chac chan khong co nhieu).
                    if cfg.LAMBDA_MASK > 0:
                        under_core = F.relu(core_target - mask_n)
                        over_halo = F.relu(mask_n - halo_target)
                        loss_mask = (under_core.mean() + cfg.LAMBDA_MASK_HALO * over_halo.mean()
                                     + l1(mask_id, torch.zeros_like(mask_id)))
                    if cfg.LAMBDA_PSEUDO > 0:
                        # V8.3: loss_pseudo - tin hieu pixel-level TRUC TIEP trong vung halo/rim (band =
                        # halo_target TRU core_target, KHONG dung vung core bao hoa hoan toan vi o do
                        # khong con du lieu that de "doan sach"). pseudo_target = anh da INPAINT san
                        # (build_pseudo_clean_targets - "va" vung halo bang thong tin tu vien NGOAI vao,
                        # xem ghi chu ham do) khi co san (pseudo_batch), fallback ve box_blur(rg) neu
                        # khong (vd thieu file cache) - box_blur YEU hon (chi lam mo TRONG chinh vung
                        # sang, khong keo gia tri xuong nhieu bang inpaint tu vung sach that ben ngoai).
                        if pseudo_batch is not None:
                            pseudo_target = pseudo_batch
                        else:
                            with torch.no_grad():
                                pseudo_target = box_blur(rg, cfg.PSEUDO_BLUR_KSIZE)
                        band = torch.clamp(halo_target - core_target, 0, 1)
                        band_sum = band.sum().clamp_min(1.0)
                        loss_pseudo = (torch.abs(fake_n - pseudo_target) * band).sum() / band_sum
                total_G = (loss_adv + cfg.LAMBDA_CYCLE * loss_cyc + cfg.LAMBDA_IDENTITY * loss_id
                           + cfg.LAMBDA_TV * loss_tv + cfg.LAMBDA_MASK * loss_mask
                           + cfg.LAMBDA_PSEUDO * loss_pseudo)
                # V4: kiem tra NaN/Inf NGAY sau khi tinh total_G (truoc backward/step) - phat hien som,
                # chua ap dung buoc cap nhat hong len trong so (ban goc kiem tra sau ca opt_G.step() va opt_D.step()).
                if not torch.isfinite(total_G):
                    raise FloatingPointError("Loss = NaN/Inf. Giam CFG.LR_G/CFG.LR_D hoac tang CFG.LAMBDA_* roi chay lai (se tiep tuc tu checkpoint).")
                total_G.backward()
                if cfg.GRAD_CLIP_NORM > 0:      # V4: chan gradient qua lon truoc khi cap nhat G+F
                    torch.nn.utils.clip_grad_norm_(itertools.chain(gen_G.parameters(), gen_F.parameters()), cfg.GRAD_CLIP_NORM)
                opt_G.step()
                if cfg.EMA_DECAY > 0:            # V4: cap nhat trong so EMA cua gen_G ngay sau moi buoc G
                    ema_step += 1
                    eff_decay = min(cfg.EMA_DECAY, (1 + ema_step) / (10 + ema_step))   # warmup: EMA "theo sat" luc dau, "muot" dan sau
                    ema_update(ema_G, gen_G, eff_decay)

                # ---------- Discriminators ----------
                for d in (disc_X, disc_Y):
                    for p in d.parameters():
                        p.requires_grad_(True)
                opt_D.zero_grad(set_to_none=True)
                fn_pool, fg_pool = pool_n.query(fake_n), pool_g.query(fake_g)
                loss_DX = 0.5 * (adv(disc_X(rn), True) + adv(disc_X(fn_pool.detach()), False))
                loss_DY = 0.5 * (adv(disc_Y(rg), True) + adv(disc_Y(fg_pool.detach()), False))
                (loss_DX + loss_DY).backward()
                if cfg.GRAD_CLIP_NORM > 0:      # V4: chan gradient qua lon truoc khi cap nhat DX+DY
                    torch.nn.utils.clip_grad_norm_(itertools.chain(disc_X.parameters(), disc_Y.parameters()), cfg.GRAD_CLIP_NORM)
                opt_D.step()

                acc["D_X"] += loss_DX.item(); acc["D_Y"] += loss_DY.item()
                acc["G_total"] += total_G.item(); acc["G_adv"] += loss_adv.item()
                acc["cycle"] += loss_cyc.item(); acc["identity"] += loss_id.item()
                acc["tv"] += loss_tv.item(); acc["mask_loss"] += float(loss_mask.detach())
                acc["pseudo_loss"] += float(loss_pseudo.detach())
                acc["att_glare"] += mask_n.mean().item()
                nb += 1
                if tqdm:
                    loop.set_postfix(G=f"{total_G.item():.2f}", DX=f"{loss_DX.item():.2f}", DY=f"{loss_DY.item():.2f}")

            row = {k: v / max(nb, 1) for k, v in acc.items()}
            row.update(evaluate(ema_G, gen_F, val_g, val_n, cfg, dev))   # V4: danh gia bang trong so EMA (muot hon)
            # V8.3: ReduceLROnPlateau can gia tri metric moi lan step() (khong nhu LambdaLR cu chi can
            # so epoch) - dung thang val_score (mode="max", xem luc khoi tao scheduler o tren).
            sch_G.step(row["val_score"]); sch_D.step(row["val_score"])
            row["epoch"] = epoch
            row["lr"] = opt_G.param_groups[0]["lr"]
            row["epoch_sec"] = time.time() - t0
            history.append(row)
            epoch_times.append(row["epoch_sec"])

            # V8: is_best doi hoi cai thien > EARLY_STOP_MIN_DELTA (khong chi > truoc do, tranh dao dong
            # nho lien tuc "reset" bo dem early-stop ma khong thuc su tien bo).
            is_best = row["val_score"] > best["score"] + cfg.EARLY_STOP_MIN_DELTA
            if is_best:
                best = {"score": row["val_score"], "epoch": epoch}
                no_improve = 0
                atomic_torch_save({"gen_G": ema_G.state_dict(), "gen_F": gen_F.state_dict(),
                                   "epoch": epoch, "metrics": row}, ck_dir / "best.pth")   # V4: luu ema_G thay vi gen_G
            else:
                no_improve += 1

            # ---- checkpoint moi epoch (tiep tuc duoc neu bi ngat)
            save_last(ck_dir / "last.pth", {
                "epoch": epoch, "G": gen_G.state_dict(), "F": gen_F.state_dict(),
                "DX": disc_X.state_dict(), "DY": disc_Y.state_dict(),
                "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                "sch_G": sch_G.state_dict(), "sch_D": sch_D.state_dict(),
                "history": history, "best": best, "no_improve": no_improve, "rng": rng_state(), "cfg": cfg_snapshot(),
                "G_ema": ema_G.state_dict(), "ema_step": ema_step})   # V4: luu ca trong so EMA + step de resume dung
            if cfg.SAVE_EVERY and epoch % cfg.SAVE_EVERY == 0:
                atomic_torch_save(gen_G.state_dict(), ck_dir / f"gen_G_epoch_{epoch:03d}.pth")
                atomic_torch_save(gen_F.state_dict(), ck_dir / f"gen_F_epoch_{epoch:03d}.pth")
                prune_epoch_files(ck_dir, cfg.KEEP_LAST_N_EPOCH_FILES)

            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            if prev_batch is not None and epoch % cfg.PREVIEW_EVERY == 0:
                save_progress_image(ema_G, prev_batch, prev_tags, epoch, prog_dir / f"epoch_{epoch:03d}.png", dev)  # V4: xem bang EMA
            if epoch % cfg.PLOT_EVERY == 0 or epoch == cfg.EPOCHS:
                plot_history(history, run_dir / "loss_curves.png")
            if writer:
                for k, v in row.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        writer.add_scalar(f"train/{k}", v, epoch)
                if prev_batch is not None and epoch % cfg.TB_IMAGE_EVERY == 0:
                    with torch.no_grad():
                        o, m = ema_G(prev_batch.to(dev))   # V4: ghi TensorBoard bang EMA
                    writer.add_images("progress/input", (prev_batch + 1) / 2, epoch)
                    writer.add_images("progress/output", ((o.cpu() + 1) / 2).clamp(0, 1), epoch)
                    writer.add_images("progress/mask", m.cpu(), epoch)

            eta = np.mean(epoch_times[-5:]) * (cfg.EPOCHS - epoch)
            log(f"Ep {epoch:03d}/{cfg.EPOCHS} | G {row['G_total']:.3f} DX {row['D_X']:.3f} DY {row['D_Y']:.3f} "
                f"cyc {row['cycle']:.3f} tv {row['tv']:.4f} pseudo {row['pseudo_loss']:.4f} | "
                f"val: sat_int_red {row['val_sat_intensity_reduction']:.3f} "
                f"(sat_red_cu {row['val_sat_reduction']:.3f}) halo_red {row['val_halo_reduction']:.3f} "
                f"pres_ssim {row['val_pres_ssim']:.3f} id_ssim {row['val_id_ssim']:.3f} score {row['val_score']:.3f}"
                f"{' *BEST*' if is_best else f' (no_improve {no_improve})'} | lr {row['lr']:.2e} | "
                f"{row['epoch_sec']:.0f}s, con ~{eta/60:.0f} phut")
            # V8: early stopping - dung neu qua EARLY_STOP_PATIENCE epoch lien tiep khong cai thien
            if cfg.EARLY_STOP_PATIENCE and no_improve >= cfg.EARLY_STOP_PATIENCE:
                log(f"==> EARLY STOPPING: {no_improve} epoch lien tiep khong cai thien val_score "
                    f"(best={best['score']:.4f} o epoch {best['epoch']}). Dung som tai epoch {epoch}/{cfg.EPOCHS}.")
                break
    except KeyboardInterrupt:
        log("\n!!! Da nhan Ctrl+C. Checkpoint gan nhat = epoch", max(len(history) and history[-1]["epoch"], 0))
        log("    Chay lai dung lenh cu de tiep tuc (epoch dang do se chay lai tu dau).")
        if writer:
            writer.close()
        raise
    except Exception:
        log("!!! Loi trong luc train:\n" + traceback.format_exc())
        log("    Checkpoint van con; sua loi roi chay lai de tiep tuc.")
        if writer:
            writer.close()
        raise

    atomic_torch_save({"gen_G": ema_G.state_dict(), "gen_F": gen_F.state_dict(), "epoch": epoch},
                      ck_dir / "final_weights.pth")   # V4: luu ema_G thay vi gen_G ; V8: epoch thuc te (co the < CFG.EPOCHS neu early-stop)
    plot_history(history, run_dir / "loss_curves.png")
    if writer:
        writer.close()
    state["train_done"] = True
    total_hours = (time.time() - t_run) / 3600
    stopped_early = bool(cfg.EARLY_STOP_PATIENCE) and no_improve >= cfg.EARLY_STOP_PATIENCE
    log(f"Huan luyen xong{' (EARLY STOP tai epoch ' + str(epoch) + ')' if stopped_early else ''}. "
        f"Best epoch = {best['epoch']} (score {best['score']:.4f}). Tong {total_hours:.2f} gio (lan chay nay).")

    # V6: xuat bieu do rieng (do phan giai cao) + thong tin mo hinh cho paper/luan van
    export_training_figures(history, run_dir, cfg)
    export_model_info(gen_G, gen_F, disc_X, disc_Y, cfg, run_dir, dev, extra={
        "best_epoch": best["epoch"], "best_val_score": best["score"],
        "epochs_trained": cfg.EPOCHS, "avg_epoch_sec": float(np.mean(epoch_times)) if epoch_times else None,
        "total_train_hours_this_run": total_hours,
        "dataset_counts": {"train_glare": len(train_g.dataset), "train_normal": len(train_n.dataset),
                            "val_glare": len(val_g.dataset), "val_normal": len(val_n.dataset)},
        "batch_size": cfg.BATCH_SIZE, "img_size": cfg.IMG_SIZE,
    })


# ==============================================================================
#  [3] HAU XU LY  (poss_processing.ipynb)
# ==============================================================================
def load_generator_for_post(cfg, run_dir: Path, dev):
    ck_dir = run_dir / "checkpoints"
    order = ["best.pth", "last.pth", "final_weights.pth"] if cfg.POST_WEIGHTS == "best" else ["last.pth", "final_weights.pth", "best.pth"]
    for name in order:
        p = ck_dir / name
        if p.exists():
            obj = torch.load(p, map_location=dev, weights_only=False)
            sd = obj.get("gen_G") or obj.get("G")
            g = UNetGenerator10(1).to(dev)
            g.load_state_dict(sd)
            g.eval()
            log(f"Hau xu ly dung trong so: {name} (epoch {obj.get('epoch', '?')})")
            return g, name
    raise FileNotFoundError(f"Khong co checkpoint nao trong {ck_dir}")


@torch.no_grad()
def run_gan(gen_G, gray, cfg, dev):
    src = extract_roi(gray) if cfg.TRAIN_ON_ROI else gray
    h, w = gray.shape
    x = cv2.resize(src, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_AREA if max(src.shape) > cfg.IMG_SIZE else cv2.INTER_LINEAR)
    t = torch.from_numpy(x).float().div(127.5).sub(1).unsqueeze(0).unsqueeze(0).to(dev)
    out, m = gen_G(t)
    out = ((out[0, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    mk = (m[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return (cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4),
            cv2.resize(mk, (w, h), interpolation=cv2.INTER_LINEAR))


def make_artifact_mask(orig, cfg, net_mask=None):
    """V8: `net_mask` (anh 0..255, mask ma chinh generator da hoc du doan - xem run_gan) duoc HOP (OR)
    vao mask co dien thay vi hau xu ly tu tinh mask rieng roi VUT BO mask cua generator nhu V6/V7.
    Neu khong sua diem nay, moi mo rong vung generator duoc phep sua (LAMBDA_MASK_HALO o buoc train)
    se vo nghia - anh xuat ra van bi "cap" lai vao dung vung nguong co dien cu."""
    _, core = cv2.threshold(orig, cfg.THRESH, 255, cv2.THRESH_BINARY)                                # 1.1
    art = cv2.dilate(core, np.ones((cfg.KERNEL_SIZE, cfg.KERNEL_SIZE), np.uint8), iterations=cfg.DILATE_ITERS)  # 1.2
    if net_mask is not None:
        _, net_bin = cv2.threshold(net_mask, cfg.NET_MASK_THRESH_U8, 255, cv2.THRESH_BINARY)
        art = cv2.bitwise_or(art, net_bin)
    blurred = cv2.GaussianBlur(art, (cfg.BLUR_KSIZE, cfg.BLUR_KSIZE), 0)                            # 1.3
    return core, art, blurred.astype(np.float32) / 255.0


def make_halo_band_mask(orig, cfg):
    """V9: mask vung 'vet sang ben ngoai' (streak/halo, sang vua ~150-240, KHONG bao hoa) - GIONG HET
    halo_target dung trong loss_pseudo luc train (xem CFG.HALO_THRESH_U8/HALO_DILATE_PX), tai dung o
    day cho postprocess_one() moi: day la vung se duoc XU LY CO DIEN (inpaint) THAY VI GAN, vi GAN
    (dù da co loss_pseudo) van giam sang vung nay YEU hon nhieu so voi inpaint co dien (xem so sanh
    thuc te trong run V8.3/V8.3b/V8.4). Tra ve (mask nhi phan uint8 - dung cho cv2.inpaint, mask lien
    tuc 0..1 da Gaussian-blur - dung de blend min bien, khong bi seam)."""
    _, halo_bin = cv2.threshold(orig, cfg.HALO_THRESH_U8, 255, cv2.THRESH_BINARY)
    k = 2 * cfg.HALO_DILATE_PX + 1
    halo_bin = cv2.dilate(halo_bin, np.ones((k, k), np.uint8), iterations=1)
    blurred = cv2.GaussianBlur(halo_bin, (cfg.BLUR_KSIZE, cfg.BLUR_KSIZE), 0)
    return halo_bin, blurred.astype(np.float32) / 255.0


def blur_background_classical(gray, cfg):
    """V7: lam mo THUAN CO DIEN (khong qua GAN) - dung cho vung NGOAI 'khung rang' trong hybrid_compose."""
    m = cfg.HYBRID_BG_METHOD
    k = max(3, cfg.HYBRID_BG_GAUSS_KSIZE | 1)
    if m == "gaussian":
        return cv2.GaussianBlur(gray, (k, k), 0)
    if m == "bilateral":
        return cv2.bilateralFilter(gray, cfg.HYBRID_BG_BILAT_D, cfg.HYBRID_BG_BILAT_SIGMA, cfg.HYBRID_BG_BILAT_SIGMA)
    if m == "nlm_gauss":
        bg = cv2.fastNlMeansDenoising(gray, None, h=cfg.HYBRID_BG_NLM_H, templateWindowSize=7, searchWindowSize=21)
        return cv2.GaussianBlur(bg, (k, k), 0)
    return gray.copy()


def hybrid_compose(orig, gan_processed, cfg):
    """V7: ghep 2 vung tach biet -
    - trong 'khung rang' (ROI Otsu 3 lop tren anh GOC, giong extract_roi_mask) -> giu ket qua GAN
      da hau xu ly (`gan_processed`, thuong la final_smooth/final).
    - ngoai khung rang (nen/khong khi/vien anh - GAN khong dang tin cay o day) -> thay bang anh GOC
      da lam mo thuan thuat toan co dien (blur_background_classical, KHONG dung GAN).
    Vien ghep duoc lam min bang Gaussian blur (HYBRID_ROI_MASK_BLUR) de "dem" khung rang vao nen tu
    nhien, khong bi seam cung. Tra ve (anh ket hop uint8, roi_mask 0..255)."""
    try:
        roi_mask, _, _ = extract_roi_mask(orig, cfg.HYBRID_ROI_OTSU_OFFSET, cfg.HYBRID_ROI_CLOSE_KERNEL,
                                          cfg.HYBRID_ROI_DILATE_ITERS, cfg.HYBRID_ROI_MIN_AREA_RATIO, 0,
                                          cfg.HYBRID_ROI_MASK_BLUR)
    except Exception:
        return gan_processed.copy(), np.full_like(orig, 255)
    bg = blur_background_classical(orig, cfg)
    w = roi_mask.astype(np.float32) / 255.0
    out = np.clip(w * gan_processed.astype(np.float32) + (1 - w) * bg.astype(np.float32), 0, 255).astype(np.uint8)
    return out, roi_mask


def bg_smooth_roi(img, cfg):
    """V8: tach rieng tu postprocess_one() de dung lai duoc cho ca stage_classical() (dam bao ca 2
    'phuong phap' GAN va co dien dung CHUNG 1 buoc lam min nen theo ROI, so sanh cong bang). Tra ve
    (anh da lam min nen, roi_mask) - neu loi (vd ROI khong tim duoc) tra ve (img goc, None)."""
    try:
        roi_mask, _, _ = extract_roi_mask(img, cfg.OTSU_OFFSET, cfg.ROI_CLOSE_KERNEL, cfg.ROI_DILATE_ITERS,
                                          cfg.ROI_MIN_AREA_RATIO, cfg.BACKGROUND_VALUE, cfg.ROI_MASK_BLUR)
        bg = cv2.fastNlMeansDenoising(img, None, h=cfg.BG_NLM_H, templateWindowSize=7, searchWindowSize=21) if cfg.BG_NLM_H > 0 else img.copy()
        if cfg.BG_GAUSS_KSIZE > 1:
            bg = cv2.GaussianBlur(bg, (cfg.BG_GAUSS_KSIZE, cfg.BG_GAUSS_KSIZE), 0)
        w = roi_mask.astype(np.float32) / 255.0
        out = np.clip(w * img + (1 - w) * bg, 0, 255).astype(np.uint8)
        return out, roi_mask
    except Exception as e:
        log(f"  (ROI loi -> giu ket qua blending: {e})")
        return img, None


def unsharp_mask(img_u8, amount, sigma):
    """V9.1: unsharp mask co dien - tra lai mot phan chi tiet/canh sac da mat do GAN + bilateral lam
    mo (xem ghi chu CFG.SHARPEN_AMOUNT/SHARPEN_SIGMA va total_variation_loss). amount<=0 -> tra ve
    nguyen anh (tat tinh nang). Cong thuc: sharp = img + amount*(img - gaussian_blur(img, sigma))."""
    if amount <= 0:
        return img_u8
    blurred = cv2.GaussianBlur(img_u8, (0, 0), sigma)
    sharp = img_u8.astype(np.float32) + amount * (img_u8.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)


def postprocess_one(orig, gan, cfg, net_mask=None):
    """V9: 3 vung tach biet thay vi 2 (theo yeu cau nguoi dung - "classical chi de lam mo vet sang
    ben ngoai, GAN de len anh da xu ly mo vien ngoai"):
      - trong M (core kim loai chac chan + mask GAN tu hoc)         -> GAN (da bilateral) - "de len".
      - trong halo_w NHUNG NGOAI M (vet sang/streak, sang vua)      -> inpaint co dien (STREAK_CLASSICAL_SMOOTH),
        KHONG con la anh goc chua xu ly nhu V8 - day la vung GAN yeu nhat (xem ghi chu make_halo_band_mask).
      - phan con lai (ngoai ca 2 vung tren)                         -> giu nguyen anh goc.
    3 trong so (w_gan, w_streak, w_orig) cong lai = 1 o moi pixel, deu la mask LIEN TUC (da Gaussian-
    blur) nen bien ghep muot, khong bi seam cung."""
    core, art, M = make_artifact_mask(orig, cfg, net_mask)
    gan_s = cv2.bilateralFilter(gan, d=cfg.BILAT_D, sigmaColor=cfg.SIGMA_COLOR, sigmaSpace=cfg.SIGMA_SPACE)  # Giai doan 2 (vung GAN)
    gan_s = unsharp_mask(gan_s, cfg.SHARPEN_AMOUNT, cfg.SHARPEN_SIGMA)  # V9.1: tra lai chi tiet da mat do bilateral+GAN
    halo_bin, halo_w = make_halo_band_mask(orig, cfg)
    if cfg.STREAK_CLASSICAL_SMOOTH:
        streak_smoothed = cv2.inpaint(orig, halo_bin, cfg.CLASSICAL_INPAINT_RADIUS, cv2.INPAINT_TELEA)
    else:
        streak_smoothed = orig.copy()          # V9 tat -> giong hanh vi V8 (vung ngoai M la anh goc)
    w_gan = M
    w_streak = halo_w * (1.0 - w_gan)          # co dien CHI nhan phan halo M chua "giu" (khong de len GAN)
    w_orig = (1.0 - w_gan) * (1.0 - halo_w)
    final = np.clip(w_gan * gan_s.astype(np.float32) + w_streak * streak_smoothed.astype(np.float32)
                     + w_orig * orig.astype(np.float32), 0, 255).astype(np.uint8)          # Giai doan 3
    res = {"original": orig, "gan": gan, "core": core, "M": M, "gan_smoothed": gan_s,
           "halo_band": halo_w, "streak_smoothed": streak_smoothed, "final": final}
    if cfg.POST_BG_SMOOTH:
        fsmooth, roi_mask = bg_smooth_roi(final, cfg)
        res["final_smooth"] = fsmooth
        if roi_mask is not None:
            res["roi_mask"] = roi_mask
    return res


def comparison_montage(res, name, panel_h=256):
    def pan(im, title, cmap=False):
        im = im if im.dtype == np.uint8 else np.clip(im * 255, 0, 255).astype(np.uint8)
        if cmap:
            im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        s = panel_h / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), panel_h))
        return _label(im, title)
    diff = np.clip(cv2.absdiff(res.get("final_smooth", res["final"]), res["original"]).astype(np.float32) * 4, 0, 255).astype(np.uint8)
    panels = [pan(res["original"], "1 original"), pan(res["core"], "2 core mask"), pan(res["M"], "3 mask M (GAN)", True)]
    if "halo_band" in res:      # V9: them panel vung streak/halo (co dien) + ket qua inpaint vung do
        panels.append(pan(res["halo_band"], "3b halo band (classical)", True))
        panels.append(pan(res["streak_smoothed"], "3c streak inpainted"))
    panels += [pan(res["gan"], "4 GAN raw"), pan(res["gan_smoothed"], "5 GAN+bilateral"),
               pan(res["final"], "6 blended (GAN core + classical streak)")]
    if "final_smooth" in res:
        panels.append(pan(res["final_smooth"], "7 +bg smooth"))
    panels.append(pan(diff, "|final-orig| x4"))
    if "hybrid" in res:      # V7: them 2 panel - khung rang ROI dung de ghep + ket qua hybrid
        panels.append(pan(res["hybrid_roi_mask"], "8 tooth ROI", True))
        panels.append(pan(res["hybrid"], "9 hybrid (teeth=GAN, bg=blur)"))
    return np.hstack(panels)


# V8: schema "dai" (long format, 1 dong = 1 anh x 1 bien the phuong phap) dung CHUNG cho ca GAN
# (stage_post, ben duoi) va co dien (stage_classical) - de aggregate_all_methods() gop lai duoc
# thanh 1 bang xep hang + bieu do so sanh duy nhat, khong phu thuoc so cot khac nhau cua tung pipeline.
LONG_METRIC_COLS = ["mask_area_pct", "sat_orig", "sat_final", "ssim_preservation", "psnr_preservation",
                     "mae_preservation", "edge_corr_preservation", "halo_reduction", "mae_full", "rmse_full",
                     "ssim_orig_vs_final", "psnr_orig_vs_final", "inference_ms"]


def _gan_long_rows(row: dict) -> list:
    """V8: chuyen 1 dong postprocess_metrics.csv (wide, GAN) thanh 1-2 dong schema 'dai' o tren -
    "GAN_final" (bat buoc) va "GAN_hybrid" (neu CFG.HYBRID_COMPOSE)."""
    out = [{
        "split": row["split"], "file": row["file"], "method": "GAN_final",
        "mask_area_pct": row.get("mask_area_pct"), "sat_orig": row.get("sat_orig"),
        "sat_final": row.get("sat_final"),
        "ssim_preservation": row.get("ssim_preservation"), "psnr_preservation": row.get("psnr_preservation"),
        "mae_preservation": row.get("mae_preservation"), "edge_corr_preservation": row.get("edge_corr_preservation"),
        "halo_reduction": row.get("halo_reduction"), "mae_full": row.get("mae_full"), "rmse_full": row.get("rmse_full"),
        "ssim_orig_vs_final": row.get("ssim_orig_vs_final"), "psnr_orig_vs_final": row.get("psnr_orig_vs_final"),
        "inference_ms": row.get("inference_ms"),
    }]
    if row.get("ssim_preservation_hybrid") is not None and not pd.isna(row.get("ssim_preservation_hybrid")):
        out.append({
            "split": row["split"], "file": row["file"], "method": "GAN_hybrid",
            "mask_area_pct": row.get("roi_area_pct", row.get("mask_area_pct")), "sat_orig": row.get("sat_orig"),
            "sat_final": row.get("sat_hybrid"),
            "ssim_preservation": row.get("ssim_preservation_hybrid"), "psnr_preservation": row.get("psnr_preservation_hybrid"),
            "mae_preservation": row.get("mae_preservation_hybrid"), "edge_corr_preservation": row.get("edge_corr_preservation_hybrid"),
            "halo_reduction": row.get("halo_reduction_hybrid"), "mae_full": row.get("mae_full_hybrid"), "rmse_full": row.get("rmse_full_hybrid"),
            "ssim_orig_vs_final": row.get("ssim_orig_vs_hybrid"), "psnr_orig_vs_final": row.get("psnr_orig_vs_hybrid"),
            "inference_ms": row.get("inference_ms"),
        })
    return out


def write_metrics_long(rows_long: list, run_dir: Path) -> None:
    """V8: ghi/gop results/metrics_long.csv (schema dai, xem LONG_METRIC_COLS) - dung chung boi ca
    stage_post (GAN) va stage_classical, la nguon du lieu duy nhat aggregate_all_methods doc vao."""
    if not rows_long:
        return
    out_root = run_dir / "results"; out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / "metrics_long.csv"
    df_new = pd.DataFrame(rows_long)
    if p.exists():
        df_new = pd.concat([pd.read_csv(p), df_new]).drop_duplicates(["split", "file", "method"], keep="last")
    df_new.to_csv(p, index=False)


def stage_post(cfg, root: Path, run_dir: Path, state: dict) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_G, wname = load_generator_for_post(cfg, run_dir, dev)
    out_root = run_dir / "results"
    metrics_path = out_root / "postprocess_metrics.csv"
    rows = []
    for split in cfg.POST_SPLITS:
        files = list_images(root / "Glare" / split)
        if cfg.POST_MAX_IMAGES_PER_SPLIT:
            files = even_pick(files, cfg.POST_MAX_IMAGES_PER_SPLIT)
        cmp_files = set(p.name for p in even_pick(files, cfg.POST_COMPARISON_MAX))
        log(f"Hau xu ly {split}: {len(files)} anh Glare")
        it = tqdm(files, desc=f"Post {split}", unit="anh") if tqdm else files
        for p in it:
            final_p = out_root / "final" / split / p.name
            if final_p.exists() and not cfg.POST_OVERWRITE:
                continue
            orig = read_gray_uint8(p)
            t_inf0 = time.perf_counter()                      # V6: do thoi gian suy luan (khong tinh hau xu ly co dien)
            gan, att = run_gan(gen_G, orig, cfg, dev)
            inference_ms = (time.perf_counter() - t_inf0) * 1000.0
            res = postprocess_one(orig, gan, cfg, net_mask=att)   # V8: dung mask cua generator, khong vut bo nua
            write_image(out_root / "gan_raw" / split / p.name, gan)
            write_image(out_root / "attention_mask" / split / p.name, att)
            write_image(final_p, res["final"])
            if "final_smooth" in res:
                write_image(out_root / "final_smooth" / split / p.name, res["final_smooth"])
            # V7: buoc ghep hybrid - khung rang (ROI Otsu) = ket qua GAN da hau xu ly, nen = blur co dien
            if cfg.HYBRID_COMPOSE:
                hyb_src = res.get(cfg.HYBRID_SOURCE, res["final"])
                hybrid, hyb_roi = hybrid_compose(orig, hyb_src, cfg)
                res["hybrid"], res["hybrid_roi_mask"] = hybrid, hyb_roi
                write_image(out_root / "final_hybrid" / split / p.name, hybrid)
            if p.name in cmp_files:
                write_image(out_root / "comparison" / split / (p.stem + "_cmp.png"), comparison_montage(res, p.name))
            sat = lambda a: float((a >= cfg.THRESH).mean())
            best_final = res.get("final_smooth", res["final"])
            keep = ~metal_region(orig, cfg.THRESH)
            halo_mask = (orig >= cfg.HALO_THRESH_U8) & (orig < cfg.THRESH)   # V8: vung streak/halo, khong tinh core

            def halo_reduction(cand):
                """V8: % giam sang trong vung streak/halo (sang vua, KHONG bao hoa) - chi so chinh de
                kiem tra co THUC SU giam duoc vet soc hay khong, giong val_halo_reduction luc train."""
                if halo_mask.sum() < 64:
                    return float("nan")
                ma = float(orig[halo_mask].mean()); mb = float(cand[halo_mask].mean())
                return max(0.0, 1.0 - mb / ma) if ma > 1e-6 else float("nan")

            def region_metrics(cand):
                """V6/V7: MAE/RMSE toan anh + SSIM/PSNR/MAE/RMSE/edge-corr tren vung KHONG kim loai
                (preservation region) - quan trong hon cho paper vi SSIM/PSNR toan anh bi vung nen
                (phan lon anh) chi phoi, khong phan anh dung chat luong vung sua."""
                diff_f = orig.astype(np.float32) - cand.astype(np.float32)
                mae_f = float(np.abs(diff_f).mean()); rmse_f = float(np.sqrt((diff_f ** 2).mean()))
                if keep.sum() > 64:
                    _, smap = ssim_fn(orig, cand, data_range=255, full=True)
                    ssim_p = float(smap[keep].mean())
                    diff_p = diff_f[keep]
                    mse_p = float((diff_p ** 2).mean())
                    psnr_p = 99.0 if mse_p < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse_p))
                    mae_p = float(np.abs(diff_p).mean())
                    edge_p = edge_corr(orig, cand, keep)
                else:
                    ssim_p = psnr_p = mae_p = edge_p = float("nan")
                return mae_f, rmse_f, ssim_p, psnr_p, mae_p, edge_p

            mae_full, rmse_full, ssim_pres, psnr_pres, mae_pres, edge_pres = region_metrics(best_final)
            row = {
                "split": split, "file": p.name, "mask_area_pct": float((res["M"] > 0).mean() * 100),
                # V9: dien tich vung "vet sang ben ngoai" duoc XU LY CO DIEN (inpaint) BEN TRONG
                # pipeline GAN - de bao cao ro trong paper GAN dam nhan bao nhieu%, co dien dam nhan
                # bao nhieu% (khong tinh vao mask_area_pct, von chi la vung GAN tu quyet dinh sua).
                "streak_area_pct": float((res.get("halo_band", 0) > 0.5).mean() * 100) if "halo_band" in res else 0.0,
                "sat_orig": sat(orig), "sat_gan": sat(gan), "sat_final": sat(res["final"]),
                "ssim_orig_vs_final": float(ssim_fn(orig, best_final, data_range=255)),
                "psnr_orig_vs_final": _psnr_safe(orig, best_final),
                "mae_full": mae_full, "rmse_full": rmse_full,
                "ssim_preservation": ssim_pres, "psnr_preservation": psnr_pres, "mae_preservation": mae_pres,
                "edge_corr_preservation": edge_pres, "halo_reduction": halo_reduction(best_final),
                "inference_ms": inference_ms,
            }
            if cfg.HYBRID_COMPOSE:
                # V7: cung bo chi so nhung cho anh hybrid - de so sanh truc tiep voi final/final_smooth
                mae_f_h, rmse_f_h, ssim_p_h, psnr_p_h, mae_p_h, edge_p_h = region_metrics(res["hybrid"])
                row.update({
                    "roi_area_pct": float((res["hybrid_roi_mask"] > 0).mean() * 100),
                    "sat_hybrid": sat(res["hybrid"]),
                    "ssim_orig_vs_hybrid": float(ssim_fn(orig, res["hybrid"], data_range=255)),
                    "psnr_orig_vs_hybrid": _psnr_safe(orig, res["hybrid"]),
                    "mae_full_hybrid": mae_f_h, "rmse_full_hybrid": rmse_f_h,
                    "ssim_preservation_hybrid": ssim_p_h, "psnr_preservation_hybrid": psnr_p_h,
                    "mae_preservation_hybrid": mae_p_h, "edge_corr_preservation_hybrid": edge_p_h,
                    "halo_reduction_hybrid": halo_reduction(res["hybrid"]),
                })
            rows.append(row)
        # V6: anh minh hoa dinh dang paper (a/b/c/d, khong overlay chu mau) cho split nay
        export_qualitative_figure(cfg, root, run_dir, split)
    if rows:
        df_new = pd.DataFrame(rows)
        if metrics_path.exists():
            df_new = pd.concat([pd.read_csv(metrics_path), df_new]).drop_duplicates(["split", "file"], keep="last")
        out_root.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(metrics_path, index=False)
        num_cols = [c for c in ["mask_area_pct", "roi_area_pct", "sat_orig", "sat_gan", "sat_final", "sat_hybrid",
                                "ssim_orig_vs_final", "psnr_orig_vs_final", "mae_full", "rmse_full",
                                "ssim_preservation", "psnr_preservation", "mae_preservation", "edge_corr_preservation",
                                "halo_reduction",
                                "ssim_orig_vs_hybrid", "psnr_orig_vs_hybrid", "mae_full_hybrid", "rmse_full_hybrid",
                                "ssim_preservation_hybrid", "psnr_preservation_hybrid", "mae_preservation_hybrid",
                                "edge_corr_preservation_hybrid", "halo_reduction_hybrid", "inference_ms"]
                   if c in df_new]
        summ = df_new.groupby("split")[num_cols].mean().round(4)
        (out_root / "summary.txt").write_text(f"weights: {wname}\n{summ.to_string()}\n", encoding="utf-8")
        log("Tom tat hau xu ly:\n" + summ.to_string())
        # V6: bang thong ke day du (min/max/mean/std/median) - de dan vao bang so lieu paper
        df_new.groupby("split")[num_cols].describe().to_csv(out_root / "postprocess_summary_stats.csv")
        export_distribution_figures(df_new, run_dir, cfg)
        export_method_comparison_figure(df_new, run_dir, cfg)   # V7: so sanh GAN(final) vs hybrid
        # V8: xuat them schema "dai" (1 dong = 1 anh x 1 bien the) - de aggregate_all_methods() gop
        # chung voi ket qua "classical" thanh 1 bang xep hang + bieu do so sanh duy nhat.
        long_rows = []
        for r in df_new.to_dict("records"):
            long_rows.extend(_gan_long_rows(r))
        write_metrics_long(long_rows, run_dir)
    state["post_done"] = True
    log("Anh ket qua:", out_root)


# ==============================================================================
#  V8: PHUONG PHAP "classical" - XU LY ANH CO DIEN (KHONG MANG NORON), port tu
#  dental_mar_classical.py, dung LAI cac ham dung chung voi GAN (make_artifact_mask, bg_smooth_roi,
#  edge_corr, _psnr_safe...) va tinh CUNG 1 bo chi so preservation-region + halo_reduction nhu GAN
#  (xem LONG_METRIC_COLS) de so sanh cong bang, khong phai tu tinh mot bo chi so khac roi tu suy dien.
# ==============================================================================
def method_inpaint(orig, art_mask, cfg, algo="telea"):
    """[A]/[B] cv2.inpaint: "va" vung kim loai bang thong tin lan truyen tu vien vao trong - khong
    dung GAN, khong "biet" giai phau that ben duoi, nhung khong co rui ro hallucinate sai."""
    k = 2 * cfg.CLASSICAL_INPAINT_DILATE_EXTRA + 1
    mask = cv2.dilate(art_mask, np.ones((k, k), np.uint8), iterations=1) if cfg.CLASSICAL_INPAINT_DILATE_EXTRA > 0 else art_mask
    flag = cv2.INPAINT_TELEA if algo == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(orig, mask, cfg.CLASSICAL_INPAINT_RADIUS, flag)


def process_one_classical(orig, cfg):
    """Tra ve core/art/M (giong make_artifact_mask cua GAN, KHONG co net_mask) + dict 3 bien the
    ('inpaint_telea', 'inpaint_ns', 'blur_blend') + dict bien the da lam min nen (neu POST_BG_SMOOTH)."""
    core, art, M = make_artifact_mask(orig, cfg, net_mask=None)
    telea = method_inpaint(orig, art, cfg, "telea")
    ns = method_inpaint(orig, art, cfg, "ns")
    # blur_blend = CHINH LA cong thuc postprocess_one() cua GAN nhung voi "gan"=orig (khong co GAN o giua)
    blur = np.clip(M * cv2.bilateralFilter(orig, d=cfg.BILAT_D, sigmaColor=cfg.SIGMA_COLOR, sigmaSpace=cfg.SIGMA_SPACE).astype(np.float32)
                    + (1 - M) * orig.astype(np.float32), 0, 255).astype(np.uint8)
    out = {"inpaint_telea": telea, "inpaint_ns": ns, "blur_blend": blur}
    out_smooth = {}
    if cfg.POST_BG_SMOOTH:
        for name, im in out.items():
            sm, _ = bg_smooth_roi(im, cfg)
            out_smooth[name] = sm
    return core, art, M, out, out_smooth


def comparison_montage_classical(orig, core, M, out, out_smooth, panel_h=256):
    def pan(im, title, cmap=False):
        im = im if im.dtype == np.uint8 else np.clip(im * 255, 0, 255).astype(np.uint8)
        im = cv2.applyColorMap(im, cv2.COLORMAP_JET) if cmap else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        s = panel_h / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), panel_h))
        return _label(im, title)
    best = out_smooth.get("inpaint_telea", out["inpaint_telea"])
    diff = np.clip(cv2.absdiff(best, orig).astype(np.float32) * 4, 0, 255).astype(np.uint8)
    panels = [pan(orig, "1 original"), pan(core, "2 metal core"), pan(M, "3 mask M", True),
              pan(out["inpaint_telea"], "4 inpaint TELEA"), pan(out["inpaint_ns"], "5 inpaint NS"),
              pan(out["blur_blend"], "6 blur+blend")]
    if out_smooth:
        panels.append(pan(out_smooth["inpaint_telea"], "7 TELEA+bg smooth"))
    panels.append(pan(diff, "|best-orig| x4"))
    return np.hstack(panels)


def stage_classical(cfg, root: Path, run_dir: Path, state: dict) -> None:
    """V8: chay ca 3 bien the co dien tren CUNG dataset/split nhu GAN (CFG.POST_SPLITS), xuat anh +
    montage so sanh + 2 file so lieu: results/metrics.csv (wide, moi bien the 1 nhom cot, giu de xem
    nhanh/debug) va results/metrics_long.csv (schema dai CHUNG voi GAN - xem LONG_METRIC_COLS)."""
    out_root = run_dir / "results"
    cmp_root = out_root / "comparison"
    metrics_path = out_root / "metrics.csv"
    rows_wide, rows_long = [], []
    sat = lambda a: float((a >= cfg.THRESH).mean())
    for split in cfg.POST_SPLITS:
        files = list_images(root / "Glare" / split)
        if cfg.POST_MAX_IMAGES_PER_SPLIT:
            files = even_pick(files, cfg.POST_MAX_IMAGES_PER_SPLIT)
        cmp_files = set(p.name for p in even_pick(files, cfg.POST_COMPARISON_MAX))
        log(f"[classical] {split}: {len(files)} anh Glare")
        it = tqdm(files, desc=f"Classical {split}", unit="anh") if tqdm else files
        n_err = 0
        for p in it:
            done_marker = out_root / "inpaint_telea" / split / p.name
            if done_marker.exists() and not cfg.POST_OVERWRITE:
                continue
            try:
                orig = read_gray_uint8(p)
                t0 = time.perf_counter()
                core, art, M, out, out_smooth = process_one_classical(orig, cfg)
                inference_ms = (time.perf_counter() - t0) * 1000.0 / 3.0   # V8: chia deu cho 3 bien the (tinh cung 1 luot)
                for name, im in out.items():
                    write_image(out_root / name / split / p.name, im)
                for name, im in out_smooth.items():
                    write_image(out_root / f"{name}_smooth" / split / p.name, im)
                if p.name in cmp_files:
                    write_image(cmp_root / split / (p.stem + "_cmp.png"),
                               comparison_montage_classical(orig, core, M, out, out_smooth))
                keep = ~metal_region(orig, cfg.THRESH)
                halo_mask = (orig >= cfg.HALO_THRESH_U8) & (orig < cfg.THRESH)

                def halo_reduction(cand):
                    if halo_mask.sum() < 64:
                        return float("nan")
                    ma = float(orig[halo_mask].mean()); mb = float(cand[halo_mask].mean())
                    return max(0.0, 1.0 - mb / ma) if ma > 1e-6 else float("nan")

                def region_metrics(cand):
                    diff_f = orig.astype(np.float32) - cand.astype(np.float32)
                    mae_f = float(np.abs(diff_f).mean()); rmse_f = float(np.sqrt((diff_f ** 2).mean()))
                    if keep.sum() > 64:
                        _, smap = ssim_fn(orig, cand, data_range=255, full=True)
                        ssim_p = float(smap[keep].mean())
                        diff_p = diff_f[keep]
                        mse_p = float((diff_p ** 2).mean())
                        psnr_p = 99.0 if mse_p < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse_p))
                        mae_p = float(np.abs(diff_p).mean())
                        edge_p = edge_corr(orig, cand, keep)
                    else:
                        ssim_p = psnr_p = mae_p = edge_p = float("nan")
                    return mae_f, rmse_f, ssim_p, psnr_p, mae_p, edge_p

                row_wide = {"split": split, "file": p.name, "mask_area_pct": float((M > 0).mean() * 100), "sat_orig": sat(orig)}
                for name in out:
                    best = out_smooth.get(name, out[name])
                    mae_full, rmse_full, ssim_pres, psnr_pres, mae_pres, edge_pres = region_metrics(best)
                    row_wide[f"sat_{name}"] = sat(best)
                    row_wide[f"ssim_preservation_{name}"] = ssim_pres
                    row_wide[f"psnr_preservation_{name}"] = psnr_pres
                    row_wide[f"mae_preservation_{name}"] = mae_pres
                    row_wide[f"edge_corr_preservation_{name}"] = edge_pres
                    row_wide[f"halo_reduction_{name}"] = halo_reduction(best)
                    rows_long.append({
                        "split": split, "file": p.name, "method": f"Classical_{name}",
                        "mask_area_pct": float((M > 0).mean() * 100), "sat_orig": sat(orig),
                        "sat_final": sat(best), "ssim_preservation": ssim_pres, "psnr_preservation": psnr_pres,
                        "mae_preservation": mae_pres, "edge_corr_preservation": edge_pres,
                        "halo_reduction": halo_reduction(best), "mae_full": mae_full, "rmse_full": rmse_full,
                        "ssim_orig_vs_final": float(ssim_fn(orig, best, data_range=255)),
                        "psnr_orig_vs_final": _psnr_safe(orig, best), "inference_ms": inference_ms,
                    })
                rows_wide.append(row_wide)
            except Exception as e:
                n_err += 1
                log(f"  [LOI] {p.name}: {e}")
                if n_err <= 3:
                    log("    " + traceback.format_exc().strip().splitlines()[-1])
        if n_err:
            log(f"  {split}: {n_err} anh loi (xem log o tren).")
    if rows_wide:
        df_new = pd.DataFrame(rows_wide)
        if metrics_path.exists():
            df_new = pd.concat([pd.read_csv(metrics_path), df_new]).drop_duplicates(["split", "file"], keep="last")
        out_root.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(metrics_path, index=False)
        num_cols = [c for c in df_new.columns if c not in ("split", "file")]
        summ = df_new.groupby("split")[num_cols].mean().round(4)
        (out_root / "summary.txt").write_text(f"3 bien the co dien (khong GAN): inpaint_telea / inpaint_ns / blur_blend\n{summ.to_string()}\n",
                                               encoding="utf-8")
        log("[classical] Tom tat (trung binh moi split):\n" + summ.to_string())
        df_new.groupby("split")[num_cols].describe().to_csv(out_root / "postprocess_summary_stats.csv")
        write_metrics_long(rows_long, run_dir)
    state["classical_done"] = True
    log("[classical] Ket qua:", out_root)


# ==============================================================================
#  V8: GOP KET QUA TAT CA PHUONG PHAP (sau khi CFG.METHODS chay xong tuan tu)
# ==============================================================================
def aggregate_all_methods(cfg, run_dirs: dict) -> Path | None:
    """V8: doc results/metrics_long.csv cua MOI phuong phap da chay xong (run_dirs: {method: run_dir},
    method la "gan"/"classical" trong CFG.METHODS) - CUNG schema (LONG_METRIC_COLS) nen gop truc tiep
    duoc, khong can quy doi. Xuat: all_methods_metrics_long.csv (tho, moi dong = 1 anh x 1 bien the),
    leaderboard.csv (mean/std/count theo method x split), leaderboard_summary.txt (doc nhanh), va
    bieu do cot nhom allmethods_<metric>.png so sanh TAT CA bien the (GAN_final/GAN_hybrid/
    Classical_inpaint_telea/...) tren cung 1 truc - day la thu muc duy nhat can xem de viet paper."""
    frames = []
    for method, run_dir in run_dirs.items():
        p = run_dir / "results" / "metrics_long.csv"
        if not p.exists():
            log(f"  [CANH BAO] Khong tim thay {p} -> bo qua phuong phap '{method}' trong bang tong hop.")
            continue
        df = pd.read_csv(p)
        df["run_method"] = method
        df["run_dir"] = str(run_dir)
        frames.append(df)
    if not frames:
        log("[CANH BAO] Khong co ket qua nao de gop (chua co results/metrics_long.csv o run nao).")
        return None
    all_df = pd.concat(frames, ignore_index=True)

    root = Path(cfg.OUTPUT_ROOT) if cfg.OUTPUT_ROOT else SCRIPT_PATH.parent / "runs"
    out_dir = root / f"ALL_METHODS_comparison_{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "all_methods_metrics_long.csv", index=False)

    num_cols = [c for c in LONG_METRIC_COLS if c in all_df]
    leaderboard = all_df.groupby(["method", "split"])[num_cols].agg(["mean", "std", "count"])
    leaderboard.to_csv(out_dir / "leaderboard.csv")
    mean_only = all_df.groupby(["method", "split"])[num_cols].mean().round(4)
    (out_dir / "leaderboard_summary.txt").write_text(mean_only.to_string(), encoding="utf-8")
    log("=" * 78)
    log("BANG XEP HANG TAT CA PHUONG PHAP (trung binh moi split):\n" + mean_only.to_string())

    key_metrics = [("halo_reduction", "Halo/streak reduction (key metric, higher=better)"),
                   ("ssim_preservation", "SSIM (non-metal region)"),
                   ("psnr_preservation", "PSNR (non-metal region, dB)"),
                   ("mae_preservation", "MAE (non-metal region, gray level, lower=better)"),
                   ("edge_corr_preservation", "Edge-preservation (gradient corr.)"),
                   ("sat_final", "Saturated-pixel fraction after processing (lower=better)")]
    methods_here = sorted(all_df["method"].unique())
    splits_here = [s for s in ("Train", "Val", "Test") if s in set(all_df["split"])] or sorted(all_df["split"].unique())
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    n_ok = 0
    for col, title in key_metrics:
        if col not in all_df:
            continue
        piv = all_df.groupby(["split", "method"])[col].mean().unstack("method").reindex(splits_here)
        piv_std = all_df.groupby(["split", "method"])[col].std().unstack("method").reindex(splits_here)
        if piv.isna().all(axis=None):
            continue
        x = np.arange(len(splits_here)); n_m = max(1, len(methods_here)); w = 0.8 / n_m
        fig, ax = plt.subplots(figsize=(max(5.5, 1.6 * n_m + 3), 4.2))
        for i, m in enumerate(methods_here):
            if m not in piv.columns:
                continue
            vals = piv[m].values
            errs = piv_std[m].values if m in piv_std else None
            ax.bar(x + (i - (n_m - 1) / 2) * w, vals, w, yerr=errs, capsize=3, label=m)
        ax.set_xticks(x); ax.set_xticklabels(splits_here)
        ax.set_ylabel(title); ax.set_title(f"Method comparison - {title}")
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"allmethods_{col}.png", dpi=cfg.PAPER_DPI)
        plt.close(fig)
        n_ok += 1
    log(f"  Da xuat {n_ok} bieu do so sanh tat ca phuong phap vao {fig_dir}")
    log("GOP KET QUA XONG:", out_dir, "<- dung thu muc nay de viet paper.")
    return out_dir


# ==============================================================================
#  QUAN LY RUN (thu muc theo ten file + gio + ngay, tu tiep tuc)
# ==============================================================================
def pick_run_dir(cfg, method: str = ""):
    root = Path(cfg.OUTPUT_ROOT) if cfg.OUTPUT_ROOT else SCRIPT_PATH.parent / "runs"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{RUN_PREFIX}_{method}" if method else RUN_PREFIX   # V8: 1 thu muc run RIENG cho tung phuong phap
    mode = str(cfg.RESUME)
    if mode not in ("auto", "new"):
        d = Path(mode)
        if not d.is_dir():
            raise FileNotFoundError(f"CFG.RESUME tro toi thu muc khong ton tai: {d}")
        return d, True
    if mode == "auto":
        cands = [d for d in root.glob(f"{prefix}_*") if (d / "run_state.json").exists()]
        for d in sorted(cands, key=lambda x: (x / "run_state.json").stat().st_mtime, reverse=True):
            st = json.load(open(d / "run_state.json", encoding="utf-8"))
            if not st.get("finished"):
                return d, True
    d = root / f"{prefix}_{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}"
    d.mkdir(parents=True)
    return d, False


def main():
    cfg = CFG
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    # V8: chay TUAN TU tung phuong phap trong CFG.METHODS (vd ["gan", "classical"]) - MOI phuong
    # phap 1 thu muc run rieng (xem pick_run_dir), tu resume rieng neu bi ngat giua chung; phuong
    # phap sau CHI bat dau khi phuong phap truoc da "finished" trong vong lap nay (khong chay cung luc -
    # dung cho 1 GPU laptop). Sau khi het danh sach -> tu dong gop ket qua (aggregate_all_methods).
    run_dirs = {}
    for method in cfg.METHODS:
        run_dir, resumed = pick_run_dir(cfg, method)
        log.open(run_dir / "log.txt")
        log("=" * 78)
        log(f"[{method}] {'TIEP TUC' if resumed else 'RUN MOI'}: {run_dir}")
        log(f"Python {sys.version.split()[0]} | torch {torch.__version__} | cuda={torch.cuda.is_available()}")
        st_path = run_dir / "run_state.json"
        state = json.load(open(st_path, encoding="utf-8")) if st_path.exists() else {}
        state.setdefault("created", datetime.now().isoformat(timespec="seconds"))
        state["method"] = method
        state["finished"] = False
        atomic_json(st_path, state)
        atomic_json(run_dir / "config_snapshot.json", cfg_snapshot())
        shutil.copy2(SCRIPT_PATH, run_dir / SCRIPT_PATH.name)      # luu ban script da dung cho run nay

        try:
            # "classical" khong train nhung van can dataset -> luon build/kiem tra du lieu cho no,
            # bat ke CFG.RUN_TRAIN/RUN_POSTPROCESS (2 co nay chi danh rieng cho phuong phap "gan").
            need_data = cfg.RUN_BUILD_DATA or cfg.RUN_TRAIN or cfg.RUN_POSTPROCESS or method != "gan"
            root = stage_data(cfg, run_dir) if need_data else Path(cfg.DATASET_DIR)
            if method == "gan":
                if cfg.RUN_TRAIN:
                    stage_train(cfg, root, run_dir, state)
                    atomic_json(st_path, state)
                if cfg.RUN_POSTPROCESS:
                    if cfg.RUN_TRAIN and not state.get("train_done"):
                        log("Train chua xong -> bo qua hau xu ly.")
                    else:
                        stage_post(cfg, root, run_dir, state)
            elif method == "classical":
                stage_classical(cfg, root, run_dir, state)
            else:
                raise ValueError(f"Phuong phap khong ho tro: {method!r} (them xu ly cho no hoac sua CFG.METHODS)")
            state["finished"] = True
            atomic_json(st_path, state)
            log(f"[{method}] HOAN TAT. Ket qua nam trong:", run_dir)
            run_dirs[method] = run_dir
        except KeyboardInterrupt:
            atomic_json(st_path, state)
            log(f"[{method}] Da dung. Chay lai `python", SCRIPT_PATH.name,
                "` de tiep tuc dung phuong phap nay (cac phuong phap da HOAN TAT truoc do se duoc bo qua, khong chay lai).")
            sys.exit(130)

    if run_dirs:
        try:
            aggregate_all_methods(cfg, run_dirs)
        except Exception as e:
            log("[CANH BAO] Loi khi gop ket qua cac phuong phap (tung phuong phap van con ket qua rieng, xem tung run_dir):", e)
            log(traceback.format_exc())
    log("=" * 78)
    log("TAT CA PHUONG PHAP HOAN TAT:", ", ".join(cfg.METHODS), "- xem ALL_METHODS_comparison_*/ de viet paper.")


if __name__ == "__main__":
    main()
