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

 Moi lan chay tao 1 thu muc:   <OUTPUT_ROOT>/train_dental_mar_HH-MM-SS_DD-MM-YYYY/
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
    TOOTHFAIRY_DIR = r"E:\Project\Dental\ToothFairy_dataset\ToothFairy_Dataset"   # co splits.json + Dataset/P*/data.npy
    # Thu muc dataset dang Glare/Normal x Train/Val/Test. Neu chua co -> tu tao tu ToothFairy.
    # (co the tro thang vao dataset CBCT_MAR goc: r"E:\Project\Dental\Dataset")
    DATASET_DIR = r"E:\Project\Dental\Dataset_TF"
    OUTPUT_ROOT = None           # None = <thu muc chua file nay>/runs
    FORCE_REBUILD = False        # True: xoa & tao lai DATASET_DIR (chi xoa neu do script nay tao ra)

    # Tiep tuc sau khi bi ngat:
    #   "auto" : tim run chua xong moi nhat trong OUTPUT_ROOT va tiep tuc, neu khong co thi tao run moi
    #   "new"  : luon tao run moi (bo qua run dang do)
    #   duong dan thu muc run cu (vd r"E:\...\runs\train_dental_mar_14-02-11_21-09-2026") de tiep tuc dung run do
    RESUME = "auto"

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
    BATCH_SIZE = 4
    # Windows: moi worker phai nap lai torch+CUDA (rat ton RAM/pagefile) -> de 0 cho an toan (anh nho, GPU moi la nut that).
    # Neu muon nhanh hon: tang pagefile Windows roi thu 2. (Val loader luon dung 0 worker.)
    NUM_WORKERS = 0 if os.name == "nt" else 4
    EPOCHS = 25                     # V3: giam tu 100 -> 25. Log thuc te cho thay sau epoch 5 diem so khong con vuot duoc nua,
                                       # train lau hon chi lam attention mask lan rong qua vung khong co kim loai (xem anh so sanh ep5 vs ep100).
    LR = 2e-4
    BETAS = (0.5, 0.999)
    DECAY_START_EPOCH = None          # None = EPOCHS//2 ; sau moc nay LR giam tuyen tinh ve 0
    GAN_LOSS = "lsgan"                # "lsgan" (on dinh hon) hoac "bce" (nhu ban goc)
    LAMBDA_CYCLE = 5.0                # V3: giam tu 10.0. Ep giu nguyen chu ky qua manh la mot nguyen nhan nghi ngo
                                       # khien sat_red dao dong manh giua cac epoch (xem lich su Ep012..Ep100 trong log).
    LAMBDA_IDENTITY = 2.0             # V3: giam tu 5.0, cung ly do tren - de generator "dam" sua vung nhieu hon.
    LAMBDA_TV = 0.1                   # Phuong phap 2 - TV loss
    LAMBDA_MASK = 2.0                 # V3: BAT len (truoc la 0.0 = tat). Ep attention mask CHI duoc sang len o vung
                                       # kim loai that (nguong METAL_THRESH_U8, gian no ~10px) va phai = 0 o noi khac.
                                       # Day la ly do chinh khien epoch 100 lan mask ra ca vung xuong ham khong co nhieu.
    IMAGE_POOL_SIZE = 50              # bo dem anh gia cho discriminator (CycleGAN chuan). 0 = tat
    AUG_ROTATE_DEG = 10               # augmentation: lat ngang + xoay +-10 do
    AUG_SCALE = 0.05
    TRAIN_ON_ROI = False              # True: cat ROI (Otsu 3 lop) tren moi anh truoc khi dua vao mang

    # Kiem dinh (validation) moi epoch
    VAL_MAX_IMAGES = 200              # toi da so anh Glare/Normal dung cho validation
    METAL_THRESH_U8 = 240             # nguong "sang bao hoa" (giong THRESH cua hau xu ly)
    # diem chon best.pth = W_SAT*(giam pixel bao hoa) + W_PRES*(SSIM giu nguyen vung khong nhieu) + W_ID*(SSIM identity)
    SCORE_WEIGHTS = {"sat": 0.5, "pres": 0.3, "id": 0.2}

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
    # Giai doan 2: bilateral
    BILAT_D = 9
    SIGMA_COLOR = 75
    SIGMA_SPACE = 75
    # Buoc lam min nen theo ROI (cell 7 cua notebook)
    POST_BG_SMOOTH = True
    OTSU_OFFSET = 0
    ROI_CLOSE_KERNEL = 9
    ROI_DILATE_ITERS = 1
    ROI_MIN_AREA_RATIO = 0.001
    BACKGROUND_VALUE = 0
    ROI_MASK_BLUR = 11
    BG_NLM_H = 15
    BG_GAUSS_KSIZE = 7


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
        tag = f"axis={ax} shape={vol.shape}" + ("  <- CHON" if ax == (cfg.SLICE_AXIS if cfg.SLICE_AXIS is not None else auto_axis) else "")
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
    def __init__(self, folder, size, train, rot_deg, scale_aug, use_roi):
        self.paths = list_images(folder)
        if not self.paths:
            raise RuntimeError(f"Khong co anh trong {folder}")
        self.size, self.train, self.rot, self.scl, self.roi = size, train, rot_deg, scale_aug, use_roi

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = read_gray_uint8(self.paths[i])
        if self.roi:
            img = extract_roi(img)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA if max(img.shape) > self.size else cv2.INTER_LINEAR)
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
            ang = random.uniform(-self.rot, self.rot)
            sc = 1.0 + random.uniform(-self.scl, self.scl)
            if abs(ang) > 1e-3 or abs(sc - 1) > 1e-3:
                M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, sc)
                img = cv2.warpAffine(np.ascontiguousarray(img), M, (self.size, self.size),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(127.5).sub_(1.0).unsqueeze(0)
        return t


def make_loader(folder, cfg, train, batch=None, shuffle=None, max_items=None):
    ds = SliceDataset(folder, cfg.IMG_SIZE, train, cfg.AUG_ROTATE_DEG, cfg.AUG_SCALE, cfg.TRAIN_ON_ROI)
    if max_items and len(ds) > max_items:
        ds.paths = even_pick(ds.paths, max_items)
    bs = batch or cfg.BATCH_SIZE
    shuffle = train if shuffle is None else shuffle
    nw = cfg.NUM_WORKERS if train else 0
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      drop_last=bool(train and len(ds) >= 2 * bs),
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0)


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
        self.dec1 = nn.ConvTranspose2d(512, 512, 4, 2, 1);   self.norm_d1 = nn.InstanceNorm2d(512)
        self.dec2 = nn.ConvTranspose2d(1024, 256, 4, 2, 1);  self.norm_d2 = nn.InstanceNorm2d(256)
        self.dec3 = nn.ConvTranspose2d(512, 128, 4, 2, 1);   self.norm_d3 = nn.InstanceNorm2d(128)
        self.dec4 = nn.ConvTranspose2d(256, 64, 4, 2, 1);    self.norm_d4 = nn.InstanceNorm2d(64)
        self.dec5 = nn.ConvTranspose2d(128, 64, 4, 2, 1);    self.norm_d5 = nn.InstanceNorm2d(64)
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


def total_variation_loss(img):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (b * c * h * w)


def gan_criterion(kind):
    if kind == "bce":
        bce = nn.BCEWithLogitsLoss()
        return lambda logit, real: bce(logit, torch.full_like(logit, 1.0 if real else 0.0))
    mse = nn.MSELoss()
    return lambda logit, real: mse(logit, torch.full_like(logit, 1.0 if real else 0.0))


# ==============================================================================
#  KIEM DINH (VALIDATION)
# ==============================================================================
def t2u8(t):
    """(B,1,H,W) [-1,1] -> list[np.uint8 HxW]"""
    a = ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5 + 0.5).numpy().astype(np.uint8)
    return [x[0] for x in a]


def metal_region(u8, thr):
    core = (u8 >= thr).astype(np.uint8) * 255
    return cv2.dilate(core, np.ones((7, 7), np.uint8), iterations=3) > 0


@torch.no_grad()
def evaluate(gen_G, gen_F, val_glare, val_normal, cfg, dev):
    gen_G.eval(); gen_F.eval()
    cyc, id_ssim, id_psnr, pres_ssim, pres_psnr, sat_red, att_g, att_n = ([] for _ in range(8))
    thr = cfg.METAL_THRESH_U8

    for rg in val_glare:
        rg = rg.to(dev)
        fake_n, m = gen_G(rg)
        rec_g, _ = gen_F(fake_n)
        cyc.append(F.l1_loss(rec_g, rg).item())
        for a, b, mm in zip(t2u8(rg), t2u8(fake_n), m.cpu().numpy()):
            att_g.append(float(mm.mean()))
            reg = metal_region(a, thr)
            keep = ~reg
            sb = float((a >= thr).mean()); sa = float((b >= thr).mean())
            if sb > 0:
                sat_red.append(max(0.0, 1.0 - sa / sb))
            if keep.sum() > 64:
                _, smap = ssim_fn(a, b, data_range=255, full=True)
                pres_ssim.append(float(smap[keep].mean()))
                mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2)[keep].mean())
                pres_psnr.append(99.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse))

    for rn in val_normal:
        rn = rn.to(dev)
        idn, m = gen_G(rn)
        fake_g, _ = gen_F(rn)
        rec_n, _ = gen_G(fake_g)
        cyc.append(F.l1_loss(rec_n, rn).item())
        for a, b, mm in zip(t2u8(rn), t2u8(idn), m.cpu().numpy()):
            att_n.append(float(mm.mean()))
            id_ssim.append(float(ssim_fn(a, b, data_range=255)))
            id_psnr.append(float(psnr_fn(a, b, data_range=255)))

    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    out = {
        "val_cycle": mean(cyc), "val_id_ssim": mean(id_ssim), "val_id_psnr": mean(id_psnr),
        "val_pres_ssim": mean(pres_ssim), "val_pres_psnr": mean(pres_psnr),
        "val_sat_reduction": mean(sat_red), "val_att_glare": mean(att_g), "val_att_normal": mean(att_n),
    }
    w = cfg.SCORE_WEIGHTS
    parts = [(w["sat"], out["val_sat_reduction"]), (w["pres"], out["val_pres_ssim"]), (w["id"], out["val_id_ssim"])]
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
    draw(ax[1, 0], ["tv", "mask_loss"], "TV / Mask loss")
    draw(ax[1, 1], ["val_sat_reduction", "val_pres_ssim", "val_id_ssim", "val_score"], "Validation (cao = tot)")
    draw(ax[1, 2], ["val_att_glare", "val_att_normal", "val_cycle"], "Attention area / val cycle")
    for a in ax.ravel():
        a.set_xlabel("epoch")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


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
    train_g = make_loader(root / "Glare" / "Train", cfg, True)
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
    opt_G = torch.optim.Adam(itertools.chain(gen_G.parameters(), gen_F.parameters()), lr=cfg.LR, betas=cfg.BETAS)
    opt_D = torch.optim.Adam(itertools.chain(disc_X.parameters(), disc_Y.parameters()), lr=cfg.LR, betas=cfg.BETAS)
    decay_start = cfg.DECAY_START_EPOCH if cfg.DECAY_START_EPOCH is not None else cfg.EPOCHS // 2
    lr_lambda = lambda e: 1.0 - max(0, e - decay_start) / float(max(1, cfg.EPOCHS - decay_start + 1))
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    history, best = [], {"score": -1e9, "epoch": 0}
    start_epoch = 1
    ck, ck_path = load_last(ck_dir, dev)
    if ck is not None:
        gen_G.load_state_dict(ck["G"]); gen_F.load_state_dict(ck["F"])
        disc_X.load_state_dict(ck["DX"]); disc_Y.load_state_dict(ck["DY"])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        sch_G.load_state_dict(ck["sch_G"]); sch_D.load_state_dict(ck["sch_D"])
        history, best = ck["history"], ck["best"]
        start_epoch = ck["epoch"] + 1
        set_rng_state(ck["rng"])
        log(f"==> TIEP TUC tu {ck_path.name}: da xong epoch {ck['epoch']}, se chay tu epoch {start_epoch}/{cfg.EPOCHS}")
    else:
        log("Bat dau huan luyen tu dau.")
    if start_epoch > cfg.EPOCHS:
        log("Da huan luyen du so epoch -> bo qua.")
        state["train_done"] = True
        return

    adv = gan_criterion(cfg.GAN_LOSS)
    l1 = nn.L1Loss()
    pool_n, pool_g = ImagePool(cfg.IMAGE_POOL_SIZE), ImagePool(cfg.IMAGE_POOL_SIZE)
    thr_t = 2.0 * cfg.METAL_THRESH_U8 / 255.0 - 1.0
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
            acc = {k: 0.0 for k in ["D_X", "D_Y", "G_total", "G_adv", "cycle", "identity", "tv", "mask_loss", "att_glare"]}
            nb = 0
            loop = tqdm(train_g, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=False, unit="it") if tqdm else train_g
            for rg in loop:
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
                loss_tv = total_variation_loss(fake_n) + total_variation_loss(fake_g)
                loss_mask = torch.zeros((), device=dev)
                if cfg.LAMBDA_MASK > 0:
                    target = F.max_pool2d((rg >= thr_t).float(), 21, 1, 10)
                    loss_mask = l1(mask_n, target) + l1(mask_id, torch.zeros_like(mask_id))
                total_G = (loss_adv + cfg.LAMBDA_CYCLE * loss_cyc + cfg.LAMBDA_IDENTITY * loss_id
                           + cfg.LAMBDA_TV * loss_tv + cfg.LAMBDA_MASK * loss_mask)
                total_G.backward()
                opt_G.step()

                # ---------- Discriminators ----------
                for d in (disc_X, disc_Y):
                    for p in d.parameters():
                        p.requires_grad_(True)
                opt_D.zero_grad(set_to_none=True)
                fn_pool, fg_pool = pool_n.query(fake_n), pool_g.query(fake_g)
                loss_DX = 0.5 * (adv(disc_X(rn), True) + adv(disc_X(fn_pool.detach()), False))
                loss_DY = 0.5 * (adv(disc_Y(rg), True) + adv(disc_Y(fg_pool.detach()), False))
                (loss_DX + loss_DY).backward()
                opt_D.step()

                if not torch.isfinite(total_G):
                    raise FloatingPointError("Loss = NaN/Inf. Giam CFG.LR hoac tang CFG.LAMBDA_* roi chay lai (se tiep tuc tu checkpoint).")

                acc["D_X"] += loss_DX.item(); acc["D_Y"] += loss_DY.item()
                acc["G_total"] += total_G.item(); acc["G_adv"] += loss_adv.item()
                acc["cycle"] += loss_cyc.item(); acc["identity"] += loss_id.item()
                acc["tv"] += loss_tv.item(); acc["mask_loss"] += float(loss_mask.detach())
                acc["att_glare"] += mask_n.mean().item()
                nb += 1
                if tqdm:
                    loop.set_postfix(G=f"{total_G.item():.2f}", DX=f"{loss_DX.item():.2f}", DY=f"{loss_DY.item():.2f}")

            row = {k: v / max(nb, 1) for k, v in acc.items()}
            row.update(evaluate(gen_G, gen_F, val_g, val_n, cfg, dev))
            sch_G.step(); sch_D.step()
            row["epoch"] = epoch
            row["lr"] = opt_G.param_groups[0]["lr"]
            row["epoch_sec"] = time.time() - t0
            history.append(row)
            epoch_times.append(row["epoch_sec"])

            is_best = row["val_score"] > best["score"]
            if is_best:
                best = {"score": row["val_score"], "epoch": epoch}
                atomic_torch_save({"gen_G": gen_G.state_dict(), "gen_F": gen_F.state_dict(),
                                   "epoch": epoch, "metrics": row}, ck_dir / "best.pth")

            # ---- checkpoint moi epoch (tiep tuc duoc neu bi ngat)
            save_last(ck_dir / "last.pth", {
                "epoch": epoch, "G": gen_G.state_dict(), "F": gen_F.state_dict(),
                "DX": disc_X.state_dict(), "DY": disc_Y.state_dict(),
                "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                "sch_G": sch_G.state_dict(), "sch_D": sch_D.state_dict(),
                "history": history, "best": best, "rng": rng_state(), "cfg": cfg_snapshot()})
            if cfg.SAVE_EVERY and epoch % cfg.SAVE_EVERY == 0:
                atomic_torch_save(gen_G.state_dict(), ck_dir / f"gen_G_epoch_{epoch:03d}.pth")
                atomic_torch_save(gen_F.state_dict(), ck_dir / f"gen_F_epoch_{epoch:03d}.pth")
                prune_epoch_files(ck_dir, cfg.KEEP_LAST_N_EPOCH_FILES)

            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            if prev_batch is not None and epoch % cfg.PREVIEW_EVERY == 0:
                save_progress_image(gen_G, prev_batch, prev_tags, epoch, prog_dir / f"epoch_{epoch:03d}.png", dev)
            if epoch % cfg.PLOT_EVERY == 0 or epoch == cfg.EPOCHS:
                plot_history(history, run_dir / "loss_curves.png")
            if writer:
                for k, v in row.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        writer.add_scalar(f"train/{k}", v, epoch)
                if prev_batch is not None and epoch % cfg.TB_IMAGE_EVERY == 0:
                    with torch.no_grad():
                        o, m = gen_G(prev_batch.to(dev))
                    writer.add_images("progress/input", (prev_batch + 1) / 2, epoch)
                    writer.add_images("progress/output", ((o.cpu() + 1) / 2).clamp(0, 1), epoch)
                    writer.add_images("progress/mask", m.cpu(), epoch)

            eta = np.mean(epoch_times[-5:]) * (cfg.EPOCHS - epoch)
            log(f"Ep {epoch:03d}/{cfg.EPOCHS} | G {row['G_total']:.3f} DX {row['D_X']:.3f} DY {row['D_Y']:.3f} "
                f"cyc {row['cycle']:.3f} tv {row['tv']:.4f} | val: sat_red {row['val_sat_reduction']:.3f} "
                f"pres_ssim {row['val_pres_ssim']:.3f} id_ssim {row['val_id_ssim']:.3f} score {row['val_score']:.3f}"
                f"{' *BEST*' if is_best else ''} | {row['epoch_sec']:.0f}s, con ~{eta/60:.0f} phut")
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

    atomic_torch_save({"gen_G": gen_G.state_dict(), "gen_F": gen_F.state_dict(), "epoch": cfg.EPOCHS},
                      ck_dir / "final_weights.pth")
    plot_history(history, run_dir / "loss_curves.png")
    if writer:
        writer.close()
    state["train_done"] = True
    log(f"Huan luyen xong. Best epoch = {best['epoch']} (score {best['score']:.4f}). Tong {(time.time() - t_run) / 3600:.2f} gio (lan chay nay).")


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


def make_artifact_mask(orig, cfg):
    _, core = cv2.threshold(orig, cfg.THRESH, 255, cv2.THRESH_BINARY)                                # 1.1
    art = cv2.dilate(core, np.ones((cfg.KERNEL_SIZE, cfg.KERNEL_SIZE), np.uint8), iterations=cfg.DILATE_ITERS)  # 1.2
    blurred = cv2.GaussianBlur(art, (cfg.BLUR_KSIZE, cfg.BLUR_KSIZE), 0)                            # 1.3
    return core, art, blurred.astype(np.float32) / 255.0


def postprocess_one(orig, gan, cfg):
    core, art, M = make_artifact_mask(orig, cfg)
    gan_s = cv2.bilateralFilter(gan, d=cfg.BILAT_D, sigmaColor=cfg.SIGMA_COLOR, sigmaSpace=cfg.SIGMA_SPACE)  # Giai doan 2
    final = np.clip(M * gan_s.astype(np.float32) + (1 - M) * orig.astype(np.float32), 0, 255).astype(np.uint8)  # Giai doan 3
    res = {"original": orig, "gan": gan, "core": core, "M": M, "gan_smoothed": gan_s, "final": final}
    if cfg.POST_BG_SMOOTH:
        try:
            roi_mask, _, _ = extract_roi_mask(final, cfg.OTSU_OFFSET, cfg.ROI_CLOSE_KERNEL, cfg.ROI_DILATE_ITERS,
                                              cfg.ROI_MIN_AREA_RATIO, cfg.BACKGROUND_VALUE, cfg.ROI_MASK_BLUR)
            bg = cv2.fastNlMeansDenoising(final, None, h=cfg.BG_NLM_H, templateWindowSize=7, searchWindowSize=21) if cfg.BG_NLM_H > 0 else final.copy()
            if cfg.BG_GAUSS_KSIZE > 1:
                bg = cv2.GaussianBlur(bg, (cfg.BG_GAUSS_KSIZE, cfg.BG_GAUSS_KSIZE), 0)
            w = roi_mask.astype(np.float32) / 255.0
            res["final_smooth"] = np.clip(w * final + (1 - w) * bg, 0, 255).astype(np.uint8)
            res["roi_mask"] = roi_mask
        except Exception as e:
            res["final_smooth"] = final
            log(f"  (ROI loi -> giu ket qua blending: {e})")
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
    panels = [pan(res["original"], "1 original"), pan(res["core"], "2 core mask"), pan(res["M"], "3 mask M", True),
              pan(res["gan"], "4 GAN raw"), pan(res["gan_smoothed"], "5 GAN+bilateral"), pan(res["final"], "6 blended")]
    if "final_smooth" in res:
        panels.append(pan(res["final_smooth"], "7 +bg smooth"))
    panels.append(pan(diff, "|final-orig| x4"))
    return np.hstack(panels)


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
            gan, att = run_gan(gen_G, orig, cfg, dev)
            res = postprocess_one(orig, gan, cfg)
            write_image(out_root / "gan_raw" / split / p.name, gan)
            write_image(out_root / "attention_mask" / split / p.name, att)
            write_image(final_p, res["final"])
            if "final_smooth" in res:
                write_image(out_root / "final_smooth" / split / p.name, res["final_smooth"])
            if p.name in cmp_files:
                write_image(out_root / "comparison" / split / (p.stem + "_cmp.png"), comparison_montage(res, p.name))
            sat = lambda a: float((a >= cfg.THRESH).mean())
            best_final = res.get("final_smooth", res["final"])
            rows.append({
                "split": split, "file": p.name, "mask_area_pct": float((res["M"] > 0).mean() * 100),
                "sat_orig": sat(orig), "sat_gan": sat(gan), "sat_final": sat(res["final"]),
                "ssim_orig_vs_final": float(ssim_fn(orig, best_final, data_range=255)),
                "psnr_orig_vs_final": float(psnr_fn(orig, best_final, data_range=255)),
            })
    if rows:
        df_new = pd.DataFrame(rows)
        if metrics_path.exists():
            df_new = pd.concat([pd.read_csv(metrics_path), df_new]).drop_duplicates(["split", "file"], keep="last")
        out_root.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(metrics_path, index=False)
        summ = df_new.groupby("split")[["mask_area_pct", "sat_orig", "sat_gan", "sat_final",
                                        "ssim_orig_vs_final", "psnr_orig_vs_final"]].mean().round(4)
        (out_root / "summary.txt").write_text(f"weights: {wname}\n{summ.to_string()}\n", encoding="utf-8")
        log("Tom tat hau xu ly:\n" + summ.to_string())
    state["post_done"] = True
    log("Anh ket qua:", out_root)


# ==============================================================================
#  QUAN LY RUN (thu muc theo ten file + gio + ngay, tu tiep tuc)
# ==============================================================================
def pick_run_dir(cfg):
    root = Path(cfg.OUTPUT_ROOT) if cfg.OUTPUT_ROOT else SCRIPT_PATH.parent / "runs"
    root.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.RESUME)
    if mode not in ("auto", "new"):
        d = Path(mode)
        if not d.is_dir():
            raise FileNotFoundError(f"CFG.RESUME tro toi thu muc khong ton tai: {d}")
        return d, True
    if mode == "auto":
        cands = [d for d in root.glob(f"{RUN_PREFIX}_*") if (d / "run_state.json").exists()]
        for d in sorted(cands, key=lambda x: (x / "run_state.json").stat().st_mtime, reverse=True):
            st = json.load(open(d / "run_state.json", encoding="utf-8"))
            if not st.get("finished"):
                return d, True
    d = root / f"{RUN_PREFIX}_{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}"
    d.mkdir(parents=True)
    return d, False


def main():
    cfg = CFG
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    run_dir, resumed = pick_run_dir(cfg)
    log.open(run_dir / "log.txt")
    log("=" * 78)
    log(f"{'TIEP TUC' if resumed else 'RUN MOI'}: {run_dir}")
    log(f"Python {sys.version.split()[0]} | torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    st_path = run_dir / "run_state.json"
    state = json.load(open(st_path, encoding="utf-8")) if st_path.exists() else {}
    state.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    state["finished"] = False
    atomic_json(st_path, state)
    atomic_json(run_dir / "config_snapshot.json", cfg_snapshot())
    shutil.copy2(SCRIPT_PATH, run_dir / SCRIPT_PATH.name)      # luu ban script da dung cho run nay

    try:
        root = stage_data(cfg, run_dir) if (cfg.RUN_BUILD_DATA or cfg.RUN_TRAIN or cfg.RUN_POSTPROCESS) else Path(cfg.DATASET_DIR)
        if cfg.RUN_TRAIN:
            stage_train(cfg, root, run_dir, state)
            atomic_json(st_path, state)
        if cfg.RUN_POSTPROCESS:
            if cfg.RUN_TRAIN and not state.get("train_done"):
                log("Train chua xong -> bo qua hau xu ly.")
            else:
                stage_post(cfg, root, run_dir, state)
        state["finished"] = True
        atomic_json(st_path, state)
        log("HOAN TAT. Toan bo ket qua nam trong:", run_dir)
    except KeyboardInterrupt:
        atomic_json(st_path, state)
        log("Da dung. Chay lai `python", SCRIPT_PATH.name, "` de tiep tuc.")
        sys.exit(130)


if __name__ == "__main__":
    main()
