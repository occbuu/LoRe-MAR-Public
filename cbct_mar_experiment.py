# -*- coding: utf-8 -*-
"""
==============================================================================
 cbct_mar_experiment.py - MOT SCRIPT DUY NHAT chay TOAN BO thi nghiem CBCT-MAR, xuat vao 1 thu muc results/
==============================================================================
 Phuong phap so sanh:
   ours_*    : PHUONG PHAP CUA BAN (V12) = YOLOv11/YOLOv12-seg tach kim loai + rang, GAN (V12) sua vung xam/den va
               quang sang quanh kim loai, giu nguyen xuong/rang, khu nhieu nhe. Luoi sieu tham so (YOLO model/epochs/
               imgsz/lr0 x nguong conf x ban kinh cuc bo x che do khu nhieu) -> nhieu ket qua de chon.
               ours_*_heur_* = cung pipeline nhung mask kim loai/rang bang luat (khong YOLO) - ablation.
   cbct_mar  : Baseline 1 - CBCT_MAR (github.com/maheshi81/CBCT_MAR): CycleGAN xu ly TOAN ANH, port nguyen ban
               (kien truc, BCE, lambda 18/15, batch 1, lr 2e-4, Resize 256); chi khac so epoch (cau hinh duoc).
   nmar      : Baseline 2 - NMAR (Meyer 2010): tach kim loai -> chieu Radon -> chuan hoa bang anh prior -> noi suy
               vet kim loai trong sinogram -> tai tao. Cung y tuong "tach kim loai + sua cuc bo" nhu phuong phap cua ban.
   limar     : Baseline 3 - LI-MAR (Kalender 1987): nhu NMAR nhung noi suy tuyen tinh truc tiep (khong prior).
 Du lieu danh gia (moi phuong phap chay tren TAT CA):
   tf_real_val / tf_synth_val : ToothFairy Val (that / nhieu tu tao co dap an) - CHI dung de CHON cau hinh tot nhat.
   tf_real_test / tf_synth_test: ToothFairy Test.
   cvan_synth / cvan_real      : CBCT CVan (doc lai DICOM, chuan hoa cuong do ve mien ToothFairy): lat sach + nhieu tu tao
                                 bang Tao_nhieu_sang.py (co dap an) / lat co kim loai that.
   truong_synth / truong_real  : CBCT Truong (DICOM 12-bit, chi 38 khoi co ten trung Dataset_AI/CBCT_Truong): cua so 0.5-99.5% cua khoi;
                                 lat sach + nhieu tu tao / lat co khoi voxel bao hoa (>=3069 HU) lon.
 Tai nguyen: bo dieu phoi tu do CPU / RAM / VRAM, chay SONG SONG cac buoc (GPU: GAN + CBCT_MAR + YOLO cung luc neu du
 VRAM; CPU: NMAR/LI-MAR, tao du lieu, hau xu ly, metrics) de may chay het cong suat nhung chua ~8% VRAM + 1 nhan
 + ~1.2GB RAM de khong tran. Moi buoc chay trong 1 tien trinh con, co log/progress rieng, TU TIEP TUC khi chay lai.

 Chay (moi truong anlk):
   conda activate /workspace/projects/envs/anlk
   cd /workspace/projects/CBCT/src
   nohup python cbct_mar_experiment.py run > ../results/nohup.log 2>&1 &     # chay toan bo (nen chay nen)
   python cbct_mar_experiment.py status [--follow]                            # theo doi tien trinh
   python cbct_mar_experiment.py grid                                         # xem luoi cau hinh + cac buoc
   python cbct_mar_experiment.py report [--ours <ten_method>]                 # lam lai bang/compie/report
   python cbct_mar_experiment.py run --help                                   # tat ca tham so
 Vi du doi tham so:
   python cbct_mar_experiment.py run --yolo-models yolo11n-seg yolo11s-seg yolo12n-seg --yolo-epochs 60 120 \
       --ours-radius 48 96 inf --gan-epochs 150 --cbctmar-epochs 200
   python cbct_mar_experiment.py run --force metrics select compie report      # tinh lai tu buoc metrics
   python cbct_mar_experiment.py run --quick --results /tmp/r_quick            # chay thu nhanh (vai anh, 1 epoch)

 Ket qua (results/):
   config.json, state.json, logs/pipeline.log, logs/tasks/<buoc>.log, progress/<buoc>.json
   data/        du lieu tao ra (nhieu tu tao ToothFairy, CVan da chuan hoa, dataset YOLO, cac tap danh gia)
   models/      ours_gan/<g>/ (checkpoint, history, tb), cbct_mar/ (checkpoint), yolo/<cfg>/ (weights, results.csv)
   outputs/<tap>/<phuong_phap>/<anh>.png   anh ket qua MOI phuong phap / cau hinh (+ _cache GAN/YOLO, _timing)
   metrics/     metrics_per_image.csv, summary.csv, leaderboard_<tap>.csv, ours_selection.csv, best_ours.json
   paper/       bang CSV + LaTeX (so sanh chinh moi tap, YOLO, thoi gian)
   compie/<tap>/<anh>.png  Original -> Phuong phap cua ban -> CBCT_MAR -> NMAR -> LI-MAR (-> Clean GT)
   REPORT.md    tom tat
==============================================================================
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
SRC = SCRIPT.parent
PROJECT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "preprocess"))

EVAL_SETS = ["tf_real_val", "tf_synth_val", "tf_real_test", "tf_synth_test", "cvan_synth", "cvan_real", "truong_synth", "truong_real"]
SPLITS = ("Train", "Val", "Test")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ==============================================================================
#  CAU HINH
# ==============================================================================
def default_config() -> dict:
    return {
        "paths": {
            "toothfairy": str(PROJECT / "Dataset_TF"),                 # Glare/Normal x Train/Val/Test (PNG)
            "cvan_dicom": str(PROJECT / "Dataset_VN" / "CBCT_CVan"),     # DICOM goc CBCT CVan
            "truong_dicom": str(PROJECT / "Dataset_VN" / "CBCT_Truong"),   # DICOM goc CBCT Truong (Sirona ORTHOPHOS SL, 12-bit, HU toi da 3071)
            "truong_only_in": str(PROJECT / "Dataset_VN" / "Dataset_AI" / "CBCT_Truong"),   # chi lay cac thu muc benh nhan CO TEN TRUNG o day
            "cbct_mar_repo": str(PROJECT / "third_party" / "CBCT_MAR"),  # git clone maheshi81/CBCT_MAR
        },
        "seed": 42,
        "quick": False,
        "resources": {
            "cpu_reserve": 1,        # nhan de danh cho he dieu hanh + bo dieu phoi
            "gpu_mem_frac": 0.92,    # dung toi da 92% VRAM (du ~8% tranh tran)
            # so buoc GPU chay CUNG LUC. Do tren P40: 3 buoc song song -> GAN 305s/epoch (rieng 108s), CBCT_MAR 410s (rieng 96s):
            # GPU chi chia thoi gian giua tien trinh (khong chong kernel) nen song song CHAM hon ~25% -> mac dinh 1 (lan luot,
            # moi buoc dung tron GPU + cudnn.benchmark), CPU van chay song song (NMAR/LI-MAR, hau xu ly, metrics...).
            "gpu_concurrency": 1,
            "ram_reserve_gb": 1.2,   # RAM de danh
            "poll_s": 4,
            "max_retries": 2,
        },
        "data": {
            "haze_prob": 0.5,             # xac suat them quang sang khuech tan (Tao_nhieu_sang HAZE) moi anh tu tao
            "tf_max_per_split": None,     # gioi han anh Normal / split khi tao nhieu (None = het; ~21% anh co chuoi rang)
            "cvan_size": 370,             # CVan 550px (0.2mm) -> 370px (~0.3mm, giong ToothFairy)
            "cvan_window_hu": [-1409.0, 3209.0],   # cua so HU: mo mem (~220HU) -> 90, rang (~2700HU) -> 227 nhu ToothFairy
            "cvan_scan_frac": [0.2, 0.85],  # vung lat axial xet (theo ti le chieu z)
            "cvan_scan_step": 3,
            "cvan_synth_per_vol": 3,      # so lat SACH / khoi de them nhieu tu tao
            "cvan_real_per_vol": 2,       # so lat co kim loai THAT / khoi
            "cvan_metal_hu": 4500.0,      # lat "co kim loai that": khoi HU >= nguong nay, dien tich >= cvan_metal_min_area
            "cvan_metal_min_area": 20,
            "cvan_clean_hu": 4000.0,      # lat "sach": co chuoi rang va khoi HU >= nguong nay < cvan_clean_max_area px
            "cvan_clean_max_area": 10,
            "cvan_max_vols": None,
            # ---- CBCT Truong (DICOM, HU that): chuan hoa cuong do cua so phan vi 0.5-99.5% cua tung khoi (cung cach process_volume/
            # Prepare_Vn_tooth_external -> mien ToothFairy). May quet 12-bit chan o 3071 HU nen "kim loai" = voxel BAO HOA (>= truong_clip_hu),
            # khong phan biet duoc bang nguong HU cao nhu CVan (dinh 5286 HU). Lat "co kim loai that": co chuoi rang + khoi bao hoa lon nhat
            # >= truong_real_blob px (o 370px); lat "sach": co chuoi rang + KHONG co voxel bao hoa nao (<= truong_clean_px).
            "truong_size": 370, "truong_scan_frac": [0.2, 0.85], "truong_scan_step": 3,
            "truong_synth_per_vol": 8, "truong_real_per_vol": 8,
            "truong_clip_hu": 3069.0, "truong_real_blob": 120, "truong_clean_px": 4,
            "truong_max_vols": None,
            "eval_max_per_set": None,     # gioi han so anh moi tap danh gia (None = het)
        },
        "yolo": {
            "models": ["yolo11n-seg", "yolo12n-seg"],
            "epochs": [80], "imgsz": [384], "batch": [16], "lr0": [0.01],
            "patience": 30, "workers": 4,
            "tooth_class": True,          # 2 lop: 0=metal, 1=tooth
            "min_area_px": 12,
            "pred_conf_floor": 0.05,      # luu du doan >= nguong nay, loc lai theo luoi ours.conf
        },
        "gan": {
            "variants": {"g0": {}},       # {ten: {THAM_SO_V12_CFG: gia_tri}} - moi bien the train 1 GAN rieng
            "epochs": 100, "patience": 20, "batch": 16, "workers": 4,
            "weights": "best",            # best (EMA, val_score cao nhat) | final
            "ckpt": None,                 # duong dan checkpoint co san -> BO QUA train (vd best.pth run V11)
        },
        "cbctmar": {
            # DUNG SO EPOCH CUA REPO GOC: code.ipynb co EPOCHS = 600 va vong lap `for epoch in range(START_EPOCH, EPOCHS + 1)` voi
            # START_EPOCH = 0 -> 601 luot (epoch 0..600). inclusive_last=True tai hien dung do. Batch 1, ~125 s/epoch tren P40 -> ~21 gio.
            # (Repo goc con nap ckpt epoch 180 truoc vong lap de tiep tuc - khong co ckpt do nen train tu dau.)
            "epochs": 600, "inclusive_last": True,
            "milestones": [60, 180, 300, 450, 600],   # epoch (chi so 0-based nhu ten file gen_G_epoch_N cua repo goc): luu du 4 mang + danh gia rieng
            "milestone_every": 30,                      # them: luu gen_G moi 30 epoch (theo doi hoi tu)
            "save_every": 10, "lambda_cycle": 18.0, "lambda_identity": 15.0,
            "lr": 2e-4, "batch": 1, "workers": 3,
            # fast_step: TOAN HOC GIONG HET code goc (kiem chung: loss trung 5 chu so, dao ham lech ~1e-5 = sai so fp32) nhung nhanh ~1.4x:
            #  (1) dung lai forward: fake_n=G(rg), fake_g=F(rn) tinh 1 LAN co grad, buoc D dung ban .detach() (G,F chua doi giua 2 lan) - bo 2/8 forward;
            #  (2) gop batch 2 cac lan chay CUNG mang doc lap (G([rg,rn]), F([rn,rg]), D([that,gia])) - G, F, D KHONG co BatchNorm/InstanceNorm
            #      nen gop batch khong doi ket qua tung mau;
            #  (3) 2 CUDA stream cho 2 nhanh doc lap (nhanh G->F va F->G) de GPU chay chong kernel.
            "fast_step": True,
            "weights": "final",           # final | bestval (val_cycle thap nhat)
            "ckpt": None,
        },
        "ours": {
            "conf": [0.25, 0.5],          # nguong tin cay YOLO
            "radius": [64, 128, "inf"],   # ban kinh (px, thang 256) quanh kim loai duoc phep sua; inf = toan vung xam/den
            "clean": ["light", "off"],    # khu nhieu nhe V12 (median+NLM+phang nen+tra ket cau) hoac tat
            "heuristic": True,            # them bien the mask bang luat (ablation khong YOLO)
        },
        "classical": {"angles": 360},
        "report_full": {"compare_with": str(PROJECT / "results")},   # thu muc ket qua CU (CBCT_MAR 60 epoch) de lap bang so sanh
        "select": {
            # "yolo": 'Ours' = bien the YOLO tot nhat (phuong phap cua ban dinh nghia co YOLO); bien the luat = ablation.
            # "all": chon trong tat ca bien the (ke ca luat).
            "ours_pool": "yolo",
            # [tap, metric, max|min] - trung binh hang (rank) tren cac metric nay (chi dung tap VAL) -> cau hinh tot nhat
            # (khong dung far_change/mae_outside: 2 chi so nay toi uu khi KHONG sua gi -> se chon nham bien the "khong lam gi")
            "metrics": [["tf_synth_val", "psnr", "max"], ["tf_synth_val", "ssim", "max"],
                        ["tf_synth_val", "mae_noise", "min"], ["tf_synth_val", "lp_noise", "min"],
                        ["tf_synth_val", "edge_bone", "max"], ["tf_real_val", "streak_ratio", "min"],
                        ["tf_real_val", "edge_prot", "max"]],
        },
        "compie": {"baselines": ["cbct_mar", "nmar", "limar"], "add_gt": True, "sheet_n": 8},
        # tai nguyen moi buoc (GiB VRAM = TRAN CUNG cua tien trinh do, so nhan CPU, GB RAM) - do thuc te tren P40:
        # GAN V12 batch16 (cudnn.benchmark tat) 15.4GB, CBCT_MAR batch1 0.9GB -> GAN + CBCT_MAR + 1 YOLO chay CUNG LUC
        "task_res": {
            "gan_train": [16.0, 5, 4.0], "cbctmar_train": [1.5, 4, 2.5], "yolo_train": [4.0, 5, 3.0],
            "yolo_train_large": [6.5, 5, 3.5], "infer": [3.0, 2, 2.0], "pool_worker_ram": 0.3, "pool_base_ram": 0.6,
        },
    }


def quick_overrides() -> dict:
    return {
        "data": {"tf_max_per_split": 70, "cvan_max_vols": 3, "truong_max_vols": 3, "eval_max_per_set": 4, "cvan_scan_step": 6,
                 "truong_scan_step": 6},
        "yolo": {"epochs": [1], "imgsz": [256], "batch": [8], "patience": 1},
        "gan": {"epochs": 1, "patience": 1},
        "cbctmar": {"epochs": 2, "inclusive_last": False, "milestones": [1], "milestone_every": 1, "save_every": 1},
        "ours": {"conf": [0.25], "radius": [64, "inf"], "clean": ["light", "off"]},
        "classical": {"angles": 60},
    }


def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict) and k not in ("variants",):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


# (co CLI, duong dan trong config, kieu) - chi ap dung co nao nguoi dung dua vao
CLI_MAP = [
    ("yolo_models", ("yolo", "models"), str, True), ("yolo_epochs", ("yolo", "epochs"), int, True),
    ("yolo_imgsz", ("yolo", "imgsz"), int, True), ("yolo_batch", ("yolo", "batch"), int, True),
    ("yolo_lr0", ("yolo", "lr0"), float, True), ("yolo_patience", ("yolo", "patience"), int, False),
    ("ours_conf", ("ours", "conf"), float, True), ("ours_radius", ("ours", "radius"), str, True),
    ("ours_clean", ("ours", "clean"), str, True),
    ("gan_epochs", ("gan", "epochs"), int, False), ("gan_patience", ("gan", "patience"), int, False),
    ("gan_batch", ("gan", "batch"), int, False), ("gan_weights", ("gan", "weights"), str, False),
    ("gan_ckpt", ("gan", "ckpt"), str, False),
    ("cbctmar_epochs", ("cbctmar", "epochs"), int, False), ("cbctmar_workers", ("cbctmar", "workers"), int, False), ("cbctmar_weights", ("cbctmar", "weights"), str, False),
    ("cbctmar_ckpt", ("cbctmar", "ckpt"), str, False),
    ("haze_prob", ("data", "haze_prob"), float, False), ("cvan_synth_per_vol", ("data", "cvan_synth_per_vol"), int, False),
    ("cvan_real_per_vol", ("data", "cvan_real_per_vol"), int, False), ("cvan_max_vols", ("data", "cvan_max_vols"), int, False),
    ("truong_synth_per_vol", ("data", "truong_synth_per_vol"), int, False), ("truong_real_per_vol", ("data", "truong_real_per_vol"), int, False),
    ("truong_max_vols", ("data", "truong_max_vols"), int, False),
    ("eval_max_per_set", ("data", "eval_max_per_set"), int, False), ("mar_angles", ("classical", "angles"), int, False),
    ("gpu_mem_frac", ("resources", "gpu_mem_frac"), float, False),
    ("gpu_concurrency", ("resources", "gpu_concurrency"), int, False), ("cpu_reserve", ("resources", "cpu_reserve"), int, False),
    ("seed", ("seed",), int, False),
]


def build_config(args, R: Path) -> dict:
    cfg = default_config()
    old = R / "config.json"
    if old.exists() and not getattr(args, "fresh_config", False):
        deep_update(cfg, json.loads(old.read_text(encoding="utf-8")))      # giu cau hinh lan truoc (tiep tuc dung)
    if getattr(args, "config", None):
        deep_update(cfg, json.loads(Path(args.config).read_text(encoding="utf-8")))
    if getattr(args, "quick", False):
        cfg["quick"] = True
        deep_update(cfg, quick_overrides())
    for flag, path, typ, is_list in CLI_MAP:
        v = getattr(args, flag, None)
        if v is None:
            continue
        v = [_parse_val(x, typ) for x in v] if is_list else _parse_val(v, typ)
        d = cfg
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = v
    if getattr(args, "gan_variants", None):
        cfg["gan"]["variants"] = json.loads(args.gan_variants)
    cfg.pop("_ours_override", None)
    return cfg


def _parse_val(x, typ):
    if isinstance(x, str) and x.lower() in ("none", "null"):
        return None
    if typ is str and isinstance(x, str) and re.fullmatch(r"\d+", x):
        return int(x)                      # radius "64" -> 64 ; "inf" giu chuoi
    return typ(x)


def load_cfg(R: Path) -> dict:
    return json.loads((R / "config.json").read_text(encoding="utf-8"))


# ==============================================================================
#  TIEN ICH
# ==============================================================================
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_dur(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "-"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Log:
    def __init__(self, path: Path | None, prefix: str = ""):
        self.path, self.prefix = path, prefix
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def __call__(self, *a):
        line = f"[{now()}]{self.prefix} " + " ".join(str(x) for x in a)
        with self.lock:
            print(line, flush=True)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_u8(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return img


def write_png(path, img) -> None:
    """Ghi PNG nguyen tu (tmp -> replace): bi ngat giua chung khong de lai file hong (buoc sau bo qua file da co)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"Khong ghi duoc {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    buf.tofile(str(tmp))
    os.replace(tmp, path)


def list_images(folder) -> list:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def even_pick(items, cap):
    items = list(items)
    if not cap or len(items) <= cap:
        return items
    idx = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def ellipse(k: int):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


class Progress:
    """Ghi results/progress/<buoc>.json (done/total/toc do/ETA) - lenh `status` doc de theo doi truc tiep."""

    def __init__(self, R: Path, task: str, total: int, desc: str = ""):
        self.path = R / "progress" / f"{_safe(task)}.json"
        self.task, self.total, self.desc = task, max(0, int(total)), desc
        self.done, self.t0, self.last = 0, time.time(), 0.0
        self.base = None          # so luong da xong LUC BAT DAU (tiep tuc sau khi ngat) -> khong tinh vao toc do/ETA
        self.extra = {}
        self._write(force=True)

    def update(self, n=1, **extra):
        self.done += n
        self.extra.update(extra)
        self._write()

    def set(self, done, total=None, **extra):
        if self.base is None:
            self.base = done
            self.t0 = time.time()
        self.done = done
        if total is not None:
            self.total = total
        self.extra.update(extra)
        self._write()

    def _write(self, force=False):
        t = time.time()
        if not force and t - self.last < 1.5 and self.done < self.total:
            return
        self.last = t
        el = t - self.t0
        made = self.done - (self.base or 0)
        rate = made / el if el > 0 and made > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        atomic_json(self.path, {"task": self.task, "desc": self.desc, "done": self.done, "total": self.total,
                                "elapsed_s": el, "rate_per_s": rate, "eta_s": eta, "updated": now(), **self.extra})


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)


def _pool_init():
    cv2.setNumThreads(1)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def run_pool(fn, items: list, workers: int, prog: Progress | None = None, log=None, fail_frac: float = 0.02):
    """Chay fn(item) song song (spawn, 1 luong/tien trinh). Loi le te (< fail_frac) chi ghi log, qua nhieu -> raise."""
    results, fails = [], []
    workers = max(1, min(int(workers), len(items))) if items else 1
    if workers <= 1:
        for it in items:
            try:
                results.append(fn(it))
            except Exception:
                fails.append((it, traceback.format_exc()))
            if prog:
                prog.update()
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_pool_init) as ex:
            futs = {ex.submit(fn, it): it for it in items}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception:
                    fails.append((futs[fut], traceback.format_exc()))
                if prog:
                    prog.update()
    if fails and log:
        for it, tb in fails[:5]:
            log(f"  [LOI] {it}: {tb.strip().splitlines()[-1]}")
        log(f"  {len(fails)}/{len(items)} muc bi loi (xem tren).")
        if len(fails) > max(1, fail_frac * len(items)):
            raise RuntimeError(f"Qua nhieu loi: {len(fails)}/{len(items)}\n{fails[0][1]}")
    return results


def gpu_query():
    """(tong MB, da dung MB, % util) cua GPU 0 qua nvidia-smi (khong can import torch)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        t, u, g = [float(x) for x in out.strip().splitlines()[0].split(",")]
        return t, u, g
    except Exception:
        return None


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 4


def limit_gpu(c=None):
    """Tran VRAM cung cho tien trinh nay (MAR_GPU_GB do bo dieu phoi dat): vuot -> OOM o CHINH buoc nay (duoc chay lai
    doc quyen), khong lan sang cac buoc GPU khac dang chay song song."""
    import torch
    gb = float(os.environ.get("MAR_GPU_GB", "0") or 0)
    if gb > 0 and torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        torch.cuda.set_per_process_memory_fraction(min(0.97, gb / tot), 0)
        if c:
            c.log(f"Tran VRAM: {gb:.1f}GiB / {tot:.1f}GiB")


def import_tn():
    import Tao_nhieu_sang as TN        # src/preprocess/Tao_nhieu_sang.py
    return TN


def import_v12():
    import train_dental_mar_V12 as V   # phuong phap cua ban: kien truc GAN + vong train
    return V


# ==============================================================================
#  PHUONG PHAP CUA BAN (V12) - PHAN XU LY ANH (chi numpy/cv2, KHONG can torch -> tien trinh con nhe RAM)
#  Hang so sao chep tu train_dental_mar_V12.CFG (buoc infer_gan kiem tra lai khop voi V12).
# ==============================================================================
P = dict(METAL_THRESH_U8=240, METAL_OPEN_PX=5, METAL_MIN_AREA_PX=20, HALO_THRESH_U8=150, HALO_DILATE_PX=20,
         KEEP_DILATE_PX=24, TOOTH_BODY_THRESH_U8=185, TOOTH_BODY_OPEN_PX=5, TOOTH_BODY_DILATE_PX=1,
         PROTECT_TISSUE_MULT=1.45, PROTECT_DILATE_PX=2, PROTECT_BLUR_SIGMA=1.5, BODY_FG_MIN=10,
         CLEAN_MEDIAN_K=5, CLEAN_NLM_H=4, CLEAN_AIR_FRAC=0.5, CLEAN_AIR_SIGMA=1.5, CLEAN_FLATTEN=True,
         CLEAN_FLAT_SIGMA_GRAY=14, CLEAN_FLAT_STRENGTH_GRAY=0.4, CLEAN_FLAT_STRENGTH_GRAY_NEAR=0.85,
         CLEAN_FLAT_NEAR_FULL_PX=60, CLEAN_FLAT_NEAR_FADE_PX=60, CLEAN_FLAT_SIGMA_AIR=6, CLEAN_FLAT_STRENGTH_AIR=0.85,
         CLEAN_FLAT_CLIP=28, CLEAN_FLAT_OUTLIER=40, CLEAN_FLAT_AIR_FRAC=0.6, CLEAN_TEXTURE_STRENGTH=0.8,
         CLEAN_TEXTURE_HP_SIGMA=1.5, CLEAN_TEXTURE_HP_CLIP=15, CLEAN_TEXTURE_CORR_SIGMA=0.7,
         SYNTH_CHG_THR=6, SYNTH_CHG_DILATE_PX=3, IMG_SIZE=256)


def _sc(img) -> float:
    return max(img.shape) / 256.0


def tooth_metal(u8, thr=None):
    """Kim loai o rang bang luat (V10): khoi bao hoa DAY + du lon (mo hinh thai loai mep vo xuong mong)."""
    thr = P["METAL_THRESH_U8"] if thr is None else thr
    sc = _sc(u8)
    k = max(3, int(round(P["METAL_OPEN_PX"] * sc)) | 1)
    min_area = max(4, int(round(P["METAL_MIN_AREA_PX"] * sc * sc)))
    m = (u8 >= thr).astype(np.uint8) * 255
    if not m.any():
        return m
    o = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(k))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
    ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
    if not ids:
        return np.zeros_like(m)
    big = (np.isin(lbl, ids) * 255).astype(np.uint8)
    return cv2.bitwise_and(cv2.dilate(big, ellipse(k + 2)), m)


def tooth_body(orig):
    sc = _sc(orig)
    k = max(3, int(round(P["TOOTH_BODY_OPEN_PX"] * sc)) | 1)
    b = cv2.GaussianBlur(orig, (3, 3), 0)
    m = (b >= P["TOOTH_BODY_THRESH_U8"]).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(k))
    d = max(0, int(round(P["TOOTH_BODY_DILATE_PX"] * sc)))
    return cv2.dilate(m, ellipse(2 * d + 1)) if d > 0 else m


def bright_mask(orig):
    """Xuong/rang theo cuong do (>= 1.45 x trung vi mo cua CHINH anh), gian 2px."""
    sc = _sc(orig)
    b = cv2.GaussianBlur(orig, (3, 3), 0)
    fg = b[b > P["BODY_FG_MIN"]]
    if fg.size == 0:
        return np.zeros_like(orig)
    m = (b >= P["PROTECT_TISSUE_MULT"] * float(np.median(fg))).astype(np.uint8) * 255
    d = max(1, int(round(P["PROTECT_DILATE_PX"] * sc)))
    return cv2.dilate(m, ellipse(2 * d + 1))


def _masked_gauss(img_f32, mask_f32, sigma):
    num = cv2.GaussianBlur(img_f32 * mask_f32, (0, 0), sigma)
    den = cv2.GaussianBlur(mask_f32, (0, 0), sigma)
    return num / np.maximum(den, 1e-3), den


def _robust_background(base_f32, mask, sigma, outlier, iters=2):
    m = mask.astype(np.float32)
    for _ in range(iters):
        f, _ = _masked_gauss(base_f32, m, sigma)
        m = mask.astype(np.float32) * (np.abs(base_f32 - f) < outlier)
    return _masked_gauss(base_f32, m, sigma)


def _air_masks(orig, sc):
    b = cv2.GaussianBlur(orig, (3, 3), 0)
    fgv = b[b > P["BODY_FG_MIN"]]
    if fgv.size == 0:
        return b, None, None, None
    tissue = float(np.median(fgv))
    air = ((b < P["CLEAN_FLAT_AIR_FRAC"] * tissue).astype(np.uint8)) * 255
    air = cv2.morphologyEx(air, cv2.MORPH_OPEN, ellipse(max(3, int(round(5 * sc)) | 1)))
    airx = cv2.dilate(air, ellipse(max(3, int(round(7 * sc)) | 1)))
    return b, tissue, air, airx


def flatten_soft(base, orig, protect, core):
    """Keo vet toi/sang RONG ve nen dong nhat cua vung xam/den; manh dan khi gan kim loai (core); +-CLIP."""
    sc = _sc(orig)
    b, tissue, air, airx = _air_masks(orig, sc)
    if tissue is None:
        return base
    prot = protect > 0
    gray = ((~prot) & (airx == 0) & (b > P["BODY_FG_MIN"])).astype(np.float32)
    airm = ((air > 0) & (~prot)).astype(np.float32)
    bf = base.astype(np.float32)
    out = bf.copy()
    if core is not None and core.any():
        dist = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 5)
        full, fade = P["CLEAN_FLAT_NEAR_FULL_PX"] * sc, max(P["CLEAN_FLAT_NEAR_FADE_PX"] * sc, 1.0)
        prox = np.clip((full + fade - dist) / fade, 0.0, 1.0).astype(np.float32)
    else:
        prox = np.zeros(orig.shape, np.float32)
    st_gray = P["CLEAN_FLAT_STRENGTH_GRAY"] + (P["CLEAN_FLAT_STRENGTH_GRAY_NEAR"] - P["CLEAN_FLAT_STRENGTH_GRAY"]) * prox
    for m, sig, st in ((gray, P["CLEAN_FLAT_SIGMA_GRAY"], st_gray), (airm, P["CLEAN_FLAT_SIGMA_AIR"], P["CLEAN_FLAT_STRENGTH_AIR"])):
        if m.sum() < 64:
            continue
        f, den = _robust_background(bf, m, sig * sc, P["CLEAN_FLAT_OUTLIER"])
        wgt = cv2.GaussianBlur(m, (0, 0), 1.5) * st * (den > 0.05)
        out = out + wgt * np.clip(f - bf, -P["CLEAN_FLAT_CLIP"], P["CLEAN_FLAT_CLIP"])
    return np.clip(out, 0, 255).astype(np.uint8)


def restore_texture(cleaned, orig, protect):
    """Cong lai nhieu tong hop cung do lech chuan nhieu that (MAD high-pass anh goc) -> anh khong phang nhu nhua."""
    sc = _sc(orig)
    o = orig.astype(np.float32)
    hp = o - cv2.GaussianBlur(o, (0, 0), P["CLEAN_TEXTURE_HP_SIGMA"] * sc)
    b, tissue, air, airx = _air_masks(orig, sc)
    if tissue is None:
        return cleaned
    prot = protect > 0
    regions = (((~prot) & (airx == 0) & (b > P["BODY_FG_MIN"])), ((air > 0) & (~prot)))
    rng = np.random.default_rng(int(orig.sum()) % (2 ** 32))
    noise = rng.standard_normal(orig.shape).astype(np.float32)
    if P["CLEAN_TEXTURE_CORR_SIGMA"] > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), P["CLEAN_TEXTURE_CORR_SIGMA"])
    noise /= max(float(noise.std()), 1e-6)
    add = np.zeros_like(noise)
    for m in regions:
        ok = m & (np.abs(hp) < P["CLEAN_TEXTURE_HP_CLIP"])
        if ok.sum() < 256:
            continue
        sigma = 1.4826 * float(np.median(np.abs(hp[ok] - np.median(hp[ok]))))
        add += sigma * cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) * noise
    return np.clip(cleaned.astype(np.float32) + P["CLEAN_TEXTURE_STRENGTH"] * add, 0, 255).astype(np.uint8)


def clean_soft(img, orig, protect, core):
    """Khu nhieu vung xam/den (V12 light): median + NLM + lam min khong khi + phang nen + tra ket cau."""
    k = max(3, P["CLEAN_MEDIAN_K"] | 1)
    med = cv2.medianBlur(img, k)
    nlm = cv2.fastNlMeansDenoising(med, None, h=P["CLEAN_NLM_H"], templateWindowSize=7, searchWindowSize=21) if P["CLEAN_NLM_H"] > 0 else med
    b = cv2.GaussianBlur(orig, (3, 3), 0)
    fg = b[b > P["BODY_FG_MIN"]]
    tissue = float(np.median(fg)) if fg.size else 100.0
    air = (b < P["CLEAN_AIR_FRAC"] * tissue).astype(np.uint8) * 255
    air = cv2.morphologyEx(air, cv2.MORPH_OPEN, ellipse(5))
    air_w = cv2.GaussianBlur(air, (0, 0), 2.0).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(nlm, (0, 0), P["CLEAN_AIR_SIGMA"]).astype(np.float32)
    soft = nlm.astype(np.float32) * (1 - air_w) + smooth * air_w
    w = 1.0 - cv2.GaussianBlur(protect, (0, 0), P["PROTECT_BLUR_SIGMA"]).astype(np.float32) / 255.0
    out = np.clip(w * soft + (1 - w) * img.astype(np.float32), 0, 255).astype(np.uint8)
    if P["CLEAN_FLATTEN"]:
        out = flatten_soft(out, orig, protect, core)
    if P["CLEAN_TEXTURE_STRENGTH"] > 0:
        out = restore_texture(out, orig, protect)
    return out


def heuristic_masks(orig):
    metal = tooth_metal(orig)
    tooth = cv2.bitwise_or(tooth_body(orig), metal)
    return metal, tooth


def ours_compose(orig, gan, metal, tooth, radius, clean):
    """PHUONG PHAP CUA BAN (V12 gan_soft) voi mask kim loai/rang cho truoc (YOLO hoac luat):
      - ket qua GAN chi dung o vung xam/den (ngoai mask bao ve xuong/rang) + quang sang quanh kim loai ngoai than rang;
      - radius (px thang 256): chi sua trong ban kinh quanh kim loai (YOLO dinh vi -> sua CUC BO, it 'bia' xa kim loai);
        'inf' = toan vung xam/den nhu V12; khong thay kim loai + radius huu han -> giu nguyen anh;
      - clean='light': khu nhieu nhe V12 trong cung vung duoc phep sua."""
    sc = _sc(orig)
    # than rang theo cuong do LUON duoc giu (hop voi mask rang YOLO): nhan 'tooth' cua YOLO chi phu 1 phan so rang ->
    # neu chi dung YOLO, rang bi bo sot gan kim loai se bi ap GAN (mo/toi rang). Voi mask luat, tooth da gom san phan nay.
    tooth = cv2.bitwise_or(tooth, tooth_body(orig))
    protect = cv2.bitwise_or(bright_mask(orig), tooth)
    soft_w = 1.0 - cv2.GaussianBlur(protect, (0, 0), P["PROTECT_BLUR_SIGMA"]).astype(np.float32) / 255.0
    local = radius not in (None, "inf", float("inf"))
    if metal.any():
        near = cv2.dilate(metal, ellipse(2 * int(round(P["KEEP_DILATE_PX"] * sc)) + 1))
        extra = ((near > 0) & (tooth == 0)).astype(np.uint8) * 255
        soft_w = np.maximum(soft_w, cv2.GaussianBlur(extra, (0, 0), 1.5).astype(np.float32) / 255.0)
        if local:
            dist = cv2.distanceTransform((metal == 0).astype(np.uint8), cv2.DIST_L2, 5)
            full, fade = float(radius) * sc, max(0.25 * float(radius) * sc, 4.0)
            gate = np.clip((full + fade - dist) / fade, 0.0, 1.0).astype(np.float32)
        else:
            gate = np.ones(orig.shape, np.float32)
    else:
        gate = np.zeros(orig.shape, np.float32) if local else np.ones(orig.shape, np.float32)
    w = soft_w * gate
    fused = w * gan.astype(np.float32) + (1 - w) * orig.astype(np.float32)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    if clean == "light" and gate.any():
        cleaned = clean_soft(fused, orig, protect, metal)       # da tu tron theo (1 - mask bao ve) ben trong
        fused = cleaned if bool((gate >= 1).all()) else np.clip(
            fused.astype(np.float32) + gate * (cleaned.astype(np.float32) - fused.astype(np.float32)), 0, 255).astype(np.uint8)
    return fused


# ==============================================================================
#  BASELINE 2/3: NMAR + LI-MAR (sinogram mo phong bang bien doi Radon cua lat 2D)
# ==============================================================================
def _pad_square(a):
    h, w = a.shape
    s = max(h, w)
    out = np.zeros((s, s), a.dtype)
    out[:h, :w] = a
    return out


def _interp_trace(Pm, T):
    out = Pm.copy()
    idx = np.arange(Pm.shape[0], dtype=np.float64)
    for j in range(Pm.shape[1]):
        t = T[:, j]
        if t.any() and (~t).sum() >= 2:
            out[t, j] = np.interp(idx[t], idx[~t], Pm[~t, j])
    return out


def classic_mar(img, metal, angles):
    """Tra ve (li_u8, nmar_u8). Metal: mask luat (tooth_metal, gian 1px). Ap dung PHAN HIEU CHINH vao anh goc:
    out = anh - FBP(P - P_hieu_chinh) (tranh sai so tai tao FBP toan anh lam thiet baseline), roi tra kim loai ve."""
    from skimage.transform import iradon, radon
    h, w = img.shape
    if not metal.any():
        return img.copy(), img.copy()
    f = _pad_square(img.astype(np.float32) / 255.0)
    m = _pad_square((cv2.dilate(metal, ellipse(3)) > 0).astype(np.float32))
    th = np.linspace(0.0, 180.0, int(angles), endpoint=False)
    Pm = radon(f, th, circle=False)
    T = radon(m, th, circle=False) > 0.5
    s = f.shape[0]
    P_li = _interp_trace(Pm, T)
    li = f - iradon(Pm - P_li, th, circle=False, filter_name="ramp", output_size=s)
    # NMAR: prior phan lop (khong khi -> 0, mo mem -> hang so, xuong giu nguyen) tu anh LI
    fg = f[f > P["BODY_FG_MIN"] / 255.0]
    tissue = float(np.median(fg)) if fg.size else 0.35
    air = li < 0.5 * tissue
    bone = li > P["PROTECT_TISSUE_MULT"] * tissue
    softm = ~air & ~bone
    prior = li.copy()
    prior[air] = 0.0
    if softm.any():
        prior[softm] = float(li[softm].mean())
    prior = cv2.GaussianBlur(prior.astype(np.float32), (0, 0), 1.0)
    Pp = radon(prior, th, circle=False)
    delta = 1.0
    Pn = Pm / (Pp + delta)
    P_nmar = _interp_trace(Pn, T) * (Pp + delta)
    nm = f - iradon(Pm - P_nmar, th, circle=False, filter_name="ramp", output_size=s)
    mm = m > 0
    li[mm] = f[mm]
    nm[mm] = f[mm]
    to_u8 = lambda a: (np.clip(a[:h, :w], 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return to_u8(li), to_u8(nm)


# ==============================================================================
#  METRICS
# ==============================================================================
def edge_corr(a, b, mask):
    if mask is None or mask.sum() < 32:
        return float("nan")
    ga = cv2.Sobel(a.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)[mask]
    gb = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)[mask]
    if ga.std() < 1e-6 or gb.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(ga, gb)[0, 1])


def _psnr(err2):
    mse = float(err2.mean()) if err2.size else float("nan")
    return 99.0 if mse < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse))


def body_mask_simple(img):
    b = cv2.GaussianBlur(img, (0, 0), 2.0)
    fg = img[img > P["BODY_FG_MIN"]]
    if fg.size == 0:
        return np.zeros(img.shape, bool)
    m = (b > 0.5 * float(np.median(fg))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(15))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cs:
        cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    return out > 0


def synth_masks(noisy, clean, metal_gt):
    valid = ~(cv2.dilate(metal_gt, ellipse(3)) > 0)
    chg = (cv2.absdiff(noisy, clean) > P["SYNTH_CHG_THR"]).astype(np.uint8) * 255
    noise = (cv2.dilate(chg, ellipse(2 * P["SYNTH_CHG_DILATE_PX"] + 1)) > 0) & valid
    prot = (cv2.bitwise_or(bright_mask(clean), tooth_body(clean)) > 0) & valid
    return valid, noise, prot


def synth_metrics(out, noisy, clean, masks):
    from skimage.metrics import structural_similarity as ssim_fn
    valid, noise, prot = masks
    o, c, n = out.astype(np.float32), clean.astype(np.float32), noisy.astype(np.float32)
    e = o - c
    _, smap = ssim_fn(clean, out, data_range=255, full=True)
    lp = lambda a: cv2.GaussianBlur(a, (0, 0), 3.0)
    e_in = float(np.abs(n - c)[noise].mean()) if noise.sum() > 16 else float("nan")
    e_out = float(np.abs(e)[noise].mean()) if noise.sum() > 16 else float("nan")
    outside = valid & ~noise
    return {
        "psnr": _psnr(e[valid] ** 2), "ssim": float(smap[valid].mean()), "mae": float(np.abs(e[valid]).mean()),
        "rmse": float(np.sqrt((e[valid] ** 2).mean())),
        "mae_noise": e_out, "psnr_noise": _psnr(e[noise] ** 2) if noise.sum() > 16 else float("nan"),
        "lp_noise": float(np.abs(lp(o) - lp(c))[noise].mean()) if noise.sum() > 16 else float("nan"),
        "mae_outside": float(np.abs(e[outside]).mean()) if outside.sum() > 16 else float("nan"),
        "gain": (1.0 - e_out / e_in) if e_in and e_in > 1e-6 else float("nan"),
        "edge_bone": edge_corr(out, clean, prot),
    }


def real_masks(inp):
    sc = _sc(inp)
    metal = tooth_metal(inp)
    body = body_mask_simple(inp)
    prot_all = (cv2.bitwise_or(bright_mask(inp), tooth_body(inp)) > 0)
    mdil = cv2.dilate(metal, ellipse(3)) > 0
    if metal.any():
        near = (cv2.dilate(metal, ellipse(2 * int(48 * sc) + 1)) > 0) & ~mdil & body
        far = body & ~(cv2.dilate(metal, ellipse(2 * int(96 * sc) + 1)) > 0)
    else:
        near = np.zeros(inp.shape, bool)
        far = body.copy()
    halo = (inp >= P["HALO_THRESH_U8"]) & (inp < P["METAL_THRESH_U8"]) & near
    return {"near_soft": near & ~prot_all, "far_soft": far & ~prot_all, "prot": prot_all & ~mdil, "halo": halo, "body": body}


def real_metrics(out, inp, mk):
    from skimage.metrics import structural_similarity as ssim_fn
    hp = lambda a: np.abs(a.astype(np.float32) - cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2.0))
    ns = mk["near_soft"]
    if ns.sum() > 64:
        e_in = float(hp(inp)[ns].mean())
        streak = float(hp(out)[ns].mean()) / max(e_in, 1e-6)
    else:
        streak = float("nan")
    halo = mk["halo"]
    if halo.sum() > 32:
        halo_red = 1.0 - float(out[halo].astype(np.float32).mean()) / max(float(inp[halo].astype(np.float32).mean()), 1e-6)
    else:
        halo_red = float("nan")
    _, smap = ssim_fn(inp, out, data_range=255, full=True)
    d = np.abs(out.astype(np.float32) - inp.astype(np.float32))
    return {
        "streak_ratio": streak, "halo_red": halo_red, "edge_prot": edge_corr(out, inp, mk["prot"]),
        "ssim_prot": float(smap[mk["prot"]].mean()) if mk["prot"].sum() > 64 else float("nan"),
        "far_change": float(d[mk["far_soft"]].mean()) if mk["far_soft"].sum() > 64 else float("nan"),
        "mean_change": float(d[mk["body"]].mean()) if mk["body"].sum() > 64 else float("nan"),
    }


# ==============================================================================
#  CAC BUOC (TASK) - moi buoc chay trong 1 tien trinh con: python cbct_mar_experiment.py _task <ten> ...
# ==============================================================================
class Ctx:
    def __init__(self, R: Path, name: str, slots: int):
        self.R, self.name, self.slots = R, name, max(1, slots)
        self.cfg = load_cfg(R)
        self.log = Log(None, prefix=f" [{name}]")
        self.quick = bool(self.cfg.get("quick"))

    def prog(self, total, desc=""):
        return Progress(self.R, self.name, total, desc)


def eval_root(R: Path, s: str) -> Path:
    return R / "data" / "eval" / s


def eval_names(R: Path, s: str) -> list:
    info = read_json(R / "data" / "eval_sets.json", {}) or {}
    return info.get(s, {}).get("images", [])


def eval_has_gt(R: Path, s: str) -> bool:
    info = read_json(R / "data" / "eval_sets.json", {}) or {}
    return bool(info.get(s, {}).get("has_gt"))


def present_sets(R: Path) -> list:
    info = read_json(R / "data" / "eval_sets.json", {}) or {}
    return [s for s in EVAL_SETS if info.get(s, {}).get("images")]


# ---------------------------------------------------------------- [1] DU LIEU ----
def _tf_synth_worker(a):
    src, out, split, fname, seed, haze_prob = a
    TN = import_tn()
    orig = read_u8(Path(src) / "Normal" / split / fname)
    rng = random.Random(stable_seed(seed, "tf", split, fname))
    haze = rng.random() < haze_prob
    TN.CFG.HAZE_ENABLE = haze
    noisy, info, core = TN.make_glare(orig, TN.CFG, rng, return_core=True)
    if not info:
        return None
    write_png(Path(out) / "images" / split / fname, noisy)
    write_png(Path(out) / "masks" / split / fname, core)
    return {"split": split, "file": fname, "n_metal_teeth": len(info), "haze": int(haze)}


def task_data_tf(c: Ctx):
    cfg = c.cfg
    src = Path(cfg["paths"]["toothfairy"])
    out = c.R / "data" / "synth_tf"
    man = out / "manifest_tao_nhieu_sang.csv"
    items = []
    for split in SPLITS:
        files = even_pick([p.name for p in list_images(src / "Normal" / split)], cfg["data"]["tf_max_per_split"])
        items += [(str(src), str(out), split, f, cfg["seed"], cfg["data"]["haze_prob"]) for f in files]
    c.log(f"Tao nhieu sang (Tao_nhieu_sang.make_glare, haze_prob={cfg['data']['haze_prob']}) cho {len(items)} anh Normal ToothFairy, {c.slots} tien trinh")
    rows = [r for r in run_pool(_tf_synth_worker, items, c.slots, c.prog(len(items), "tao nhieu ToothFairy"), c.log) if r]
    df = pd.DataFrame(rows).sort_values(["split", "file"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(man, index=False)
    c.log("Cap (nhieu, sach) theo split:", df.groupby("split").size().to_dict(), "->", out)


def _cvan_volume_worker(a):
    vol_dir, files, vid, out, d = a
    import pydicom
    TN = import_tn()
    hdr = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            z = float(ds.ImagePositionPatient[2]) if getattr(ds, "ImagePositionPatient", None) is not None else float(getattr(ds, "InstanceNumber", 0) or 0)
            hdr.append((z, f))
        except Exception:
            continue
    hdr.sort()
    n = len(hdr)
    if n < 50:
        return {"vid": vid, "n": n, "note": "qua it lat"}, []
    lo_hu, hi_hu = d["cvan_window_hu"]
    size = d["cvan_size"]
    f0, f1 = d["cvan_scan_frac"]
    E = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def comp_area(mask):
        m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, E(3))
        nn, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        return int(max([st[i, 4] for i in range(1, nn)], default=0))

    cands = []
    for k in range(int(f0 * n), int(f1 * n), int(d["cvan_scan_step"])):
        ds = pydicom.dcmread(hdr[k][1])
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        h0, w0 = hu.shape
        s = size / max(h0, w0)
        hs = cv2.resize(hu, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA)
        u8 = (np.clip((hs - lo_hu) / (hi_hu - lo_hu), 0, 1) * 255.0 + 0.5).astype(np.uint8)
        teeth = TN.segment_teeth(u8, TN.CFG).any()
        a_metal = comp_area(hs >= d["cvan_metal_hu"])
        a_clean = comp_area(hs >= d["cvan_clean_hu"])
        cands.append((k, u8, teeth, a_metal, a_clean))
    real = [x for x in cands if x[3] >= d["cvan_metal_min_area"]]
    clean = [x for x in cands if x[2] and x[3] == 0 and x[4] < d["cvan_clean_max_area"]]
    rows = []
    for x in even_pick(real, d["cvan_real_per_vol"]):
        name = f"{vid}_s{x[0]:04d}.png"
        write_png(Path(out) / "real" / name, x[1])
        rows.append({"vid": vid, "file": name, "slice": x[0], "type": "real", "metal_area": x[3]})
    rng = random.Random(stable_seed(d.get("seed", 42), "cvan", vid))
    picked = 0
    # uu tien cac lat cach deu; neu make_glare khong tao duoc (khong tach duoc chuoi rang) thi thu lat khac
    first = even_pick(list(range(len(clean))), d["cvan_synth_per_vol"])
    order = [clean[i] for i in first] + [clean[i] for i in range(len(clean)) if i not in set(first)]
    for x in order:
        if picked >= d["cvan_synth_per_vol"]:
            break
        haze = rng.random() < d["haze_prob"]
        TN.CFG.HAZE_ENABLE = haze
        noisy, info, core = TN.make_glare(x[1], TN.CFG, rng, return_core=True)
        if not info:
            continue
        name = f"{vid}_s{x[0]:04d}.png"
        write_png(Path(out) / "clean" / name, x[1])
        write_png(Path(out) / "synth" / name, noisy)
        write_png(Path(out) / "synth_metal" / name, core)
        rows.append({"vid": vid, "file": name, "slice": x[0], "type": "synth", "haze": int(haze), "n_metal_teeth": len(info)})
        picked += 1
    return {"vid": vid, "n": n, "n_cand": len(cands), "n_real_cand": len(real), "n_clean_cand": len(clean),
            "n_real": sum(r["type"] == "real" for r in rows), "n_synth": picked}, rows


def cvan_volumes(root: Path) -> list:
    """(thu_muc_nhom, [file dcm], ma_an_danh, ten_goc) - moi thu muc chua truc tiep .dcm = 1 lan chup."""
    vols = []
    for pdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        groups = {}
        for f in sorted(pdir.rglob("*.dcm")):
            groups.setdefault(f.parent, []).append(str(f))
        m = re.match(r"\s*(\d+)", pdir.name)
        pid = f"CV{int(m.group(1)):03d}" if m else f"CVX{stable_seed(pdir.name) % 1000:03d}"
        for i, (g, fl) in enumerate(sorted(groups.items())):
            vid = pid if len(groups) == 1 else f"{pid}v{i + 1}"
            vols.append((str(g), fl, vid, str(g.relative_to(root))))
    return vols


def task_data_cvan(c: Ctx):
    cfg = c.cfg
    d = dict(cfg["data"])
    d["seed"] = cfg["seed"]
    root = Path(cfg["paths"]["cvan_dicom"])
    out = c.R / "data" / "cvan"
    vols = cvan_volumes(root)
    if d.get("cvan_max_vols"):
        vols = even_pick(vols, d["cvan_max_vols"])
    c.log(f"CVan: {len(vols)} lan chup DICOM trong {root}; cua so HU {d['cvan_window_hu']} -> uint8, resize {d['cvan_size']}px, {c.slots} tien trinh")
    out.mkdir(parents=True, exist_ok=True)
    # ma an danh -> thu muc goc (co ten benh nhan): CHI de truy vet noi bo, KHONG dua vao paper
    pd.DataFrame([{"vid": v[2], "source_dir": v[3]} for v in vols]).to_csv(out / "_id_map_PRIVATE.csv", index=False)
    items = [(v[0], v[1], v[2], str(out), d) for v in vols]
    res = run_pool(_cvan_volume_worker, items, c.slots, c.prog(len(items), "doc DICOM CVan"), c.log)
    stats = pd.DataFrame([r[0] for r in res])
    rows = [x for r in res for x in r[1]]
    stats.to_csv(out / "volumes.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    c.log(f"CVan xong: {sum(r['type'] == 'synth' for r in rows)} lat sach + nhieu tu tao, {sum(r['type'] == 'real' for r in rows)} lat kim loai that")


def _hu_slice(path) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(str(path))
    return ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)


def _truong_volume_worker(a):
    """1 khoi DICOM CBCT Truong -> uint8 mien ToothFairy + chon lat theo HU that (khong suy ra tu PNG):
      cua so = phan vi 0.5-99.5% cua khoi (lay mau vol[::4,::4,::4] nhu process_volume) -> resize 370px;
      lat 'co kim loai that' = co chuoi rang + khoi voxel bao hoa (>= truong_clip_hu) >= truong_real_blob px;
      lat 'sach' = co chuoi rang + khong co voxel bao hoa (<= truong_clean_px) -> them nhieu tu tao (Tao_nhieu_sang)."""
    import pydicom
    vol_dir, vid, out, d, seed = a
    TN = import_tn()
    files = sorted(Path(vol_dir).glob("*.dcm"))
    hdr = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            hdr.append((float(ds.ImagePositionPatient[2]) if getattr(ds, "ImagePositionPatient", None) is not None
                        else float(getattr(ds, "InstanceNumber", 0) or 0), f))
        except Exception:
            continue
    hdr.sort()
    order = [f for _, f in hdr]
    n = len(order)
    if n < 50:
        return {"vid": vid, "n": n, "note": "qua it lat"}, []
    sub = np.concatenate([_hu_slice(order[i])[::4, ::4].ravel() for i in range(0, n, 4)])
    lo, hi = (float(x) for x in np.percentile(sub, [0.5, 99.5]))
    size = d["truong_size"]
    f0, f1 = d["truong_scan_frac"]
    E3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cands = []
    for k in range(int(f0 * n), int(f1 * n), int(d["truong_scan_step"])):
        hs = cv2.resize(_hu_slice(order[k]), (size, size), interpolation=cv2.INTER_AREA)
        u8 = (np.clip((hs - lo) / (hi - lo + 1e-9), 0, 1) * 255.0 + 0.5).astype(np.uint8)
        teeth = bool(TN.segment_teeth(u8, TN.CFG).any())
        m = cv2.morphologyEx((hs >= d["truong_clip_hu"]).astype(np.uint8) * 255, cv2.MORPH_OPEN, E3)
        nn, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        blob = int(max([st[i, 4] for i in range(1, nn)], default=0))
        cands.append((k, u8, teeth, blob, int((hs >= d["truong_clip_hu"]).sum())))
    real = [x for x in cands if x[2] and x[3] >= d["truong_real_blob"]]
    clean = [x for x in cands if x[2] and x[4] <= d["truong_clean_px"]]
    rows = []
    for k in even_pick(list(range(len(real))), d["truong_real_per_vol"]):
        x = real[k]
        name = f"{vid}_s{x[0]:04d}.png"
        write_png(Path(out) / "real" / name, x[1])
        rows.append({"vid": vid, "file": name, "slice": x[0], "type": "real", "clipped_blob_px": x[3], "clipped_px": x[4]})
    rng = random.Random(stable_seed(seed, "truong", vid))
    first = even_pick(list(range(len(clean))), d["truong_synth_per_vol"])
    order2 = [clean[i] for i in first] + [clean[i] for i in range(len(clean)) if i not in set(first)]
    picked = 0
    for x in order2:
        if picked >= d["truong_synth_per_vol"]:
            break
        haze = rng.random() < d["haze_prob"]
        TN.CFG.HAZE_ENABLE = haze
        noisy, info, core = TN.make_glare(x[1], TN.CFG, rng, return_core=True)
        if not info:
            continue
        name = f"{vid}_s{x[0]:04d}.png"
        write_png(Path(out) / "clean" / name, x[1])
        write_png(Path(out) / "synth" / name, noisy)
        write_png(Path(out) / "synth_metal" / name, core)
        rows.append({"vid": vid, "file": name, "slice": x[0], "type": "synth", "haze": int(haze), "n_metal_teeth": len(info)})
        picked += 1
    return {"vid": vid, "n": n, "win_lo_hu": round(lo, 1), "win_hi_hu": round(hi, 1), "n_cand": len(cands),
            "n_teeth": sum(x[2] for x in cands), "n_real_cand": len(real), "n_clean_cand": len(clean),
            "n_real": sum(r["type"] == "real" for r in rows), "n_synth": picked}, rows


def task_data_truong(c: Ctx):
    cfg = c.cfg
    d = dict(cfg["data"])
    root, only = Path(cfg["paths"]["truong_dicom"]), Path(cfg["paths"]["truong_only_in"])
    keep = {p.name for p in only.iterdir() if p.is_dir()} if only.is_dir() else None
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (keep is None or p.name in keep) and any(p.glob("*.dcm")))
    if d.get("truong_max_vols"):
        dirs = even_pick(dirs, d["truong_max_vols"])
    out = c.R / "data" / "truong"
    out.mkdir(parents=True, exist_ok=True)
    ids = {p: f"TR{i + 1:03d}" for i, p in enumerate(dirs)}
    pd.DataFrame([{"vid": v, "source_dir": p.name} for p, v in ids.items()]).to_csv(out / "_id_map_PRIVATE.csv", index=False)
    c.log(f"Truong: {len(dirs)} khoi DICOM (chi cac thu muc co ten trung voi {only.name}: {len(keep) if keep is not None else 'tat ca'}) trong {root}; "
          f"resize {d['truong_size']}px, {c.slots} tien trinh")
    items = [(str(p), v, str(out), d, cfg["seed"]) for p, v in ids.items()]
    res = run_pool(_truong_volume_worker, items, c.slots, c.prog(len(items), "doc DICOM Truong"), c.log)
    stats = pd.DataFrame([r[0] for r in res]).sort_values("vid")
    rows = [x for r in res for x in r[1]]
    stats.to_csv(out / "volumes.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    c.log(f"Truong xong: {sum(r['type'] == 'synth' for r in rows)} lat sach + nhieu tu tao, {sum(r['type'] == 'real' for r in rows)} lat kim loai that "
          f"(cua so HU trung vi {stats['win_lo_hu'].median():.0f}..{stats['win_hi_hu'].median():.0f})")


def task_eval_sets(c: Ctx):
    cfg = c.cfg
    R = c.R
    tf = Path(cfg["paths"]["toothfairy"])
    cap = cfg["data"]["eval_max_per_set"]
    info = {}

    def put(s, pairs, has_gt):
        pairs = even_pick(sorted(pairs, key=lambda x: x[0]), cap)
        root = eval_root(R, s)
        for name, inp, gt, mgt in pairs:
            write_png(root / "input" / name, read_u8(inp))
            if gt is not None:
                write_png(root / "gt" / name, read_u8(gt))
                write_png(root / "metal_gt" / name, read_u8(mgt))
        info[s] = {"has_gt": has_gt, "images": [p[0] for p in pairs]}

    for split, s in (("Val", "tf_real_val"), ("Test", "tf_real_test")):
        put(s, [(p.name, p, None, None) for p in list_images(tf / "Glare" / split)], False)
    man = pd.read_csv(R / "data" / "synth_tf" / "manifest_tao_nhieu_sang.csv")
    for split, s in (("Val", "tf_synth_val"), ("Test", "tf_synth_test")):
        names = man.loc[man["split"] == split, "file"].tolist()
        put(s, [(n, R / "data" / "synth_tf" / "images" / split / n, tf / "Normal" / split / n,
                 R / "data" / "synth_tf" / "masks" / split / n) for n in names], True)
    cm = R / "data" / "cvan" / "manifest.csv"
    if cm.exists():
        df = pd.read_csv(cm)
        cv = R / "data" / "cvan"
        put("cvan_synth", [(n, cv / "synth" / n, cv / "clean" / n, cv / "synth_metal" / n)
                           for n in df.loc[df["type"] == "synth", "file"]], True)
        put("cvan_real", [(n, cv / "real" / n, None, None) for n in df.loc[df["type"] == "real", "file"]], False)
    tm = R / "data" / "truong" / "manifest.csv"
    if tm.exists():
        df = pd.read_csv(tm)
        tr = R / "data" / "truong"
        put("truong_synth", [(n, tr / "synth" / n, tr / "clean" / n, tr / "synth_metal" / n)
                             for n in df.loc[df["type"] == "synth", "file"]], True)
        put("truong_real", [(n, tr / "real" / n, None, None) for n in df.loc[df["type"] == "real", "file"]], False)
    atomic_json(R / "data" / "eval_sets.json", info)
    c.log("Tap danh gia:", {k: len(v["images"]) for k, v in info.items()})


def _mask_to_polys(mask, min_area):
    polys = []
    n, lbl, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    h, w = mask.shape
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cs, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.8, True).reshape(-1, 2)
        if len(cnt) < 3:
            continue
        polys.append([(float(x) / w, float(y) / h) for x, y in cnt])
    return polys


def _yolo_label_worker(a):
    kind, img_path, clean_path, metal_path, out_img, out_lbl, tooth_class, min_area = a
    TN = import_tn()
    img = read_u8(img_path)
    if kind == "real":
        metal = tooth_metal(img)
        teeth = TN.segment_teeth(img, TN.CFG)
    else:
        metal = read_u8(metal_path)
        teeth = TN.segment_teeth(read_u8(clean_path), TN.CFG)
    teeth = cv2.bitwise_and(teeth, cv2.bitwise_not(cv2.dilate(metal, ellipse(3))))
    lines = [("0 " + " ".join(f"{x:.5f} {y:.5f}" for x, y in p)) for p in _mask_to_polys(metal, min_area)]
    if tooth_class:
        lines += [("1 " + " ".join(f"{x:.5f} {y:.5f}" for x, y in p)) for p in _mask_to_polys(teeth, min_area)]
    write_png(out_img, img)
    Path(out_lbl).parent.mkdir(parents=True, exist_ok=True)
    Path(out_lbl).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def task_yolo_data(c: Ctx):
    """Dataset YOLO-seg: anh Glare THAT (nhan tu dong bang luat) + anh nhieu tu tao (nhan kim loai CHINH XAC)."""
    cfg = c.cfg
    tf = Path(cfg["paths"]["toothfairy"])
    ydir = c.R / "data" / "yolo"
    man = pd.read_csv(c.R / "data" / "synth_tf" / "manifest_tao_nhieu_sang.csv")
    yc = cfg["yolo"]
    items = []
    for split, ys in (("Train", "train"), ("Val", "val")):
        reals = list_images(tf / "Glare" / split)
        if c.quick:
            reals = reals[:24]
        for p in reals:
            items.append(("real", str(p), None, None, str(ydir / "images" / ys / f"real_{p.name}"),
                          str(ydir / "labels" / ys / f"real_{p.stem}.txt"), yc["tooth_class"], yc["min_area_px"]))
        for n in man.loc[man["split"] == split, "file"]:
            items.append(("synth", str(c.R / "data" / "synth_tf" / "images" / split / n), str(tf / "Normal" / split / n),
                          str(c.R / "data" / "synth_tf" / "masks" / split / n), str(ydir / "images" / ys / f"synth_{n}"),
                          str(ydir / "labels" / ys / f"synth_{Path(n).stem}.txt"), yc["tooth_class"], yc["min_area_px"]))
    run_pool(_yolo_label_worker, items, c.slots, c.prog(len(items), "gan nhan YOLO"), c.log)
    names = {0: "metal", 1: "tooth"} if yc["tooth_class"] else {0: "metal"}
    (ydir / "data.yaml").write_text(
        f"path: {ydir}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items()),
        encoding="utf-8")
    c.log(f"Dataset YOLO: {len(items)} anh -> {ydir}")


# ------------------------------------------------------------ [2] HUAN LUYEN ----
def _watch_csv(path: Path, prog: Progress, total: int, stop: threading.Event, parse):
    while not stop.wait(10):
        try:
            if path.exists():
                df = pd.read_csv(path)
                if len(df):
                    prog.set(len(df), total, **parse(df.iloc[-1]))
        except Exception:
            pass


def task_gan_train(c: Ctx, g: str):
    """GAN cua phuong phap ban: dung NGUYEN vong train V12 (train_dental_mar_V12.stage_train) voi du lieu nhieu tu tao."""
    import torch
    cfg, gc = c.cfg, c.cfg["gan"]
    run_dir = c.R / "models" / "ours_gan" / g
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if gc.get("ckpt"):
        shutil.copy2(gc["ckpt"], run_dir / "checkpoints" / "external.pth")
        c.log(f"Dung checkpoint co san {gc['ckpt']} -> bo qua huan luyen")
        return
    V = import_v12()
    C = V.CFG
    C.DATASET_DIR = cfg["paths"]["toothfairy"]
    C.USE_SYNTH, C.SYNTH_DIR = True, str(c.R / "data" / "synth_tf")
    C.EPOCHS, C.EARLY_STOP_PATIENCE = int(gc["epochs"]), int(gc["patience"])
    C.LR_DECAY_PATIENCE = max(1, int(gc["patience"]) // 2)
    C.BATCH_SIZE, C.SYNTH_BATCH = int(gc["batch"]), max(1, int(gc["batch"]) // 2)
    C.NUM_WORKERS = int(gc["workers"])
    C.MAX_TRAIN_IMAGES = 32 if c.quick else None
    C.VAL_MAX_IMAGES = 8 if c.quick else 200
    C.POST_MODE = "gan_soft"
    for k, v in (gc["variants"].get(g) or {}).items():
        setattr(C, k, v)
    # benchmark: nhanh hon ~10% nhung giu gan het VRAM -> chi bat khi GPU chay 1 buoc/lan
    C.CUDNN_BENCHMARK = int(cfg["resources"].get("gpu_concurrency", 1)) <= 1
    torch.set_num_threads(max(1, c.slots - int(gc["workers"])))
    limit_gpu(c)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    V.log.open(run_dir / "log.txt")
    prog = c.prog(C.EPOCHS, "epoch GAN")
    stop = threading.Event()
    th = threading.Thread(target=_watch_csv, daemon=True, args=(run_dir / "history.csv", prog, C.EPOCHS, stop, lambda r: {
        "val_score": round(float(r.get("val_score", float("nan"))), 4), "syn_gain": round(float(r.get("val_synth_gain", float("nan"))), 4)}))
    th.start()
    state = {}
    try:
        V.stage_train(C, Path(C.DATASET_DIR), run_dir, state)
    finally:
        stop.set()
    ck = torch.load(run_dir / "checkpoints" / "best.pth", map_location="cpu", weights_only=False)
    atomic_json(run_dir / "train_summary.json", {"best_epoch": ck.get("epoch"), "best_metrics": ck.get("metrics", {}),
                                                  "peak_gpu_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None})
    c.log(f"GAN {g} xong, best epoch {ck.get('epoch')}")


def cbctmar_nets():
    """Kien truc NGUYEN BAN CBCT_MAR (code.ipynb): UNet 5 tang ConvTranspose + tanh; PatchGAN 3 lop + sigmoid."""
    import torch
    import torch.nn as nn

    class UNetGenerator10(nn.Module):
        def __init__(self, in_channels=1):
            super().__init__()
            self.relu = nn.ReLU(True)
            self.tanh = nn.Tanh()
            self.enc1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
            self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
            self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
            self.enc4 = nn.Conv2d(256, 512, 4, 2, 1)
            self.enc5 = nn.Conv2d(512, 512, 4, 2, 1)
            self.dec1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
            self.dec2 = nn.ConvTranspose2d(1024, 256, 4, 2, 1)
            self.dec3 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
            self.dec4 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
            self.dec5 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
            self.out_conv = nn.Conv2d(64, in_channels, 3, 1, 1)

        def forward(self, x):
            e1 = self.relu(self.enc1(x)); e2 = self.relu(self.enc2(e1)); e3 = self.relu(self.enc3(e2))
            e4 = self.relu(self.enc4(e3)); e5 = self.relu(self.enc5(e4))
            d1 = self.relu(self.dec1(e5))
            d2 = self.relu(self.dec2(torch.cat([d1, e4], 1)))
            d3 = self.relu(self.dec3(torch.cat([d2, e3], 1)))
            d4 = self.relu(self.dec4(torch.cat([d3, e2], 1)))
            d5 = self.relu(self.dec5(torch.cat([d4, e1], 1)))
            return self.tanh(self.out_conv(d5))

    class Discriminator3(nn.Module):
        def __init__(self, in_channels=1):
            super().__init__()
            self.model = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid())

        def forward(self, x):
            return self.model(x)

    return UNetGenerator10, Discriminator3


def cbctmar_transforms():
    from torchvision import transforms
    tr = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    ev = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    return tr, ev


class _PngDS:
    def __init__(self, folder, tf):
        self.paths = [str(p) for p in list_images(folder)]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("L"))


def cbctmar_fast_step(nets, opts, adv, l1, lc, li, rn, rg, streams):
    """1 iteration huan luyen CBCT_MAR, TOAN HOC GIONG code goc (xem CFG cbctmar.fast_step). Tra ve tensor 9 loss:
    [loss_DX, loss_DY, loss_G, loss_F, cycle_G, cycle_F, id_G, id_F, total_G]."""
    import torch
    gen_G, gen_F, disc_X, disc_Y = nets
    oG, oF, oX, oY = opts
    sA, sB, d_params = streams
    cur = torch.cuda.current_stream()
    # ---- forward chung (co grad): G([rg, rn]) = [fake_n, id_n] ; F([rn, rg]) = [fake_g, id_g]  (2 nhanh doc lap tren 2 stream)
    sA.wait_stream(cur); sB.wait_stream(cur)
    with torch.cuda.stream(sA):
        o_g = gen_G(torch.cat([rg, rn]))
    with torch.cuda.stream(sB):
        o_f = gen_F(torch.cat([rn, rg]))
    cur.wait_stream(sA); cur.wait_stream(sB)
    fake_n, id_n = o_g[0:1], o_g[1:2]
    fake_g, id_g = o_f[0:1], o_f[1:2]
    # ---- buoc D (code goc: D_X roi D_Y, moi cai zero_grad -> backward -> step), dung ban detach cua fake
    for p_ in d_params:
        p_.requires_grad_(True)
    px = disc_X(torch.cat([rn, fake_n.detach()]))
    py = disc_Y(torch.cat([rg, fake_g.detach()]))
    loss_DX = 0.5 * (adv(px[0:1], torch.ones_like(px[0:1])) + adv(px[1:2], torch.zeros_like(px[1:2])))
    loss_DY = 0.5 * (adv(py[0:1], torch.ones_like(py[0:1])) + adv(py[1:2], torch.zeros_like(py[1:2])))
    oX.zero_grad(); loss_DX.backward(); oX.step()
    oY.zero_grad(); loss_DY.backward(); oY.step()
    # ---- buoc G+F (D da cap nhat; grad cua D khong can -> tat de nhanh hon, ket qua G/F khong doi)
    for p_ in d_params:
        p_.requires_grad_(False)
    oG.zero_grad(); oF.zero_grad()
    sA.wait_stream(cur); sB.wait_stream(cur)
    with torch.cuda.stream(sA):
        rec_n = gen_G(fake_g)            # nhanh F(rn) -> G
        dg = disc_Y(fake_g)
    with torch.cuda.stream(sB):
        rec_g = gen_F(fake_n)            # nhanh G(rg) -> F
        dn = disc_X(fake_n)
    cur.wait_stream(sA); cur.wait_stream(sB)
    loss_G = adv(dn, torch.ones_like(dn))
    loss_F = adv(dg, torch.ones_like(dg))
    cycle_G = l1(rec_n, rn)
    cycle_F = l1(rec_g, rg)
    id_G_l = l1(id_n, rn)
    id_F_l = l1(id_g, rg)
    total_G = loss_G + loss_F + lc * (cycle_G + cycle_F) + li * (id_G_l + id_F_l)
    total_G.backward()
    oG.step(); oF.step()
    for p_ in d_params:
        p_.requires_grad_(True)
    return torch.stack([v.detach() for v in (loss_DX, loss_DY, loss_G, loss_F, cycle_G, cycle_F, id_G_l, id_F_l, total_G)])


def task_cbctmar_train(c: Ctx):
    """Baseline 1: huan luyen CBCT_MAR dung nguyen vong lap cua code.ipynb (D_X, D_Y truoc; roi G+F), batch 1."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    cfg, mc = c.cfg, c.cfg["cbctmar"]
    out = c.R / "models" / "cbct_mar"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if mc.get("ckpt"):
        shutil.copy2(mc["ckpt"], out / "checkpoints" / "external_gen_G.pth")
        c.log(f"Dung checkpoint co san {mc['ckpt']}")
        return
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = bool(mc.get("fast_step"))   # gop batch 2 -> chon thuat toan conv toi uu (GPU dung rieng cho buoc nay)
    torch.set_num_threads(max(1, c.slots - int(mc["workers"])))
    limit_gpu(c)
    G_, D_ = cbctmar_nets()
    gen_G, gen_F, disc_X, disc_Y = G_().to(dev), G_().to(dev), D_().to(dev), D_().to(dev)
    tr, ev = cbctmar_transforms()
    tf = Path(cfg["paths"]["toothfairy"])
    kw = dict(batch_size=int(mc["batch"]), num_workers=int(mc["workers"]), pin_memory=True, persistent_workers=int(mc["workers"]) > 0)
    nl = DataLoader(_PngDS(tf / "Normal" / "Train", tr), shuffle=True, **kw)
    gl = DataLoader(_PngDS(tf / "Glare" / "Train", tr), shuffle=True, **kw)
    nvl = DataLoader(_PngDS(tf / "Normal" / "Val", ev), batch_size=1, shuffle=False)
    gvl = DataLoader(_PngDS(tf / "Glare" / "Val", ev), batch_size=1, shuffle=False)
    adv, l1 = nn.BCELoss(), nn.L1Loss()
    lc, li = float(mc["lambda_cycle"]), float(mc["lambda_identity"])
    b = (0.5, 0.999)
    oG = torch.optim.Adam(gen_G.parameters(), lr=mc["lr"], betas=b)
    oF = torch.optim.Adam(gen_F.parameters(), lr=mc["lr"], betas=b)
    oX = torch.optim.Adam(disc_X.parameters(), lr=mc["lr"], betas=b)
    oY = torch.optim.Adam(disc_Y.parameters(), lr=mc["lr"], betas=b)
    E = int(mc["epochs"])
    E_total = E + 1 if mc.get("inclusive_last") else E      # repo goc: range(START_EPOCH, EPOCHS + 1)
    milestones = set(int(x) for x in mc.get("milestones", []))
    m_every = int(mc.get("milestone_every") or 0)
    start, best_val = 0, float("inf")
    last = out / "checkpoints" / "last.pth"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        for m, k in ((gen_G, "G"), (gen_F, "F"), (disc_X, "DX"), (disc_Y, "DY"), (oG, "oG"), (oF, "oF"), (oX, "oX"), (oY, "oY")):
            m.load_state_dict(ck[k])
        start, best_val = ck["epoch"] + 1, ck.get("best_val", float("inf"))
        c.log(f"Tiep tuc CBCT_MAR tu epoch {start}")
    csv = out / "training_losses.csv"
    prog = c.prog(E_total, "epoch CBCT_MAR")
    prog.set(start, E_total)
    max_it = 40 if c.quick else None
    keys = ["disc_X_loss", "disc_Y_loss", "loss_G", "loss_F", "cycle_loss_G", "cycle_loss_F", "loss_identity_G",
            "loss_identity_F", "total_loss_G"]
    d_params = list(itertools.chain(disc_X.parameters(), disc_Y.parameters()))
    fast_streams = (torch.cuda.Stream(), torch.cuda.Stream(), d_params) if mc.get("fast_step") else None
    c.log(f"CBCT_MAR: {E_total} luot (epoch {start}..{E_total - 1}), batch {mc['batch']}, lambda_cycle {lc}, lambda_identity {li}, "
          f"lr {mc['lr']}, workers {mc['workers']}; moc luu 4 mang tai epoch {sorted(milestones)}, gen_G moi {m_every} epoch")
    for epoch in range(start, E_total):
        t0 = time.time()
        for m in (gen_G, gen_F, disc_X, disc_Y):
            m.train()
        acc = torch.zeros(len(keys), device=dev)      # cong don loss tren GPU, chi dong bo 1 lan / epoch (bo .item() moi iteration)
        nb = 0
        for rn, rg in zip(nl, gl):
            rn, rg = rn.to(dev, non_blocking=True), rg.to(dev, non_blocking=True)
            if mc.get("fast_step"):
                acc += cbctmar_fast_step((gen_G, gen_F, disc_X, disc_Y), (oG, oF, oX, oY), adv, l1, lc, li, rn, rg, fast_streams)
                nb += 1
                if max_it and nb >= max_it:
                    break
                continue
            fake_n = gen_G(rg).detach()
            fake_g = gen_F(rn).detach()
            oX.zero_grad()
            pr, pf = disc_X(rn), disc_X(fake_n)
            loss_DX = 0.5 * (adv(pr, torch.ones_like(pr)) + adv(pf, torch.zeros_like(pf)))
            loss_DX.backward(); oX.step()
            oY.zero_grad()
            pr, pf = disc_Y(rg), disc_Y(fake_g)
            loss_DY = 0.5 * (adv(pr, torch.ones_like(pr)) + adv(pf, torch.zeros_like(pf)))
            loss_DY.backward(); oY.step()
            oG.zero_grad(); oF.zero_grad()
            # Buoc G+F: trong code goc D cung nhan gradient (bi bo vi D.zero_grad() truoc lan backward cua D) -> tat requires_grad
            # cua D o day cho ket qua G/F GIONG HET (dao ham theo tham so G,F khong doi) nhung khong tinh grad trong so cua D.
            for p_ in d_params:
                p_.requires_grad_(False)
            fake_n = gen_G(rg)
            fake_g = gen_F(rn)
            dn, dg = disc_X(fake_n), disc_Y(fake_g)
            loss_G = adv(dn, torch.ones_like(dn))
            loss_F = adv(dg, torch.ones_like(dg))
            cycle_G = l1(gen_G(fake_g), rn)
            cycle_F = l1(gen_F(fake_n), rg)
            id_G = l1(gen_G(rn), rn)
            id_F = l1(gen_F(rg), rg)
            total_G = loss_G + loss_F + lc * (cycle_G + cycle_F) + li * (id_G + id_F)
            total_G.backward(); oG.step(); oF.step()
            for p_ in d_params:
                p_.requires_grad_(True)
            acc += torch.stack([v.detach() for v in (loss_DX, loss_DY, loss_G, loss_F, cycle_G, cycle_F, id_G, id_F, total_G)])
            nb += 1
            if max_it and nb >= max_it:
                break
        sums = {k: float(v) / max(nb, 1) for k, v in zip(keys, acc.tolist())}
        gen_G.eval(); gen_F.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for rn, rg in zip(nvl, gvl):
                rn, rg = rn.to(dev), rg.to(dev)
                tot += l1(gen_G(gen_F(rn)), rn).item() + l1(gen_F(gen_G(rg)), rg).item()
                cnt += 2
                if max_it and cnt >= 8:
                    break
        val_cyc = tot / max(cnt, 1)
        row = {"epoch": epoch, **sums, "val_cycle": val_cyc, "epoch_sec": time.time() - t0}
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        if val_cyc < best_val:
            best_val = val_cyc
            torch.save(gen_G.state_dict(), out / "checkpoints" / "gen_G_bestval.pth")
        # ten file theo QUY UOC REPO GOC: gen_G_epoch_{epoch} voi epoch la chi so 0-based cua vong lap
        if epoch in milestones:
            for m, k in ((gen_G, "gen_G"), (gen_F, "gen_F"), (disc_X, "disc_X"), (disc_Y, "disc_Y")):
                torch.save(m.state_dict(), out / "checkpoints" / f"{k}_epoch_{epoch}.pth")
        elif m_every and epoch > 0 and epoch % m_every == 0:
            torch.save(gen_G.state_dict(), out / "checkpoints" / f"gen_G_epoch_{epoch}.pth")
        if epoch in milestones or (m_every and epoch % m_every == 0) or epoch == E_total - 1:
            _cbctmar_progress_img(gen_G, ev, tf / "Glare" / "Val", out / "progress" / f"epoch_{epoch:03d}.png", dev)
        torch.save({"G": gen_G.state_dict(), "F": gen_F.state_dict(), "DX": disc_X.state_dict(), "DY": disc_Y.state_dict(),
                    "oG": oG.state_dict(), "oF": oF.state_dict(), "oX": oX.state_dict(), "oY": oY.state_dict(),
                    "epoch": epoch, "best_val": best_val}, str(last) + ".tmp")
        os.replace(str(last) + ".tmp", last)
        prog.set(epoch + 1, E_total, val_cycle=round(val_cyc, 4), G=round(sums["total_loss_G"], 3), epoch_sec=round(row["epoch_sec"], 1))
        c.log(f"Epoch {epoch}/{E_total - 1} | DX {sums['disc_X_loss']:.3f} DY {sums['disc_Y_loss']:.3f} G {sums['total_loss_G']:.3f} "
              f"ValCycle {val_cyc:.3f} | {row['epoch_sec']:.0f}s")
    for m, k in ((gen_G, "gen_G"), (gen_F, "gen_F"), (disc_X, "disc_X"), (disc_Y, "disc_Y")):
        torch.save(m.state_dict(), out / "checkpoints" / f"{k}_final.pth")
    atomic_json(out / "train_summary.json", {"epochs_constant": E, "passes": E_total, "best_val_cycle": best_val,
                                             "peak_gpu_gb": torch.cuda.max_memory_reserved() / 1e9})


def _cbctmar_progress_img(G, ev, folder, path, dev):
    import torch
    from PIL import Image
    ps = list_images(folder)
    if not ps:
        return
    img = Image.open(ps[0]).convert("L")
    with torch.no_grad():
        o = G(ev(img).unsqueeze(0).to(dev))[0, 0].cpu().numpy()
    o = cv2.resize(((o + 1) / 2 * 255).clip(0, 255).astype(np.uint8), img.size, interpolation=cv2.INTER_LANCZOS4)
    write_png(path, np.hstack([np.array(img), o]))


def yolo_cfgs(cfg) -> list:
    y = cfg["yolo"]
    out = []
    for m, e, s, b, lr in product(y["models"], y["epochs"], y["imgsz"], y["batch"], y["lr0"]):
        out.append({"name": f"{m}_e{e}_s{s}_b{b}_lr{lr:g}", "model": m, "epochs": int(e), "imgsz": int(s), "batch": int(b), "lr0": float(lr)})
    return out


def _yolo_short(name: str) -> str:
    return name.replace("yolo", "y").replace("-seg", "")


def _yolo_pretrained(model: str, wdir: Path):
    """Tra ve doi tuong YOLO: <model>.pt neu co trong so san (YOLO11-seg), neu khong dung <model>.yaml + chuyen trong so
    tu mo hinh detect cung co (vd YOLO12-seg: yolo12n-seg.yaml + yolo12n.pt)."""
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset
    wdir.mkdir(parents=True, exist_ok=True)
    try:
        p = attempt_download_asset(str(wdir / f"{model}.pt"))
        if Path(p).exists():
            return YOLO(str(p)), f"{model}.pt"
    except Exception:
        pass
    base = model.replace("-seg", "")
    m = YOLO(f"{model}.yaml")
    try:
        p = attempt_download_asset(str(wdir / f"{base}.pt"))
        if Path(p).exists():
            m = m.load(str(p))
            return m, f"{model}.yaml + {base}.pt"
    except Exception:
        pass
    return m, f"{model}.yaml (tu dau)"


def task_yolo_train(c: Ctx, name: str):
    import torch
    cfg = c.cfg
    y = next(x for x in yolo_cfgs(cfg) if x["name"] == name)
    proj = c.R / "models" / "yolo"
    run = proj / name
    data = c.R / "data" / "yolo" / "data.yaml"
    from ultralytics import YOLO
    limit_gpu(c)
    prog = c.prog(y["epochs"], f"epoch {name}")
    last = run / "weights" / "last.pt"
    t0 = time.time()
    resume = last.exists()
    if resume:
        model, src = YOLO(str(last)), "resume last.pt"
    else:
        model, src = _yolo_pretrained(y["model"], proj / "_pretrained")
    c.log(f"YOLO {name}: khoi tao tu {src}")
    if not resume:
        run.mkdir(parents=True, exist_ok=True)
        (run / "init_source.txt").write_text(src, encoding="utf-8")

    def cb(tr):
        m = getattr(tr, "metrics", {}) or {}
        prog.set(tr.epoch + 1, tr.epochs, mask_map50=round(float(m.get("metrics/mAP50(M)", float("nan"))), 4))
    model.add_callback("on_fit_epoch_end", cb)
    kw = dict(data=str(data), epochs=y["epochs"], imgsz=y["imgsz"], batch=y["batch"], lr0=y["lr0"],
              patience=int(cfg["yolo"]["patience"]), device=0, workers=int(cfg["yolo"]["workers"]), project=str(proj),
              name=name, exist_ok=True, amp=False, seed=int(cfg["seed"]), deterministic=False, degrees=10.0, fliplr=0.5,
              hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, cache=False, plots=True, verbose=False)
    if resume:
        try:
            model.train(resume=True)
        except Exception as e:
            if (run / "weights" / "best.pt").exists() and "finished" in str(e).lower():
                c.log("Da train xong truoc do -> chi danh gia lai")
            else:
                c.log(f"Khong resume duoc ({e}) -> train lai tu dau")
                model, src = _yolo_pretrained(y["model"], proj / "_pretrained")
                (run / "init_source.txt").write_text(src, encoding="utf-8")
                resume = False
                model.add_callback("on_fit_epoch_end", cb)
                model.train(**kw)
    else:
        model.train(**kw)
    best = run / "weights" / "best.pt"
    if resume and (run / "init_source.txt").exists():    # ghi dung trong so KHOI TAO ban dau, khong phai "last.pt"
        src = (run / "init_source.txt").read_text(encoding="utf-8").strip() + " (tiep tuc sau khi bi ngat)"
    vm = YOLO(str(best)).val(data=str(data), imgsz=y["imgsz"], batch=y["batch"], device=0, plots=False, verbose=False)
    res = {"name": name, **y, "init": src, "train_time_s": time.time() - t0,
           "mask_map50": float(vm.seg.map50), "mask_map50_95": float(vm.seg.map),
           "box_map50": float(vm.box.map50), "box_map50_95": float(vm.box.map),
           "mask_map50_95_per_class": [float(x) for x in vm.seg.maps],
           "peak_gpu_gb": torch.cuda.max_memory_reserved() / 1e9}
    atomic_json(run / "train_done.json", res)
    c.log(f"YOLO {name} xong: mask mAP50 {res['mask_map50']:.3f}, mAP50-95 {res['mask_map50_95']:.3f}")


# ------------------------------------------------------------ [3] SUY LUAN ----
def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def task_infer_gan(c: Ctx, g: str):
    import torch
    limit_gpu(c)
    V = import_v12()
    diff = {k: (v, getattr(V.CFG, k)) for k, v in P.items() if hasattr(V.CFG, k) and getattr(V.CFG, k) != v}
    if diff:
        c.log(f"[CANH BAO] hang so hau xu ly khac V12: {diff}")
    run_dir = c.R / "models" / "ours_gan" / g
    ck_dir = run_dir / "checkpoints"
    order = ["external.pth", "best.pth", "final_weights.pth"] if c.cfg["gan"]["weights"] == "best" else ["external.pth", "final_weights.pth", "best.pth"]
    ck = next(ck_dir / n for n in order if (ck_dir / n).exists())
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    dev = torch.device("cuda")
    G = V.UNetGenerator10(1).to(dev)
    G.load_state_dict(obj.get("gen_G") or obj.get("G"))
    G.eval()
    c.log(f"GAN {g}: trong so {ck.name} (epoch {obj.get('epoch', '?')})")
    sets = present_sets(c.R)
    total = sum(len(eval_names(c.R, s)) for s in sets)
    prog = c.prog(total, f"GAN {g} suy luan")
    S = P["IMG_SIZE"]
    for s in sets:
        cache = c.R / "outputs" / s / "_cache" / f"gan_{g}"
        rows = []
        names = eval_names(c.R, s)
        for batch in _batches(names, 32):
            todo = [n for n in batch if not (cache / n).exists()]
            if todo:
                imgs = [read_u8(eval_root(c.R, s) / "input" / n) for n in todo]
                x = np.stack([cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA if max(im.shape) > S else cv2.INTER_LINEAR) for im in imgs])
                t = torch.from_numpy(x).float().div(127.5).sub(1).unsqueeze(1).to(dev)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    out, m = G(t)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000 / len(todo)
                out = ((out[:, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                mk = (m[:, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                for n, im, o, a in zip(todo, imgs, out, mk):
                    h, w = im.shape
                    write_png(cache / n, cv2.resize(o, (w, h), interpolation=cv2.INTER_LANCZOS4))
                    write_png(cache / "_att" / n, cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR))
                    rows.append({"file": n, "ms": dt})
            prog.update(len(batch))
        _append_timing(c.R / "outputs" / s / "_timing" / f"gan_{g}.csv", rows)


def _append_timing(path: Path, rows: list):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df]).drop_duplicates("file", keep="last")
    df.to_csv(path, index=False)


def _cbctmar_infer(c: Ctx, ck: Path, method: str, timing: str):
    """Suy luan generator G cua CBCT_MAR (trong so `ck`) tren MOI tap danh gia -> outputs/<tap>/<method>/ (bo qua anh da co)."""
    import torch
    from PIL import Image
    G_, _ = cbctmar_nets()
    dev = torch.device("cuda")
    G = G_().to(dev)
    G.load_state_dict(torch.load(ck, map_location=dev))
    G.eval()
    c.log(f"CBCT_MAR -> {method}: trong so {ck.name}")
    _, ev = cbctmar_transforms()
    sets = present_sets(c.R)
    prog = c.prog(sum(len(eval_names(c.R, s)) for s in sets), f"{method} suy luan")
    for s in sets:
        od = c.R / "outputs" / s / method
        rows = []
        names = eval_names(c.R, s)
        for batch in _batches(names, 64):
            todo = [n for n in batch if not (od / n).exists()]
            if todo:
                imgs = [Image.open(eval_root(c.R, s) / "input" / n).convert("L") for n in todo]
                t = torch.stack([ev(im) for im in imgs]).to(dev)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    out = G(t)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000 / len(todo)
                out = ((out[:, 0].cpu().numpy() + 1) / 2).clip(0, 1)
                for n, im, o in zip(todo, imgs, out):
                    pil = Image.fromarray((o * 255).round().astype(np.uint8)).resize(im.size, Image.Resampling.LANCZOS)
                    write_png(od / n, np.array(pil))
                    rows.append({"file": n, "ms": dt})
            prog.update(len(batch))
        _append_timing(c.R / "outputs" / s / "_timing" / f"{timing}.csv", rows)


def task_infer_cbctmar(c: Ctx):
    limit_gpu(c)
    ck_dir = c.R / "models" / "cbct_mar" / "checkpoints"
    w = c.cfg["cbctmar"]["weights"]
    cands = ["external_gen_G.pth", "gen_G_bestval.pth", "gen_G_final.pth"] if w == "bestval" else ["external_gen_G.pth", "gen_G_final.pth", "gen_G_bestval.pth"]
    ck = next(ck_dir / n for n in cands if (ck_dir / n).exists())
    _cbctmar_infer(c, ck, "cbct_mar", "cbct_mar")


def cbctmar_epoch_methods(R: Path, cfg) -> list:
    """[(epoch, ten_method, duong_dan_ckpt)] cho cac moc epoch da co checkpoint gen_G_epoch_<N>.pth."""
    ck_dir = R / "models" / "cbct_mar" / "checkpoints"
    out = []
    for n in sorted(int(x) for x in cfg["cbctmar"].get("milestones", [])):
        f = ck_dir / f"gen_G_epoch_{n}.pth"
        if f.exists():
            out.append((n, f"cbct_mar_e{n:03d}", f))
    return out


def task_infer_cbctmar_epochs(c: Ctx):
    limit_gpu(c)
    items = cbctmar_epoch_methods(c.R, c.cfg)
    c.log(f"CBCT_MAR theo moc epoch: {[i[0] for i in items]}")
    for n, m, f in items:
        _cbctmar_infer(c, f, m, m)


def task_infer_yolo(c: Ctx, name: str):
    from ultralytics import YOLO
    import torch
    limit_gpu(c)
    y = next(x for x in yolo_cfgs(c.cfg) if x["name"] == name)
    model = YOLO(str(c.R / "models" / "yolo" / name / "weights" / "best.pt"))
    floor = float(c.cfg["yolo"]["pred_conf_floor"])
    sets = present_sets(c.R)
    prog = c.prog(sum(len(eval_names(c.R, s)) for s in sets), f"YOLO {name} suy luan")
    for s in sets:
        cache = c.R / "outputs" / s / "_cache" / f"yolo_{name}"
        vis = c.R / "outputs" / s / f"yolo_masks_{_yolo_short(name)}"
        rows = []
        names = eval_names(c.R, s)
        for batch in _batches(names, 16):
            todo = [n for n in batch if not (cache / (Path(n).stem + ".npz")).exists()]
            if todo:
                imgs = [read_u8(eval_root(c.R, s) / "input" / n) for n in todo]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                res = model.predict([cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in imgs], imgsz=y["imgsz"], conf=floor,
                                    iou=0.6, retina_masks=True, verbose=False, device=0)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000 / len(todo)
                for n, im, r in zip(todo, imgs, res):
                    h, w = im.shape
                    if r.masks is not None and len(r.boxes):
                        mk = r.masks.data.cpu().numpy() > 0.5
                        if mk.shape[1:] != (h, w):
                            mk = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in mk])
                        cls = r.boxes.cls.cpu().numpy().astype(np.int16)
                        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
                    else:
                        mk, cls, conf = np.zeros((0, h, w), bool), np.zeros(0, np.int16), np.zeros(0, np.float32)
                    cache.mkdir(parents=True, exist_ok=True)
                    tmp = cache / (Path(n).stem + f".{os.getpid()}.tmp.npz")
                    np.savez_compressed(tmp, masks=np.packbits(mk, axis=-1), shape=np.array(mk.shape), cls=cls, conf=conf)
                    os.replace(tmp, cache / (Path(n).stem + ".npz"))
                    metal, tooth = yolo_masks_from(mk, cls, conf, 0.25, (h, w))
                    v = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                    v[tooth > 0] = (0.5 * v[tooth > 0] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
                    v[metal > 0] = (0.4 * v[metal > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
                    write_png(vis / n, v)
                    rows.append({"file": n, "ms": dt})
            prog.update(len(batch))
        _append_timing(c.R / "outputs" / s / "_timing" / f"yolo_{name}.csv", rows)


def load_yolo_npz(path: Path):
    z = np.load(path)
    shp = tuple(int(x) for x in z["shape"])
    mk = np.unpackbits(z["masks"], axis=-1, count=shp[2]).astype(bool).reshape(shp) if shp[0] else np.zeros(shp, bool)
    return mk, z["cls"], z["conf"]


def yolo_masks_from(mk, cls, conf, thr, hw):
    metal = np.zeros(hw, np.uint8)
    tooth = np.zeros(hw, np.uint8)
    for m, k, p in zip(mk, cls, conf):
        if p < thr:
            continue
        if int(k) == 0:
            metal[m] = 255
        else:
            tooth[m] = 255
    sc = max(hw) / 256.0
    d = max(0, int(round(P["TOOTH_BODY_DILATE_PX"] * sc)))
    tooth = cv2.bitwise_or(tooth, metal)
    if d:
        tooth = cv2.dilate(tooth, ellipse(2 * d + 1))
    return metal, tooth


def ours_variants(cfg) -> list:
    """Luoi bien the phuong phap cua ban: (gan x nguon mask x conf x radius x clean)."""
    oc = cfg["ours"]
    srcs = [(yc["name"], _yolo_short(yc["name"])) for yc in yolo_cfgs(cfg)]
    out = []
    for g in cfg["gan"]["variants"]:
        for (ysrc, ys), conf, rad, cl in product(srcs, oc["conf"], oc["radius"], oc["clean"]):
            out.append({"method": f"ours_{g}_{ys}_c{int(round(conf * 100)):02d}_r{rad}_{cl}", "gan": g, "mask": ysrc,
                        "conf": conf, "radius": rad, "clean": cl})
        if oc.get("heuristic", True):
            for rad, cl in product(oc["radius"], oc["clean"]):
                out.append({"method": f"ours_{g}_heur_r{rad}_{cl}", "gan": g, "mask": "heur", "conf": None, "radius": rad, "clean": cl})
    return out


def _classical_worker(a):
    R, s, n, angles = a
    R = Path(R)
    outs = [R / "outputs" / s / m / n for m in ("limar", "nmar")]
    if all(p.exists() for p in outs):
        return None
    img = read_u8(eval_root(R, s) / "input" / n)
    t0 = time.perf_counter()
    li, nm = classic_mar(img, tooth_metal(img), angles)
    dt = (time.perf_counter() - t0) * 1000
    write_png(outs[0], li)
    write_png(outs[1], nm)
    return {"file": n, "ms": dt}


def task_classical(c: Ctx, s: str):
    names = eval_names(c.R, s)
    items = [(str(c.R), s, n, int(c.cfg["classical"]["angles"])) for n in names]
    c.log(f"NMAR + LI-MAR ({c.cfg['classical']['angles']} goc) cho {s}: {len(items)} anh, {c.slots} tien trinh")
    rows = [r for r in run_pool(_classical_worker, items, c.slots, c.prog(len(items), f"NMAR/LI-MAR {s}"), c.log) if r]
    # NMAR va LI-MAR tinh CHUNG 1 lan chieu Radon -> chia deu thoi gian cho moi phuong phap
    _append_timing(c.R / "outputs" / s / "_timing" / "nmar.csv", [{"file": r["file"], "ms": r["ms"] / 2} for r in rows])
    _append_timing(c.R / "outputs" / s / "_timing" / "limar.csv", [{"file": r["file"], "ms": r["ms"] / 2} for r in rows])


def _ours_worker(a):
    R, s, n, variants = a
    R = Path(R)
    todo = [v for v in variants if not (R / "outputs" / s / v["method"] / n).exists()]
    if not todo:
        return []
    img = read_u8(eval_root(R, s) / "input" / n)
    hw = img.shape
    gans = {}
    ycache = {}
    rows = []
    t0 = time.perf_counter()
    hm = heuristic_masks(img)
    heur_ms = (time.perf_counter() - t0) * 1000
    for v in todo:
        if v["gan"] not in gans:
            gans[v["gan"]] = read_u8(R / "outputs" / s / "_cache" / f"gan_{v['gan']}" / n)
        t0 = time.perf_counter()
        if v["mask"] == "heur":
            metal, tooth = hm
            extra = heur_ms
        else:
            if v["mask"] not in ycache:
                ycache[v["mask"]] = load_yolo_npz(R / "outputs" / s / "_cache" / f"yolo_{v['mask']}" / (Path(n).stem + ".npz"))
            metal, tooth = yolo_masks_from(*ycache[v["mask"]], float(v["conf"]), hw)
            extra = 0.0
        out = ours_compose(img, gans[v["gan"]], metal, tooth, v["radius"], v["clean"])
        write_png(R / "outputs" / s / v["method"] / n, out)
        rows.append({"method": v["method"], "file": n, "post_ms": (time.perf_counter() - t0) * 1000 + extra})
    return rows


def task_ours_post(c: Ctx, s: str):
    vs = ours_variants(c.cfg)
    names = eval_names(c.R, s)
    items = [(str(c.R), s, n, vs) for n in names]
    c.log(f"Phuong phap cua ban: {len(vs)} bien the x {len(names)} anh ({s}), {c.slots} tien trinh")
    res = run_pool(_ours_worker, items, c.slots, c.prog(len(items), f"ours {s}"), c.log)
    rows = [r for rr in res for r in rr]
    if rows:
        df = pd.DataFrame(rows)
        for m, g in df.groupby("method"):
            _append_timing(c.R / "outputs" / s / "_timing" / f"post_{m}.csv", g[["file"]].assign(ms=g["post_ms"]).to_dict("records"))


# ------------------------------------------------------------ [4] DANH GIA ----
def list_methods(R: Path, s: str) -> list:
    d = R / "outputs" / s
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("_") and not p.name.startswith("yolo_masks_"))


def _metrics_worker(a):
    R, s, n, methods, has_gt, seg_cfgs = a
    R = Path(R)
    root = eval_root(R, s)
    inp = read_u8(root / "input" / n)
    rows = []
    if has_gt:
        gt, mgt = read_u8(root / "gt" / n), read_u8(root / "metal_gt" / n)
        mk = synth_masks(inp, gt, mgt)
        for m in ["input"] + methods:
            p = root / "input" / n if m == "input" else R / "outputs" / s / m / n
            if p.exists():
                rows.append({"set": s, "file": n, "method": m, **synth_metrics(read_u8(p), inp, gt, mk)})
        # chat luong tach kim loai (Dice/IoU voi mask kim loai gia lap CHINH XAC)
        g = mgt > 0
        segs = [("seg_heur", tooth_metal(inp) > 0)]
        for yname, confs in seg_cfgs:
            f = R / "outputs" / s / "_cache" / f"yolo_{yname}" / (Path(n).stem + ".npz")
            if f.exists():
                z = load_yolo_npz(f)
                for cf in confs:
                    segs.append((f"seg_{_yolo_short(yname)}_c{int(round(cf * 100)):02d}", yolo_masks_from(*z, cf, inp.shape)[0] > 0))
        for m, pm in segs:
            inter, uni = float((pm & g).sum()), float((pm | g).sum())
            rows.append({"set": s, "file": n, "method": m, "dice": 2 * inter / max(pm.sum() + g.sum(), 1),
                         "iou": inter / max(uni, 1)})
    else:
        mk = real_masks(inp)
        for m in ["input"] + methods:
            p = root / "input" / n if m == "input" else R / "outputs" / s / m / n
            if p.exists():
                rows.append({"set": s, "file": n, "method": m, **real_metrics(read_u8(p), inp, mk)})
    return rows


def _timing_table(R: Path, s: str, cfg) -> pd.DataFrame:
    """ms/anh moi phuong phap: ours = GAN + (YOLO) + hau xu ly; con lai doc truc tiep."""
    tdir = R / "outputs" / s / "_timing"
    read = lambda f: pd.read_csv(f).set_index("file")["ms"] if f.exists() else None
    rows = []
    extra = sorted(f.stem for f in tdir.glob("cbct_mar_e*.csv")) if tdir.is_dir() else []
    for m in ["cbct_mar", "nmar", "limar"] + extra:
        t = read(tdir / f"{m}.csv")
        if t is not None:
            rows += [{"file": k, "method": m, "time_ms": v} for k, v in t.items()]
    for v in ours_variants(cfg):
        post = read(tdir / f"post_{v['method']}.csv")
        gan = read(tdir / f"gan_{v['gan']}.csv")
        if post is None or gan is None:
            continue
        tot = post.add(gan, fill_value=0)
        if v["mask"] != "heur":
            yt = read(tdir / f"yolo_{v['mask']}.csv")
            if yt is not None:
                tot = tot.add(yt, fill_value=0)
        rows += [{"file": k, "method": v["method"], "time_ms": x} for k, x in tot.items()]
    return pd.DataFrame(rows)


def task_metrics(c: Ctx, s: str):
    methods = list_methods(c.R, s)
    has_gt = eval_has_gt(c.R, s)
    seg_cfgs = [(y["name"], c.cfg["ours"]["conf"]) for y in yolo_cfgs(c.cfg)]
    names = eval_names(c.R, s)
    items = [(str(c.R), s, n, methods, has_gt, seg_cfgs) for n in names]
    c.log(f"Metrics {s}: {len(methods) + 1} phuong phap x {len(names)} anh, {c.slots} tien trinh")
    res = run_pool(_metrics_worker, items, c.slots, c.prog(len(items), f"metrics {s}"), c.log)
    df = pd.DataFrame([r for rr in res for r in rr])
    tt = _timing_table(c.R, s, c.cfg)
    if len(tt):
        df = df.merge(tt, on=["file", "method"], how="left")
    out = c.R / "metrics" / "per_set"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{s}.csv", index=False)


def _method_label(m: str, best: str | None = None) -> str:
    if m == "input":
        return "Original"
    if best and m == best:
        return "Ours (best)"
    mm = re.fullmatch(r"cbct_mar_e(\d+)", m)
    if mm:
        return f"CBCT_MAR (epoch {int(mm.group(1))})"
    return {"cbct_mar": "CBCT_MAR [B1]", "cbct_mar_60ep_run1": "CBCT_MAR (60 epoch, lan chay cu)", "nmar": "NMAR [B2]",
            "limar": "LI-MAR [B3]", "gt": "Clean (GT)"}.get(m, m)


def task_select(c: Ctx):
    R, cfg = c.R, c.cfg
    per = [pd.read_csv(f) for f in sorted((R / "metrics" / "per_set").glob("*.csv"))]
    df = pd.concat(per, ignore_index=True)
    df.to_csv(R / "metrics" / "metrics_per_image.csv", index=False)
    num = [x for x in df.columns if x not in ("set", "file", "method")]
    summ = df.groupby(["set", "method"])[num].agg(["mean", "std", "count"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(R / "metrics" / "summary.csv", index=False)
    means = df.groupby(["set", "method"])[num].mean()
    # ---- chon cau hinh "ours" tot nhat CHI tren tap VAL (khong nhin tap test) - trung binh hang tren cac metric
    vs = [v["method"] for v in ours_variants(cfg)]
    rank_cols = {}
    for s, met, direction in cfg["select"]["metrics"]:
        if (s not in means.index.get_level_values(0)) or met not in means.columns:
            continue
        col = means.loc[s][met].reindex(vs)
        if col.notna().sum() == 0:
            continue
        rank_cols[f"{s}:{met}"] = col.rank(ascending=(direction == "min"))
        rank_cols[f"{s}:{met}:value"] = col
    sel = pd.DataFrame(rank_cols, index=vs)
    rk = [k for k in sel.columns if not k.endswith(":value")]
    sel["mean_rank"] = sel[rk].mean(axis=1) if rk else np.nan
    params0 = {v["method"]: v for v in ours_variants(cfg)}
    for k in ("gan", "mask", "conf", "radius", "clean"):
        sel[k] = [params0[m][k] for m in sel.index]
    sel = sel.sort_values("mean_rank")
    sel.index.name = "method"
    sel.to_csv(R / "metrics" / "ours_selection.csv")
    forced = cfg.get("_ours_override")
    params = {v["method"]: v for v in ours_variants(cfg)}
    order = [m for m in sel.index if pd.notna(sel.loc[m, "mean_rank"])] or list(sel.index)
    pool = cfg["select"].get("ours_pool", "yolo")
    cand = [m for m in order if pool == "all" or params[m]["mask"] != "heur"] or order
    best = forced if forced else (cand[0] if cand else None)
    bp = params.get(best, {})
    heur_twin = f"ours_{bp.get('gan')}_heur_r{bp.get('radius')}_{bp.get('clean')}" if bp else None
    best_per_mask = {}
    for m in order:
        best_per_mask.setdefault(params[m]["mask"], m)
    atomic_json(R / "metrics" / "best_ours.json", {"method": best, "params": bp, "ablation_no_yolo": heur_twin,
                                                   "best_per_mask": best_per_mask, "best_overall": order[0] if order else None,
                                                   "ours_pool": pool, "selected_on": cfg["select"]["metrics"], "forced": bool(forced)})
    c.log(f"Cau hinh 'Ours' (chon tren VAL, pool={pool}): {best} | tot nhat moi nguon mask: {best_per_mask}")
    # ---- bang xep hang day du moi tap + bang so sanh chinh cho paper
    paper = R / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    def ours_label(m):
        v = params.get(m, {})
        if not v:
            return m
        src = "w/o YOLO (rule masks)" if v["mask"] == "heur" else f"+ {_yolo_short(v['mask']).split('_')[0].upper()} (conf {v['conf']})"
        return f"Ours {src}, r={v['radius']}, clean={v['clean']}" + (" [selected]" if m == best else "")
    main_methods = []
    for m in (["input", best] + [best_per_mask[k] for k in best_per_mask if k != "heur"] + [heur_twin, best_per_mask.get("heur"),
              "cbct_mar", "nmar", "limar"]):
        if m and m not in main_methods:
            main_methods.append(m)
    for s in means.index.get_level_values(0).unique():
        m = means.loc[s]
        m.to_csv(R / "metrics" / f"leaderboard_{s}.csv")
        synth = "psnr" in m.columns and m["psnr"].notna().any()
        cols = (["psnr", "ssim", "mae", "mae_noise", "lp_noise", "gain", "mae_outside", "edge_bone", "time_ms"] if synth
                else ["streak_ratio", "halo_red", "edge_prot", "ssim_prot", "far_change", "time_ms"])
        cols = [x for x in cols if x in m.columns]
        sd = df[df["set"] == s].groupby("method")[cols].std()
        rows = []
        for mm in main_methods:
            if mm is None or mm not in m.index:
                continue
            lab = ours_label(mm) if mm in params else _method_label(mm)
            rows.append({"Method": lab, **{k: f"{m.loc[mm, k]:.4g} ± {sd.loc[mm, k]:.3g}" if pd.notna(m.loc[mm, k]) else "-" for k in cols}})
        t = pd.DataFrame(rows)
        t.to_csv(paper / f"table_{s}.csv", index=False)
        try:
            (paper / f"table_{s}.tex").write_text(t.to_latex(index=False, escape=True), encoding="utf-8")
        except Exception:
            pass
    # ---- YOLO: mAP (val YOLO) + Dice kim loai tren tap nhieu tu tao + thoi gian
    yrows = []
    for y in yolo_cfgs(cfg):
        d = read_json(R / "models" / "yolo" / y["name"] / "train_done.json", {}) or {}
        row = {"yolo": y["name"], "train_time": fmt_dur(d.get("train_time_s")), "mask_mAP50": d.get("mask_map50"),
               "mask_mAP50-95": d.get("mask_map50_95"), "box_mAP50": d.get("box_map50")}
        for s in ("tf_synth_test", "cvan_synth", "truong_synth"):
            for i, cf in enumerate(cfg["ours"]["conf"]):
                k = f"seg_{_yolo_short(y['name'])}_c{int(round(cf * 100)):02d}"
                if s in means.index.get_level_values(0) and k in means.loc[s].index:
                    row[f"metal_dice_{s}" if i == 0 else f"metal_dice_{s}@conf{cf}"] = means.loc[s].loc[k, "dice"]
        yrows.append(row)
    hrow = {"yolo": "rule-based (tooth_metal, khong train)"}
    for s in ("tf_synth_test", "cvan_synth", "truong_synth"):
        if s in means.index.get_level_values(0) and "seg_heur" in means.loc[s].index:
            hrow[f"metal_dice_{s}"] = means.loc[s].loc["seg_heur", "dice"]
    pd.DataFrame(yrows + [hrow]).to_csv(paper / "table_yolo.csv", index=False)
    # ---- thoi gian moi phuong phap (ms/anh) + thoi gian huan luyen
    tm = df.dropna(subset=["time_ms"]) if "time_ms" in df.columns else pd.DataFrame()
    if len(tm):
        tm[tm["method"].isin([x for x in main_methods if x])].groupby(["method", "set"])["time_ms"].mean().unstack().to_csv(paper / "table_time_ms.csv")
    st = read_json(R / "state.json", {}) or {}
    pd.DataFrame([{"task": k, "status": v.get("status"), "elapsed_s": v.get("elapsed_s"), "slots": v.get("slots"),
                   "gpu_gb_est": v.get("gpu")} for k, v in (st.get("tasks") or {}).items()]).to_csv(paper / "table_task_time.csv", index=False)


def _panel(img, label, w=None):
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    if w and v.shape[1] != w:
        v = cv2.resize(v, (w, int(round(v.shape[0] * w / v.shape[1]))), interpolation=cv2.INTER_AREA)
    bar = np.zeros((30, v.shape[1], 3), np.uint8)
    scale = 0.55 if v.shape[1] >= 300 else 0.42
    cv2.putText(bar, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, v])


def _compie_worker(a):
    R, s, n, methods, labels, has_gt = a
    R = Path(R)
    root = eval_root(R, s)
    panels = []
    for m, lab in zip(methods, labels):
        p = root / "input" / n if m == "input" else (root / "gt" / n if m == "gt" else R / "outputs" / s / m / n)
        if m == "gt" and not has_gt:
            continue
        img = read_u8(p) if p.exists() else np.zeros_like(read_u8(root / "input" / n))
        panels.append(_panel(img, lab if p.exists() else f"{lab} (chua co)"))
    h = max(x.shape[0] for x in panels)
    panels = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in panels]
    write_png(R / "compie" / s / n, np.hstack(panels))
    return n


def task_compie(c: Ctx):
    R, cfg = c.R, c.cfg
    best = (read_json(R / "metrics" / "best_ours.json", {}) or {}).get("method") or ours_variants(cfg)[0]["method"]
    methods = ["input", best] + list(cfg["compie"]["baselines"]) + (["gt"] if cfg["compie"]["add_gt"] else [])
    labels = ["Original", "Ours"] + [_method_label(m) for m in cfg["compie"]["baselines"]] + ["Clean (GT)"]
    items = []
    for s in present_sets(R):
        items += [(str(R), s, n, methods, labels, eval_has_gt(R, s)) for n in eval_names(R, s)]
    c.log(f"compie: {len(items)} anh, thu tu {labels}")
    run_pool(_compie_worker, items, c.slots, c.prog(len(items), "compie"), c.log)
    k = int(cfg["compie"]["sheet_n"])
    for s in present_sets(R):
        ims = [read_u8(R / "compie" / s / n) for n in even_pick(eval_names(R, s), k) if (R / "compie" / s / n).exists()]
        if ims:
            w = max(x.shape[1] for x in ims)
            ims = [cv2.copyMakeBorder(x, 0, 4, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=255) for x in ims]
            sheet = np.vstack(ims)
            write_png(R / "compie" / f"_sheet_{s}.png", cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA))


def task_report(c: Ctx):
    R = c.R
    best = read_json(R / "metrics" / "best_ours.json", {}) or {}
    st = read_json(R / "state.json", {}) or {}
    has_b1 = any((R / "outputs" / s / "cbct_mar").is_dir() and any((R / "outputs" / s / "cbct_mar").iterdir()) for s in present_sets(R))
    L = ["# Bao cao thi nghiem CBCT-MAR", "", f"Tao luc: {now()}  |  thu muc: `{R}`", "",
         "Trang thai: " + ("DAY DU (co CBCT_MAR)." if has_b1 else "BAN SOM - CBCT_MAR (baseline 1) chua train xong; bang/compie se "
                           "tu cap nhat khi xong."), ""]
    L += ["## Cau hinh tot nhat cua phuong phap cua ban (chon tren tap VAL)", "",
          f"- 'Ours' (bien the {'YOLO' if best.get('ours_pool') == 'yolo' else 'bat ky'} tot nhat): `{best.get('method')}`",
          f"- Tham so: `{json.dumps(best.get('params'), ensure_ascii=False)}`",
          f"- Tot nhat theo tung nguon mask: `{json.dumps(best.get('best_per_mask'), ensure_ascii=False)}`",
          f"- Tot nhat tong the (ke ca khong YOLO): `{best.get('best_overall')}`",
          f"- Ablation khong YOLO (cung r/clean): `{best.get('ablation_no_yolo')}`",
          "- Xep hang day du (trung binh hang tren cac metric tap VAL): `metrics/ours_selection.csv`", ""]
    for s in EVAL_SETS:
        f = R / "paper" / f"table_{s}.csv"
        if f.exists():
            t = pd.read_csv(f)
            L += [f"## {s} ({len(eval_names(R, s))} anh)", "", _md_table(t), ""]
    f = R / "paper" / "table_yolo.csv"
    if f.exists():
        L += ["## YOLO (tach kim loai/rang)", "", _md_table(pd.read_csv(f)), ""]
    rows = [{"task": k, "status": v.get("status"), "time": fmt_dur(v.get("elapsed_s"))} for k, v in (st.get("tasks") or {}).items()]
    if rows:
        L += ["## Thoi gian cac buoc", "", _md_table(pd.DataFrame(rows)), ""]
    L += ["## Chu thich metric", "",
          "- Tap nhieu tu tao (co anh sach): psnr/ssim/mae tinh tren toan anh tru kim loai gia lap; mae_noise/lp_noise = sai so trong "
          "vung nhieu (lp = sau loc Gauss 3px, do quang sang/vet rong); gain = 1 - mae_noise/mae_noise(input); mae_outside = thay doi "
          "ngoai vung nhieu (giu nguyen); edge_bone = tuong quan bien Sobel o xuong/rang so voi anh sach.",
          "- Tap that (khong co anh sach): streak_ratio = nang luong tan so cao vung mo mem gan kim loai so voi input (thap = it vet soc); "
          "halo_red = % giam sang quang sang; edge_prot/ssim_prot = giu cau truc xuong/rang so voi input; far_change = thay doi vung "
          "mo mem XA kim loai (thap = it 'bia').",
          "- time_ms: thoi gian suy luan / anh (GPU cho mang, CPU 1 luong cho hau xu ly/NMAR/LI-MAR).", "",
          "## Thu muc", "", "- `compie/<tap>/`: Original -> Ours -> CBCT_MAR -> NMAR -> LI-MAR (-> Clean GT)",
          "- `outputs/<tap>/<method>/`: anh ket qua MOI bien the (chon anh dua vao paper)",
          "- `metrics/`, `paper/`: bang so lieu (CSV, LaTeX)", "- `models/`: checkpoint GAN (ours), CBCT_MAR, YOLO"]
    (R / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    c.log(f"REPORT.md -> {R / 'REPORT.md'}")


# ---------------------------------------------------------------- REPORT_FULL ----
SYN_KEYS = [("psnr", "PSNR (dB)", "max"), ("ssim", "SSIM", "max"), ("mae_noise", "MAE vùng nhiễu", "min"),
            ("lp_noise", "MAE vùng nhiễu (lọc Gauss 3px)", "min"), ("gain", "Gain", "max"),
            ("edge_bone", "Biên xương/răng", "max"), ("mae_outside", "Thay đổi ngoài vùng nhiễu", "min")]
REAL_KEYS = [("streak_ratio", "Vệt sọc còn lại", "min"), ("halo_red", "Giảm quầng sáng", "max"),
             ("edge_prot", "Giữ biên xương/răng", "max"), ("ssim_prot", "SSIM xương/răng", "max"), ("far_change", "Thay đổi xa kim loại", "min")]
MAIN_KEYS = {"synth": ["psnr", "lp_noise", "edge_bone"], "real": ["streak_ratio", "edge_prot"]}


def _set_kind(s: str) -> str:
    return "synth" if s.endswith("_synth") or "_synth_" in s else "real"


def _fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}g}"


def _get_mean(df, s, m, key):
    if df is None:
        return float("nan")
    r = df[(df["set"] == s) & (df["method"] == m)]
    col = f"{key}_mean"
    return float(r.iloc[0][col]) if len(r) and col in r.columns and pd.notna(r.iloc[0][col]) else float("nan")


def _dir_size(p: Path) -> str:
    try:
        out = subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True, timeout=120).stdout.split()[0]
    except Exception:
        out = "?"
    return out


def task_report_full(c: Ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, cfg = c.R, c.cfg
    pf = R / "paper_full"
    (pf / "figures").mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(R / "metrics" / "summary.csv")
    old_dir = Path(cfg.get("report_full", {}).get("compare_with", ""))
    old = pd.read_csv(old_dir / "metrics" / "summary.csv") if (old_dir / "metrics" / "summary.csv").exists() else None
    best = (read_json(R / "metrics" / "best_ours.json", {}) or {}).get("method")
    sets = [s for s in EVAL_SETS if s in set(summ["set"])]
    mc = cfg["cbctmar"]
    tl = R / "models" / "cbct_mar" / "training_losses.csv"
    tdf = pd.read_csv(tl) if tl.exists() else pd.DataFrame()
    n_pass = int(mc["epochs"]) + (1 if mc.get("inclusive_last") else 0)
    done_pass = len(tdf)
    final = done_pass >= n_pass
    epoch_ms = cbctmar_epoch_methods(R, cfg)
    L = ["# Báo cáo đầy đủ — so sánh công bằng với CBCT_MAR (huấn luyện đúng số epoch của repo gốc)", "",
         f"Tạo lúc: {now()}  |  thư mục: `{R}`", "",
         f"**Trạng thái: {'HOÀN TẤT' if final else 'TẠM THỜI'}** — CBCT_MAR đã huấn luyện **{done_pass}/{n_pass}** lượt "
         f"(epoch 0..{max(done_pass - 1, 0)}); các mốc epoch đã có checkpoint và được đánh giá: "
         f"{[e for e, _, _ in epoch_ms] or 'chưa có'}.", ""]
    # ---- 1. tom tat thay doi
    L += ["## 1. Thay đổi so với thư mục kết quả cũ", "",
          f"- Thư mục cũ (`{old_dir}`): CBCT_MAR huấn luyện **60 epoch**. Thư mục này: **{n_pass} lượt** (`EPOCHS = {mc['epochs']}`, "
          f"`range(START_EPOCH, EPOCHS + 1)` với `START_EPOCH = 0` như `code.ipynb` của repo gốc).",
          "- Phương pháp của bạn, NMAR, LI-MAR, dữ liệu, tập đánh giá và cách chọn cấu hình 'Ours' (trên tập Val): **giữ nguyên**. "
          "Mọi số liệu của các phương pháp đó phải trùng hệt bản cũ (kiểm tra ở mục 4).",
          "- Ảnh CBCT_MAR trong `compie/`, `outputs/*/cbct_mar/` và các bảng trong `paper/` đã được tính lại với mô hình mới; kết quả lần chạy 60 epoch "
          "cũ được giữ làm phương pháp `cbct_mar_60ep_run1` để đối chiếu.", ""]
    # ---- 2. tai hien
    L += ["## 2. Mức độ tái hiện CBCT_MAR (đối chiếu với `code.ipynb`)", "",
          "| Thành phần | Repo gốc (`code.ipynb`) | Bản chạy | Ghi chú |", "|---|---|---|---|",
          "| Kiến trúc G / D | UNet 5 tầng (ConvTranspose + tanh), PatchGAN 3 lớp + sigmoid | Giống hệt | port nguyên lớp |",
          "| Loss | BCE (adv), L1 (cycle, identity); λ_cycle=18, λ_identity=15 | Giống hệt | |",
          "| Tối ưu | 4 Adam riêng (G, F, D_X, D_Y), lr 2e-4, betas (0.5, 0.999) | Giống hệt | |",
          "| Batch | 1, `zip(normal_loader, glare_loader)` → 1 lượt = min(số ảnh) bước | Giống hệt | 1109 bước/epoch trên dữ liệu này |",
          "| Tiền xử lý | Resize 256, RandomHorizontalFlip, RandomRotation(10), Normalize 0.5 | Giống hệt (torchvision) | |",
          "| Thứ tự cập nhật | D_X, D_Y trước; rồi G+F cùng lúc | Giống hệt | |",
          f"| Số epoch | `EPOCHS = 600`, `range(0, 601)` | **{n_pass} lượt** | {'đủ' if final else 'đang chạy'} |",
          "| Khởi tạo | Nạp `gen_*_epoch_180.pth` (huấn luyện trước) rồi lặp từ epoch 0 | **Train từ đầu** | không có checkpoint gốc; giả định số epoch đúng bằng EPOCHS |",
          "| Dữ liệu | ToothFairy đã xử lý của tác giả | ToothFairy tự xử lý (`Dataset_TF`) | khác nguồn ảnh, cùng miền |",
          "| Tối ưu tốc độ (toán học giống hệt) | 8 lần forward G/F mỗi bước, tuần tự | dùng lại forward (bỏ 2/8), gộp batch 2 các lần chạy cùng mạng độc lập (G, F, D không có BatchNorm/InstanceNorm nên kết quả từng mẫu không đổi), 2 CUDA stream cho 2 nhánh độc lập, tắt grad của D trong bước G+F | kiểm chứng: loss trùng 5–6 chữ số, đạo hàm lệch ~1e-5 (sai số fp32); nhanh ×1,49 (126 → 85 s/epoch trên P40) |",
          "| Validation | `validate_cycle` mỗi epoch | Giống hệt | ghi `training_losses.csv` |",
          "| Trọng số dùng để đánh giá | `gen_G_final.pth` | `gen_G_final.pth` | ngoài ra đánh giá thêm các mốc epoch |", ""]
    # ---- 3. huan luyen
    if len(tdf):
        sec = tdf["epoch_sec"] if "epoch_sec" in tdf else pd.Series([np.nan])
        L += ["## 3. Quá trình huấn luyện CBCT_MAR", "",
              f"- Số lượt đã chạy: {done_pass}/{n_pass}; thời gian trung vị: {sec.median():.0f} s/epoch; tổng: {fmt_dur(float(sec.sum()))}.",
              f"- `val_cycle`: đầu {tdf['val_cycle'].iloc[0]:.4f} → cuối {tdf['val_cycle'].iloc[-1]:.4f} (thấp nhất {tdf['val_cycle'].min():.4f} "
              f"ở epoch {int(tdf.loc[tdf['val_cycle'].idxmin(), 'epoch'])}).",
              "- Hình: `paper_full/figures/cbctmar_training.png`.", ""]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(tdf["epoch"], tdf["total_loss_G"], lw=1); ax[0].set_title("total_loss_G (G+F)")
        ax[1].plot(tdf["epoch"], tdf["val_cycle"], lw=1, color="C1"); ax[1].set_title("val_cycle (L1, tập Val)")
        ax[2].plot(tdf["epoch"], tdf["disc_X_loss"], lw=1, label="D_X"); ax[2].plot(tdf["epoch"], tdf["disc_Y_loss"], lw=1, label="D_Y")
        ax[2].set_title("Discriminator loss"); ax[2].legend()
        for a_ in ax:
            a_.set_xlabel("epoch"); a_.grid(alpha=.3)
            for e in mc.get("milestones", []):
                a_.axvline(e, color="gray", ls=":", lw=.8)
        fig.tight_layout(); fig.savefig(pf / "figures" / "cbctmar_training.png", dpi=200); plt.close(fig)
    # ---- 4. bang chinh + so sanh voi cu
    L += ["## 4. Kết quả chính và so sánh với thư mục cũ", "",
          "Mỗi bảng: giá trị trung bình trên tập. `Ours` = cấu hình chọn trên Val. `CBCT_MAR (cũ)` = huấn luyện 60 epoch (thư mục cũ); "
          "`CBCT_MAR (mới)` = bản đủ epoch.", ""]
    rows_delta, rows_rank, chg_lines = [], [], []
    same_max = 0.0
    for s_ in sets:
        keys = SYN_KEYS if _set_kind(s_) == "synth" else REAL_KEYS
        meths = [("Original", "input"), ("Ours", best), ("CBCT_MAR (cũ, 60 epoch)", None), ("CBCT_MAR (mới)", "cbct_mar"),
                 ("NMAR [B2]", "nmar"), ("LI-MAR [B3]", "limar")]
        tab = []
        for lab, m in meths:
            row = {"Phương pháp": lab}
            for k, name, _d in keys:
                if lab.startswith("CBCT_MAR (cũ"):
                    v = _get_mean(old, s_, "cbct_mar", k)
                    if np.isnan(v):
                        v = _get_mean(summ, s_, "cbct_mar_60ep_run1", k)
                else:
                    v = _get_mean(summ, s_, m, k)
                row[name] = _fmt(v)
            tab.append(row)
        t = pd.DataFrame(tab)
        t.to_csv(pf / f"table_compare_{s_}.csv", index=False)
        try:
            (pf / f"table_compare_{s_}.tex").write_text(t.to_latex(index=False, escape=True), encoding="utf-8")
        except Exception:
            pass
        n_img = len(eval_names(R, s_))
        L += [f"### {s_} ({n_img} ảnh)", "", _md_table(t), ""]
        for k, name, d in keys:
            o = _get_mean(old, s_, "cbct_mar", k)
            n_ = _get_mean(summ, s_, "cbct_mar", k)
            rows_delta.append({"set": s_, "metric": k, "CBCT_MAR_cu_60ep": o, "CBCT_MAR_moi": n_, "thay_doi": n_ - o,
                               "thay_doi_%": (n_ - o) / abs(o) * 100 if o and not np.isnan(o) else np.nan, "tot_hon_neu": d})
            if o and not np.isnan(o) and not np.isnan(n_) and abs(n_ - o) / abs(o) > 0.05 and k in MAIN_KEYS[_set_kind(s_)]:
                better = (n_ > o) if d == "max" else (n_ < o)
                chg_lines.append(f"- `{s_}` — {name}: CBCT_MAR {_fmt(o)} → {_fmt(n_)} ({(n_ - o) / abs(o) * 100:+.1f}%, {'tốt hơn' if better else 'kém đi'}).")
            if k in MAIN_KEYS[_set_kind(s_)]:
                ours = _get_mean(summ, s_, best, k)
                win = lambda v: ("Ours" if ((ours > v) if d == "max" else (ours < v)) else "CBCT_MAR") if not (np.isnan(v) or np.isnan(ours)) else "-"
                rows_rank.append({"set": s_, "metric": k, "Ours": ours, "CBCT_MAR_cu": o, "CBCT_MAR_moi": n_, "thang_cu": win(o), "thang_moi": win(n_)})
        for m in ("nmar", "limar", best):
            for k, _n, _d in keys:
                a1, a2 = _get_mean(summ, s_, m, k), _get_mean(old, s_, m, k)
                if not (np.isnan(a1) or np.isnan(a2)):
                    same_max = max(same_max, abs(a1 - a2))
    dd = pd.DataFrame(rows_delta)
    dd.to_csv(pf / "table_delta_cbctmar_vs_old.csv", index=False)
    rk = pd.DataFrame(rows_rank)
    rk.to_csv(pf / "table_ours_vs_cbctmar_ranking.csv", index=False)
    L += [f"**Kiểm tra không đổi:** chênh lệch lớn nhất giữa thư mục mới và cũ của Ours/NMAR/LI-MAR trên mọi tập, mọi metric = **{same_max:.3g}** "
          "(≈ 0 nghĩa là chỉ CBCT_MAR thay đổi).", ""]
    # ---- 5. so voi epoch
    L += ["## 5. CBCT_MAR theo số epoch", "",
          "Ghi chú: tên mốc theo quy ước repo gốc (`gen_G_epoch_N`, N là chỉ số 0-based của vòng lặp, tức sau N+1 lượt huấn luyện).", ""]
    ep_methods = [(0, "cbct_mar_60ep_run1", "60 lượt (lần chạy cũ)")] if (summ["method"] == "cbct_mar_60ep_run1").any() else []
    ep_methods += [(e, m, f"epoch {e}") for e, m, _ in epoch_ms]
    if final:
        ep_methods += [(10 ** 6, "cbct_mar", "cuối (final)")]
    if ep_methods:
        fig, axs = plt.subplots(len(sets), 3, figsize=(13, 2.6 * len(sets)), squeeze=False)
        for i_, s_ in enumerate(sets):
            keys = SYN_KEYS if _set_kind(s_) == "synth" else REAL_KEYS
            tab = []
            for _e, m, lab in ep_methods:
                row = {"Mốc": lab}
                for k, name, _d in keys:
                    row[name] = _fmt(_get_mean(summ, s_, m, k))
                tab.append(row)
            t = pd.DataFrame(tab)
            t.to_csv(pf / f"table_cbctmar_epochs_{s_}.csv", index=False)
            try:
                (pf / f"table_cbctmar_epochs_{s_}.tex").write_text(t.to_latex(index=False, escape=True), encoding="utf-8")
            except Exception:
                pass
            L += [f"### {s_}", "", _md_table(t), ""]
            for j_, k in enumerate(MAIN_KEYS[_set_kind(s_)][:3]):
                ax = axs[i_][j_]
                xs = [e for e, m, _ in ep_methods if e < 10 ** 6 and e > 0]
                ys = [_get_mean(summ, s_, m, k) for e, m, _ in ep_methods if 0 < e < 10 ** 6]
                ax.plot(xs, ys, "o-", label="CBCT_MAR")
                for ref, col in ((best, "C3"), ("nmar", "C2"), ("input", "gray")):
                    v = _get_mean(summ, s_, ref, k)
                    if not np.isnan(v):
                        ax.axhline(v, color=col, ls="--", lw=1, label={best: "Ours", "nmar": "NMAR", "input": "Original"}[ref])
                ax.set_title(f"{s_}: {k}", fontsize=8); ax.grid(alpha=.3)
                if i_ == 0 and j_ == 0:
                    ax.legend(fontsize=7)
            for j_ in range(len(MAIN_KEYS[_set_kind(s_)][:3]), 3):
                axs[i_][j_].axis("off")
        fig.tight_layout(); fig.savefig(pf / "figures" / "cbctmar_vs_epoch.png", dpi=200); plt.close(fig)
        L += ["Hình: `paper_full/figures/cbctmar_vs_epoch.png`.", ""]
    # ---- 6. cap nhat noi dung paper
    L += ["## 6. Nội dung paper cần cập nhật so với thư mục cũ", "",
          "**a) Mô tả baseline:** ghi CBCT_MAR được huấn luyện đúng cấu hình repo gốc (batch 1, λ_cycle=18, λ_identity=15, Adam 2e-4, "
          f"{n_pass} lượt tương ứng `EPOCHS = {mc['epochs']}`), dữ liệu ToothFairy của chúng tôi, huấn luyện từ đầu (repo gốc nạp checkpoint epoch 180 mà không công bố).", "",
          "**b) Số liệu CBCT_MAR thay đổi đáng kể (>5%, chỉ các metric chính):**", ""]
    L += chg_lines or ["- (chưa có thay đổi vượt 5% hoặc chưa có mô hình mới)"]
    L += ["", "**c) Ours có còn tốt hơn CBCT_MAR không (metric chính từng tập):**", "", _md_table(rk) if len(rk) else "(chưa có)", ""]
    flips = rk[rk["thang_cu"] != rk["thang_moi"]] if len(rk) else rk
    L += ["Các dòng đổi người thắng so với bản cũ: " + (", ".join(f"{r.set}/{r.metric} ({r.thang_cu}→{r.thang_moi})" for r in flips.itertuples()) or "không có"), "",
          "Đối chiếu số liệu chi tiết: `paper_full/table_delta_cbctmar_vs_old.csv`. Các bảng chính (đã có CBCT_MAR mới): `paper/table_<tập>.csv/.tex`.", ""]
    # ---- 7. han che
    L += ["## 7. Hạn chế cần nêu trong paper", "",
          "- NMAR/LI-MAR là bản mô phỏng trên lát cắt 2D bằng biến đổi Radon (không có sinogram cone-beam thật).",
          "- Nhiễu tự tạo (`Tao_nhieu_sang.py`) không phải nhiễu thật; tập ảnh thật chỉ có chỉ số giữ cấu trúc/năng lượng vệt, không có ảnh sạch.",
          "- CVan: cửa sổ HU tự chọn từ DICOM. Truong: DICOM 12-bit chặn ở 3071 HU nên kim loại xác định bằng khối voxel bão hòa.",
          "- CBCT_MAR gốc huấn luyện tiếp từ checkpoint epoch 180 không công bố; bản này train từ đầu cùng số epoch.", ""]
    # ---- 8. tai gi
    L += ["## 8. Cần tải thư mục nào để viết paper", "",
          "| Thư mục / file | Nội dung | Dung lượng |", "|---|---|---|"]
    items = [("REPORT_FULL.md", "báo cáo này"), ("paper_full/", "bảng so sánh cũ↔mới, theo epoch, hình huấn luyện (CSV/LaTeX/PNG)"),
             ("paper/", "bảng chính mỗi tập (CSV/LaTeX)"), ("compie/", "ảnh ghép Original→Ours→CBCT_MAR→NMAR→LI-MAR (→GT)"),
             ("metrics/", "số liệu từng ảnh, xếp hạng cấu hình"), ("config.json", "cấu hình chạy"),
             ("models/cbct_mar/", "checkpoint + log huấn luyện CBCT_MAR"), ("outputs/", "ảnh kết quả mọi phương pháp/biến thể (lớn, tùy chọn)")]
    for name, desc in items:
        pth = R / name
        L.append(f"| `{name}` | {desc} | {_dir_size(pth) if pth.exists() else '-'} |")
    L += ["", "Tối thiểu để viết paper: `REPORT_FULL.md`, `paper_full/`, `paper/`, `compie/`, `metrics/`, `config.json`.", ""]
    (R / "REPORT_FULL.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    c.log(f"REPORT_FULL.md -> {R / 'REPORT_FULL.md'} ({'HOAN TAT' if final else 'TAM THOI'})")



def _md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind == "f":
            df[col] = df[col].map(lambda x: "-" if pd.isna(x) else f"{x:.4g}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(x) for x in r) + " |" for r in df.astype(str).values]
    return "\n".join([head, sep] + body)


# ==============================================================================
#  DO THI BUOC + BO DIEU PHOI (song song theo CPU / RAM / VRAM)
# ==============================================================================
class Task:
    def __init__(self, name, deps, kind, prio, gpu=0.0, cpu=1, ram=0.5, min_cpu=1):
        self.name, self.deps, self.kind, self.prio = name, list(deps), kind, prio
        self.gpu, self.cpu, self.ram, self.min_cpu = gpu, cpu, ram, min_cpu


def build_tasks(cfg) -> list:
    tr = cfg["task_res"]
    T = []
    T.append(Task("data_tf", [], "pool", 10, min_cpu=2))
    T.append(Task("data_cvan", [], "pool", 11, min_cpu=2))
    T.append(Task("data_truong", [], "pool", 11, min_cpu=2))
    T.append(Task("eval_sets", ["data_tf", "data_cvan", "data_truong"], "light", 12))
    T.append(Task("yolo_data", ["data_tf"], "pool", 13, min_cpu=2))
    gnames = list(cfg["gan"]["variants"])
    for g in gnames:
        T.append(Task(f"gan_train:{g}", ["data_tf"], "gpu", 0, *tr["gan_train"]))
    T.append(Task("cbctmar_train", [], "gpu", 3, *tr["cbctmar_train"]))
    for y in yolo_cfgs(cfg):
        big = not re.search(r"\d+n-", y["model"])
        T.append(Task(f"yolo_train:{y['name']}", ["yolo_data"], "gpu", 1, *(tr["yolo_train_large"] if big else tr["yolo_train"])))
    for g in gnames:
        T.append(Task(f"infer_gan:{g}", [f"gan_train:{g}", "eval_sets"], "gpu", 2, *tr["infer"]))
    T.append(Task("infer_cbctmar", ["cbctmar_train", "eval_sets"], "gpu", 4, *tr["infer"]))
    T.append(Task("infer_cbctmar_epochs", ["cbctmar_train", "eval_sets"], "gpu", 5, *tr["infer"]))
    for y in yolo_cfgs(cfg):
        T.append(Task(f"infer_yolo:{y['name']}", [f"yolo_train:{y['name']}", "eval_sets"], "gpu", 2, *tr["infer"]))
    gpu_infer = [f"infer_gan:{g}" for g in gnames] + [f"infer_yolo:{y['name']}" for y in yolo_cfgs(cfg)]
    # CBCT_MAR (batch 1, ~1.5h) KHONG chan ket qua: metrics/chon cau hinh/compie/report chay NGAY khi phuong phap cua ban +
    # NMAR/LI-MAR xong (ban "som"); khi CBCT_MAR xong -> tinh lai metrics + bang + compie + report (ban "final", ghi de).
    for s in EVAL_SETS:
        T.append(Task(f"classical:{s}", ["eval_sets"], "pool", 20, min_cpu=2))
        T.append(Task(f"ours_post:{s}", gpu_infer, "pool", 15, min_cpu=2))
        T.append(Task(f"metrics:{s}", [f"ours_post:{s}", f"classical:{s}"], "pool", 16, min_cpu=2))
    T.append(Task("select", [f"metrics:{s}" for s in EVAL_SETS], "light", 17))
    T.append(Task("compie", ["select"], "pool", 18, min_cpu=2))
    T.append(Task("report", ["compie"], "light", 19))
    for s in EVAL_SETS:
        T.append(Task(f"metrics_final:{s}", ["infer_cbctmar", "infer_cbctmar_epochs", f"metrics:{s}"], "pool", 21, min_cpu=2))
    T.append(Task("select_final", [f"metrics_final:{s}" for s in EVAL_SETS] + ["report"], "light", 22))
    T.append(Task("compie_final", ["select_final"], "pool", 23, min_cpu=2))
    T.append(Task("report_final", ["compie_final"], "light", 24))
    T.append(Task("report_full", ["report_final"], "light", 25))
    return T


def build_tasks_interim(cfg) -> list:
    """Bao cao TAM THOI trong luc CBCT_MAR con dang train: suy luan cac moc epoch da co checkpoint -> metrics -> chon -> REPORT_FULL.
    Chay SONG SONG voi buoc train (khong dung/khong ghi de trang thai cua lan chay chinh): `python cbct_mar_experiment.py interim`."""
    T = [Task("infer_cbctmar_epochs", [], "gpu", 1, *cfg["task_res"]["infer"])]
    for s in EVAL_SETS:
        T.append(Task(f"metrics_final:{s}", ["infer_cbctmar_epochs"], "pool", 2, min_cpu=2))
    T.append(Task("select_final", [f"metrics_final:{s}" for s in EVAL_SETS], "light", 3))
    T.append(Task("report_full", ["select_final"], "light", 4))
    return T


def run_task_inproc(R: Path, name: str, slots: int, done_dir: Path | None = None):
    c = Ctx(R, name, slots)
    head, _, arg = name.partition(":")
    head = head[:-len("_final")] if head.endswith("_final") else head      # *_final = chay lai cung ham sau khi co CBCT_MAR
    t0 = time.time()
    c.log(f"BAT DAU (slots={slots}, pid={os.getpid()})")
    try:
        if head in ("classical", "ours_post", "metrics") and arg not in present_sets(R):
            c.log(f"Tap {arg} khong co anh -> bo qua")
        else:
            fn = {"data_tf": task_data_tf, "data_cvan": task_data_cvan, "data_truong": task_data_truong, "eval_sets": task_eval_sets, "yolo_data": task_yolo_data,
                  "cbctmar_train": task_cbctmar_train, "infer_cbctmar": task_infer_cbctmar, "select": task_select,
                  "infer_cbctmar_epochs": task_infer_cbctmar_epochs, "report_full": task_report_full,
                  "compie": task_compie, "report": task_report}.get(head)
            if fn:
                fn(c)
            else:
                {"gan_train": task_gan_train, "infer_gan": task_infer_gan, "yolo_train": task_yolo_train,
                 "infer_yolo": task_infer_yolo, "classical": task_classical, "ours_post": task_ours_post,
                 "metrics": task_metrics}[head](c, arg)
    except Exception:
        c.log("LOI:\n" + traceback.format_exc())
        raise
    el = time.time() - t0
    atomic_json((done_dir or (R / "state" / "done")) / f"{_safe(name)}.json", {"task": name, "elapsed_s": el, "finished": now(), "slots": slots})
    c.log(f"XONG sau {fmt_dur(el)}")


def stop_other_processes(log):
    import psutil
    me = psutil.Process()
    protect = {me.pid} | {p.pid for p in me.parents()}
    pats = ("train_dental_mar", "benchmark_mar_methods", "cbct_mar_experiment.py")
    victims = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if p.pid in protect or not cmd or "python" not in Path(cmd[0]).name:
                continue
            if any(pt in " ".join(cmd) for pt in pats):
                victims.append(p)
        except Exception:
            continue
    allp = []
    for p in victims:
        try:
            allp += [p] + p.children(recursive=True)
        except Exception:
            allp.append(p)
    allp = [p for p in {p.pid: p for p in allp}.values() if p.pid not in protect]
    if not allp:
        return
    log(f"Dung {len(allp)} tien trinh dang chay: " + ", ".join(f"{p.pid}" for p in allp[:20]))
    for p in allp:
        try:
            p.terminate()
        except Exception:
            pass
    _, alive = psutil.wait_procs(allp, timeout=15)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass


def orchestrate(args):
    import psutil
    R = Path(args.results).resolve()
    R.mkdir(parents=True, exist_ok=True)
    interim = bool(getattr(args, "interim", False))
    sfx = "_interim" if interim else ""
    log = Log(R / "logs" / f"pipeline{sfx}.log")
    cfg = build_config(args, R)
    cfg["_ours_override"] = getattr(args, "ours_override", None)
    if not getattr(args, "no_stop_others", False):
        stop_other_processes(log)
    lock = R / f".run{sfx}.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text())
            if pid != os.getpid() and psutil.pid_exists(pid):
                sys.exit(f"Dang co tien trinh khac (pid {pid}) chay tren {R}. Dung no truoc hoac bo --no-stop-others.")
        except ValueError:
            pass
    lock.write_text(str(os.getpid()))
    if not interim:
        atomic_json(R / "config.json", cfg)
        atomic_json(R / "config_history" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", cfg)
    tasks = build_tasks_interim(cfg) if interim else build_tasks(cfg)
    by = {t.name: t for t in tasks}
    done_dir = R / "state" / ("done_interim" if interim else "done")
    if interim:                       # bao cao tam thoi luon chay lai tu dau (checkpoint moi hon lan truoc)
        shutil.rmtree(done_dir, ignore_errors=True)
    force = set(args.force or [])
    if force:
        # --force <ten hoac tien to>: chay lai buoc do + moi buoc phu thuoc (dem theo do thi)
        hit = {t.name for t in tasks if any(t.name == f or t.name.startswith(f + ":") or t.name.split(":")[0] == f for f in force)}
        changed = True
        while changed:
            changed = False
            for t in tasks:
                if t.name not in hit and any(d in hit for d in t.deps):
                    hit.add(t.name)
                    changed = True
        for n in hit:
            (done_dir / f"{_safe(n)}.json").unlink(missing_ok=True)
        log(f"Chay lai (force): {sorted(hit)}")
    only = set(args.stages or [])
    if only:
        keep = {t.name for t in tasks if any(t.name == f or t.name.split(":")[0] == f for f in only)}
        tasks = [t for t in tasks if t.name in keep]
    run_names = {t.name for t in tasks}
    ncpu = cpu_count()
    slots_total = max(1, ncpu - int(cfg["resources"]["cpu_reserve"]))
    g = gpu_query()
    gpu_total = (g[0] / 1024.0) if g else 0.0
    gpu_budget = gpu_total * float(cfg["resources"]["gpu_mem_frac"])
    ram_total = psutil.virtual_memory().total / 1e9
    log("=" * 90)
    log(f"KET QUA: {R}")
    log(f"Tai nguyen: CPU {ncpu} nhan (dung {slots_total}), RAM {ram_total:.1f}GB, GPU {gpu_total:.1f}GB (ngan sach {gpu_budget:.1f}GB)")
    log(f"Luoi YOLO: {[y['name'] for y in yolo_cfgs(cfg)]}")
    log(f"Bien the phuong phap cua ban: {len(ours_variants(cfg))} | GAN: {list(cfg['gan']['variants'])} | "
        f"tap danh gia: {EVAL_SETS}")
    state_file = R / f"state{sfx}.json"
    state = read_json(state_file, {}) or {}
    st = state.setdefault("tasks", {})
    state.update({"orchestrator_pid": os.getpid(), "started": now(), "results": str(R), "finished": None})
    for t in tasks:
        info = read_json(done_dir / f"{_safe(t.name)}.json")
        st[t.name] = {"status": "done", "elapsed_s": info.get("elapsed_s"), "slots": info.get("slots")} if info else {"status": "pending"}
    running = {}   # name -> (Popen, slots, gpu, ram, t0, logfile)
    retries = {}
    exclusive = set()
    stopping = {"flag": False}

    def on_sig(sig, frm):
        stopping["flag"] = True
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    def save():
        state["tasks"] = st
        state["updated"] = now()
        atomic_json(state_file, state)

    poll = float(cfg["resources"]["poll_s"])
    gpu_conc = int(cfg["resources"].get("gpu_concurrency", 1))
    psutil.cpu_percent(interval=None)
    while True:
        if stopping["flag"]:
            log("Nhan tin hieu dung -> dung cac buoc dang chay (chay lai lenh cu de tiep tuc).")
            for n, (p, *_r) in running.items():
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except Exception:
                    pass
                st[n]["status"] = "interrupted"
            save()
            break
        # --- thu hoi buoc da ket thuc
        for n in list(running):
            p, sl, gg, rr, t0, lf = running[n]
            rc = p.poll()
            if rc is None:
                continue
            del running[n]
            el = time.time() - t0
            if rc == 0 and (done_dir / f"{_safe(n)}.json").exists():
                st[n].update({"status": "done", "elapsed_s": el, "slots": sl, "gpu": gg})
                log(f"XONG  {n:34s} {fmt_dur(el)}")
            else:
                tail = ""
                try:
                    tail = Path(lf).read_text(encoding="utf-8", errors="ignore")[-4000:]
                except Exception:
                    pass
                oom = "out of memory" in tail.lower() or "outofmemory" in tail.lower()
                retries[n] = retries.get(n, 0) + 1
                if retries[n] <= int(cfg["resources"]["max_retries"]):
                    if oom:
                        exclusive.add(n)
                    st[n].update({"status": "pending", "last_error": "OOM" if oom else f"exit {rc}"})
                    log(f"LOI   {n} (exit {rc}{', OOM -> chay lai khi GPU rong' if oom else ''}) -> thu lai lan {retries[n]}")
                else:
                    st[n].update({"status": "failed", "elapsed_s": el})
                    log(f"THAT BAI {n} - xem {lf}")
        # --- trang thai phu thuoc (buoc khong nam trong lan chay nay -> can da xong truoc do)
        def dep_status(d):
            if d in run_names:
                return st[d]["status"]
            return "done" if (done_dir / f"{_safe(d)}.json").exists() else "missing"
        for t in tasks:
            if st[t.name]["status"] == "pending" and any(dep_status(d) in ("failed", "blocked", "missing") for d in t.deps):
                st[t.name]["status"] = "blocked"
                log(f"CHAN  {t.name} (phu thuoc loi/thieu: {[d for d in t.deps if dep_status(d) != 'done']})")
        pend = [t for t in tasks if st[t.name]["status"] == "pending"]
        if not pend and not running:
            break
        # --- tai nguyen con lai
        used_gpu = sum(v[2] for v in running.values())
        free_cpu = slots_total - sum(v[1] for v in running.values())
        free_ram = psutil.virtual_memory().available / 1e9 - float(cfg["resources"]["ram_reserve_gb"])
        # nhan CPU RANH THUC TE (buoc GPU thuong dung it hon so nhan duoc gan) -> buoc CPU dung phan ranh nay
        idle_cpu = int(ncpu * (1.0 - psutil.cpu_percent(interval=None) / 100.0)) - int(cfg["resources"]["cpu_reserve"])
        gq = gpu_query()
        gpu_real_free = ((gq[0] - gq[1]) / 1024.0) if gq else 0.0
        excl_running = any(x in exclusive for x in running)
        ready = sorted([t for t in pend if all(dep_status(d) == "done" for d in t.deps)], key=lambda t: t.prio)
        waiting_gpu = []
        per_w, base_w = float(cfg["task_res"]["pool_worker_ram"]), float(cfg["task_res"]["pool_base_ram"])
        for t in ready:
            if t.kind == "gpu":
                n_gpu = n_gpu_running(running, by)
                if excl_running or (t.name in exclusive and n_gpu > 0) or n_gpu >= gpu_conc \
                        or used_gpu + t.gpu > gpu_budget or t.gpu > gpu_real_free - 0.5 or free_ram < t.ram:
                    waiting_gpu.append(t)
                    continue
                sl = max(2, min(t.cpu, free_cpu))
                ram_use = t.ram
            elif t.kind == "light":
                if free_ram < 0.5 and running:
                    continue
                sl, ram_use = 1, 0.5
            else:
                # KHONG giu cho CPU cho buoc GPU dang cho VRAM (no chua chay duoc -> giu cho = CPU ngoi khong);
                # buoc GPU khi duoc chay se dung chung CPU (hoi vuot so nhan mot chut, khong tran RAM/VRAM)
                sl = int(min(max(free_cpu, idle_cpu), (free_ram - base_w) / per_w))
                if sl < t.min_cpu:
                    if running:
                        continue
                    sl = max(1, sl)
                ram_use = base_w + sl * per_w
            lf = R / "logs" / "tasks" / f"{_safe(t.name)}.log"
            lf.parent.mkdir(parents=True, exist_ok=True)
            gpu_cap = (gpu_budget if (t.name in exclusive or gpu_conc <= 1) else t.gpu) if t.kind == "gpu" else 0.0
            env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(sl), MKL_NUM_THREADS=str(sl),
                       OPENBLAS_NUM_THREADS=str(sl), MAR_GPU_GB=f"{gpu_cap:.2f}")
            fh = open(lf, "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, str(SCRIPT), "_task", t.name, "--results", str(R), "--slots", str(sl),
                                  "--done-dir", str(done_dir)],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=str(SRC))
            running[t.name] = (p, sl, t.gpu, ram_use, time.time(), str(lf))
            st[t.name].update({"status": "running", "pid": p.pid, "slots": sl, "gpu": t.gpu, "started": now()})
            used_gpu += t.gpu
            free_cpu -= sl
            free_ram -= ram_use
            idle_cpu -= sl
            gpu_real_free -= t.gpu
            if t.name in exclusive:
                excl_running = True
            log(f"CHAY  {t.name:34s} cpu={sl} gpu~{t.gpu:.1f}GB ram~{ram_use:.1f}GB (dang chay: {len(running)})")
        save()
        time.sleep(poll)
    state["finished"] = now()
    save()
    try:
        lock.unlink()
    except Exception:
        pass
    bad = [n for n, v in st.items() if v["status"] in ("failed", "blocked")]
    log("=" * 90)
    if stopping["flag"]:
        log("DA DUNG giua chung - chay lai `python cbct_mar_experiment.py run` de tiep tuc tu cho da dung.")
    else:
        log("HOAN TAT" + (f" - co buoc loi/bi chan: {bad}" if bad else " - tat ca cac buoc thanh cong") + f". Xem {R / 'REPORT.md'}")


def n_gpu_running(running, by) -> int:
    return sum(1 for n in running if by.get(n) is not None and by[n].kind == "gpu")


# ==============================================================================
#  THEO DOI
# ==============================================================================
def status(args):
    R = Path(args.results).resolve()
    while True:
        st = read_json(R / "state.json", {}) or {}
        lines = [f"=== {R}  (cap nhat {st.get('updated', '-')}, bat dau {st.get('started', '-')}, ket thuc {st.get('finished') or '-'})"]
        tasks = st.get("tasks") or {}
        cnt = {}
        for v in tasks.values():
            cnt[v["status"]] = cnt.get(v["status"], 0) + 1
        lines.append("Buoc: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
        for n, v in tasks.items():
            if v["status"] in ("running", "failed", "blocked", "interrupted"):
                pr = read_json(R / "progress" / f"{_safe(n)}.json", {}) or {}
                extra = {k: pr[k] for k in pr if k not in ("task", "desc", "done", "total", "elapsed_s", "rate_per_s", "eta_s", "updated")}
                pg = f"{pr.get('done', 0)}/{pr.get('total', '?')} ETA {fmt_dur(pr.get('eta_s'))}" if pr else ""
                lines.append(f"  {v['status']:11s} {n:34s} cpu={v.get('slots', '-')} gpu~{v.get('gpu', 0) or 0:.0f}GB "
                             f"{fmt_dur(time.time() - datetime.strptime(v['started'], '%Y-%m-%d %H:%M:%S').timestamp()) if v.get('started') and v['status'] == 'running' else ''} {pg} {extra if extra else ''}")
        done = [(n, v) for n, v in tasks.items() if v["status"] == "done"]
        if done:
            lines.append("Da xong: " + ", ".join(f"{n}({fmt_dur(v.get('elapsed_s'))})" for n, v in done))
        g = gpu_query()
        try:
            import psutil
            vm = psutil.virtual_memory()
            lines.append(f"CPU {psutil.cpu_percent(interval=0.5):.0f}% | RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f}GB"
                         + (f" | GPU {g[2]:.0f}% VRAM {g[1] / 1024:.1f}/{g[0] / 1024:.1f}GB" if g else ""))
        except Exception:
            pass
        lp = R / "logs" / "pipeline.log"
        if lp.exists():
            lines.append("--- pipeline.log (cuoi):")
            lines += Path(lp).read_text(encoding="utf-8", errors="ignore").splitlines()[-args.lines:]
        out = "\n".join(lines)
        if args.follow:
            os.system("clear")
        print(out, flush=True)
        if not args.follow:
            break
        time.sleep(args.interval)


def print_grid(args):
    R = Path(args.results).resolve()
    cfg = build_config(args, R)
    print("Buoc (task) va phu thuoc:")
    for t in build_tasks(cfg):
        print(f"  [{t.kind:5s}] {t.name:34s} <- {', '.join(t.deps) or '-'}")
    print("\nCau hinh YOLO:", *[y["name"] for y in yolo_cfgs(cfg)], sep="\n  ")
    vs = ours_variants(cfg)
    print(f"\n{len(vs)} bien the phuong phap cua ban:", *[v["method"] for v in vs], sep="\n  ")


# ==============================================================================
#  MAIN / CLI
# ==============================================================================
def add_common(p):
    p.add_argument("--results", default=str(PROJECT / "results"), help="thu muc ket qua duy nhat (mac dinh <project>/results)")


def main():
    ap = argparse.ArgumentParser(description="Chay toan bo thi nghiem CBCT-MAR (phuong phap cua ban + CBCT_MAR + NMAR + LI-MAR).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Vi du:\n  nohup python cbct_mar_experiment.py run > ../results/nohup.log 2>&1 &\n"
                                        "  python cbct_mar_experiment.py status -f\n  python cbct_mar_experiment.py grid\n"
                                        "  python cbct_mar_experiment.py report --ours <ten_method>")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="chay (hoac tiep tuc) toan bo pipeline")
    add_common(r)
    r.add_argument("--config", help="file JSON ghi de cau hinh (cung cau truc voi results/config.json)")
    r.add_argument("--fresh-config", action="store_true", help="bo qua results/config.json cu, dung mac dinh + tham so moi")
    r.add_argument("--quick", action="store_true", help="chay thu nhanh: vai anh, 1 epoch (kiem tra pipeline)")
    r.add_argument("--stages", nargs="*", help="chi chay cac buoc nay (ten hoac tien to, vd: data_tf yolo_train metrics)")
    r.add_argument("--force", nargs="*", help="chay LAI cac buoc nay (+ moi buoc phu thuoc), vd: --force metrics")
    r.add_argument("--no-stop-others", action="store_true", help="KHONG dung cac tien trinh train/benchmark khac dang chay")
    r.add_argument("--yolo-models", nargs="+", dest="yolo_models", help="vd: yolo11n-seg yolo11s-seg yolo12n-seg yolo12s-seg")
    r.add_argument("--yolo-epochs", nargs="+", dest="yolo_epochs")
    r.add_argument("--yolo-imgsz", nargs="+", dest="yolo_imgsz")
    r.add_argument("--yolo-batch", nargs="+", dest="yolo_batch")
    r.add_argument("--yolo-lr0", nargs="+", dest="yolo_lr0")
    r.add_argument("--yolo-patience", dest="yolo_patience")
    r.add_argument("--ours-conf", nargs="+", dest="ours_conf", help="nguong conf YOLO, vd 0.25 0.5")
    r.add_argument("--ours-radius", nargs="+", dest="ours_radius", help="ban kinh sua cuc bo (px thang 256) hoac inf")
    r.add_argument("--ours-clean", nargs="+", dest="ours_clean", help="light | off")
    r.add_argument("--gan-epochs", dest="gan_epochs")
    r.add_argument("--gan-patience", dest="gan_patience")
    r.add_argument("--gan-batch", dest="gan_batch")
    r.add_argument("--gan-weights", dest="gan_weights", help="best | final")
    r.add_argument("--gan-ckpt", dest="gan_ckpt", help="checkpoint GAN co san -> bo qua train GAN")
    r.add_argument("--gan-variants", dest="gan_variants", help='JSON {ten: {THAM_SO_V12: gia_tri}}, vd \'{"g0":{},"g1":{"LAMBDA_PAIRED":20}}\'')
    r.add_argument("--cbctmar-epochs", dest="cbctmar_epochs", help="EPOCHS cua CBCT_MAR (mac dinh 600 = repo goc, chay 601 luot)")
    r.add_argument("--cbctmar-workers", dest="cbctmar_workers")
    r.add_argument("--cbctmar-weights", dest="cbctmar_weights", help="final | bestval")
    r.add_argument("--cbctmar-ckpt", dest="cbctmar_ckpt")
    r.add_argument("--haze-prob", dest="haze_prob")
    r.add_argument("--cvan-synth-per-vol", dest="cvan_synth_per_vol")
    r.add_argument("--cvan-real-per-vol", dest="cvan_real_per_vol")
    r.add_argument("--cvan-max-vols", dest="cvan_max_vols")
    r.add_argument("--truong-synth-per-vol", dest="truong_synth_per_vol")
    r.add_argument("--truong-real-per-vol", dest="truong_real_per_vol")
    r.add_argument("--truong-max-vols", dest="truong_max_vols")
    r.add_argument("--eval-max-per-set", dest="eval_max_per_set")
    r.add_argument("--mar-angles", dest="mar_angles", help="so goc chieu Radon cho NMAR/LI-MAR")
    r.add_argument("--gpu-mem-frac", dest="gpu_mem_frac", help="ti le VRAM toi da duoc dung (mac dinh 0.92)")
    r.add_argument("--gpu-concurrency", dest="gpu_concurrency", help="so buoc GPU chay cung luc (mac dinh 1 - nhanh nhat tren P40)")
    r.add_argument("--cpu-reserve", dest="cpu_reserve", help="so nhan CPU de danh (mac dinh 1)")
    r.add_argument("--seed", dest="seed")
    s = sp.add_parser("status", help="xem tien trinh")
    add_common(s)
    s.add_argument("--follow", "-f", action="store_true")
    s.add_argument("--interval", type=float, default=10)
    s.add_argument("--lines", type=int, default=15)
    gr = sp.add_parser("grid", help="in cac buoc + luoi cau hinh (khong chay)")
    add_common(gr)
    for a in ("--yolo-models", "--yolo-epochs", "--yolo-imgsz", "--yolo-batch", "--yolo-lr0", "--ours-conf", "--ours-radius", "--ours-clean"):
        gr.add_argument(a, nargs="+", dest=a[2:].replace("-", "_"))
    gr.add_argument("--quick", action="store_true")
    rp = sp.add_parser("report", help="tinh lai metrics -> chon cau hinh -> bang -> compie -> REPORT.md tu output da co")
    add_common(rp)
    rp.add_argument("--ours", help="ep dung method nay lam 'Ours' trong bang/compie (ten thu muc trong outputs/<tap>/)")
    rp.add_argument("--no-metrics", action="store_true", help="khong tinh lai metrics, chi chon/bang/compie/report")
    im = sp.add_parser("interim", help="bao cao TAM THOI (moc epoch CBCT_MAR da co checkpoint) - chay song song voi buoc train dang chay")
    add_common(im)
    t = sp.add_parser("_task")
    t.add_argument("name")
    add_common(t)
    t.add_argument("--slots", type=int, default=1)
    t.add_argument("--done-dir", default=None)
    args = ap.parse_args()
    if args.cmd == "run":
        orchestrate(args)
    elif args.cmd == "status":
        status(args)
    elif args.cmd == "grid":
        print_grid(args)
    elif args.cmd == "report":
        ns = argparse.Namespace(results=args.results, config=None, fresh_config=False, quick=False, stages=None,
                                force=(["select"] if args.no_metrics else ["metrics"]), no_stop_others=True,
                                ours_override=args.ours or None)
        orchestrate(ns)
    elif args.cmd == "_task":
        run_task_inproc(Path(args.results).resolve(), args.name, args.slots, Path(args.done_dir) if args.done_dir else None)
    elif args.cmd == "interim":
        ns = argparse.Namespace(results=args.results, config=None, fresh_config=False, quick=False, stages=None, force=None,
                                no_stop_others=True, ours_override=None, interim=True)
        orchestrate(ns)


if __name__ == "__main__":
    main()
