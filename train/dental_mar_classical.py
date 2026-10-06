# -*- coding: utf-8 -*-
"""
==============================================================================
 dental_mar_classical.py  -  GIAM NHIEU KIM LOAI BANG XU LY ANH CO DIEN (KHONG GAN)
==============================================================================
 Muc dich: baseline KHONG dung mang neural, chi dung thuat toan xu ly anh co dien,
 de so sanh voi ket qua CycleGAN (train_dental_mar_V*.py). Chay:

        python dental_mar_classical.py

 (khong can them tham so - moi cai dat nam trong class CFG ngay ben duoi)

 3 phuong an duoc tinh cho MOI anh Glare, de so sanh truc tiep:
   [A] inpaint_telea : cv2.inpaint() thuat toan Telea - "va" vung kim loai bang
                        thong tin lan truyen tu vien vao trong. Day la ky thuat
                        MAR co dien pho bien nhat khi chi co anh da tai tao (khong
                        co sinogram goc).
   [B] inpaint_ns     : cv2.inpaint() thuat toan Navier-Stokes - tuong tu Telea
                        nhung lan truyen theo phuong trinh dong chat long, thuong
                        muot hon nhung cham hon.
   [C] blur_blend     : lam mo (bilateral) + hoa tron theo mask - CHINH LA cach
                        hau xu ly dang dung sau GAN (postprocess_one trong file
                        train_dental_mar_V*.py), nhung ap THANG len anh goc thay
                        vi anh GAN. Giu de ban thay ro GAN dong gop them duoc gi.

 Uu diem so voi GAN: khong can train, khong can GPU, chay tren TOAN BO anh gay,
 khong co rui ro "hallucinate" sai giai phau. Nhuoc diem: chi "va/lam mo" vung
 kim loai dua tren pixel xung quanh - khong "biet" xuong/rang that trong do trong
 nhu the nao, va it xu ly duoc vet soc (streak) lan ra ngoai vung kim loai loi.

 Moi lan chay tao 1 thu muc:  <OUTPUT_ROOT>/dental_mar_classical_HH-MM-SS_DD-MM-YYYY/
        results/<method>/<split>/*.png   comparison/<split>/*_cmp.png
        metrics.csv   summary.txt   log.txt   config_snapshot.json

 Cai thu vien (1 lan, dung chung voi file GAN):
   E:\\Project\\.venv\\Scripts\\activate
   pip install numpy pandas opencv-python scikit-image tqdm
==============================================================================
"""
from __future__ import annotations

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import cv2
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
except ImportError as _e:  # pragma: no cover
    sys.exit(
        f"\n[LOI] Thieu thu vien: {_e}\n"
        "Hay activate venv roi chay:\n"
        "   pip install numpy pandas opencv-python scikit-image tqdm\n"
    )

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ==============================================================================
#  CFG  -  TAT CA CAI DAT O DAY
# ==============================================================================
class CFG:
    # ------------------------------ duong dan --------------------------------
    # Tro thang vao thu muc dataset dang Glare/Normal x Train/Val/Test (do
    # train_dental_mar_V*.py tao ra tu ToothFairy, hoac dataset CBCT_MAR goc).
    DATASET_DIR = r"E:\Project\Dental\Dataset_TF"
    OUTPUT_ROOT = None           # None = <thu muc chua file nay>/runs_classical

    # ------------------------------ chay tren nhom nao ------------------------
    SPLITS = ["Train"]           # doi thanh ["Train", "Val", "Test"] neu muon chay het
    MAX_IMAGES_PER_SPLIT = None  # None = chay het toan bo Glare trong moi split
    OVERWRITE = False            # False: bo qua anh da xuat ket qua tu lan chay truoc (thu muc results/)
    COMPARISON_MAX = 60          # so anh so sanh 6-panel xuat MOI split (xem nhanh, khong xuat het cho nhanh)

    # ------------------------------ [1] MASK vung kim loai --------------------
    THRESH = 240                 # nguong "sang bao hoa" = kim loai (giong pipeline GAN)
    KERNEL_SIZE = 7
    DILATE_ITERS = 3             # so lan gian no mask loi (kernel KERNEL_SIZE)
    BLUR_KSIZE = 15              # lam mem bien mask cho blur_blend (le)

    # ------------------------------ [A/B] INPAINT ------------------------------
    INPAINT_RADIUS = 7           # ban kinh lay mau xung quanh de "va" (px)
    INPAINT_DILATE_EXTRA = 2     # gian no THEM mask truoc khi inpaint, tranh con vien sang sot lai

    # ------------------------------ [C] BLUR + BLEND ---------------------------
    BILAT_D = 9
    SIGMA_COLOR = 75
    SIGMA_SPACE = 75

    # ------------------------------ lam min nen theo ROI (tuy chon) -----------
    # Chi ap dung cho 1 phuong an "chinh" (POST_BG_SMOOTH_METHOD) de do tot thoi gian chay.
    POST_BG_SMOOTH = True
    POST_BG_SMOOTH_METHOD = "inpaint_telea"   # "inpaint_telea" | "inpaint_ns" | "blur_blend"
    OTSU_OFFSET = 0
    ROI_CLOSE_KERNEL = 9
    ROI_DILATE_ITERS = 1
    ROI_MIN_AREA_RATIO = 0.001
    BACKGROUND_VALUE = 0
    ROI_MASK_BLUR = 11
    BG_NLM_H = 15                 # 0 = tat khu nhieu NLM (nhanh hon nhieu)
    BG_GAUSS_KSIZE = 7


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS_ALL = ("Train", "Val", "Test")
SCRIPT_PATH = Path(__file__).resolve()
RUN_PREFIX = SCRIPT_PATH.stem
METHODS = ("inpaint_telea", "inpaint_ns", "blur_blend")


# ==============================================================================
#  TIEN ICH CHUNG (giong file train_dental_mar_V*.py de nhat quan)
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


def even_pick(items: list, cap) -> list:
    if cap is None or len(items) <= cap:
        return list(items)
    idx = np.linspace(0, len(items) - 1, cap).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


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


def _label(img_bgr, text):
    cv2.putText(img_bgr, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


# ==============================================================================
#  ROI (Otsu 3 lop) - dung cho buoc lam min nen tuy chon, copy tu file GAN
# ==============================================================================
def multi_otsu_3(values_u8: np.ndarray):
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


def bg_smooth(final: np.ndarray, cfg) -> np.ndarray:
    """Lam min vung nen (ngoai cung ham) theo ROI Otsu - tuy chon, giong pipeline GAN."""
    roi_mask, _, _ = extract_roi_mask(final, cfg.OTSU_OFFSET, cfg.ROI_CLOSE_KERNEL, cfg.ROI_DILATE_ITERS,
                                      cfg.ROI_MIN_AREA_RATIO, cfg.BACKGROUND_VALUE, cfg.ROI_MASK_BLUR)
    bg = cv2.fastNlMeansDenoising(final, None, h=cfg.BG_NLM_H, templateWindowSize=7, searchWindowSize=21) if cfg.BG_NLM_H > 0 else final.copy()
    if cfg.BG_GAUSS_KSIZE > 1:
        bg = cv2.GaussianBlur(bg, (cfg.BG_GAUSS_KSIZE, cfg.BG_GAUSS_KSIZE), 0)
    w = roi_mask.astype(np.float32) / 255.0
    return np.clip(w * final + (1 - w) * bg, 0, 255).astype(np.uint8)


# ==============================================================================
#  MASK vung kim loai + 3 PHUONG AN GIAM NHIEU (khong GAN)
# ==============================================================================
def make_artifact_mask(orig, cfg):
    _, core = cv2.threshold(orig, cfg.THRESH, 255, cv2.THRESH_BINARY)
    art = cv2.dilate(core, np.ones((cfg.KERNEL_SIZE, cfg.KERNEL_SIZE), np.uint8), iterations=cfg.DILATE_ITERS)
    blurred = cv2.GaussianBlur(art, (cfg.BLUR_KSIZE, cfg.BLUR_KSIZE), 0)
    return core, art, blurred.astype(np.float32) / 255.0


def method_inpaint(orig, art_mask, cfg, algo="telea"):
    """[A]/[B] cv2.inpaint: 'va' vung kim loai bang thong tin lan truyen tu vien vao trong."""
    k = 2 * cfg.INPAINT_DILATE_EXTRA + 1
    mask = cv2.dilate(art_mask, np.ones((k, k), np.uint8), iterations=1) if cfg.INPAINT_DILATE_EXTRA > 0 else art_mask
    flag = cv2.INPAINT_TELEA if algo == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(orig, mask, cfg.INPAINT_RADIUS, flag)


def method_blur_blend(orig, M, cfg):
    """[C] Bilateral filter + hoa tron theo mask M - CHINH LA hau xu ly dang dung sau GAN,
    nhung ap thang len anh goc (khong co GAN o giua)."""
    smoothed = cv2.bilateralFilter(orig, d=cfg.BILAT_D, sigmaColor=cfg.SIGMA_COLOR, sigmaSpace=cfg.SIGMA_SPACE)
    return np.clip(M * smoothed.astype(np.float32) + (1 - M) * orig.astype(np.float32), 0, 255).astype(np.uint8)


def process_one(orig, cfg):
    core, art, M = make_artifact_mask(orig, cfg)
    out = {
        "inpaint_telea": method_inpaint(orig, art, cfg, "telea"),
        "inpaint_ns": method_inpaint(orig, art, cfg, "ns"),
        "blur_blend": method_blur_blend(orig, M, cfg),
    }
    bg_out = None
    if cfg.POST_BG_SMOOTH:
        try:
            bg_out = bg_smooth(out[cfg.POST_BG_SMOOTH_METHOD], cfg)
        except Exception as e:
            log(f"  (ROI/bg-smooth loi -> bo qua cho anh nay: {e})")
    return core, art, M, out, bg_out


def comparison_montage(orig, core, M, out, bg_out, cfg, panel_h=256):
    def pan(im, title, cmap=False):
        im = im if im.dtype == np.uint8 else np.clip(im * 255, 0, 255).astype(np.uint8)
        im = cv2.applyColorMap(im, cv2.COLORMAP_JET) if cmap else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        s = panel_h / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), panel_h))
        return _label(im, title)

    diff = np.clip(cv2.absdiff(out["inpaint_telea"], orig).astype(np.float32) * 4, 0, 255).astype(np.uint8)
    panels = [pan(orig, "1 original"), pan(core, "2 metal core"), pan(M, "3 mask M", True),
              pan(out["inpaint_telea"], "4 inpaint TELEA"), pan(out["inpaint_ns"], "5 inpaint NS"),
              pan(out["blur_blend"], "6 blur+blend")]
    if bg_out is not None:
        panels.append(pan(bg_out, f"7 {cfg.POST_BG_SMOOTH_METHOD}+bg"))
    panels.append(pan(diff, "|TELEA-orig| x4"))
    return np.hstack(panels)


# ==============================================================================
#  CHAY TREN TOAN BO DATASET
# ==============================================================================
def stage_classical(cfg, run_dir: Path) -> None:
    root = Path(cfg.DATASET_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"Khong tim thay dataset: {root} (kiem tra CFG.DATASET_DIR)")
    out_root = run_dir / "results"
    cmp_root = run_dir / "comparison"
    metrics_path = run_dir / "metrics.csv"
    rows = []
    sat = lambda a: float((a >= cfg.THRESH).mean())

    for split in cfg.SPLITS:
        files = list_images(root / "Glare" / split)
        if not files:
            log(f"  [CANH BAO] Khong co anh Glare trong {root / 'Glare' / split} -> bo qua split '{split}'")
            continue
        if cfg.MAX_IMAGES_PER_SPLIT:
            files = even_pick(files, cfg.MAX_IMAGES_PER_SPLIT)
        cmp_files = set(p.name for p in even_pick(files, cfg.COMPARISON_MAX))
        log(f"Xu ly {split}: {len(files)} anh Glare")
        it = tqdm(files, desc=f"Classical {split}", unit="anh") if tqdm else files
        n_err = 0
        for p in it:
            done_marker = out_root / "inpaint_telea" / split / p.name
            if done_marker.exists() and not cfg.OVERWRITE:
                continue
            try:
                orig = read_gray_uint8(p)
                core, art, M, out, bg_out = process_one(orig, cfg)
                for name, im in out.items():
                    write_image(out_root / name / split / p.name, im)
                if bg_out is not None:
                    write_image(out_root / f"{cfg.POST_BG_SMOOTH_METHOD}_bg" / split / p.name, bg_out)
                if p.name in cmp_files:
                    write_image(cmp_root / split / (p.stem + "_cmp.png"),
                               comparison_montage(orig, core, M, out, bg_out, cfg))
                row = {"split": split, "file": p.name, "mask_area_pct": float((M > 0).mean() * 100),
                       "sat_orig": sat(orig)}
                for name, im in out.items():
                    row[f"sat_{name}"] = sat(im)
                    row[f"ssim_{name}_vs_orig"] = float(ssim_fn(orig, im, data_range=255))
                    row[f"psnr_{name}_vs_orig"] = float(psnr_fn(orig, im, data_range=255)) if not np.array_equal(orig, im) else 99.0
                if bg_out is not None:
                    row["sat_bg_smooth"] = sat(bg_out)
                rows.append(row)
            except Exception as e:
                n_err += 1
                log(f"  [LOI] {p.name}: {e}")
                if n_err <= 3:
                    log("    " + traceback.format_exc().strip().splitlines()[-1])
        if n_err:
            log(f"  {split}: {n_err} anh loi (xem log o tren).")

    if rows:
        df_new = pd.DataFrame(rows)
        if metrics_path.exists():
            df_new = pd.concat([pd.read_csv(metrics_path), df_new]).drop_duplicates(["split", "file"], keep="last")
        run_dir.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(metrics_path, index=False)
        num_cols = [c for c in df_new.columns if c not in ("split", "file")]
        summ = df_new.groupby("split")[num_cols].mean().round(4)
        (run_dir / "summary.txt").write_text(f"config: {cfg.POST_BG_SMOOTH_METHOD} (+bg smooth)\n{summ.to_string()}\n",
                                               encoding="utf-8")
        log("Tom tat (trung binh moi split):\n" + summ.to_string())
    else:
        log("Khong co anh nao duoc xu ly (co the tat ca da xong tu lan chay truoc - xem CFG.OVERWRITE).")
    log("Ket qua:", out_root, "| So sanh:", cmp_root)


# ==============================================================================
#  MAIN
# ==============================================================================
def pick_run_dir(cfg):
    """Giong RESUME='auto' cua file GAN: neu co run truoc CHUA xong (khong co DONE.json) thi
    dung lai thu muc do (anh da xuat se duoc bo qua nho CFG.OVERWRITE=False) - tranh xu ly lai
    tu dau neu bi ngat giua chung tren tap du lieu lon."""
    root = Path(cfg.OUTPUT_ROOT) if cfg.OUTPUT_ROOT else SCRIPT_PATH.parent / "runs_classical"
    root.mkdir(parents=True, exist_ok=True)
    cands = [d for d in root.glob(f"{RUN_PREFIX}_*") if d.is_dir() and not (d / "DONE.json").exists()]
    if cands:
        d = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        return d, True
    d = root / f"{RUN_PREFIX}_{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}"
    d.mkdir(parents=True)
    return d, False


def main():
    cfg = CFG
    run_dir, resumed = pick_run_dir(cfg)
    log.open(run_dir / "log.txt")
    log("=" * 78)
    log(f"{'TIEP TUC' if resumed else 'RUN MOI'}:", run_dir)
    log("Splits:", cfg.SPLITS, "| Dataset:", cfg.DATASET_DIR)
    atomic_json(run_dir / "config_snapshot.json", cfg_snapshot())
    t0 = time.time()
    try:
        stage_classical(cfg, run_dir)
        atomic_json(run_dir / "DONE.json", {"finished": datetime.now().isoformat(timespec="seconds")})
        log(f"HOAN TAT trong {(time.time() - t0) / 60:.1f} phut. Toan bo ket qua nam trong: {run_dir}")
    except KeyboardInterrupt:
        log("Da dung (Ctrl+C). Chay lai file nay se TIEP TUC tu day (bo qua anh da xong).")
        sys.exit(130)


if __name__ == "__main__":
    main()
