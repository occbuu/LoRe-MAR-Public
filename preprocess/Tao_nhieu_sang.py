# -*- coding: utf-8 -*-
"""
==============================================================================
 tao_nhieu_sang.py - GIA LAP NHIEU SANG KIM LOAI (METAL GLARE) TREN ANH NORMAL
==============================================================================
 Doc anh Normal (sach) tu Dataset_TF/Normal/{Train,Val,Test}, chon vai RANG that va bien chinh cac rang
 do thanh "rang kim loai": rang bao hoa 255 + phat sang (bloom) + tia sang toa ra TU CHINH rang do.
 KHONG dung template/hinh dang ngoai. Moi rang co kieu nhieu rieng (so tia, do dai, do sang, do rong
 vang sang random rieng cho tung rang).

 V12: vung cho phep nhieu = CHI vung xam (mo mem/xuong), nguong thich nghi theo tung anh, loai ca
      vien toi dan ngoai vung xam.
 V11: vet soc DAY hon (rong ~2-4px, doa rong dan, khong nhon nhu kim); nhieu bi CHAN trong vung anh
      CBCT (body_mask), khong lem ra nen den.
 V10 (sua theo phan hoi):
   - Bo template Glare (khong con dan hinh la vao anh). Kim loai = CHINH mask rang segment duoc.
   - Nhieu SANG RUC: rang = 255 (khong bi lam mo/giam mau), co vang sang quanh rang + tia sang.
   - Bo dai TOI noi cac rang (gay vet den lum la). Tia toi chi con rat nhe, hiem.
   - Moi rang 1 kieu nhieu: "bloom" (vang sang lon, it tia), "star" (nhieu tia manh), "mixed".

 Ket qua ghi vao CFG.OUT_DIR:
   images/<split>/<ten>.png   - anh Normal da them nhieu (ten giu nguyen ten anh Normal goc)
   compare/<split>/<ten>.png  - anh ghep canh nhau: TRAI = Normal goc | PHAI = da them nhieu
   _preview/contact_sheet_*.png
   manifest_tao_nhieu_sang.csv

 Chay:
   conda activate /workspace/projects/envs/anlk
   python src/preprocess/Tao_nhieu_sang.py
==============================================================================
"""
import os
import random
from pathlib import Path

import numpy as np
import cv2
import pandas as pd


# ------------------------------------------------------------------ CFG ----
class CFG:
    _IS_WIN = (os.name == "nt")
    SRC_NORMAL_DIR = os.environ.get(
        "MAR_SRC_NORMAL_DIR",
        r"E:\Project\Dental\Dataset_TF\Normal" if _IS_WIN
        else "/workspace/projects/CBCT/Dataset_TF/Normal")
    OUT_DIR = os.environ.get(
        "MAR_TAO_NHIEU_OUT",
        r"E:\Project\Dental\Dataset_TF\tao_nhieu_sang" if _IS_WIN
        else "/workspace/projects/CBCT/Dataset_TF/tao_nhieu_sang")

    N_IMAGES = None        # None = het anh moi split; dat vd 30 de test nhanh
    SEED = 42

    # ---- segment RANG (chuoi blob sang, gon, xep doc cung ham) ----
    TOOTH_THRESHOLDS = (205, 225, 240)   # thu lan luot cac nguong sang; nguong dau cho ra chuoi rang hop le thi dung
    TOOTH_BLOB_MIN_AREA_PX = 50
    TOOTH_BLOB_MAX_AREA_PX = 1500
    TOOTH_BLOB_MAX_SIDE_PX = 55
    TOOTH_BLOB_MAX_ASPECT = 2.0
    TOOTH_BLOB_MIN_SOLIDITY = 0.78
    TOOTH_MIN_BLOBS = 4                  # can >= 4 rang gon xep thanh chuoi moi coi la co hang rang
    TOOTH_NEIGHBOR_DIST_PX = 45          # moi rang phai co >= 2 rang lang gieng trong khoang nay
    TOOTH_DILATE_PX = 1                  # gian mask rang de phu het vien rang

    # ---- chon rang kim loai ----
    N_GROUPS_RANGE = (1, 2)              # so nhom rang kim loai / anh
    GROUP_RADIUS_RANGE = (40, 90)        # ban kinh (px) gom cac rang ke nhau thanh 1 nhom
    GROUP_MAX_TEETH = 6
    METAL_VALUE = 255                    # rang kim loai bao hoa hoan toan

    # ---- moi rang 1 kieu nhieu rieng (chon ngau nhien theo xac suat p) ----
    #   rays: so tia | amp: do sang tia | width: sigma tia (px) | length: do dai tia (px)
    #   bloom_amp / bloom_sigma: do sang / ban kinh vang sang quanh rang
    STYLES = {
        "bloom": dict(p=0.25, rays=(3, 6),  amp=(40, 75), width=(2.0, 3.0), length=(35, 70), bloom_amp=(45, 80), bloom_sigma=(3, 5)),
        "star":  dict(p=0.45, rays=(6, 12), amp=(35, 75), width=(1.5, 2.6), length=(40, 95), bloom_amp=(25, 55), bloom_sigma=(2, 4)),
        "mixed": dict(p=0.30, rays=(4, 9),  amp=(40, 80), width=(1.7, 2.8), length=(35, 80), bloom_amp=(35, 65), bloom_sigma=(2.5, 4.5)),
    }
    RAY_WIDEN_PER_PX = 0.015              # vet soc doa rong dan theo khoang cach (px rong them / px di xa) - vet DAY, khong nhon nhu kim
    BODY_TISSUE_FRAC = 0.5               # V12: nguong vung XAM = 0.5 * trung vi do sang cua phan mo (pixel > BODY_FG_MIN)
    BODY_FG_MIN = 10                     # pixel <= gia tri nay bi coi la nen den khi tinh trung vi mo
    BODY_BLUR_SIGMA = 3.0                # lam mem truoc khi nguong (bo qua vien toi dan / nhieu le te)
    BODY_CLOSE_PX = 9
    BODY_ERODE_PX = 3                    # co vung anh vao 1 chut de nhieu khong lem sat mep nen den
    # V13: QUANG SANG KHUECH TAN (haze) - anh Glare that co dai sang mo, rong (nhieu chuc px) tu kim loai chay vao vung
    # den/xam (vd vet lem vao o den ben trong ham). Cac kieu nhieu cu chi co bloom hep + tia mong nen mang khong hoc
    # duoc cach xoa quang sang nay. HAZE_ENABLE=1 (env MAR_HAZE) them: (a) quang sang rong bao ca nhom kim loai,
    # (b) vai "cai chem" sang rat rong toa vao ben trong. Mac dinh TAT de khong doi hanh vi cu.
    HAZE_ENABLE = os.environ.get("MAR_HAZE", "0") == "1"
    HAZE_SIGMA = (8, 18)                 # sigma (px) quang sang bao nhom kim loai
    HAZE_AMP = (22, 50)                  # do sang dinh cua quang sang
    HAZE_WEDGES = (2, 5)                 # so cai chem sang rong / anh
    HAZE_WEDGE_WIDTH = (4.0, 8.0)        # sigma (px) tai goc cai chem
    HAZE_WEDGE_WIDEN_PER_PX = 0.06       # doa rong theo khoang cach
    HAZE_WEDGE_LENGTH = (35, 75)         # px
    HAZE_WEDGE_AMP = (18, 42)
    RAY_DARK_PROB = 0.12                 # ty le tia TOI (rat nhe) - giu thap de khong sinh vet den lum la
    RAY_DARK_AMP = (10, 28)               # bien do tia toi (nho)

    PREVIEW_N = 16
    PREVIEW_PANEL_H = 220
    PREVIEW_PER_SHEET = 4


# ------------------------------------------------------------- helpers -----
def read_gray_uint8(path) -> np.ndarray:
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
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


# ------------------------------------------------------- segment RANG -------
def _tooth_blobs_at(gray: np.ndarray, cfg, thresh: int):
    """Segment RANG theo 1 nguong sang: giu blob nho, gon (dien tich / ti le canh / solidity), roi chi giu
    blob co >= 2 blob lang gieng (rang xep thanh chuoi doc cung ham; manh xuong vo/vanh xuong la dai,
    mong, roi rac nen bi loai). Tra ve (mask uint8 0/255, so blob)."""
    empty = np.zeros(gray.shape, np.uint8)
    b = cv2.GaussianBlur(gray, (3, 3), 0)
    m = ((b >= thresh) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = []
    for i in range(1, n):
        a, w, h = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if a < cfg.TOOTH_BLOB_MIN_AREA_PX or a > cfg.TOOTH_BLOB_MAX_AREA_PX or max(w, h) > cfg.TOOTH_BLOB_MAX_SIDE_PX:
            continue
        if max(w, h) / max(min(w, h), 1) > cfg.TOOTH_BLOB_MAX_ASPECT:
            continue
        cs, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea)))
        if a / max(hull, 1.0) < cfg.TOOTH_BLOB_MIN_SOLIDITY:
            continue
        keep.append(i)
    if len(keep) < cfg.TOOTH_MIN_BLOBS:
        return empty, 0
    C = cen[keep]
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    ok = [keep[j] for j in range(len(keep)) if (D[j] < cfg.TOOTH_NEIGHBOR_DIST_PX).sum() - 1 >= 2]
    if len(ok) < cfg.TOOTH_MIN_BLOBS:
        return empty, 0
    return (np.isin(lbl, ok) * 255).astype(np.uint8), len(ok)


def segment_teeth(gray: np.ndarray, cfg) -> np.ndarray:
    """Mask RANG (uint8 0/255). Khong tim thay chuoi rang -> mask rong (anh se duoc giu nguyen)."""
    for t in cfg.TOOTH_THRESHOLDS:
        m, k = _tooth_blobs_at(gray, cfg, t)
        if k:
            if cfg.TOOTH_DILATE_PX > 0:
                kk = 2 * cfg.TOOTH_DILATE_PX + 1
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
            return m
    return np.zeros(gray.shape, np.uint8)


# ------------------------------------------------------------- gia lap ------
def body_mask(gray: np.ndarray, cfg) -> np.ndarray:
    """Vung anh CBCT that (float32 0..1), loai nen den bao quanh: nguong thich nghi = BODY_TISSUE_FRAC * trung vi mo (chi vung XAM, bo vien toi dan), dong lo nho,
    lay thanh phan lon nhat, to day (fill) cac lo ben trong (khoang khi/thanh quan nam trong vung anh
    van duoc tinh), co vao BODY_ERODE_PX, lam mem mep. Nhieu chi duoc phep xuat hien trong mask nay."""
    fg = gray[gray > cfg.BODY_FG_MIN]
    if fg.size == 0:
        return np.zeros(gray.shape, np.float32)
    thr = cfg.BODY_TISSUE_FRAC * float(np.median(fg))    # thich nghi tung anh: vung XAM (mo), loai vien toi dan ben ngoai
    m = (cv2.GaussianBlur(gray, (0, 0), cfg.BODY_BLUR_SIGMA) > thr).astype(np.uint8) * 255
    k = 2 * cfg.BODY_CLOSE_PX + 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return np.zeros(gray.shape, np.float32)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    if cfg.BODY_ERODE_PX > 0:
        ke = 2 * cfg.BODY_ERODE_PX + 1
        filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
    return cv2.GaussianBlur(filled.astype(np.float32) / 255.0, (0, 0), 1.5)


def _pick_style(cfg, rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for name, s in cfg.STYLES.items():
        acc += s["p"]
        if r <= acc:
            return name
    return next(iter(cfg.STYLES))


def make_glare(orig: np.ndarray, cfg, rng: random.Random, return_core: bool = False) -> tuple:
    """Chon 1-2 nhom rang ke nhau -> bien thanh rang kim loai: rang = 255, vang sang + tia sang toa ra
    TU CHINH rang do (moi rang 1 kieu rieng). Tra ve (anh uint8, list thong tin tung rang kim loai);
    return_core=True -> tra them mask kim loai gia lap (uint8 0/255) lam nhan chinh xac (vd cho YOLO)."""
    h, w = orig.shape
    tooth_u8 = segment_teeth(orig, cfg)
    n_lbl, tlbl, _, tcen = cv2.connectedComponentsWithStats(tooth_u8, connectivity=8)
    if n_lbl <= 1:
        return (orig.copy(), [], np.zeros_like(orig)) if return_core else (orig.copy(), [])

    metal_ids = set()
    for _ in range(rng.randint(*cfg.N_GROUPS_RANGE)):
        seed = rng.randrange(1, n_lbl)
        d = np.linalg.norm(tcen[1:] - tcen[seed], axis=1)
        radius = rng.uniform(*cfg.GROUP_RADIUS_RANGE)
        near = [int(i) + 1 for i in np.argsort(d) if d[i] <= radius][:cfg.GROUP_MAX_TEETH]
        metal_ids.update(near)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = np.zeros((h, w), np.float32)      # phan CONG them (sang)
    dark = np.zeros((h, w), np.float32)      # phan TRU di (rat nhe)
    core = np.zeros((h, w), bool)
    info = []

    for tid in sorted(metal_ids):
        tmask = (tlbl == tid)
        core |= tmask
        ty, tx = np.where(tmask)
        style = _pick_style(cfg, rng)
        st = cfg.STYLES[style]

        # (1) vang sang (bloom) quanh CHINH rang: blur mask rang -> sang nhat sat rang, giam dan ra ngoai
        sigma = rng.uniform(*st["bloom_sigma"])
        bloom = cv2.GaussianBlur(tmask.astype(np.float32), (0, 0), sigma)
        bloom /= max(float(bloom.max()), 1e-6)
        glow += rng.uniform(*st["bloom_amp"]) * bloom

        # (2) tia: moi tia xuat phat tu 1 diem NGAU NHIEN TREN rang do, huong/do sang/do dai rieng
        n_rays = rng.randint(*st["rays"])
        for _r in range(n_rays):
            k = rng.randrange(len(ty))
            cy, cx = float(ty[k]), float(tx[k])
            ang = rng.uniform(0, 2 * np.pi)
            wid = rng.uniform(*st["width"])
            length = rng.uniform(*st["length"])
            c, s = np.cos(ang), np.sin(ang)
            dx, dy = xx - cx, yy - cy
            t = dx * c + dy * s
            perp = -dx * s + dy * c
            tp = np.clip(t, 0, None)
            wt = wid + cfg.RAY_WIDEN_PER_PX * tp                      # doa rong dan -> vet day, khong nhon
            # do sang gan nhu DEU doc vet roi tat dan o cuoi (khong giam nhanh thanh mui kim)
            prof = np.exp(-(perp ** 2) / (2 * wt ** 2)) * np.exp(-(tp / length) ** 2) * (t > 0)
            if rng.random() < cfg.RAY_DARK_PROB:
                dark += rng.uniform(*cfg.RAY_DARK_AMP) * prof
            else:
                glow += rng.uniform(*st["amp"]) * prof
        info.append({"style": style, "cx": round(float(tx.mean()), 1), "cy": round(float(ty.mean()), 1),
                     "tooth_area_px": int(tmask.sum()), "n_rays": n_rays})

    if cfg.HAZE_ENABLE and core.any():
        hz = cv2.GaussianBlur(core.astype(np.float32), (0, 0), rng.uniform(*cfg.HAZE_SIGMA))
        glow += rng.uniform(*cfg.HAZE_AMP) * hz / max(float(hz.max()), 1e-6)
        cys, cxs = np.where(core)
        for _r in range(rng.randint(*cfg.HAZE_WEDGES)):
            k = rng.randrange(len(cys))
            cy, cx = float(cys[k]), float(cxs[k])
            ang = rng.uniform(0, 2 * np.pi)
            c, s_ = np.cos(ang), np.sin(ang)
            dx, dy = xx - cx, yy - cy
            t = dx * c + dy * s_
            perp = -dx * s_ + dy * c
            tp = np.clip(t, 0, None)
            wt = rng.uniform(*cfg.HAZE_WEDGE_WIDTH) + cfg.HAZE_WEDGE_WIDEN_PER_PX * tp
            length = rng.uniform(*cfg.HAZE_WEDGE_LENGTH)
            glow += rng.uniform(*cfg.HAZE_WEDGE_AMP) * np.exp(-(perp ** 2) / (2 * wt ** 2)) * np.exp(-(tp / length) ** 2) * (t > 0)
    body = body_mask(orig, cfg)                # nhieu chi nam trong vung anh CBCT, khong lem ra nen den
    out = np.clip(orig.astype(np.float32) + (glow - dark) * body, 0, 255)
    out[core] = cfg.METAL_VALUE                # rang kim loai luon sang bao hoa, khong bi lam mo
    if return_core:
        return out.astype(np.uint8), info, core.astype(np.uint8) * 255
    return out.astype(np.uint8), info


def make_contact_sheet(pairs: list, out_path: Path, h: int) -> None:
    rows = []
    for name, a, b in pairs:
        wa = max(1, int(a.shape[1] * h / a.shape[0]))
        pa = cv2.cvtColor(cv2.resize(a, (wa, h)), cv2.COLOR_GRAY2BGR)
        pb = cv2.cvtColor(cv2.resize(b, (wa, h)), cv2.COLOR_GRAY2BGR)
        cv2.putText(pa, "goc (Normal)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pb, "da them nhieu", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pb, name[-18:], (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack([pa, pb]))
    if not rows:
        return
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))


# ------------------------------------------------------------------ main ---
def _find_splits(src_root: Path) -> list:
    """SRC_NORMAL_DIR co the la .../Normal (co Train/Val/Test) hoac .../Normal/Train (tu tim split anh em)."""
    names = ("Train", "Val", "Test")
    if any((src_root / n).is_dir() for n in names):
        base = src_root
    elif src_root.name in names:
        base = src_root.parent
    else:
        return [(src_root.name, src_root)] if src_root.is_dir() else []
    return [(sp, base / sp) for sp in names if (base / sp).is_dir()]


def main():
    cfg = CFG
    src_root = Path(cfg.SRC_NORMAL_DIR)
    out_root = Path(cfg.OUT_DIR)
    splits = _find_splits(src_root)
    if not splits:
        raise FileNotFoundError(f"Khong tim thay anh Normal trong {src_root}. Kiem tra CFG.SRC_NORMAL_DIR.")

    rng = random.Random(cfg.SEED)
    rows, preview_buf = [], []
    n_total = n_noisy = 0
    for sp, folder in splits:
        files = list_images(folder)
        if cfg.N_IMAGES:
            files = files[:cfg.N_IMAGES]
        print(f"[{sp}] {len(files)} anh Normal tu {folder}")
        for i, p in enumerate(files):
            orig = read_gray_uint8(p)
            synth, info = make_glare(orig, cfg, rng)
            write_image(out_root / "images" / sp / p.name, synth)
            n_total += 1
            if info:
                n_noisy += 1
                rows += [{"split": sp, "file": p.name, **t} for t in info]
                pair = np.hstack([orig, np.full((orig.shape[0], 4), 255, np.uint8), synth])
                write_image(out_root / "compare" / sp / p.name, pair)
                if len(preview_buf) < cfg.PREVIEW_N:
                    preview_buf.append((p.name, orig, synth))
            if (i + 1) % 100 == 0 or (i + 1) == len(files):
                print(f"  ... {sp}: {i + 1}/{len(files)} anh")

    for start in range(0, len(preview_buf), cfg.PREVIEW_PER_SHEET):
        make_contact_sheet(preview_buf[start:start + cfg.PREVIEW_PER_SHEET],
                           out_root / "_preview" / f"contact_sheet_{start // cfg.PREVIEW_PER_SHEET:02d}.png",
                           cfg.PREVIEW_PANEL_H)
    if rows:
        pd.DataFrame(rows).to_csv(out_root / "manifest_tao_nhieu_sang.csv", index=False)

    print("=" * 78)
    print(f"XONG. {n_noisy}/{n_total} anh co them nhieu; {n_total - n_noisy} anh GIU NGUYEN (khong tach duoc chuoi rang).")
    print("Anh da them nhieu      :", out_root / "images" / "<split>")
    print("Anh SO SANH (goc|nhieu):", out_root / "compare" / "<split>")
    print("Contact sheet          :", out_root / "_preview")
    print("Manifest               :", out_root / "manifest_tao_nhieu_sang.csv")


if __name__ == "__main__":
    main()
