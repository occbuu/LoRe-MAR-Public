# -*- coding: utf-8 -*-
"""
==============================================================================
 prepare_vn_tooth_external.py - CHUAN BI TAP DU LIEU NGOAI (VN_Tooth) DE KIEM TRA NHIEU
==============================================================================
 Muc dich: 2 bo CBCT DICOM ngoai ("CBCT CVan", "CBCT Truong") KHONG thuoc ToothFairy, dung lam
 TAP TEST HOAN TOAN BEN NGOAI (chua co nhan Glare/Normal - muc dich chinh la XUAT ANH PNG de TU MAT
 kiem tra xem cac ca nay co von da co nhieu sang kim loai (glare) hay khong; neu KHONG co thi se can
 tu them nhieu tong hop sau, viec do KHONG nam trong script nay).

 Script nay CHI lam 2 viec:
   [1] Doc tung series DICOM (1 thu muc = 1 benh nhan, moi file *.dcm = 1 lat) -> ghep thanh 1 khoi
       3D theo dung thu tu khong gian that (ImagePositionPatient/SliceLocation, KHONG dua vao ten
       file vi thu tu ten file khong dam bao dung thu tu lat that).
   [2] Chuan hoa cuong do -> uint8 CUNG KIEU voi train_dental_mar_V8.py (percentile P_LO/P_HI tren
       mau con, ham _to_u8 giong het process_volume() trong file do) - de phan bo pixel cua tap ngoai
       nay KHOP voi domain ma model da duoc huan luyen, roi ghi PNG moi lat + 1 anh "contact sheet"
       xem nhanh (vai lat dai dien) cho tung benh nhan de duyet nhanh hang tram ca ma khong can mo
       tung anh.

 Ket qua ghi vao CFG.OUT_ROOT:
   <OUT_ROOT>/<ten_bo_du_lieu>/<ten_thu_muc_benh_nhan>/slice_0000.png, slice_0001.png, ...
   <OUT_ROOT>/_preview/<ten_bo_du_lieu>/<ten_thu_muc_benh_nhan>.png   (anh xem nhanh, vai lat/benh nhan)
   <OUT_ROOT>/manifest_external.csv                                   (bang tong hop: so lat, kich thuoc...)

 LUU Y QUAN TRONG (da kiem tra truoc khi giao script nay): tai thoi diem viet, TOAN BO cac thu muc
 benh nhan trong "CBCT Truong" (vd 11574719, 11603829, ...) dang RONG (0 file), trong khi "CBCT CVan"
 co day du *.dcm. Script se TU DONG bo qua va IN CANH BAO ro rang cho tung ca rong nhu vay (khong bi
 crash), nhung can kiem tra lai xem du lieu that cua "CBCT Truong" co dang nam o 1 vi tri khac (vd 1
 file .zip/.rar chua giai nen) hay chua duoc chuyen vao day.

 Chay:
   E:\\Project\\.venv\\Scripts\\activate
   pip install pydicom opencv-python numpy pandas
   (neu pydicom bao loi doc pixel_array vi transfer syntax nen - thuong khong can voi du lieu nay vi
   kich thuoc file gan nhu dong deu giua cac lat, tuc la KHONG nen - nhung neu can):
   pip install pylibjpeg pylibjpeg-libjpeg python-gdcm
   python prepare_vn_tooth_external.py
==============================================================================
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import pandas as pd

try:
    import pydicom
except ImportError:
    sys.exit("Thieu thu vien pydicom. Cai: pip install pydicom")


# ------------------------------------------------------------------ CFG ----
class CFG:
    # ten bo du lieu -> thu muc goc (moi thu muc con ben trong = 1 benh nhan)
    SRC_DATASETS = {
        "CBCT_CVan": r"E:\Project\Dental\ToothFairy_dataset\VN_Tooth\Dataset\CBCT CVan",
        "CBCT_Truong": r"E:\Project\Dental\ToothFairy_dataset\VN_Tooth\Dataset\CBCT Trường",
    }
    OUT_ROOT = r"E:\Project\Dental\ToothFairy_dataset\VN_Tooth\Dataset_AI"

    # Chuan hoa cuong do -> uint8: GIONG HET CFG.P_LO/P_HI/SUBSAMPLE va _to_u8() trong
    # train_dental_mar_V8.py (xem process_volume) - de tap ngoai nay cung phan bo pixel voi du lieu
    # da dung huan luyen model.
    P_LO, P_HI = 0.5, 99.5
    SUBSAMPLE = 4          # lay mau 1/4 moi chieu khi tinh percentile (nhanh, giong file chinh)

    SLICE_STEP = 1         # xuat MOI lat (khong bo qua lat nao) - day la du lieu ngoai, can xem HET
                            # de tu kiem tra nhieu, khac voi ToothFairy (SLICE_STEP=2) da co nhan san.
    MIN_FG_FRAC = 0.03      # bo cac lat gan nhu RONG (qua it mo/xuong, vd ngoai vung chup) - nguong
                            # THAP hon file train (0.15) vi chua biet truoc dac diem CBCT ngoai nay.

    PREVIEW_N = 12          # so lat dai dien lay vao anh "contact sheet" xem nhanh / benh nhan
    PREVIEW_PANEL_H = 220   # chieu cao (px) moi o trong contact sheet


# ------------------------------------------------------------- helpers -----
def _to_u8(sl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Y HET _to_u8() trong train_dental_mar_V8.py - giu nguyen cong thuc de dong bo domain."""
    return (np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _z_key(ds) -> float:
    """Khoa sap xep lat theo VI TRI KHONG GIAN THAT, uu tien ImagePositionPatient[2] (chinh xac
    nhat), roi SliceLocation, cuoi cung InstanceNumber - KHONG bao gio dua vao ten file (thu tu ten
    file vd DCT0000.dcm khong dam bao la thu tu lat that trong khong gian)."""
    if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
        try:
            return float(ds.SliceLocation)
        except Exception:
            pass
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def load_dicom_series(patient_dir: Path):
    """Doc tat ca *.dcm trong 1 thu muc benh nhan, sap xep dung thu tu khong gian, ghep thanh khoi
    3D (Z, H, W) da ap dung RescaleSlope/RescaleIntercept (neu co - dua ve dung don vi HU/CT that).
    Tra ve None neu thu muc rong hoac khong doc duoc file .dcm nao."""
    files = sorted(patient_dir.glob("*.dcm"))
    if not files:
        return None, "rong (khong co file .dcm)"
    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            _ = ds.pixel_array  # ep giai ma ngay de bat loi som (vd thieu plugin nen) tai day
        except Exception as e:
            continue
        datasets.append(ds)
    if not datasets:
        return None, f"co {len(files)} file .dcm nhung KHONG doc duoc pixel_array file nao (co the thieu plugin giai nen - xem huong dan dau file)"
    datasets.sort(key=_z_key)
    arrs = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arrs.append(arr * slope + intercept)
    shapes = Counter(a.shape for a in arrs)
    common_shape, n_common = shapes.most_common(1)[0]
    if len(shapes) > 1:
        n_drop = len(arrs) - n_common
        arrs = [a for a in arrs if a.shape == common_shape]
        note = f"canh bao: bo {n_drop} lat khac kich thuoc voi da so ({common_shape})"
    else:
        note = None
    vol = np.stack(arrs, axis=0)  # (Z, H, W)
    return vol, note


def process_patient(vol: np.ndarray, out_dir: Path, cfg) -> tuple:
    """Chuan hoa + ghi PNG moi lat (theo SLICE_STEP/MIN_FG_FRAC). Tra ve (list z da luu, lo, hi)."""
    s = max(1, cfg.SUBSAMPLE)
    sub = np.asarray(vol[::s, ::s, ::s], dtype=np.float32)
    lo, hi = (float(x) for x in np.percentile(sub, [cfg.P_LO, cfg.P_HI]))
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for z in range(0, vol.shape[0], cfg.SLICE_STEP):
        u8 = _to_u8(vol[z], lo, hi)
        if (u8 > 20).mean() < cfg.MIN_FG_FRAC:
            continue
        cv2.imwrite(str(out_dir / f"slice_{z:04d}.png"), u8)
        kept.append(z)
    return kept, lo, hi


def make_preview(vol: np.ndarray, kept_zs: list, lo: float, hi: float, out_path: Path, cfg) -> None:
    """Anh 'contact sheet' - vai lat dai dien cach deu nhau cua benh nhan, ghep ngang 1 hang, de
    duyet nhanh HANG TRAM ca ma khong can mo tung PNG rieng."""
    if not kept_zs:
        return
    n = min(cfg.PREVIEW_N, len(kept_zs))
    idxs = np.linspace(0, len(kept_zs) - 1, n).astype(int)
    zs = sorted({kept_zs[i] for i in idxs})
    h = cfg.PREVIEW_PANEL_H
    panels = []
    for z in zs:
        u8 = _to_u8(vol[z], lo, hi)
        w = max(1, int(u8.shape[1] * h / u8.shape[0]))
        panel = cv2.resize(u8, (w, h))
        cv2.putText(panel, f"z={z}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
        panels.append(panel)
    row = np.concatenate(panels, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), row)


# ------------------------------------------------------------------ main ---
def main():
    cfg = CFG
    out_root = Path(cfg.OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for ds_name, src in cfg.SRC_DATASETS.items():
        src_p = Path(src)
        if not src_p.is_dir():
            print(f"[{ds_name}] BO QUA - khong tim thay thu muc: {src_p}")
            continue
        patients = sorted([d for d in src_p.iterdir() if d.is_dir()])
        print(f"[{ds_name}] {len(patients)} benh nhan trong {src_p}")
        empty_or_failed = []
        for pdir in patients:
            pid = pdir.name
            vol, note = load_dicom_series(pdir)
            if vol is None:
                empty_or_failed.append((pid, note))
                continue
            out_dir = out_root / ds_name / pid
            kept, lo, hi = process_patient(vol, out_dir, cfg)
            preview_path = out_root / "_preview" / ds_name / f"{pid}.png"
            make_preview(vol, kept, lo, hi, preview_path, cfg)
            print(f"  [{pid}] khoi {tuple(vol.shape)} -> {len(kept)}/{vol.shape[0]} lat PNG"
                  + (f"  ({note})" if note else ""))
            summary.append({"dataset": ds_name, "patient": pid, "shape": list(vol.shape),
                             "n_slices_total": int(vol.shape[0]), "n_slices_saved": len(kept),
                             "lo": lo, "hi": hi, "note": note or ""})
        if empty_or_failed:
            print(f"  [CANH BAO] {len(empty_or_failed)}/{len(patients)} benh nhan KHONG xuat duoc anh:")
            for pid, note in empty_or_failed[:15]:
                print(f"      - {pid}: {note}")
            if len(empty_or_failed) > 15:
                print(f"      ... va {len(empty_or_failed) - 15} ca khac (xem manifest_external.csv / errors_external.csv)")
            pd.DataFrame([{"dataset": ds_name, "patient": p, "reason": n} for p, n in empty_or_failed]) \
                .to_csv(out_root / f"errors_{ds_name}.csv", index=False)

    if summary:
        pd.DataFrame(summary).to_csv(out_root / "manifest_external.csv", index=False)
    print("=" * 78)
    print("XONG. Anh PNG day du:", out_root)
    print("Xem NHANH (contact sheet vai lat/benh nhan) tai:", out_root / "_preview")
    print("Bang tong hop:", out_root / "manifest_external.csv")


if __name__ == "__main__":
    main()
