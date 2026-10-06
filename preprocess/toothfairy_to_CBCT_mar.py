"""
Map ToothFairy_Dataset (data.npy / gt_*.npy per patient)  ->  CBCT_MAR layout

    Dataset/
      Glare/  {Train,Val,Test}/slice_<patient>_<idx>.png   (slices WITH metal artefact)
      Normal/ {Train,Val,Test}/slice_<patient>_<idx>.png   (clean slices)

Usage (Windows, from E:\\Project\\Dental):
    pip install numpy opencv-python tqdm
    python toothfairy_to_cbct_mar.py --inspect            # 1) look at shapes / value ranges first
    python toothfairy_to_cbct_mar.py --out Dataset_TF     # 2) convert

Train/Val/Test comes from ToothFairy_Dataset/splits.json (split by PATIENT -> no leakage).
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

ROOT = Path(r"E:\Project\Dental\ToothFairy_dataset\ToothFairy_Dataset")
VOL_DIR = ROOT / "Dataset"
SPLITS = ROOT / "splits.json"

# ---- tuneable -------------------------------------------------------------
AXIS = 2               # axis to slice along (axial). Check with --inspect!
P_LO, P_HI = 0.5, 99.5 # window used for uint8 conversion (per volume)
METAL_PCT = 99.9       # "metal" = voxels above this percentile of the volume ...
METAL_MIN_PIXELS = 300 # ... a slice is Glare if it has >= this many such pixels
BONE_ROI_MIN = 0.02    # skip near-empty slices (top/bottom of the scan)
STEP = 1               # keep every STEP-th slice
# ---------------------------------------------------------------------------


def patient_no(pid):            # "P105" -> 105
    return int(re.sub(r"\D", "", pid))


def to_uint8(sl, lo, hi):
    return (np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def inspect(n=3):
    for p in sorted(VOL_DIR.iterdir())[:n]:
        d = np.load(p / "data.npy", mmap_mode="r")
        g = np.load(p / "gt_sparse.npy", mmap_mode="r")
        print(p.name, "data", d.shape, d.dtype, "min/max", d.min(), d.max(),
              "| gt", g.shape, g.dtype, "labels", np.unique(g)[:10])


def convert(out_dir):
    splits = json.load(open(SPLITS))            # keys: train / val / test
    counter = 0                                  # global running index (like slice_003_9922)
    rows = []
    for split_key, split_name in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        for pid in tqdm(splits.get(split_key, []), desc=split_name):
            f = VOL_DIR / pid / "data.npy"
            if not f.exists():
                continue
            vol = np.load(f)                     # ~177 MB per patient
            vol = np.moveaxis(vol, AXIS, 0)      # -> (slices, H, W)
            lo, hi = np.percentile(vol, [P_LO, P_HI])
            metal_thr = np.percentile(vol, METAL_PCT)
            for z in range(0, vol.shape[0], STEP):
                sl = vol[z]
                img = to_uint8(sl, lo, hi)
                if (img > 30).mean() < BONE_ROI_MIN:
                    continue
                is_glare = (sl >= metal_thr).sum() >= METAL_MIN_PIXELS
                grp = "Glare" if is_glare else "Normal"
                d = Path(out_dir) / grp / split_name
                d.mkdir(parents=True, exist_ok=True)
                name = f"slice_{patient_no(pid):03d}_{counter}.png"
                cv2.imwrite(str(d / name), img)
                rows.append(f"{grp},{split_name},{pid},{z},{name}")
                counter += 1
            del vol
    (Path(out_dir) / "manifest.csv").write_text(
        "group,split,patient,z,file\n" + "\n".join(rows), encoding="utf-8")
    print("done:", counter, "slices ->", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default="Dataset_TF")
    a = ap.parse_args()
    inspect() if a.inspect else convert(a.out)