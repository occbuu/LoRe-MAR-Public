# -*- coding: utf-8 -*-
"""
==============================================================================
 migrate_checkpoint_v8.py - CHUYEN checkpoint Colab (62 epoch) vao dung run
 cuc bo cua train_dental_mar_V8.py de "chay tiep" duoi ham loss/diem so DA SUA.
==============================================================================
Chay 1 lan (dung venv da cai torch, giong luc chay train_dental_mar_V8.py):

    python migrate_checkpoint_v8.py

Roi chay lai binh thuong:

    python train_dental_mar_V8.py

RESUME="auto" (mac dinh trong CFG) se tu nhan ra run nay CHUA xong va tiep tuc
tu epoch 63/600, dung ham loss + tieu chi chon best.pth DA SUA cua V8 (mask
loss 1-chieu + halo-aware, hop mask generator vao hau xu ly, val_score co
halo_reduction) - khong phai lap lai dung tieu chi CU da khien no bi "early
stop" o epoch 62 voi best=epoch 2 (chi 2 epoch dau, gan nhu chua sua duoc gi).

Vi sao can script nay (khong the chi doi ten file):
  - Checkpoint tren Colab dung ten key DIFFERENT: "D_X"/"D_Y"/"ema_G" thay vi
    "DX"/"DY"/"G_ema" cua ban train_dental_mar_V8.py hien tai -> phai remap.
  - "no_improve" cu = 59 (theo tieu chi diem so CU). Neu giu nguyen, luc resume
    train_dental_mar_V8.py se thay 59 >= EARLY_STOP_PATIENCE (40) va DUNG NGAY,
    khong train them epoch nao ca. Script nay RESET no_improve = 0 de cho no
    mot "cua so" patience moi, dua tren tieu chi diem so DA SUA.
  - best.pth cu (epoch 2, chi ~2 epoch, gan nhu chua sua gi) khong dang tin
    cay bang trong so da train 62 epoch (last.pth) - script nay ghi lai
    best.pth tu chinh trong so epoch 62 (EMA) de neu vai chuc epoch tiep theo
    van chua vuot duoc moc diem cu, hau xu ly (stage_post) van dung trong so
    TOT NHAT hien co (epoch 62) chu khong phai epoch 2 cu.
==============================================================================
"""
import shutil
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    sys.exit("Thieu torch - hay activate dung venv da dung de chay train_dental_mar_V8.py roi chay lai.")

# ---------------------------------------------------------------------------
# 1) SUA 2 DUONG DAN NAY NEU CAN (mac dinh da dien theo may cua ban)
# ---------------------------------------------------------------------------
SRC_CHECKPOINT = Path(r"E:\Project\Dental\src\train\runs\Dataset_TF_runs\mar_v8\checkpoints\last.pth")
# Thu muc run CUC BO da co san (RUN MOI luc 01:09:37 23-09-2026, da Ctrl+C o epoch 3) -
# se ghi CHE checkpoints/best.pth va checkpoints/last.pth cua no bang checkpoint da chuyen doi.
TARGET_RUN_DIR = Path(r"E:\Project\Dental\src\train\runs\train_dental_mar_V8_gan_01-09-37_23-09-2026")
# ---------------------------------------------------------------------------

if not SRC_CHECKPOINT.exists():
    sys.exit(f"[LOI] Khong thay checkpoint nguon: {SRC_CHECKPOINT}")
if not TARGET_RUN_DIR.is_dir():
    sys.exit(f"[LOI] Khong thay thu muc run dich: {TARGET_RUN_DIR}")

ck_dir = TARGET_RUN_DIR / "checkpoints"
ck_dir.mkdir(parents=True, exist_ok=True)

print("Dang doc checkpoint Colab (co the mat vai chuc giay, file ~350MB)...")
old = torch.load(SRC_CHECKPOINT, map_location="cpu", weights_only=False)
print(f"  -> epoch={old['epoch']} | best cu={old['best']} | no_improve cu={old['no_improve']}")

# Backup 2 file cu (neu co) truoc khi ghi de, de an toan
for name in ("last.pth", "best.pth"):
    p = ck_dir / name
    if p.exists():
        bak = p.with_suffix(".pth.bak_before_migrate")
        shutil.copy2(p, bak)
        print(f"  Da backup {p.name} -> {bak.name}")

new_last = {
    "epoch": old["epoch"],
    "G": old["G"], "F": old["F"],
    "DX": old["D_X"], "DY": old["D_Y"],
    "opt_G": old["opt_G"], "opt_D": old["opt_D"],
    "sch_G": old["sch_G"], "sch_D": old["sch_D"],
    "history": old["history"],
    "best": old["best"],       # giu moc diem cu (0.6577 @ epoch 2) lam "vach" de vuot qua
    "no_improve": 0,           # QUAN TRONG: reset ve 0 (xem giai thich dau file)
    "rng": old["rng"],
    "cfg": old["cfg"],
    "G_ema": old["ema_G"],
    "ema_step": 100_000,       # bo qua warmup EMA - model da truong thanh (62 epoch)
}
torch.save(new_last, ck_dir / "last.pth")
print(f"  Da ghi {ck_dir / 'last.pth'}")

new_best = {
    "gen_G": old["ema_G"],
    "gen_F": old["F"],
    "epoch": old["epoch"],
    "metrics": old["history"][-1],
}
torch.save(new_best, ck_dir / "best.pth")
print(f"  Da ghi {ck_dir / 'best.pth'} (tam thoi = trong so epoch {old['epoch']}, se duoc thay khi co epoch moi tot hon)")

print("\nXONG. Chay lai:  python train_dental_mar_V8.py")
print("Se thay dong: '==> TIEP TUC tu last.pth: da xong epoch 62, se chay tu epoch 63/600'")
