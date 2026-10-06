# -*- coding: utf-8 -*-
"""
==============================================================================
 reset_best_after_score_fix.py
==============================================================================
Chay 1 lan SAU KHI ban da thay file train_dental_mar_V8.py moi (ban sua
val_sat_intensity_reduction) - truoc khi chay lai train_dental_mar_V8.py.

VI SAO CAN SCRIPT NAY:
  Ban thay: den epoch 71 van chua "cai thien" (best van la score=0.6577 tu
  epoch 2 cu, luc con o Colab), va val_sat_reduction "nhay loan" giua cac
  epoch (0.15 <-> 0.99). Da kiem tra history.csv thuc te va xac nhan:
    - val_sat_reduction (ty le SO PIXEL vuot nguong 240) co std ~0.25 SUOT
      CA 71 EPOCH (khong phai loi do resume) - vi mau so (so pixel bao hoa
      trong 1 anh) qua nho, chi can vai pixel dao qua lai la ty le nhay manh.
    - Day la 1 trong 4 thanh phan cua val_score (SCORE_WEIGHTS['sat']) nen
      lam ca diem tong hop nhay loan theo, va epoch 2 (luc model con GAN
      NHU IDENTITY, chua sua duoc gi nhieu) tinh co "trung" mot lan sat_red
      cao bat thuong (0.99) nen dat diem cao gia tao (0.6577) - 1 moc KHONG
      cong bang de cac epoch sau (dang THUC SU hoc sua streak, xem
      val_halo_reduction tang tu 0.045 -> 0.085 qua 62 epoch dau) phai vuot qua.
  File train_dental_mar_V8.py moi da SUA: dung "val_sat_intensity_reduction"
  (do % giam CUONG DO TRUNG BINH trong vung core, cong thuc lien tuc nhu
  halo_reduction, on dinh hon nhieu - khong con la ty le so pixel nua) de
  tinh diem, thay cho "val_sat_reduction" cu. Vi cong thuc tinh diem DA DOI,
  moc "best" cu (0.6577, tinh theo cong thuc CU) khong con so sanh dung nghia
  voi diem MOI nua. Script nay RESET "best" va "no_improve" trong checkpoint
  hien tai de lan chay tiep theo duoc cham theo dung 1 cong thuc, cong bang.
  (Trong so G/F/DX/DY/optimizer/scheduler - phan model dang hoc - GIU NGUYEN,
  khong mat gi ca, chi reset phan "so sach" theo diem so.)
==============================================================================
"""
import shutil
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    sys.exit("Thieu torch - hay activate dung venv da dung de chay train_dental_mar_V8.py roi chay lai.")

# Sua duong dan nay neu run cua ban khac
CK_PATH = Path(r"E:\Project\Dental\src\train\runs\train_dental_mar_V8_gan_01-09-37_23-09-2026\checkpoints\last.pth")

if not CK_PATH.exists():
    sys.exit(f"[LOI] Khong thay checkpoint: {CK_PATH}")

bak = CK_PATH.with_suffix(".pth.bak_before_score_reset")
shutil.copy2(CK_PATH, bak)
print(f"Da backup -> {bak}")

ck = torch.load(CK_PATH, map_location="cpu", weights_only=False)
print(f"Truoc: epoch={ck['epoch']} best={ck['best']} no_improve={ck['no_improve']}")

ck["best"] = {"score": -1e9, "epoch": ck["epoch"]}   # reset moc "tot nhat" - se duoc thiet lap lai
                                                       # tu epoch dau tien chay duoi cong thuc moi
ck["no_improve"] = 0                                  # reset bo dem - cho 1 "cua so" patience moi, cong bang

torch.save(ck, CK_PATH)
print(f"Sau:   epoch={ck['epoch']} best={ck['best']} no_improve={ck['no_improve']}")
print("\nXONG. Chay lai: python train_dental_mar_V8.py (dung ban file MOI da sua diem so)")
print("Epoch tiep theo chay xong se tu dong thanh *BEST* dau tien duoi cong thuc moi.")
