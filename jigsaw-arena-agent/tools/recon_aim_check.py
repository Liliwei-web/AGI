import glob, os
import numpy as np
from PIL import Image

REC = os.environ.get("RECON_DIR", "logs/recon")   # 侦察产物目录（原始实验位于赛场工作区）

def arr(p, w=1000):
    im = np.asarray(Image.open(p).convert("L")).astype(np.float32)
    return im[:, :w]

groups = {
    "arr_right_cell (aimed at 3 different slots)": sorted(glob.glob(os.path.join(REC, "probe_c66f5f1f26_arr_right_cell_*.jpg"))),
    "calib 5 same-station shots": sorted(glob.glob(os.path.join(REC, "probe_f5e5000736_calib_*.jpg")))[:5],
    "aim probe shots (20260909)": [],
}
for name, fs in groups.items():
    if len(fs) < 2:
        continue
    print("=" * 20, name, len(fs))
    for f in fs:
        print("   ", os.path.basename(f))
    M = [arr(f) for f in fs]
    for i in range(len(M)):
        print("   " + " ".join("%6.2f" % np.abs(M[i] - M[j]).mean() for j in range(len(M))))

# also: same image content but different file? check the two pframe poster shots
