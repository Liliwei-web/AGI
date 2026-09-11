import glob, os, re
import numpy as np
from PIL import Image

REC = os.environ.get("RECON_DIR", "logs/recon")   # 侦察产物目录（原始实验位于赛场工作区）
fs = sorted(glob.glob(os.path.join(REC, "probe_c518498f40_turn_*.jpg")))
print("frames:", len(fs))
labels = []
for f in fs:
    m = re.search(r"turn_(\d+)_(-?\d+)_", os.path.basename(f))
    labels.append((int(m.group(1)), int(m.group(2))))
M = [np.asarray(Image.open(f).convert("L")).astype(np.float32)[:, :1000] for f in fs]
print("idx turn  luma")
for i, f in enumerate(fs):
    print("  %2d %4d  %6.1f   %s" % (labels[i][0], labels[i][1], M[i].mean(), os.path.basename(f)[-24:]))

print()
print("pairwise mean|diff| (rows/cols = frame idx):")
hdr = "      " + " ".join("%5d" % i for i in range(len(M)))
print(hdr)
for i in range(len(M)):
    print("  %3d " % i + " ".join("%5.1f" % np.abs(M[i] - M[j]).mean() for j in range(len(M))))

print()
print("expected repeats: 0<->6, 0<->7, 1<->8, 2<->9, 3<->10, 4<->11, 5<->12(n/a)")
for a, b in [(0, 6), (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 11), (6, 7)]:
    if b < len(M):
        print("   %2d vs %2d : %6.2f" % (a, b, np.abs(M[a] - M[b]).mean()))
