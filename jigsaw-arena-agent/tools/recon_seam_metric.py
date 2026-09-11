import glob, os, re
import numpy as np
from PIL import Image

REC = os.environ.get("RECON_DIR", "logs/recon")   # 侦察产物目录（原始实验位于赛场工作区）
pat = re.compile(r"probe_[0-9a-f]+_seam_(a|b)_(\d+)_(\d+)__(\d+)_(\d+)_([a-z])_\d{8}_\d+\.jpg$")
pairs = {}
for f in glob.glob(os.path.join(REC, "probe_fe7bedf7b2_seam_*.jpg")):
    m = pat.search(os.path.basename(f))
    if not m: continue
    key = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6))
    pairs.setdefault(key, {})[m.group(1)] = f

def load(path, pane_w=1000):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)[:, :pane_w]

def dprofile(im, axis, off=4, half=200):
    """axis=z -> vertical neighbours -> horizontal seam -> scan rows (d over y).
       axis=y -> horizontal neighbours -> vertical seam -> scan columns (d over x)."""
    h, w, _ = im.shape
    cy, cx = h // 2, w // 2
    sub = im[cy - half:cy + half, cx - half:cx + half]
    if axis == "z":
        d = np.abs(sub[:-2*off, :] - sub[2*off:, :]).mean(axis=(1, 2))
    else:
        d = np.abs(sub[:, :-2*off] - sub[:, 2*off:]).mean(axis=(0, 2))
    return d, sub.shape[0] // 2 if axis == "z" else sub.shape[1] // 2

print("%-28s %-6s %-24s %-24s" % ("seam", "axis", "A  peak/med (off)", "B  peak/med (off)"))
rows = []
for key in sorted(pairs):
    p = pairs[key]
    if "a" not in p or "b" not in p: continue
    res = {}
    for arr in ("a", "b"):
        im = load(p[arr]); d, c = dprofile(im, key[4])
        win = d[c-45:c+45]; med = float(np.median(d))
        i = int(win.argmax()); peak = float(win[i])
        res[arr] = (peak / med, i - 45, peak, med)
    rows.append((key, res))
    print("%-28s %-6s A: %6.2f (%4d) peak=%5.1f  B: %6.2f (%4d) peak=%5.1f" % (
        "(%d,%d)->(%d,%d)" % key[:4], key[4], res["a"][0], res["a"][1], res["a"][2], res["b"][0], res["b"][1], res["b"][2]))

sep = [r for r in rows]
print()
print("separation (B/A) per seam:")
for key, res in sep:
    print("  (%d,%d)->(%d,%d) %s : %.2f" % (key[0], key[1], key[2], key[3], key[4], res["b"][0] / res["a"][0]))

