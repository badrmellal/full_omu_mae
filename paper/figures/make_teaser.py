"""Generate the Figure-1 teaser: 2D DINOv2 patches -> camera-LiDAR projection
-> 3D per-voxel feature volume (85% masked) -> reconstruct. Rendered to PNG so
it always displays (no LaTeX/TikZ compile dependency)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

np.random.seed(3)
turbo = mpl.colormaps["turbo"]

fig = plt.figure(figsize=(13.0, 4.4), dpi=200)
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------- (a) 2D image
axI = fig.add_axes([0.015, 0.20, 0.215, 0.60])
F = np.random.rand(16, 16)
axI.imshow(F, cmap="turbo", interpolation="bilinear", extent=[0, 16, 0, 16])
for k in range(17):
    axI.axhline(k, color="white", lw=0.4, alpha=0.45)
    axI.axvline(k, color="white", lw=0.4, alpha=0.45)
for (px, py, c) in [(6, 10, "k"), (7, 10, "k"), (6, 9, "k")]:
    axI.add_patch(Rectangle((px, py), 1, 1, fill=False, edgecolor=c, lw=1.8))
axI.set_xlim(0, 16); axI.set_ylim(0, 16)
axI.set_xticks([]); axI.set_yticks([])
axI.set_title("(a) camera image $\\rightarrow$\nfrozen DINOv2 patches\n$16{\\times}16{\\times}768$",
              fontsize=9.5, linespacing=1.25)

# ---------------------------------------------------------------- (b) 3D voxels
axV = fig.add_axes([0.34, 0.0, 0.40, 1.0], projection="3d")
nx, ny, nz = 12, 12, 5
filled = np.zeros((nx, ny, nz), bool)
filled[:, :, 0] = True               # ground plane
filled[2:5, 7:10, 0:3] = True        # building
filled[8:10, 2:4, 0:2] = True        # vehicle
filled[5:7, 4:6, 0:2] = True         # vegetation clump

fx, fy, fz = np.indices((nx, ny, nz))
feat = np.sin(fx * 0.6) + np.cos(fy * 0.55) + fz * 0.35
feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-9)

fc = turbo(feat)
visible = (np.random.rand(nx, ny, nz) < 0.15) & filled   # ~15% visible, 85% masked
fc[~visible] = (0.82, 0.82, 0.84, 0.32)                  # masked -> faint grey
fc[visible, 3] = 0.97                                    # visible -> solid feature colour

axV.voxels(filled, facecolors=fc, edgecolor=(0.35, 0.35, 0.35, 0.25), linewidth=0.3)
axV.view_init(elev=24, azim=-60)
axV.set_axis_off()
try:
    axV.set_box_aspect((nx, ny, nz * 1.7))
except Exception:
    pass
axV.set_title("(b) per-voxel DINOv2 feature volume   $128{\\times}128{\\times}32$\n"
              "coloured = visible · grey = 85% masked",
              fontsize=9.5, y=0.93, linespacing=1.25)

# ---------------------------------------------------------------- arrows/labels
def farrow(x0, y0, x1, y1, color="#222", lw=2.2, ms=20):
    a = FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                        shrinkA=0, shrinkB=0)
    fig.add_artist(a)

# projection 2D -> 3D
farrow(0.245, 0.50, 0.345, 0.50, color="#1f4e9b")
fig.text(0.296, 0.60, "camera--LiDAR\nprojection $(\\mathbf{P}_2,\\,\\mathbf{Tr})$",
         ha="center", va="center", fontsize=8.8, color="#1f4e9b", linespacing=1.2)

# 3D -> reconstruct
farrow(0.735, 0.50, 0.805, 0.50, color="#222")
fig.text(0.905, 0.50,
         "(c) range-aware mask\n$\\rho=0.85$  $+$  3D CNN\n$\\Rightarrow$ reconstruct\nocc. $+$ DINOv2 feats\nat masked voxels",
         ha="center", va="center", fontsize=9.0, linespacing=1.35,
         bbox=dict(boxstyle="round,pad=0.5", fc="#efeaf7", ec="#7a5ea8", lw=1.2))

fig.savefig("teaser.png", bbox_inches="tight", facecolor="white", pad_inches=0.06)
print("wrote teaser.png")
