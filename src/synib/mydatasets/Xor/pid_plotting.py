import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe

SQRT3_2 = np.sqrt(3) / 2
EPS = 1e-12

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def ternary_xy(a, b, c):
    x = b + 0.5 * c
    y = SQRT3_2 * c
    return x, y

def draw_triangle(ax, left, right, top, grid_levels=(0.2, 0.4, 0.6, 0.8)):
    tri = np.array([[0, 0], [1, 0], [0.5, SQRT3_2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], lw=1.6)

    # subtle grid
    for t in grid_levels:
        x1, y1 = ternary_xy(1 - t, 0, t); x2, y2 = ternary_xy(0, 1 - t, t)
        ax.plot([x1, x2], [y1, y2], lw=0.5, alpha=0.22, color="0.55")
        x1, y1 = ternary_xy(t, 1 - t, 0); x2, y2 = ternary_xy(t, 0, 1 - t)
        ax.plot([x1, x2], [y1, y2], lw=0.5, alpha=0.22, color="0.55")
        x1, y1 = ternary_xy(1 - t, t, 0); x2, y2 = ternary_xy(0, t, 1 - t)
        ax.plot([x1, x2], [y1, y2], lw=0.5, alpha=0.22, color="0.55")

    # vertex labels INSIDE corners (prevents cross-panel collisions)
    outline = [pe.withStroke(linewidth=2, foreground="white")]
    kw = dict(fontweight="semibold", path_effects=outline)

    ax.text(0.02, -0.03, left,  transform=ax.transAxes, ha="left",  va="bottom", **kw)
    ax.text(0.98, -0.03, right, transform=ax.transAxes, ha="right", va="bottom", **kw)
    ax.text(0.55, 0.92, top,   transform=ax.transAxes, ha="left", va="top", **kw)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, SQRT3_2 + 0.08)
    ax.set_axis_off()

def educational_accuracy(U1, U2, Syn, Red, seed=2):
    rng = np.random.default_rng(seed)
    uniq = U1 + U2
    imbalance = np.abs(U1 - U2) / (uniq + EPS)
    red_sweet = np.exp(-((Red - 0.20) ** 2) / (2 * 0.12 ** 2))
    acc = (
        0.55 + 0.32 * Syn + 0.08 * Red + 0.05 * red_sweet
        - 0.06 * (imbalance ** 2)
        + rng.normal(0, 0.02, size=len(U1))
    )
    return np.clip(acc, 0.45, 0.95)

def size_from_keep(keep, smin=7, smax=16):
    return smin + (smax - smin) * np.clip(keep, 0, 1)

# ----------------------------
# Demo data (replace with your real arrays)
# ----------------------------
rng = np.random.default_rng(0)
N = 900
P = rng.dirichlet([1, 1, 1, 1], size=N)
U1, U2, Syn, Red = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
acc = educational_accuracy(U1, U2, Syn, Red, seed=2)  # replace with real accuracy if you have it

# Projections
keep1 = U1 + U2 + Red   # (Unique1, Unique2, Redundancy)
x1, y1 = ternary_xy(U1/(keep1+EPS), U2/(keep1+EPS), Red/(keep1+EPS))

keep2 = U1 + U2 + Syn   # (Unique1, Unique2, Synergy)
x2, y2 = ternary_xy(U1/(keep2+EPS), U2/(keep2+EPS), Syn/(keep2+EPS))

keep3 = U1 + Red + Syn  # (Unique1, Redundancy, Synergy)
x3, y3 = ternary_xy(U1/(keep3+EPS), Red/(keep3+EPS), Syn/(keep3+EPS))

# ----------------------------
# Layout: triangles row + caption row; dedicated colorbar column
# ----------------------------
fig = plt.figure(figsize=(7.4, 3.05))
gs = fig.add_gridspec(
    2, 4,
    height_ratios=[1.0, 0.14],
    width_ratios=[1, 1, 1, 0.055],   # slightly wider bar column
    wspace=0.35, hspace=0.05
)

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])
txtax = fig.add_subplot(gs[1, 0:3])
txtax.axis("off")

# Triangles
draw_triangle(axes[0], "Unique 1", "Unique 2", "Redundancy")
draw_triangle(axes[1], "Unique 1", "Unique 2", "Synergy")
draw_triangle(axes[2], "Unique 1", "Redundancy", "Synergy")

# Panel tags only
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(0.03, 0.97, lab, transform=ax.transAxes,
            ha="left", va="top", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.6", lw=0.6, alpha=0.95))

norm = Normalize(vmin=float(acc.min()), vmax=float(acc.max()))
scatter_kw = dict(cmap="cividis", norm=norm, edgecolors="none", rasterized=True)

sc0 = axes[0].scatter(x1, y1, c=acc, s=size_from_keep(keep1), **scatter_kw)
sc1 = axes[1].scatter(x2, y2, c=acc, s=size_from_keep(keep2), **scatter_kw)
sc2 = axes[2].scatter(x3, y3, c=acc, s=size_from_keep(keep3), **scatter_kw)

# A/B/C letters only (no circles), with strong white halo
examples = {
    "A": (0.50, 0.50, 0.00),
    "B": (1/3, 1/3, 1/3),
    "C": (0.70, 0.10, 0.20),
}
for tag, (a, b, c) in examples.items():
    ex, ey = ternary_xy(a, b, c)
    axes[0].text(
        ex, ey, tag,
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color="black",
        path_effects=[pe.withStroke(linewidth=4.5, foreground="white")],
        zorder=10
    )

# Bottom row text (kept clean; remove if you want ultra-minimal)
# txtax.text(
#     0.0, 0.65,
#     "Examples in (a) are (Unique 1, Unique 2, Redundancy): "
#     "A=(0.50,0.50,0.00),  B=(0.33,0.33,0.33),  C=(0.70,0.10,0.20).",
#     ha="left", va="center", fontsize=8
# )
# txtax.text(
#     0.0, 0.10,
#     "Marker size ∝ sum of shown components (larger = less mass in the omitted component).",
#     ha="left", va="center", fontsize=8
# )

# Create the same path effect used in the figure
outline = [pe.withStroke(linewidth=2, foreground="white")]

# Base text for the caption
# txtax.text(
#     0.0, 0.65,
#     "Examples in (a) are (Unique 1, Unique 2, Redundancy):",
#     ha="left", va="center", fontsize=8
# )

# Coordinates for the specific labels (offset to the right of the base text)
# Adjust the x-coordinates (0.42, 0.58, 0.76) based on your final figure width
label_y = 4.8
label_x = 0.22
txtax.text(label_x+0.0, label_y+1.2, "Examples:", fontweight="bold", fontsize=9, ha="left", va="center")
txtax.text(label_x+0.0, label_y+0.8, "A", fontweight="bold", fontsize=9, path_effects=outline, ha="left", va="center")
txtax.text(label_x+0.02, label_y+0.8, "=(0.5, 0.5, 0.0),", ha="left", va="center", fontsize=8)

txtax.text(label_x+0.0, label_y+0.4, "B", fontweight="bold", fontsize=9, path_effects=outline, ha="left", va="center")
txtax.text(label_x+0.02, label_y+0.4, "=(0.33, 0.33, 0.33),", ha="left", va="center", fontsize=8)

txtax.text(label_x+0.0, label_y, "C", fontweight="bold", fontsize=9, path_effects=outline, ha="left", va="center")
txtax.text(label_x+0.02, label_y, "=(0.7, 0.1, 0.2).", ha="left", va="center", fontsize=8)

# txtax.text(
#     0.0, 0.15,
#     "Marker size ∝ sum of shown components (larger = less mass in the omitted component).",
#     ha="left", va="center", fontsize=8
# )


# Colorbar: dedicated axis => right position + same height as triangles
pos = cax.get_position()
# Shift 'bottom' up and reduce 'height'
new_pos = [pos.x0, pos.y0 + 0.1, pos.width, pos.height * 0.7]
cax.set_position(new_pos)
cb = fig.colorbar(sc2, cax=cax, shrink=0.5)
cb.set_label("Accuracy", rotation=90, labelpad=8, fontweight="semibold")
cb.formatter = FormatStrFormatter("%.2f")
cb.update_ticks()

# plt.savefig("pid_icml_ternary_triplet_final.pdf", bbox_inches="tight", pad_inches=0.01)
plt.savefig("pid_icml_ternary_triplet_final.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
