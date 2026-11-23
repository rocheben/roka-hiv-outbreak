"""Figure 2 du manuscrit. Scénarios d'introduction et de transmission comparés (A-D).
Sortie : figures/fig2_scenarios.png (+ PDF et TIFF dans figures/eid/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
from roka.figstyle import plt, save

C_IDX, C_INF, C_SUS = "#c0392b", "#f0a868", "#ffffff"
C_SYR, C_VIAL = "#2e6fb7", "#2a9d6f"


def person(ax, x, y, col, r=0.28, lw=1.2):
    ax.add_patch(Circle((x, y), r, fc=col, ec="black", lw=lw, zorder=3))


def syringe(ax, x, y, col=C_SYR, contaminated=True):
    ax.add_patch(FancyBboxPatch((x - 0.35, y - 0.12), 0.7, 0.24, boxstyle="round,pad=0.02",
                                fc=col if contaminated else "white", ec=col, lw=1.3, zorder=3))
    ax.plot([x + 0.35, x + 0.6], [y, y], color=col, lw=1.3, zorder=3)


def vial(ax, x, y, contaminated=True):
    ax.add_patch(FancyBboxPatch((x - 0.22, y - 0.35), 0.44, 0.6, boxstyle="round,pad=0.03",
                                fc=C_VIAL if contaminated else "white", ec=C_VIAL, lw=1.3, zorder=3))
    ax.add_patch(Rectangle((x - 0.12, y + 0.25), 0.24, 0.14, fc="#444444", ec="none", zorder=3))


def arrow(ax, p, q, col="black", lw=1.2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=col, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2,
                                 shrinkA=9, shrinkB=9))


def frame(ax, title, sub):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.text(0.1, 0.25, sub, fontsize=8.2, va="bottom", ha="left", wrap=True)


fig = plt.figure(figsize=(12, 6.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.75])
A, B = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
Cc, D = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
F = fig.add_subplot(gs[:, 2])

# ---------------------------------------------------------------- A cas index
frame(A, "A. Index case", "One chronically infected patient, ordinary injection\n"
      "frequency. Transmission by syringes reused\nbetween patients (chains).")
person(A, 1.2, 4.5, C_IDX)
syringe(A, 3.3, 4.5)
person(A, 5.4, 5.6, C_INF)
person(A, 5.4, 3.4, C_SUS)
syringe(A, 7.3, 5.6)
person(A, 9.2, 5.6, C_INF)
person(A, 9.2, 4.2, C_SUS)
arrow(A, (1.2, 4.5), (3.0, 4.5))
arrow(A, (3.8, 4.5), (5.4, 5.6))
arrow(A, (3.8, 4.5), (5.4, 3.4), ls=":")
arrow(A, (5.4, 5.6), (7.0, 5.6))
arrow(A, (7.8, 5.6), (9.2, 5.6))
arrow(A, (7.8, 5.6), (9.2, 4.2), ls=":")

# ---------------------------------------------------------------- B patient fréquent
frame(B, "B. Frequent-patient index case", "The index patient receives injections much more\n"
      "often than other patients and repeatedly\nrecontaminates the syringes of the practitioner.")
person(B, 1.2, 4.5, C_IDX, r=0.36, lw=2)
for k, y in enumerate((5.8, 4.5, 3.2)):
    syringe(B, 3.6, y)
    arrow(B, (1.2, 4.5), (3.25, y), lw=2.2, col=C_IDX)
for y in (6.1, 5.2, 4.5, 3.8, 2.9):
    person(B, 6.3, y, C_INF if y != 3.8 else C_SUS, r=0.24)
for y0 in (5.8, 4.5, 3.2):
    arrow(B, (4.1, y0), (6.3, y0 + 0.3))
    arrow(B, (4.1, y0), (6.3, y0 - 0.4))
syringe(B, 8.2, 5.2)
arrow(B, (6.3, 5.2), (7.9, 5.2))
person(B, 9.5, 5.2, C_INF, r=0.24)
arrow(B, (8.7, 5.2), (9.5, 5.2))

# ---------------------------------------------------------------- C flacons
frame(Cc, "C. Multidose vials", "A syringe reused after an infected patient\n"
      "contaminates a vial, which infects the following\npatients until it is replaced.")
person(Cc, 1.0, 4.5, C_IDX)
syringe(Cc, 2.8, 4.5)
vial(Cc, 4.8, 4.5)
arrow(Cc, (1.0, 4.5), (2.5, 4.5))
arrow(Cc, (3.3, 4.5), (4.6, 4.5))
for y in (6.0, 5.0, 4.0, 3.0):
    person(Cc, 7.0, y, C_INF if y != 4.0 else C_SUS, r=0.24)
    arrow(Cc, (5.0, 4.5), (7.0, y))
vial(Cc, 9.1, 4.5, contaminated=False)
arrow(Cc, (5.3, 2.4), (8.8, 3.4), ls="--", col=C_VIAL, rad=0.25)
Cc.text(7.1, 1.95, "replacement", fontsize=7.5, color=C_VIAL, ha="center")

# ---------------------------------------------------------------- D combinaison
frame(D, "D. Frequent patient and multidose vials", "Combination of B and C: the frequent index\n"
      "patient recontaminates syringes and vials.")
person(D, 1.1, 4.5, C_IDX, r=0.36, lw=2)
syringe(D, 3.1, 5.6)
vial(D, 3.2, 3.3)
arrow(D, (1.1, 4.5), (2.8, 5.6), lw=2.2, col=C_IDX)
arrow(D, (1.1, 4.5), (3.0, 3.4), lw=2.2, col=C_IDX)
for y in (6.2, 5.3, 4.4, 3.5, 2.6):
    person(D, 6.0, y, C_INF if y != 4.4 else C_SUS, r=0.24)
arrow(D, (3.6, 5.6), (6.0, 6.2))
arrow(D, (3.6, 5.6), (6.0, 5.3))
for y in (4.4, 3.5, 2.6):
    arrow(D, (3.4, 3.3), (6.0, y))
syringe(D, 8.0, 3.5)
arrow(D, (6.0, 3.5), (7.7, 3.5))
person(D, 9.5, 3.5, C_INF, r=0.24)
arrow(D, (8.5, 3.5), (9.5, 3.5))

# ---------------------------------------------------------------- F éléments communs
F.set_xlim(0, 10)
F.set_ylim(0, 14)
F.set_aspect("equal")
F.axis("off")
F.set_title("Legend and common elements", loc="left", fontsize=10.5, fontweight="bold")
items = [
    (lambda y: person(F, 0.7, y, C_IDX), "HIV index case (chronic infection)"),
    (lambda y: person(F, 0.7, y, C_INF), "Infected individual"),
    (lambda y: person(F, 0.7, y, C_SUS), "Susceptible individual"),
    (lambda y: syringe(F, 0.7, y), "Contaminated syringe"),
    (lambda y: vial(F, 0.7, y), "Contaminated multidose vial"),
]
for i, (draw, lab) in enumerate(items):
    y = 11.5 - 1.3 * i
    draw(y)
    F.text(1.6, y, lab, va="center", fontsize=8.5)
F.text(0.2, 4.6, "In every scenario: sexual transmission of HIV\n"
       "among 15–64 year-olds, recency assessed at the test date.\n"
       "In B, C and D: HCV already present in part of the\n"
       "population, estimated date at which the practitioner\n"
       "stopped injecting.",
       fontsize=8.2, va="top")
save(fig, "fig2_scenarios.png")
