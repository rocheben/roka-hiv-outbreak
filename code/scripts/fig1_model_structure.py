"""Figure 1 du manuscrit. Structure du modèle individu-centré : états des individus,
matériel du praticien, chronologie et statistiques résumées utilisées pour l'ABC.
Sortie : figures/fig1_model_structure.png (+ PDF et TIFF dans figures/eid/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from roka.figstyle import plt, save

C_S, C_HIV, C_HCV, C_CO = "#ffffff", "#f4a261", "#8ecae6", "#b5a1d6"
C_SYR, C_VIAL, C_GREY = "#2e6fb7", "#2a9d6f", "#6b6b6b"

fig, ax = plt.subplots(figsize=(14, 8.2))
ax.set_xlim(0, 28)
ax.set_ylim(0, 16.4)
ax.axis("off")


def box(x, y, w, h, txt, fc, ec="black", fs=10, bold=False, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arr(p, q, lab="", col="black", rad=0.0, lw=1.3, ls="-", off=(0, 0), fs=9, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=col, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3, zorder=1))
    if lab:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], lab, fontsize=fs, color=col,
                ha=ha, va="center", zorder=3)


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.3",
                                fc="#f7f7f7", ec="#bbbbbb", lw=1, zorder=0))
    ax.text(x + 0.3, y + h - 0.45, title, fontsize=11.5, fontweight="bold", va="center")


# ------------------------------------------------------------------ A. individus
panel(0.3, 5.2, 12.4, 10.9, "A. Villagers (3 age classes: <15, 15–64, ≥65 years)")
box(0.9, 11.2, 2.6, 1.3, "Susceptible", C_S, fs=10.5)
box(5.3, 13.1, 2.9, 1.3, "HIV acute", C_HIV)
box(9.3, 13.1, 2.9, 1.3, "HIV chronic", C_HIV)
box(5.3, 9.3, 2.9, 1.3, "HCV only", C_HCV)
box(5.3, 6.2, 2.9, 1.3, "HIV acute\n+ HCV", C_CO)
box(9.3, 6.2, 2.9, 1.3, "HIV chronic\n+ HCV", C_CO)
arr((3.5, 12.2), (5.3, 13.6), "HIV: β$_{HIV}$\n(injection)\nor sexual", off=(-0.35, 0.55), fs=8.5)
arr((3.5, 11.5), (5.3, 10.0), "HCV: β$_{HCV}$", off=(-0.2, -0.45), fs=8.5)
arr((8.2, 13.75), (9.3, 13.75), "θ", off=(0, 0.3))
arr((8.2, 6.85), (9.3, 6.85), "θ", off=(0, 0.3))
arr((6.75, 13.1), (6.75, 10.6), "β$_{HCV}$", off=(0.55, 0.25), fs=8.5)
arr((10.75, 13.1), (10.75, 7.5), "β$_{HCV}$", off=(0.55, 1.2), fs=8.5)
arr((6.75, 9.3), (6.75, 7.5), "β$_{HIV}$ or\nsexual", off=(0.85, 0), fs=8.5)
ax.text(0.8, 9.9, "Injections at rate γ$_1$, γ$_2$, γ$_3$\n(by age class)", fontsize=9, va="top")
ax.text(0.8, 8.3, "Sexual transmission of HIV\namong 15–64 year-olds:\nβ$_{SI}$ (acute), β$_{SU}$ (chronic)", fontsize=9, va="top")
box(0.9, 5.7, 3.4, 1.5, "Index patient\n(chronic HIV, injected\nat rate γ$_{index}$)", C_HIV, lw=2.2, fs=9)
ax.text(4.6, 5.55, "θ = 1/30 day$^{-1}$ (end of acute phase). No recovery, no demography.", fontsize=8.5, color=C_GREY)

# ------------------------------------------------------------------ B. matériel
panel(15.3, 5.2, 12.4, 10.9, "B. Material of the practitioner")
ax.text(15.8, 14.8, "Syringe pool (N$_{syr}$)", fontsize=10.5, fontweight="bold", color=C_SYR)
for i, (lab, fc) in enumerate((("clean", "white"), ("HIV", C_HIV), ("HCV", C_HCV), ("HIV + HCV", C_CO))):
    box(15.9 + i * 2.9, 12.9, 2.5, 1.1, lab, fc, ec=C_SYR, lw=1.8, fs=9.5)
arr((20.1, 12.85), (16.9, 12.85), "loss of virus: σ$_{HIV}$, σ$_{HCV}$", col=C_SYR, rad=-0.35, off=(0, -1.05), fs=9)
ax.text(16.4, 9.95, "Multidose vials (N$_{vial}$), used with probability π", fontsize=10.5, fontweight="bold",
        color=C_VIAL)
for i, (lab, fc) in enumerate((("clean", "white"), ("HIV", C_HIV), ("HCV", C_HCV), ("HIV + HCV", C_CO))):
    box(15.9 + i * 2.9, 8.1, 2.5, 1.1, lab, fc, ec=C_VIAL, lw=1.8, fs=9.5)
arr((20.1, 8.05), (16.9, 8.05), "replacement after mean duration L$_{vial}$", col=C_VIAL, rad=-0.35,
    off=(0, -1.05), fs=9)
ax.text(15.8, 5.75, "Injections stop at t$_{stop}$ (estimated). Sexual transmission continues.",
        fontsize=8.5, color=C_GREY)

# ------------------------------------------------------------------ injection : échanges
box(12.95, 9.9, 2.1, 2.9, "Injection", "#fff3c4", fs=10.5, bold=True)
ax.add_patch(FancyArrowPatch((12.95, 11.35), (12.25, 11.35), arrowstyle="<|-|>", mutation_scale=12, lw=1.3, zorder=1))
ax.text(12.6, 11.75, "any\nvillager", ha="center", va="bottom", fontsize=8.5)
arr((15.05, 12.2), (15.9, 12.9), "1 − π", col=C_SYR, off=(0.2, -0.35), fs=9)
arr((15.05, 10.4), (15.9, 9.2), "π", col=C_VIAL, off=(0.45, 0.1), fs=10)
ax.text(14.0, 9.45, "material → patient: β$_{HIV}$, β$_{HCV}$\npatient → material:\nω$_{HIV,I}$, ω$_{HIV,U}$, ω$_{HCV}$",
        fontsize=8.8, ha="center", va="top")

# ------------------------------------------------------------------ C. chronologie et observations
panel(0.3, 0.2, 27.4, 4.6, "C. Timeline, observations and inference")
y = 2.55
ax.plot([1.2, 18.6], [y, y], color="black", lw=1.5)

# evenements documentes (au-dessus de l'axe), dans l'ordre chronologique
for x, dy, top in ((1.6, 0.32, "Introduction of the\nindex patient"),
                   (9.6, 1.15, "First cases diagnosed\n(early December 2014)"),
                   (12.9, 0.32, "Arrest of the practitioner\n(end of December 2014)"),
                   (16.2, 1.15, "Testing of the exposed population\n(Dec 2014\u2013Feb 2015)")):
    ax.plot([x, x], [y - 0.18, y + 0.18], color="black", lw=1.5)
    if dy > 0.5:
        ax.plot([x, x], [y + 0.18, y + dy - 0.03], color=C_GREY, lw=0.7)
    ax.text(x, y + dy, top, ha="center", va="bottom", fontsize=8.8)

# quantites estimees par le modele (en dessous de l'axe)
ax.plot([1.6, 1.6], [y - 0.55, y - 0.18], color=C_GREY, lw=0.7, ls=":")
ax.text(1.6, y - 0.6, "t$_{start}$, estimated\nHCV already present\nin a fraction p$_{HCV,0}$",
        ha="center", va="top", fontsize=8.5, color=C_GREY)
ax.plot([12.9, 12.9], [y - 0.92, y - 0.18], color=C_GREY, lw=0.7, ls=":", zorder=3)
ax.add_patch(FancyBboxPatch((8.3, y - 0.92), 5.1, 0.32,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=C_VIAL, ec="none", alpha=0.3, zorder=1))
ax.plot([9.4, 9.4], [y - 0.92, y - 0.60], color=C_VIAL, lw=2.2, zorder=2)
ax.text(10.85, y - 1.02, "Stop of injections (t$_{stop}$, estimated)\n"
        "median 16 Nov 2014, 95% CI 1 Nov\u201330 Dec 2014",
        ha="center", va="top", fontsize=8.5, color=C_VIAL)
ax.text(16.2, y - 0.3, "Recency classes,\nsampling of 199 HIV lineages",
        ha="center", va="top", fontsize=8.5, color=C_GREY)

box(19.4, 0.8, 3.6, 2.9, "Summary statistics\n\u2022 21 recency counts\n  by age class\n\u2022 tMRCA\n\u2022 LTT peak", "white",
    fs=8.8)
box(23.8, 1.25, 3.5, 2.0, "ABC-SMC\nparameters and\nmodel evidence", "#fff3c4", fs=9.5, bold=True)
arr((18.6, y), (19.4, y))
arr((23.0, y), (23.8, y))
ax.text(9.9, 0.45, "Every infection and contamination creates a viral lineage: the genealogy is recorded "
        "during the simulation.", ha="center", fontsize=8.5, style="italic")
save(fig, "fig1_model_structure.png")
