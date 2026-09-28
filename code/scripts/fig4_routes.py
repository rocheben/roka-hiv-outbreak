"""Figure 4 : voies de transmission (modèle retenu).
A-B : infections VIH et VHC par voie (prédictif a posteriori).
C : cas VIH à la détection selon les scénarios contrefactuels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
import numpy as np
from roka.figstyle import plt, save
from roka.posterior import posterior_predictive, counterfactuals, CF_NAMES

pp = posterior_predictive()
rh, rc = pp["routes_hiv"], pp["routes_hcv"]            # vial, seringue, sexe, index/préexistant
COL = ["#2a9d6f", "#2e6fb7", "#8e6bbf", "#7f7f7f"]
rng = np.random.default_rng(3)


def panel(ax, data, labels, cols, title, ylab):
    for k, (v, c) in enumerate(zip(data, cols)):
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=c, alpha=0.3, lw=0)
        b = ax.boxplot([v], positions=[k], widths=0.5, showfliers=False, patch_artist=True)
        b["boxes"][0].set_facecolor("none")
        b["boxes"][0].set_edgecolor("black")
        for part in ("whiskers", "caps", "medians"):
            for x in b[part]:
                x.set_color("black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylab)


fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw=dict(width_ratios=[3, 3, 4]))
panel(axes[0], [rh[:, 0], rh[:, 1], rh[:, 2]], ["Multidose\nvials", "Syringes shared\nbetween patients",
      "Sexual\ntransmission"], COL[:3], "A. HIV infections by route", "Number of infections")
panel(axes[1], [rc[:, 0], rc[:, 1], rc[:, 3]], ["Multidose\nvials", "Syringes shared\nbetween patients",
      "Infected before\nthe outbreak"], [COL[0], COL[1], COL[3]], "B. HCV infections by route",
      "Number of infections")
cf = counterfactuals()
LAB = ["Fitted\nmodel", "Without sexual\ntransmission", "Without\nmultidose vials",
       "Without syringe\nsharing"]
panel(axes[2], [cf[c][:, :, 10].sum(1) for c in CF_NAMES], LAB,
      ["#555555", COL[2], COL[0], COL[1]], "C. HIV cases at testing", "Number of HIV-infected individuals")
save(fig, "fig4_routes.png")
