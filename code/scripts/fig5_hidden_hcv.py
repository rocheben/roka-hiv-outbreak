"""Figure 5 : infections par le VHC à la date de test (extrapolation du modèle
retenu), en distinguant les infections diagnostiquées (co-infections VIH-VHC,
seuls les individus VIH+ ayant été testés pour le VHC) et non diagnostiquées
(VHC seul)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
import numpy as np
from roka.figstyle import plt, save, AGE
from roka.posterior import posterior_predictive

c = posterior_predictive()["counts"]
C_DX, C_UDX = "#e07b39", "#9aa5b1"
fig, ax = plt.subplots(figsize=(7, 4.4))
for a in range(3):
    for k, (col, v) in enumerate(((C_DX, c[:, a, 8]), (C_UDX, c[:, a, 9]))):
        b = ax.boxplot([v], positions=[a * 3 + k], widths=0.7, showfliers=False, patch_artist=True)
        b["boxes"][0].set_facecolor(col)
        b["medians"][0].set_color("black")
ax.set_xticks([0.5, 3.5, 6.5])
ax.set_xticklabels(AGE)
ax.set_ylabel("HCV-infected individuals at testing")
ax.bar([0], [0], color=C_DX, label="Diagnosed (HIV/HCV coinfected, tested for HCV)")
ax.bar([0], [0], color=C_UDX, label="Not diagnosed (HCV only, not tested)")
ax.legend(fontsize=8, loc="upper left")
save(fig, "fig5_hidden_hcv.png")
