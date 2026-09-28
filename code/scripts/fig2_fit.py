"""Figure 2 : ajustement du modèle retenu (prédictif a posteriori) aux données
de récence, par âge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
import numpy as np
from roka.figstyle import plt, save, AGE
from roka.abc import OBS_HIV, OBS_HCV
from roka.posterior import posterior_predictive

S = posterior_predictive()["stats"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [12, 9]})
lh = [f"{AGE[a]}\n{b}" for a in range(3) for b in ("0-30 d", "30-100 d", "100-130 d", ">130 d")]
lc = [f"{AGE[a]}\n{b}" for a in range(3) for b in ("<60 d", "60-180 d", ">180 d")]
for ax, sl, obs, labs, ttl in ((axes[0], slice(0, 12), OBS_HIV.ravel(), lh, "A. HIV cases by time since infection"),
                               (axes[1], slice(12, 21), OBS_HCV.ravel(), lc, "B. HIV/HCV coinfections by time since HCV infection")):
    ax.boxplot(S[:, sl], showfliers=False, widths=0.6)
    ax.scatter(np.arange(1, len(obs) + 1), obs, color="#d62728", zorder=3, s=25, label="Observed")
    ax.set_xticks(np.arange(1, len(obs) + 1))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_title(ttl, loc="left")
    ax.set_ylabel("Number of cases with known recency")
axes[0].legend(fontsize=8)
save(fig, "fig2_fit.png")
