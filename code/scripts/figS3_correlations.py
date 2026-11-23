"""Appendix Figure 3. Corrélations a posteriori entre paramètres du modèle retenu
(échelle log pour les paramètres à prior log-uniforme).
Sortie : figures/figS3_correlations.png."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
import numpy as np
from roka.figstyle import plt, save, PARAM_SYMBOLS
from roka.posterior import posterior
from roka.paths import CACHE
from roka import abc as A

m, X, _, _ = posterior()
est = A.EST_BY_MODEL[m]
Y = A.to_y(m, X)
R = np.corrcoef(Y.T)
np.save(CACHE / "posterior_correlations.npy", R)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(est)))
ax.set_yticks(range(len(est)))
lab = [PARAM_SYMBOLS.get(k, k) for k in est]
ax.set_xticklabels(lab, rotation=90, fontsize=9)
ax.set_yticklabels(lab, fontsize=9)
fig.colorbar(im, ax=ax, label="Pearson correlation")
save(fig, "figS3_correlations.png")
