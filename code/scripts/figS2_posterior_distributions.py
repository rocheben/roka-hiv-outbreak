"""Figure S2 : distributions a priori et a posteriori des paramètres du modèle retenu."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
import numpy as np
from roka.figstyle import plt, save, PARAM_LABELS
from roka.posterior import posterior
from roka import abc as A

LAB = PARAM_LABELS
m, X, W, _ = posterior()
est = A.EST_BY_MODEL[m]
lo, hi = A.bounds(m)
fig, axes = plt.subplots(4, 5, figsize=(14, 10))
for i, ax in enumerate(axes.ravel()):
    if i >= len(est):
        ax.axis("off")
        continue
    k = est[i]
    v, a, b, lab = X[:, i], lo[i], hi[i], LAB[k]
    if k in A.LOG_PRIOR:
        v, a, b, lab = np.log10(v), np.log10(a), np.log10(b), "log$_{10}$ " + lab
    ax.hist(v, bins=np.linspace(a, b, 30), density=True, color="#1f77b4", alpha=0.7, label="Posterior")
    ax.plot([a, a, b, b], [0, 1 / (b - a), 1 / (b - a), 0], color="k", lw=1, label="Prior (unconstrained)")
    ax.set_title(lab)
    ax.set_yticks([])
axes[0, 0].legend(fontsize=7)
save(fig, "figS2_posterior_distributions.png")
