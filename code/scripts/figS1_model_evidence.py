"""Appendix Figure 1. Évidence ABC de chaque modèle en fonction de la tolérance.
Sortie : figures/figS1_model_evidence.png."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
from roka.figstyle import plt, save
from roka.posterior import evidence_table, REALISTIC, SENSITIVITY, SELECTED
from roka import abc as A

E = evidence_table()
fig, ax = plt.subplots(figsize=(8, 4.8))
cm = plt.get_cmap("tab10")
SENS_COL = {k: c for k, c in zip(SENSITIVITY, ("#333333", "#999999"))}
for i, k in enumerate(REALISTIC + SENSITIVITY):
    if k not in E:
        continue
    z = E[k]["Z"]
    eps = sorted(z, reverse=True)
    sens = k in SENSITIVITY
    ax.semilogy(eps, [z[e] for e in eps], ":" if sens else "-", marker="o", ms=3,
                color=SENS_COL[k] if sens else cm(i % 10),
                lw=2.5 if k == A.MODEL_KEYS[SELECTED] else 1.2,
                label=E[k]["label"] + (" (sensitivity)" if sens else ""))
ax.invert_xaxis()
ax.set_xlabel("ABC tolerance")
ax.set_ylabel("ABC evidence P(distance ≤ tolerance | model)")
ax.legend(fontsize=7, loc="lower left")
save(fig, "figS1_model_evidence.png")
