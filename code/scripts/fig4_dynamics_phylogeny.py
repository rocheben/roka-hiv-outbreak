"""Figure 4 du manuscrit. Dynamique simulée du modèle retenu et comparaison avec les
analyses phylogénétiques publiées (tMRCA, pic LTT).
Sortie : figures/fig4_dynamics_phylogeny.png (+ PDF et TIFF dans figures/eid/)."""
import sys
import datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
import numpy as np
from roka.figstyle import plt, save, C_HIV, C_HCV
from roka.posterior import posterior_predictive, posterior
from roka.dates import T_LAST, ym, rouet
from roka import abc as A

pp, R = posterior_predictive(), rouet()
m, X, _, _ = posterior()
stop = T_LAST - X[:, A.EST_BY_MODEL[m].index("stop_d")] / 365.25
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
xs = T_LAST - pp["grid"] / 365.25
for tr, col, lab in ((pp["traj_hiv"], C_HIV, "HIV"), (pp["traj_hcv"], C_HCV, "HCV (new infections)")):
    ax[0].fill_between(xs, np.percentile(tr, 2.5, 0), np.percentile(tr, 97.5, 0), color=col, alpha=0.2)
    ax[0].plot(xs, np.median(tr, 0), color=col, lw=2, label=f"{lab}, cumulative (model)")
ax[0].axvspan(np.percentile(stop, 2.5), np.percentile(stop, 97.5), color="grey", alpha=0.18,
              label="End of injections (model, 95% CI)")
top = np.percentile(pp["traj_hcv"], 97.5, 0).max()
for key, ls, lab in (("hiv_ltt_rise", ":", "LTT rise"), ("hiv_ltt_peak", "--", "LTT peak")):
    x = ym(R[key])
    ax[0].axvline(x, color="k", ls=ls, lw=1)
    ax[0].text(x, top * (0.95 if ls == ":" else 0.8), f" {lab}\n (Rouet et al.)", fontsize=8)
ax[0].set_xlim(T_LAST - 2.0, T_LAST + 0.02)
ax[0].set_xticks([2013.0, 2013.5, 2014.0, 2014.5, 2015.0])
ax[0].set_xticklabels(["Jan 2013", "Jul 2013", "Jan 2014", "Jul 2014", "Jan 2015"])
ax[0].set_ylabel("Individuals")
ax[0].legend(fontsize=7.5, loc="upper left")
ax[0].set_title("A. Simulated outbreak", loc="left")
ph = T_LAST - pp["phylo"] / 365.25
ok = np.isfinite(ph).all(1)
ax[1].violinplot([ph[ok, 0]], positions=[0], showmedians=True)
ax[1].violinplot([ph[ok, 1]], positions=[2.3], showmedians=True)
m_, lo, hi = ym(R["hiv_tmrca"]["mean"]), ym(R["hiv_tmrca"]["hpd_low"]), ym(R["hiv_tmrca"]["hpd_high"])
ax[1].errorbar([1], [m_], yerr=[[m_ - lo], [hi - m_]], fmt="o", color="k", capsize=4)
ax[1].plot([3.3], [ym(R["hiv_ltt_peak"])], "o", color="k")
ax[1].set_xticks([0, 1, 2.3, 3.3])
ax[1].set_xticklabels(["tMRCA\nmodel", "tMRCA\nRouet et al.", "Peak of\nbranching\nmodel", "LTT peak\nRouet et al."])
ax[1].set_ylabel("Calendar year")
ax[1].set_title("B. HIV genealogy: model vs BEAST", loc="left")
save(fig, "fig4_dynamics_phylogeny.png")
