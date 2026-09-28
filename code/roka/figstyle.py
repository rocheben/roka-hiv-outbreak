"""Style commun des figures (consignes EID : police Arial, figures en fichiers séparés).
Chaque figure est écrite en PNG (aperçu, Appendix) et, dans figures/eid/, en PDF vectoriel
et en TIFF 600 dpi pour la soumission."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Arial ; Liberation Sans (métriques identiques) si Arial n'est pas installée
_sans = [f for f in ("Arial", "Liberation Sans", "DejaVu Sans")
         if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": _sans or ["sans-serif"],
                     "font.size": 10, "axes.titlesize": 10, "savefig.dpi": 300,
                     "pdf.fonttype": 42})
AGE = ["<15 years", "15–64 years", "≥65 years"]
C_HIV, C_HCV = "#1f77b4", "#e07b39"


def save(fig, name):
    from .paths import FIGURES
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=300)
    if not name.startswith("figS"):
        eid = FIGURES / "eid"
        eid.mkdir(exist_ok=True)
        stem = name.rsplit(".", 1)[0]
        fig.savefig(eid / f"{stem}.pdf")
        fig.savefig(eid / f"{stem}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("écrit :", FIGURES / name)


# Symboles des paramètres (identiques aux notations du texte et des tableaux)
PARAM_SYMBOLS = {"beta_H": "$\\beta_{HIV}$", "beta_C": "$\\beta_{HCV}$", "sigma_H": "$\\sigma_{HIV}$",
                 "sigma_C": "$\\sigma_{HCV}$", "omega_A": "$\\omega_{HIV,I}$", "omega_U": "$\\omega_{HIV,U}$",
                 "omega_C": "$\\omega_{HCV}$", "g1": "$\\gamma_1$", "g2": "$\\gamma_2$", "g3": "$\\gamma_3$",
                 "t_max": "$t_{start}$", "b_SI": "$\\beta_{SI}$", "b_SU": "$\\beta_{SU}$", "n_syr": "$N_{syr}$",
                 "gam_idx": "$\\gamma_{index}$", "pi": "$\\pi$", "n_vial": "$N_{vial}$", "vial_life": "$L_{vial}$",
                 "p_C0": "$p_{HCV,0}$", "stop_d": "$t_{stop}$", "tau_C": "$\\tau_{HCV}$"}
_UNITS = {"sigma_H": " (/day)", "sigma_C": " (/day)", "g1": " (/day)", "g2": " (/day)", "g3": " (/day)",
          "b_SI": " (/day/pair)", "b_SU": " (/day/pair)", "gam_idx": " (/day)", "vial_life": " (days)",
          "t_max": " (days before last sequence)", "stop_d": " (days before last sequence)"}
PARAM_LABELS = {k: v + _UNITS.get(k, "") for k, v in PARAM_SYMBOLS.items()}
