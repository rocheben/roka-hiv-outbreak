"""Estimations a posteriori (médiane et IC 95 %) de tous les modèles ABC,
en unités interprétables. Écrit resultats/parameter_estimates_all_models.csv."""
import sys
import pickle
import datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from roka import abc as A
from roka.paths import ABC_DIR, RESULTS

LAST = dt.date(2015, 1, 28)
LAB = {
    "beta_H": ("Transmission HIV per injection (contaminated material)", "probability", lambda v, x: v),
    "beta_C": ("Transmission HCV per injection (contaminated material)", "probability", lambda v, x: v),
    "sigma_H": ("HIV persistence in syringe", "days", lambda v, x: 1 / v),
    "sigma_C": ("HCV persistence in syringe", "days", lambda v, x: 1 / v),
    "omega_A": ("Syringe contamination by acute HIV patient", "probability", lambda v, x: v),
    "omega_U": ("Syringe contamination by chronic HIV patient", "probability", lambda v, x: v),
    "omega_C": ("Syringe contamination by HCV patient", "probability", lambda v, x: v),
    "g1": ("Interval between injections, <15 years", "days", lambda v, x: 1 / v),
    "g2": ("Interval between injections, 15-64 years", "days", lambda v, x: 1 / v),
    "g3": ("Interval between injections, >=65 years", "days", lambda v, x: 1 / v),
    "gam_idx": ("Interval between injections, index case", "days", lambda v, x: 1 / v),
    "t_max": ("Start of HIV transmission", "date", lambda v, x: v),
    "tau_C": ("Start of HCV transmission", "date", lambda v, x: v),
    "b_SI": ("Sexual transmission rate, acute HIV", "per year per pair", lambda v, x: v * 365),
    "b_SU": ("Sexual transmission rate, chronic HIV", "per year per pair", lambda v, x: v * 365),
    "n_syr": ("Number of syringes", "count", lambda v, x: v),
    "pi": ("Fraction of injections from vials", "fraction", lambda v, x: v),
    "p_C0": ("Pre-existing HCV prevalence", "fraction", lambda v, x: v),
    "stop_d": ("Stop of injections", "date", lambda v, x: v),
    "n_vial": ("Number of multidose vials", "count", lambda v, x: v),
    "vial_life": ("Duration of use of a contaminated vial", "days", lambda v, x: v),
}


def fmt(k, q):
    if k == "t_max":
        return [str(LAST - dt.timedelta(days=int(round(v)))) for v in q[::-1]]
    if k == "stop_d":
        return [str(LAST - dt.timedelta(days=int(round(v)))) for v in q[::-1]]
    return [f"{v:.3g}" for v in q]


rows = []
for m, key in enumerate(A.MODEL_KEYS):
    f = ABC_DIR / f"abc_{key}.pkl"
    if not f.exists():
        continue
    g = pickle.load(open(f, "rb"))[-1]
    X, est = g["x"], A.EST_BY_MODEL[m]
    for k in LAB:
        if k not in est:
            continue
        v = X[:, est.index(k)]
        if k == "tau_C":                      # date = début VIH + tau_C
            tm = X[:, est.index("t_max")]
            d = tm - v                        # jours avant la dernière séquence
            q = np.percentile(d, [97.5, 50, 2.5])
            vals = [str(LAST - dt.timedelta(days=int(round(x)))) for x in q]
        else:
            q = np.percentile(LAB[k][2](v, X), [2.5, 50, 97.5])
            vals = fmt(k, q)
        rows.append([A.MODEL_LABELS[m], g["eps"], LAB[k][0], LAB[k][1]] + vals)
with open(RESULTS / "parameter_estimates_all_models.csv", "w") as fo:
    fo.write("model;final_tolerance;parameter;unit;q2.5;median;q97.5\n")
    for r in rows:
        fo.write(";".join(str(x) for x in r) + "\n")
print("écrit :", RESULTS / "parameter_estimates_all_models.csv")
