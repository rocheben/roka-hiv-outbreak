"""Tableaux et valeurs citées dans le manuscrit : resultats/numbers.json,
resultats/table1_parameters.csv, resultats/table_model_comparison.csv.
Toutes les valeurs des documents proviennent de numbers.json."""
import sys
import json
import datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from roka.paths import RESULTS, DATA
from roka import abc as A
from roka.posterior import (posterior, posterior_predictive, counterfactuals, evidence_table, ci,
                            CF_NAMES, SELECTED, REALISTIC, SENSITIVITY, sensitivity_predictive,
                            load_history, available_models)
from roka.dates import rouet
from roka.data import load_design

LAST = dt.date(2015, 1, 28)
day = lambda d: LAST - dt.timedelta(days=int(round(float(d))))   # même arrondi que parameter_table.py
iso = lambda d: day(d).isoformat()
n = {}

# ------------------------------------------------------------ données
D = load_design()
n["data"] = dict(hiv_by_recency=A.OBS_HIV.tolist(), hcv_by_recency=A.OBS_HCV.tolist(),
                 hiv_total_recency=int(A.OBS_HIV.sum()), hcv_total_recency=int(A.OBS_HCV.sum()),
                 rho_hiv=A.RHO_HIV, rho_hcv=A.RHO_HCV, design=D,
                 phylo_obs_days=A.OBS_PHY.tolist(), sd_tmrca_days=float(A.SD_TMRCA),
                 sd_peak_days=float(np.sqrt(A.VAR_PHY[1])), n_seq=A.N_SEQ)
T = json.load(open(DATA / "test_dates.json"))
n["data"]["test_dates"] = dict(first=iso(max(T["days_before_last"])), last=iso(min(T["days_before_last"])))
seq = json.load(open(DATA / "observed_sequence_stats.json"))
n["data"]["alignment"] = dict(n=seq["n_sequences"], length=seq["alignment_length"])
tt = {}
for k in ("none", "skyline"):
    for line in open(DATA / "sequences" / f"treetime_hiv_{k}" / "dates.tsv"):
        if line.startswith("NODE_0000000"):
            p = line.split("\t")
            tt[k] = dict(date=p[1], lo=float(p[3]), hi=float(p[4]))
n["treetime"] = tt

# ------------------------------------------------------------ comparaison des modèles
E = evidence_table()
S = A.SETTINGS
n["abc"] = dict(n_particles=S["n_particles"], eps0=S["eps_schedule"][0], eps_factor=0.94,
                max_moves=S["max_moves"], min_acc=S["min_acceptance"],
                total_sims=int(sum(sum(g["n_sim"] for g in load_history(m)) for m in available_models())))


def probs(eps, keys):
    ks = [k for k in keys if k in E and eps in E[k]["Z"]]
    z = np.array([E[k]["Z"][eps] for k in ks])
    return {k: float(v) for k, v in zip(ks, z / z.sum())}


keys_all = [k for k in REALISTIC if k in E]
common = min(e for e in E["index_case"]["Z"])                       # tolérance atteinte par tous
sel_key = A.MODEL_KEYS[SELECTED]
eps_sel = E[sel_key]["final_eps"]
n["model_comparison"] = dict(
    models={k: dict(label=E[k]["label"], final_eps=E[k]["final_eps"]) for k in keys_all},
    eps_common=common, probs_common=probs(common, keys_all),
    eps_selected=eps_sel, probs_selected=probs(eps_sel, keys_all))
with open(RESULTS / "table_model_comparison.csv", "w") as f:
    f.write(f"model;lowest_tolerance;P(model) at eps={common};P(model) at eps={eps_sel}\n")
    for k in keys_all:
        f.write(";".join([E[k]["label"], f"{E[k]['final_eps']}",
                          f"{n['model_comparison']['probs_common'].get(k, float('nan')):.3f}",
                          f"{n['model_comparison']['probs_selected'].get(k, float('nan')):.3f}"]) + "\n")

# ------------------------------------------------------------ Table 1 (modèle retenu)
m, X, W, G = posterior()
est = A.EST_BY_MODEL[m]
n["selected"] = dict(key=sel_key, label=A.MODEL_LABELS[m], final_eps=float(G["eps"]),
                     unique=int(G.get("unique", len(X))), n_particles=len(X))
CONV = {"sigma_H": ("inv", "days"), "sigma_C": ("inv", "days"), "g1": ("inv", "days"),
        "g2": ("inv", "days"), "g3": ("inv", "days"), "gam_idx": ("inv", "days"),
        "b_SI": ("py", "per year per pair"), "b_SU": ("py", "per year per pair"),
        "t_max": ("date", "date"), "stop_d": ("date", "date"), "vial_life": ("id", "days"),
        "n_syr": ("id", "count"), "n_vial": ("id", "count"), "p_C0": ("id", "fraction"),
        "pi": ("id", "fraction")}
t1 = {}
for i, k in enumerate(est):
    v = X[:, i]
    tr, unit = CONV.get(k, ("id", "probability"))
    if tr == "date":
        q = [iso(np.percentile(v, 97.5)), iso(np.median(v)), iso(np.percentile(v, 2.5))]
    else:
        vv = {"inv": 1 / v, "py": v * 365, "id": v}[tr]
        q = ci(vv)
    t1[k] = dict(q=q, unit=unit)
n["table1"] = t1
with open(RESULTS / "table1_parameters.csv", "w") as f:
    f.write("parameter;unit;q2.5;median;q97.5\n")
    for k, d in t1.items():
        f.write(f"{k};{d['unit']};" + ";".join(str(x) if isinstance(x, str) else f"{x:.4g}" for x in d["q"]) + "\n")
ix = {k: i for i, k in enumerate(est)}
n["derived"] = dict(
    ratio_betaC_betaH=ci(X[:, ix["beta_C"]] / X[:, ix["beta_H"]]),
    ratio_omegaA_omegaU=ci(X[:, ix["omega_A"]] / X[:, ix["omega_U"]]),
    index_vs_adult_rate=ci(X[:, ix["gam_idx"]] / X[:, ix["g2"]]),
    duration_days=ci(X[:, ix["t_max"]] - X[:, ix["stop_d"]]))
corr = np.corrcoef(A.to_y(m, X).T)
iu = np.triu_indices(len(est), 1)
top = np.argsort(-np.abs(corr[iu]))[:4]
n["top_correlations"] = [[est[iu[0][t]], est[iu[1][t]], float(corr[iu][t])] for t in top]

# ------------------------------------------------------------ prédictif a posteriori
pp = posterior_predictive()
c = pp["counts"]
rh, rc = pp["routes_hiv"], pp["routes_hcv"]
tot_h = np.maximum(rh.sum(1), 1)
tot_c = np.maximum(rc.sum(1), 1)
ph = pp["phylo"]
okp = np.isfinite(ph).all(1)
n["pp"] = dict(
    n_draws=int(len(c)),
    hiv_by_age=[ci(c[:, a, 10]) for a in range(3)], hiv_total=ci(c[:, :, 10].sum(1)),
    coinf_by_age=[ci(c[:, a, 8]) for a in range(3)], coinf_total=ci(c[:, :, 8].sum(1)),
    hcv_only_by_age=[ci(c[:, a, 9]) for a in range(3)], hcv_only_total=ci(c[:, :, 9].sum(1)),
    hcv_total=ci(c[:, :, 11].sum(1)),
    extinct=float(np.mean(c[:, :, 10].sum(1) < 20)),
    routes_hiv_n=dict(vials=ci(rh[:, 0]), syringes=ci(rh[:, 1]), sexual=ci(rh[:, 2])),
    routes_hiv_frac=dict(vials=ci(rh[:, 0] / tot_h), syringes=ci(rh[:, 1] / tot_h), sexual=ci(rh[:, 2] / tot_h)),
    routes_hcv_n=dict(vials=ci(rc[:, 0]), syringes=ci(rc[:, 1]), pre=ci(rc[:, 3])),
    routes_hcv_frac=dict(vials=ci(rc[:, 0] / tot_c), syringes=ci(rc[:, 1] / tot_c), pre=ci(rc[:, 3] / tot_c)),
    index_share=ci(pp["index_share"]),
    tmrca=[iso(np.percentile(ph[okp, 0], 97.5)), iso(np.median(ph[okp, 0])), iso(np.percentile(ph[okp, 0], 2.5))],
    peak=[iso(np.percentile(ph[okp, 1], 97.5)), iso(np.median(ph[okp, 1])), iso(np.percentile(ph[okp, 1], 2.5))],
    dist_epi=ci(np.sqrt(pp["dist_epi"])))
cf = counterfactuals()
n["counterfactuals"] = {k: dict(hiv=ci(cf[k][:, :, 10].sum(1)), coinf=ci(cf[k][:, :, 8].sum(1)),
                                hcv=ci(cf[k][:, :, 11].sum(1))) for k in CF_NAMES}

# ------------------------------------------------------------ autres modèles (estimations clés)
other = {}
for k in ("freq_index_c", "index_case"):
    mm = A.MODEL_KEYS.index(k)
    Xk = load_history(mm)[-1]["x"]
    e = A.EST_BY_MODEL[mm]
    d = dict(start=[iso(np.percentile(Xk[:, e.index("t_max")], q)) for q in (97.5, 50, 2.5)])
    if "stop_d" in e:
        d["stop"] = [iso(np.percentile(Xk[:, e.index("stop_d")], q)) for q in (97.5, 50, 2.5)]
    d["n_syr"] = ci(Xk[:, e.index("n_syr")])
    d["hiv_persist"] = ci(1 / Xk[:, e.index("sigma_H")])
    other[k] = d
n["other_models"] = other

# ------------------------------------------------------------ sensibilité : fréquence d'injection du cas index
sens = {}
ks = [sel_key] + SENSITIVITY
eps_s = max(E[k]["final_eps"] for k in ks)        # tolérance commune
z = np.array([E[k]["Z"][eps_s] for k in ks])
for k, pk in zip(ks, z / z.sum()):
    mm = A.MODEL_KEYS.index(k)
    Xk = load_history(mm)[-1]["x"]
    e = A.EST_BY_MODEL[mm]
    d = dict(label=E[k]["label"], final_eps=E[k]["final_eps"], prob=float(pk),
             bf_selected=float(E[sel_key]["Z"][eps_s] / E[k]["Z"][eps_s]),
             interval=ci(1 / Xk[:, e.index("gam_idx")]), adult_interval=ci(1 / Xk[:, e.index("g2")]),
             pi=ci(Xk[:, e.index("pi")]),
             start=[iso(np.percentile(Xk[:, e.index("t_max")], q)) for q in (97.5, 50, 2.5)],
             stop=[iso(np.percentile(Xk[:, e.index("stop_d")], q)) for q in (97.5, 50, 2.5)])
    if k != sel_key:
        r = sensitivity_predictive(k)
        c_, rh_ = r["counts"], r["routes_hiv"]
        t_ = np.maximum(rh_.sum(1), 1)
        d.update(hiv=ci(c_[:, :, 10].sum(1)), coinf=ci(c_[:, :, 8].sum(1)), hcv_only=ci(c_[:, :, 9].sum(1)),
                 vials=ci(rh_[:, 0] / t_), sexual=ci(rh_[:, 2] / t_), extinct=float(np.mean(c_[:, :, 10].sum(1) < 20)))
    sens[k] = d
n["sensitivity"] = dict(eps=eps_s, models=sens)
n["rouet"] = rouet()
n["priors"] = dict(log={k: list(v) for k, v in A.LOG_PRIOR.items()}, unif={k: list(v) for k, v in A.UNIF_PRIOR.items()})
n["evidence"] = {k: dict(label=E[k]["label"], Z={f"{e:.2f}": z for e, z in E[k]["Z"].items()}) for k in keys_all + [k for k in SENSITIVITY if k in E]}
n["est_by_model"] = {A.MODEL_KEYS[i]: A.EST_BY_MODEL[i] for i in range(len(A.MODEL_KEYS))}
n["abc"]["settings"] = dict(var_ltt_peak=S["var_ltt_peak"], eps_schedule=S["eps_schedule"])
json.dump(n, open(RESULTS / "numbers.json", "w"), indent=1, ensure_ascii=False, default=float)
print("écrit :", RESULTS / "numbers.json")
