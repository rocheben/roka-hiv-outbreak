"""Posterior ABC du modèle retenu et simulations a posteriori (caches dans
resultats/cache). Chaque cache est recalculé s'il est absent (graines fixées)."""
import pickle
import numpy as np

from .paths import ABC_DIR, CACHE
from . import abc as A
from .core import Sim, P, KIND_HOST, ROUTE_INDEX, ROUTE_NEEDLE, ROUTE_SEX, ROUTE_VIAL, ROUTE_PRE

SELECTED = A.MODEL_KEYS.index("freq_index_vials_c")     # modèle retenu
REALISTIC = ["index_case", "freq_index", "vials_index", "freq_index_vials",
             "freq_index_c", "freq_index_vials_c"]          # introduction unique
SENSITIVITY = ["idx3d_vials_c", "idx7d_vials_c"]             # cas index injecté moins souvent
N_SENS = 300
GRID_DAYS = np.arange(0, 3 * 365 + 1, 5)                   # jours avant la dernière séquence
N_PP, N_CF = 500, 300
CF_NAMES = ["Fitted model", "No sexual transmission", "No multi-dose vials",
            "No syringe sharing between patients"]


def load_history(m):
    return pickle.load(open(ABC_DIR / f"abc_{A.MODEL_KEYS[m]}.pkl", "rb"))


def available_models():
    return [m for m, k in enumerate(A.MODEL_KEYS) if (ABC_DIR / f"abc_{k}.pkl").exists()]


def posterior(m=SELECTED):
    """(m, particules, poids, génération finale). Poids égaux (ABC-SMC MCMC)."""
    g = load_history(m)[-1]
    X = g["x"]
    return m, X, np.ones(len(X)) / len(X), g


def draws(n, seed, m=SELECTED):
    m, X, W, _ = posterior(m)
    idx = np.random.default_rng(seed).choice(len(X), size=n)
    return m, X[idx]


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]


def wquant(v, w, qs=(0.025, 0.5, 0.975)):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return [float(np.interp(q, c / c[-1], v[o])) for q in qs]


def evidence_table():
    """Évidence Z(eps) de chaque modèle disponible : {clé: {eps: Z}} et tolérance finale."""
    out = {}
    for m in available_models():
        h = load_history(m)
        out[A.MODEL_KEYS[m]] = dict(label=A.MODEL_LABELS[m], final_eps=float(h[-1]["eps"]),
                                    Z={float(g["eps"]): float(g["Z"]) for g in h})
    return out


def _routes(s, virus):
    sel = (s.kind == KIND_HOST) & (s.virus == virus)
    r = s.route[sel]
    return np.array([np.sum(r == ROUTE_VIAL), np.sum(r == ROUTE_NEEDLE), np.sum(r == ROUTE_SEX),
                     np.sum((r == ROUTE_INDEX) | (r == ROUTE_PRE))], float)


def _index_share(s):
    """Part des infections VIH par injection dues à un matériel contaminé
    directement par le cas index."""
    hiv = np.where((s.kind == KIND_HOST) & (s.virus == 0))[0]
    idx = hiv[s.route[hiv] == ROUTE_INDEX]
    inj = hiv[(s.route[hiv] == ROUTE_VIAL) | (s.route[hiv] == ROUTE_NEEDLE)]
    if len(idx) == 0 or len(inj) == 0:
        return np.nan
    return float(np.mean(s.src[s.parent[inj]] == s.owner[idx[0]]))


def _cumulative(s, virus, grid):
    sel = (s.kind == KIND_HOST) & (s.virus == virus) & (s.route != ROUTE_PRE) & (s.route != ROUTE_INDEX)
    b = np.sort(s.birth[sel])
    return np.searchsorted(b, s.t_max - grid, side="right")


def posterior_predictive(force=False):
    f = CACHE / "posterior_predictive.npz"
    if f.exists() and not force:
        return dict(np.load(f))
    m, X = draws(N_PP, seed=7)
    res = {k: [] for k in ("counts", "stats", "routes_hiv", "routes_hcv", "phylo", "index_share",
                           "traj_hiv", "traj_hcv", "dist_epi", "dist_phy")}
    for j, x in enumerate(X):
        s = Sim(A.full_par(m, x), A.POP, A.MODELS[m], 50_000 + j)
        d1, d2, st = A.distance_parts(s)
        res["counts"].append(s.counts)
        res["stats"].append(A.epi_stats(s))
        res["routes_hiv"].append(_routes(s, 0))
        res["routes_hcv"].append(_routes(s, 1))
        ps, _ = s.phylo_stats(A.N_SEQ)
        res["phylo"].append(ps if ps is not None else [np.nan, np.nan])
        res["index_share"].append(_index_share(s))
        res["traj_hiv"].append(_cumulative(s, 0, GRID_DAYS))
        res["traj_hcv"].append(_cumulative(s, 1, GRID_DAYS))
        res["dist_epi"].append(d1)
        res["dist_phy"].append(d2)
    out = {k: np.array(v) for k, v in res.items()}
    out.update(grid=GRID_DAYS, params=X, model=m)
    np.savez_compressed(f, **out)
    return out


def counterfactuals(force=False):
    f = CACHE / "counterfactuals.npz"
    if f.exists() and not force:
        return dict(np.load(f))
    m, X = draws(N_CF, seed=11)
    ipi = A.EST_BY_MODEL[m].index("pi")
    out = {}
    for name in CF_NAMES:
        cs = []
        for j, x in enumerate(X):
            x = x.copy()
            pool_on, sex_on = True, True
            if name == "No sexual transmission":
                sex_on = False
            elif name == "No multi-dose vials":
                x[ipi] = 0.0
            elif name == "No syringe sharing between patients":
                pool_on = False
            s = Sim(A.full_par(m, x), A.POP, A.MODELS[m], 70_000 + j, pool_on=pool_on, sex_on=sex_on)
            cs.append(s.counts)
        out[name] = np.array(cs)
    np.savez_compressed(f, **out)
    return out


def sensitivity_predictive(key, force=False):
    """Simulations a posteriori d'un modèle de sensibilité (fréquence d'injection du cas index)."""
    f = CACHE / f"sensitivity_{key}.npz"
    if f.exists() and not force:
        return dict(np.load(f))
    m = A.MODEL_KEYS.index(key)
    m, X = draws(N_SENS, seed=13, m=m)
    res = {k: [] for k in ("counts", "routes_hiv", "routes_hcv", "phylo")}
    for j, x in enumerate(X):
        s = Sim(A.full_par(m, x), A.POP, A.MODELS[m], 90_000 + j)
        res["counts"].append(s.counts)
        res["routes_hiv"].append(_routes(s, 0))
        res["routes_hcv"].append(_routes(s, 1))
        ps, _ = s.phylo_stats(A.N_SEQ)
        res["phylo"].append(ps if ps is not None else [np.nan, np.nan])
    out = {k: np.array(v) for k, v in res.items()}
    out.update(params=X, model=m)
    np.savez_compressed(f, **out)
    return out
