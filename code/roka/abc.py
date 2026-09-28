"""
ABC-SMC sur l'IBM (core.py), un modèle à la fois, avec estimation de
l'évidence ABC pour le choix de modèle.

Modèles (population exposée au praticien) :
  voir MODEL_LABELS : cas index (un adulte infecté par le VIH à t = 0, un
  par le VHC à t = tau_C), éventuellement patient fréquent (taux d'injection
  gam_idx) et flacons multidoses renouvelés ; source persistante (modèle V2)
  et ses variantes gardées comme références

Données utilisées pour l'estimation (distance) :
  * 21 effectifs épidémiologiques (Roka_data.txt) : VIH+ par âge et classe de
    récence (12), co-infectés par âge et classe de récence VHC (9), évalués à
    la date de test de chaque individu et multipliés par la fraction de cas
    dont la récence est connue ;
  * 2 statistiques phylogénétiques publiées par Rouet et al. (2018) pour le
    cluster VIH env : date de l'ancêtre commun (tMRCA) et date du pic du
    graphe LTT. Côté simulation, elles sont calculées sur la généalogie des
    199 individus échantillonnés à leur date de test.
  d^2 = sum_epi (s - y)^2 / (y + 1) + sum_phylo (s - o)^2 / v
  v(tMRCA) = (largeur de l'intervalle HPD à 95% / 3.92)^2 ;
  v(pic LTT) = variance entre simulations répétées (étude pilote), faute
  d'incertitude publiée (data/abc_settings.json).
"""
import json
import time
import pickle
import numpy as np

from .core import Sim, P, PNAMES
from .data import load_design, load_recency
from .paths import DATA

_D = load_design()
OBS_HIV, OBS_HCV = load_recency()
OBS = np.concatenate([OBS_HIV.ravel(), OBS_HCV.ravel()])
RHO_HIV = _D["hiv_cases_with_recency"] / _D["hiv_cases_with_status"]
RHO_HCV = _D["coinfected_with_hcv_recency"] / _D["coinfected_cases"]
POP = np.array(_D["population_exposed"])
POPS = {"exposed": POP, "village": np.array(_D["population_village"])}
THETA_ACUTE = 1 / _D["acute_phase_days"]

from .dates import rouet, ym, T_LAST
N_SEQ = 199                        # séquences env du cluster (data/sequences)
_R = rouet()
# jours avant la dernière séquence (dates publiées au mois, placées au 15)
OBS_PHY = np.array([(T_LAST - ym(_R["hiv_tmrca"]["mean"])) * 365.25,
                    (T_LAST - ym(_R["hiv_ltt_peak"])) * 365.25])
SD_TMRCA = (ym(_R["hiv_tmrca"]["hpd_high"]) - ym(_R["hiv_tmrca"]["hpd_low"])) * 365.25 / 3.92
SETTINGS = json.load(open(DATA / "abc_settings.json"))
VAR_PHY = np.array([SD_TMRCA ** 2, SETTINGS["var_ltt_peak"]], float)

KSCALE = SETTINGS.get("kernel_scale", 1.0)     # covariance MCMC = KSCALE x 2.38^2/dim x covariance des particules
MODELS = [3, 1, 1, 3, 1, 1, 3, 4, 4, 3, 4, 4, 4]   # scénario de core.py
MODEL_LABELS = ["Index case", "Persistent contaminated source",
                "Persistent source, heterogeneous exposure",
                "Index case, heterogeneous exposure",
                "Persistent source, pre-existing HCV",
                "Persistent source, estimated stop date",
                "Frequent-patient index case",
                "Index case with multidose vials",
                "Frequent-patient index case with multidose vials",
                "Frequent-patient index case, biological constraints",
                "Frequent-patient index case with multidose vials, biological constraints",
                "Index case injected at most every 3 days, multidose vials, biological constraints",
                "Index case injected at most weekly, multidose vials, biological constraints"]
MODEL_KEYS = ["index_case", "persistent_source", "source_hetero", "index_hetero",
              "source_hcv_pre", "source_stop", "freq_index", "vials_index", "freq_index_vials",
              "freq_index_c", "freq_index_vials_c", "idx3d_vials_c", "idx7d_vials_c"]
COMMON = ["beta_H", "beta_C", "sigma_H", "sigma_C", "omega_A", "omega_U", "omega_C",
          "g1", "g2", "g3", "t_max", "b_SI", "b_SU", "tau_C", "n_syr"]
SRC = ["P_i", "pi"]
HET = ["kappa", "f_hi"]
EXTRA = ["p_C0", "stop_d"]
NOTAU = [k for k in COMMON if k != "tau_C"]          # VHC préexistant et date d'arrêt estimée
VIALS = ["pi", "n_vial", "vial_life"]
EST_BY_MODEL = [COMMON, COMMON + SRC, COMMON + SRC + HET, COMMON + HET,
                COMMON + SRC + ["p_C0"], COMMON + SRC + ["stop_d"],
                COMMON + ["gam_idx"] + EXTRA, COMMON + VIALS + EXTRA,
                COMMON + ["gam_idx"] + VIALS + EXTRA,
                NOTAU + ["gam_idx"] + EXTRA, NOTAU + ["gam_idx"] + VIALS + EXTRA,
                NOTAU + ["gam_idx"] + VIALS + EXTRA, NOTAU + ["gam_idx"] + VIALS + EXTRA]
# Contraintes biologiques (modèles contraints) : contamination d'une seringue au
# moins aussi probable en phase aiguë qu'en phase chronique, persistance du VHC
# dans une seringue au moins égale à celle du VIH, transmission du VHC par
# injection au moins égale à celle du VIH. Le VHC étant déjà présent, il n'y a
# pas de cas index VHC (tau_C non estimé).
CONSTRAINED = {9, 10, 11, 12}
# Analyses de sensibilité : fréquence d'injection du cas index limitée
# (au plus une injection tous les 3 jours, ou toutes les semaines).
BOUND_OVERRIDE = {11: {"gam_idx": (1 / 365, 1 / 3)}, 12: {"gam_idx": (1 / 365, 1 / 7)}}
# valeurs des paramètres non estimés
DEFAULTS = {"stop_d": 28.0, "p_C0": 0.0, "kappa": 1.0, "f_hi": 0.0, "sig_VH": 0.0,
            "sig_VC": 0.0, "mu": 4.1e-3, "P_i": 0.0, "pi": 0.0, "gam_idx": 0.0,
            "n_vial": 0.0, "vial_life": 0.0, "tau_C": 1e9}

# priors : log-uniforme (taux, probabilités, nombre de seringues), uniforme
# (durées, pi), normal tronqué (horloge moléculaire, Rouet et al.)
LOG_PRIOR = {"beta_H": (1e-3, 1), "beta_C": (1e-3, 1), "sigma_H": (1e-2, 5), "sigma_C": (1e-2, 5),
             "omega_A": (1e-3, 1), "omega_U": (1e-3, 1), "omega_C": (1e-3, 1),
             "g1": (1 / 365, 1), "g2": (1 / 365, 1), "g3": (1 / 365, 1),
             "b_SI": (1e-7, 0.1 / 365), "b_SU": (1e-7, 0.1 / 365), "n_syr": (1, 1000),
             "sig_VH": (1e-4, 5), "sig_VC": (1e-4, 5), "kappa": (1, 100),
             "gam_idx": (1 / 365, 2), "n_vial": (1, 30), "vial_life": (0.5, 90)}
UNIF_PRIOR = {"t_max": (30, 4 * 365), "tau_C": (-5 * 365, 2 * 365), "P_i": (0, 2 * 365), "pi": (0, 1),
              "f_hi": (0, 1), "p_C0": (0, 0.3), "stop_d": (28, 120)}
MU_PRIOR = dict(mean=4.1e-3, sd=9.3e-4, lo=1e-3, hi=1e-2)


def _phi(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


_MU_MASS = (_phi((MU_PRIOR["hi"] - MU_PRIOR["mean"]) / MU_PRIOR["sd"])
            - _phi((MU_PRIOR["lo"] - MU_PRIOR["mean"]) / MU_PRIOR["sd"]))


def prior_range(m, k):
    if k in BOUND_OVERRIDE.get(m, {}):
        return BOUND_OVERRIDE[m][k]
    return LOG_PRIOR.get(k) or UNIF_PRIOR.get(k) or (MU_PRIOR["lo"], MU_PRIOR["hi"])


def bounds(m):
    est = EST_BY_MODEL[m]
    lo, hi = [], []
    for k in est:
        a, b = prior_range(m, k)
        lo.append(a)
        hi.append(b)
    return np.array(lo, float), np.array(hi, float)


def log_idx(m):
    return np.array([i for i, k in enumerate(EST_BY_MODEL[m]) if k in LOG_PRIOR], int)


def to_y(m, x):
    y = np.array(x, float).copy()
    li = log_idx(m)
    y[..., li] = np.log(y[..., li])
    return y


def to_x(m, y):
    x = np.array(y, float).copy()
    li = log_idx(m)
    x[..., li] = np.exp(x[..., li])
    return x


def constraints_ok(m, x):
    if m not in CONSTRAINED:
        return True
    e = EST_BY_MODEL[m]
    g = lambda k: x[e.index(k)]
    return g("omega_A") >= g("omega_U") and g("sigma_C") <= g("sigma_H") and g("beta_C") >= g("beta_H")


def prior_sample(m, rng):
    while True:
        x = _prior_sample(m, rng)
        if constraints_ok(m, x):
            return x


def _prior_sample(m, rng):
    est = EST_BY_MODEL[m]
    x = np.empty(len(est))
    for i, k in enumerate(est):
        if k in LOG_PRIOR:
            a, b = prior_range(m, k)
            x[i] = np.exp(rng.uniform(np.log(a), np.log(b)))
        elif k in UNIF_PRIOR:
            x[i] = rng.uniform(*prior_range(m, k))
        else:
            while True:
                v = rng.normal(MU_PRIOR["mean"], MU_PRIOR["sd"])
                if MU_PRIOR["lo"] <= v <= MU_PRIOR["hi"]:
                    break
            x[i] = v
    return x


def prior_logdens_y(m, x):
    """Log-densité du prior dans l'espace de travail y (log pour LOG_PRIOR)."""
    lo, hi = bounds(m)
    if np.any(x < lo) or np.any(x > hi) or not constraints_ok(m, x):
        return -np.inf
    ld = 0.0
    for i, k in enumerate(EST_BY_MODEL[m]):
        if k in LOG_PRIOR:
            ld -= np.log(np.log(hi[i]) - np.log(lo[i]))
        elif k in UNIF_PRIOR:
            ld -= np.log(hi[i] - lo[i])
        else:
            z = (x[i] - MU_PRIOR["mean"]) / MU_PRIOR["sd"]
            ld += -0.5 * z * z - np.log(MU_PRIOR["sd"] * np.sqrt(2 * np.pi) * _MU_MASS)
    return ld


def full_par(m, x):
    par = np.zeros(len(PNAMES))
    for k, v in DEFAULTS.items():
        par[P[k]] = v
    for i, k in enumerate(EST_BY_MODEL[m]):
        par[P[k]] = x[i]
    par[P["theta"]] = THETA_ACUTE
    return par


def epi_stats(sim):
    c = sim.counts
    return np.concatenate([(c[:, 0:4] * RHO_HIV).ravel(), (c[:, 4:7] * RHO_HCV).ravel()])


def distance_parts(sim, eps=np.inf):
    """(d_epi^2, d_phylo^2, statistiques). Les statistiques phylogénétiques ne
    sont calculées que si la partie épidémiologique est déjà sous la tolérance."""
    if not sim.ok:
        return np.inf, np.inf, None
    se = epi_stats(sim)
    d1 = float(np.sum((se - OBS) ** 2 / (OBS + 1.0)))
    if d1 > eps ** 2:
        return d1, np.inf, None
    sp, _ = sim.phylo_stats(N_SEQ)
    if sp is None:
        return d1, np.inf, None
    d2 = float(np.sum((sp - OBS_PHY) ** 2 / VAR_PHY))
    return d1, d2, np.concatenate([se, sp])


def simulate_distance(m, x, seed, eps=np.inf):
    s = Sim(full_par(m, x), POP, MODELS[m], seed)
    d1, d2, _ = distance_parts(s, eps)
    return float(np.sqrt(d1 + d2))


def _worker(args):
    m, x, seed, eps = args
    return simulate_distance(m, x, seed, eps)


def ess(w):
    w = np.asarray(w, float)
    w = w / w.sum()
    return 1.0 / np.sum(w ** 2)


# --------------------------------------------------------------------------
class Progress:
    """Barre de progression texte (une ligne, mise à jour sur place)."""

    def __init__(self, label, total):
        self.label, self.total, self.t0 = label, total, time.time()

    def update(self, done, extra=""):
        f = min(done / self.total, 1.0) if self.total else 1.0
        el = time.time() - self.t0
        eta = el / f - el if f > 0 else float("nan")
        bar = "#" * int(30 * f) + "-" * (30 - int(30 * f))
        print(f"\r{self.label} [{bar}] {100 * f:5.1f}%  {extra}  "
              f"écoulé {el / 60:5.1f} min, reste ~{eta / 60:5.1f} min   ", end="", flush=True)

    def close(self):
        print(flush=True)


def prior_ratio_ok(m, x_new, x_old, rng):
    """Acceptation Metropolis-Hastings sur le prior (noyau symétrique en y)."""
    lp1 = prior_logdens_y(m, x_new)
    if not np.isfinite(lp1):
        return False
    return np.log(rng.random()) < lp1 - prior_logdens_y(m, x_old)


def run_smc(m, pool, out_file, n_particles, eps_schedule, max_sims, min_acc,
            batch=400, seed=1, log=print, max_moves=40, c_target=0.01):
    """ABC-SMC à poids égaux avec mouvements MCMC (Drovandi et Pettitt 2011,
    Del Moral et al. 2012), calendrier de tolérances fixé, reprise automatique.

    Génération 0 : tirages du prior jusqu'à n_particles simulations sous eps_0,
    Z(eps_0) = fraction acceptée. Génération t : les particules dont la
    distance est déjà sous eps_t sont conservées (fraction r_t, Z(eps_t) =
    Z(eps_{t-1}) x r_t), les autres sont remplacées par des copies des
    survivantes, puis toutes subissent R mouvements MCMC-ABC (proposition
    gaussienne de covariance KSCALE x 2.38^2/dim x covariance des particules, acceptation si
    la nouvelle simulation est sous eps_t et selon le rapport des priors).
    R est choisi pour que chaque particule ait bougé avec une probabilité
    1 - c_target, d'après le taux d'acceptation du premier mouvement."""
    rng = np.random.default_rng(seed + 1000 * m)
    hist = pickle.load(open(out_file, "rb")) if out_file.exists() else []
    if hist and not hist[-1].get("complete", True):
        log(f"model {m}: generation {len(hist) - 1} incomplete, recomputed")
        hist = hist[:-1]
    seed_ctr = (seed + 1) * 100_000_000 + m * 10_000_000 + 1_000_000 * len(hist) + 500_000
    lo, hi = bounds(m)
    N = n_particles
    for k, eps in enumerate(eps_schedule):
        if k < len(hist):
            continue
        prog = Progress(f"{MODEL_LABELS[m]:<31s} gen {k:2d} eps={eps:6.2f}", N)
        if k == 0:
            acc_x, acc_d, n_sim = [], [], 0
            while len(acc_x) < N and n_sim < max_sims:
                xs = [prior_sample(m, rng) for _ in range(batch)]
                seeds = seed_ctr + np.arange(batch)
                seed_ctr += batch
                d = np.array(pool.map(_worker, [(m, x, s, eps) for x, s in zip(xs, seeds)], chunksize=8))
                n_sim += batch
                for x, dd in zip(xs, d):
                    if dd <= eps:
                        acc_x.append(x)
                        acc_d.append(dd)
                prog.update(len(acc_x), f"acc {len(acc_x) / n_sim:.4f}, sims {n_sim}")
            prog.close()
            X, Dd = np.array(acc_x[:N]), np.array(acc_d[:N])
            Z = len(acc_x) / n_sim
            g = dict(model=m, eps=eps, x=X, d=Dd, w=np.ones(len(X)) / max(len(X), 1), n_sim=n_sim,
                     acc_rate=len(acc_x) / n_sim, Z=Z, r=1.0, moves=0, mh_acc=np.nan,
                     complete=len(X) >= N)
        else:
            prev = hist[-1]
            X, Dd = prev["x"].copy(), prev["d"].copy()
            alive = np.where(Dd <= eps)[0]
            r = len(alive) / len(X)
            Z = prev["Z"] * r
            if len(alive) < 5:
                prog.close()
                log(f"model {m} gen {k}: eps={eps:.2f} only {len(alive)} particles below the tolerance, stop")
                break
            dead = np.where(Dd > eps)[0]
            src = rng.choice(alive, size=len(dead))
            X[dead], Dd[dead] = X[src], Dd[src]
            Y = to_y(m, X)
            C = KSCALE * 2.38 ** 2 / Y.shape[1] * np.cov(Y[alive].T) + 1e-10 * np.eye(Y.shape[1])
            Lc = np.linalg.cholesky(C)
            moved = np.zeros(N, bool)
            n_sim, n_acc_mh, n_try = 0, 0, 0
            R = max_moves
            step = 0
            while step < R and n_sim < max_sims:
                props, idx = [], []
                for i in range(N):
                    xn = to_x(m, Y[i] + Lc @ rng.standard_normal(Y.shape[1]))
                    if np.any(xn < lo) or np.any(xn > hi) or not prior_ratio_ok(m, xn, X[i], rng):
                        n_try += 1
                        continue
                    props.append(xn)
                    idx.append(i)
                seeds = seed_ctr + np.arange(len(props))
                seed_ctr += len(props)
                d = np.array(pool.map(_worker, [(m, x, s, eps) for x, s in zip(props, seeds)], chunksize=8)) \
                    if props else np.array([])
                n_sim += len(props)
                acc_step = 0
                for i, xn, dd in zip(idx, props, d):
                    n_try += 1
                    if dd <= eps:
                        X[i], Dd[i], Y[i] = xn, dd, to_y(m, xn)
                        moved[i] = True
                        acc_step += 1
                n_acc_mh += acc_step
                step += 1
                if step == 1:
                    p = max(acc_step / N, 1e-4)
                    R = int(min(max_moves, max(2, np.ceil(np.log(c_target) / np.log(1 - p)))))
                prog.update(int(N * step / R), f"MCMC {step}/{R}, moved {moved.mean():.2f}, sims {n_sim}")
            prog.close()
            mh = n_acc_mh / max(n_try, 1)
            g = dict(model=m, eps=eps, x=X, d=Dd, w=np.ones(N) / N, n_sim=n_sim, acc_rate=mh,
                     Z=Z, r=r, moves=step, mh_acc=mh, moved=float(moved.mean()),
                     unique=int(len(np.unique(X[:, 0]))), complete=True)
        hist.append(g)
        pickle.dump(hist, open(out_file, "wb"))
        log(f"model {m} gen {k}: eps={eps:.2f} kept={g['r']:.2f} MH-acc={g['acc_rate']:.4f} "
            f"moves={g['moves']} unique={g.get('unique', len(g['x']))} sims={g['n_sim']} Z={g['Z']:.3e}")
        if not g["complete"]:
            break
        if k > 0 and (g["mh_acc"] < min_acc or g["unique"] < N / 3):
            log(f"model {m}: MCMC acceptance too low or too few distinct particles, stop")
            break
    return hist
