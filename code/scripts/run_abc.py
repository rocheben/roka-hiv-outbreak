"""ABC-SMC des modèles de roka/abc.py (MODEL_LABELS), l'un après l'autre,
avec barre de progression et reprise automatique.

Usage (depuis n'importe quel dossier) :
    python3 scripts/run_abc.py            # tous les modèles (roka/abc.py, MODEL_LABELS)
    python3 scripts/run_abc.py 6 7 8      # modèles 6, 7 et 8 seulement
    python3 scripts/run_abc.py --test     # essai rapide (quelques minutes)

Résultats : resultats/abc/abc_<modèle>.pkl et resultats/abc/abc_log.txt.
Relancer la même commande reprend là où le calcul s'était arrêté.
"""
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # dossier code/ (paquet roka)
from roka.paths import ABC_DIR
from roka import abc as A


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    test = "--test" in sys.argv
    models = [int(a) for a in args] or list(range(len(A.MODELS)))
    S = A.SETTINGS
    n_part = 100 if test else S["n_particles"]
    eps = S["eps_schedule"][:5] if test else S["eps_schedule"]
    max_sims = 20000 if test else S["max_sims_per_generation"]
    out_dir = ABC_DIR / "test" if test else ABC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ncpu = os.cpu_count() or 2
    logf = open(out_dir / "abc_log.txt", "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"start: models {models}, {n_part} particles, {ncpu} cores, test={test}")
    with Pool(ncpu) as pool:
        # compilation numba dans chaque processus
        pool.map(A._worker, [(0, A.prior_sample(0, __import__('numpy').random.default_rng(i)), i, 1.0)
                             for i in range(ncpu)])
        for m in models:
            f = out_dir / f"abc_{A.MODEL_KEYS[m]}.pkl"
            A.run_smc(m, pool, f, n_part, eps, max_sims, S["min_acceptance"],
                      batch=S["batch"], log=log, max_moves=S.get("max_moves", 40))
    log("finished")


if __name__ == "__main__":
    main()
