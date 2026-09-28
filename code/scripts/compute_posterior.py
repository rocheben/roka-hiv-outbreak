"""Calcule les caches de simulations a posteriori du modèle retenu
(resultats/cache/*.npz). Usage : python scripts/compute_posterior.py [--force]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roka.posterior import posterior_predictive, counterfactuals, sensitivity_predictive, SENSITIVITY

force = "--force" in sys.argv
posterior_predictive(force); print("posterior_predictive : ok", flush=True)
counterfactuals(force); print("counterfactuals : ok", flush=True)
for k in SENSITIVITY:
    sensitivity_predictive(k, force); print(f"sensitivity {k} : ok", flush=True)
