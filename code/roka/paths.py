"""Chemins du projet, relatifs à l'emplacement de ce fichier (exécutable partout)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # code/roka/paths.py -> racine du projet
DATA = ROOT / "data"
RESULTS = ROOT / "resultats"
ABC_DIR = RESULTS / "abc"
CACHE = RESULTS / "cache"
FIGURES = ROOT / "figures"
MANUSCRIPT = ROOT                                # documents à la racine

for d in (RESULTS, ABC_DIR, CACHE, FIGURES):
    d.mkdir(parents=True, exist_ok=True)
