import os
from pathlib import Path

# cache de compilation numba hors du dossier du projet (Google Drive) :
# évite de synchroniser des fichiers compilés propres à une machine
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "roka_numba"))
