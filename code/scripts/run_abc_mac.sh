#!/usr/bin/env bash
# Lance l'ABC sur le Mac : environnement Python dédié (hors Google Drive),
# installation des dépendances au premier lancement, puis calcul avec tous les
# coeurs. caffeinate empêche la mise en veille pendant le calcul.
# Relancer la même commande reprend le calcul là où il s'est arrêté.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"          # code/scripts
VENV="$HOME/.venvs/roka"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Première utilisation : création de l'environnement Python dans $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r "$HERE/../requirements.txt"
fi
export NUMBA_CACHE_DIR="$HOME/.cache/roka_numba"
# caffeinate n'existe que sur macOS ; ailleurs le calcul est lancé directement
if command -v caffeinate >/dev/null; then KEEP="caffeinate -i"; else KEEP=""; fi
$KEEP "$VENV/bin/python" "$HERE/run_abc.py" "$@"
