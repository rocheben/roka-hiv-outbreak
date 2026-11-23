#!/usr/bin/env bash
# Régénère caches, figures et tableaux à partir des résultats ABC
# (resultats/abc/). Ajouter --abc pour relancer aussi l'inférence (tous les scénarios).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"          # code/scripts
# environnement Python créé par run_abc_mac.sh s'il existe, sinon python3 du système
PY="$HOME/.venvs/roka/bin/python"; [ -x "$PY" ] || PY=python3
if [ "$1" = "--abc" ]; then "$PY" "$HERE/run_abc.py"; fi
"$PY" "$HERE/compute_posterior.py" --force
for f in fig1_model_structure fig2_scenarios fig3_fit fig4_dynamics_phylogeny fig5_routes \
         fig6_hidden_hcv figS1_model_evidence figS2_posterior_distributions figS3_correlations; do
  "$PY" "$HERE/$f.py"
done
"$PY" "$HERE/parameter_table.py"
"$PY" "$HERE/make_tables.py"
# Les deux documents (.docx) de la racine sont maintenus directement dans Word ; ce script
# ne les régénère pas. Il met à jour les figures, les tableaux et toutes les valeurs citées
# (resultats/numbers.json), à reporter ensuite dans les documents.
echo "Figures, tableaux et valeurs à jour. Les .docx de la racine ne sont pas régénérés."
