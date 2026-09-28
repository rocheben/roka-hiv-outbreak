#!/usr/bin/env bash
# Régénère caches, tables, figures et documents à partir des résultats ABC
# (resultats/abc/). Ajouter --abc pour relancer aussi l'inférence (tous les scénarios).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"          # code/scripts
CODE="$(dirname "$HERE")"                        # code
# environnement Python créé par run_abc_mac.sh s'il existe, sinon python3 du système
PY="$HOME/.venvs/roka/bin/python"; [ -x "$PY" ] || PY=python3
if [ "$1" = "--abc" ]; then "$PY" "$HERE/run_abc.py"; fi
"$PY" "$HERE/compute_posterior.py" --force
for f in fig1_scenarios fig2_fit fig3_dynamics_phylogeny fig4_routes fig5_hidden_hcv \
         figS1_model_evidence figS2_posterior_distributions figS3_correlations; do
  "$PY" "$HERE/$f.py"
done
"$PY" "$HERE/parameter_table.py"
"$PY" "$HERE/make_tables.py"
# documents (nécessite Node.js et le paquet docx : cd code/manuscript && npm install)
if command -v node >/dev/null; then
  node "$CODE/manuscript/build_manuscript.js"
  node "$CODE/manuscript/build_supplement.js"
else
  echo "Node.js absent : résultats, figures et tableaux sont à jour, mais les deux .docx n'ont pas été régénérés."
  echo "Installer Node.js (https://nodejs.org ou 'brew install node'), puis : cd code/manuscript && npm install"
fi
