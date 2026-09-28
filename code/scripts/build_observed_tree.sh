#!/usr/bin/env bash
# Arbre des 199 séquences env du cluster épidémique (Rouet et al. 2018).
# 1) FastTree (GTR+Gamma) sur les séquences de Roka et les contrôles,
#    enracinement sur les contrôles, extraction du cluster (scripts Python ci-dessous)
# 2) datation TreeTime, horloge 4.1e-3 (écart type 9.3e-4) substitutions/site/an,
#    sans prior démographique (none) et avec prior skyline (skyline) pour la sensibilité.
# Nécessite FastTree, TreeTime (pip install phylo-treetime) et DendroPy.
set -e
cd "$(dirname "$0")/../../data/sequences"
python3 ../../code/scripts/prepare_sequences.py
FastTree -nt -gtr -gamma env_all_renamed.fasta > env_all_fasttree.nwk
python3 ../../code/scripts/prepare_sequences.py --extract
treetime --tree hiv_cluster_ml.nwk --aln hiv_cluster.fasta --dates dates.tsv \
  --clock-rate 0.0041 --clock-std-dev 0.00093 --keep-root --confidence --outdir treetime_hiv_none
treetime --tree hiv_cluster_ml.nwk --aln hiv_cluster.fasta --dates dates.tsv \
  --clock-rate 0.0041 --clock-std-dev 0.00093 --keep-root --coalescent skyline --confidence --outdir treetime_hiv_skyline


python3 ../../code/scripts/observed_sequence_stats.py
