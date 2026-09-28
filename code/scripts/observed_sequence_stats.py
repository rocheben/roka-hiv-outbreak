"""Statistiques des séquences observées (cluster VIH env, 199 séquences) :
sites polymorphes, singletons repliés, nombre moyen de différences par paire.
Sites retenus : au moins 80% de bases A/C/G/T. Écrit data/observed_sequence_stats.json."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from roka.paths import DATA

seqs = [l.strip().upper() for l in open(DATA / "sequences" / "hiv_cluster.fasta") if not l.startswith(">")]
A = np.array([list(s) for s in seqs])
n, L = A.shape
valid = np.isin(A, list("ACGT"))
colok = valid.mean(0) >= 0.8
S = single = 0
for j in np.where(colok)[0]:
    u, c = np.unique(A[valid[:, j], j], return_counts=True)
    if len(u) > 1:
        S += 1
        if len(u) == 2 and c.min() == 1:
            single += 1
B, V = A[:, colok], valid[:, colok]
tot = npair = 0
for i in range(n - 1):
    d = ((B[i] != B[i + 1:]) & V[i] & V[i + 1:]).sum(1)
    tot += d.sum()
    npair += len(d)
out = {"_source": "199 séquences env C2V3 du cluster épidémique (data/sequences/hiv_cluster.fasta), "
                  "sites avec au moins 80% de bases A/C/G/T.",
       "n_sequences": int(n), "alignment_length": int(L), "effective_length": int(colok.sum()),
       "segregating_sites": int(S), "singletons_folded": int(single),
       "mean_pairwise_differences": float(tot / npair)}
json.dump(out, open(DATA / "observed_sequence_stats.json", "w"), indent=1)
print(out)
