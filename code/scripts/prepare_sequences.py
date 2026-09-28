"""Préparation des séquences env : renommage, dates, extraction du cluster
épidémique (Roka hors NCHADS171, 184, 185 et 116, isolés dans Rouet et al.)."""
import sys
from pathlib import Path
import dendropy

D = Path(__file__).resolve().parents[2] / "data" / "sequences"
EXCL = {"NCHADS171_roka", "NCHADS184_roka", "NCHADS185_roka", "NCHADS116_roka"}

recs, name, seq = [], None, []
for l in open(D / "env_roka_controls_aligned.fasta"):
    l = l.strip()
    if l.startswith(">"):
        if name:
            recs.append((name, "".join(seq)))
        name, seq = l[1:], []
    else:
        seq.append(l)
recs.append((name, "".join(seq)))
seqs = {}
for n, s in recs:
    p = n.split("|")
    seqs[p[0] + "_" + p[1]] = (s, p[3])

if "--extract" not in sys.argv:
    with open(D / "env_all_renamed.fasta", "w") as f, open(D / "dates.tsv", "w") as d:
        d.write("name\tdate\n")
        for k, (s, dt) in seqs.items():
            f.write(f">{k}\n{s}\n")
            d.write(f"{k}\t{dt}\n")
else:
    t = dendropy.Tree.get(path=str(D / "env_all_fasttree.nwk"), schema="newick", preserve_underscores=True)
    ctrl = [l for l in t.leaf_nodes() if l.taxon.label.endswith("_control")]
    t.reroot_at_edge(ctrl[0].edge, update_bipartitions=True)
    keep = [l.taxon for l in t.leaf_nodes() if l.taxon.label.endswith("_roka") and l.taxon.label not in EXCL]
    mrca = t.mrca(taxa=keep)
    assert len(mrca.leaf_nodes()) == len(keep), "le cluster n'est pas monophylétique"
    sub = dendropy.Tree(seed_node=mrca.extract_subtree(), taxon_namespace=t.taxon_namespace)
    sub.write(path=str(D / "hiv_cluster_ml.nwk"), schema="newick", suppress_rooting=True)
    with open(D / "hiv_cluster.fasta", "w") as f:
        for k in sorted(x.label for x in keep):
            f.write(f">{k}\n{seqs[k][0]}\n")
    print(len(keep), "séquences dans le cluster")
