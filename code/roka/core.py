"""
Roka VIH + VHC : modèle individu-centré (IBM) au niveau de l'injection,
compilé avec numba, et modèle champ moyen (ODE) correspondant.

Unité de temps : le JOUR.

Mécanisme (une injection = un contact praticien-villageois)
-----------------------------------------------------------
* Un villageois de la classe d'âge a reçoit des injections au taux gamma_a.
  Chaque injection utilise une aiguille tirée au hasard dans un pool de
  n_needles aiguilles.
* Si l'aiguille porte le VIH (resp. le VHC) et que l'hôte est négatif,
  l'hôte est infecté avec la probabilité beta_H (resp. beta_C). Les deux
  tirages sont indépendants : une aiguille co-contaminée peut transmettre les
  deux virus en une seule injection (probabilité beta_H * beta_C).
* Si l'hôte porte le VIH (resp. le VHC) et que l'aiguille est propre pour ce
  virus, l'aiguille est contaminée avec la probabilité omega_A ou omega_U
  selon le stade VIH (resp. omega_C).
* Une aiguille perd le VIH (resp. le VHC) au taux sigma_H (resp. sigma_C),
  soit une persistance moyenne de 1/sigma jours.
* Transmission sexuelle du VIH chez les adultes (15-64 ans), action de masse :
  taux b_SI * (adultes en phase aiguë) + b_SU * (adultes en phase latente).
* Phase aiguë VIH : sortie au taux theta = 1/30 par jour (fixé, comme dans le
  script de 2016).
* Pas de mortalité ni de démographie (fenêtre de 1 à 3 ans, comme en 2016).

Introduction (HIV at t = 0, HCV at t = tau_C, tau_C may be negative)
* scénario 1, source persistante : pendant [0, P_i) pour le VIH et
  [tau_C, tau_C + P_i) pour le VHC, une fraction pi des injections utilise
  du matériel contaminé par la souche introduite (modèle V2).
* scénario 2, individus importés au taux pi par an pendant les mêmes périodes.
* scénario 3, cas index : un seul adulte infecté de longue date (VIH en phase
  latente) à t = 0, un seul adulte infecté par le VHC à t = tau_C. Toute la
  suite passe par les seringues partagées entre patients et la voie sexuelle.

* scénario 4, flacons multidoses : cas index comme le scénario 3 ; une
  fraction pi des injections utilise un des n_vial flacons, qui peut être
  contaminé par un patient infecté (seringue réutilisée) et contamine alors
  les patients suivants ; chaque flacon contaminé est remplacé par un flacon
  neuf au taux 1 / vial_life.
Cas index patient fréquent (gam_idx > 0) : le cas index VIH reçoit des
injections au taux gam_idx (par jour), différent de celui des autres adultes.
Options : exposition hétérogène (fraction f_hi de chaque âge reçoit kappa fois
plus d'injections), VHC déjà présent (fraction p_C0 infectée de longue date),
date d'arrêt du praticien stop_d (jours avant la dernière séquence).

Arrêt du praticien : toutes les injections cessent à t_max - stop_d (défaut T_STOP_DAYS)
(arrestation fin décembre 2014, t_max = 28 janvier 2015, dernière séquence).
Le nombre de seringues n_syr est un paramètre.

Chaque infection et chaque contamination d'aiguille crée une lignée virale
(table des lignées) : la généalogie est obtenue sans reconstruction a
posteriori.
"""

import numpy as np
from numba import njit

# indices des paramètres
PNAMES = ["beta_H", "beta_C", "sigma_H", "sigma_C", "omega_A", "omega_U",
          "omega_C", "g1", "g2", "g3", "t_max", "P_i", "pi", "b_SI", "b_SU",
          "theta", "tau_C", "n_syr", "mu", "sig_VH", "sig_VC", "kappa", "f_hi",
          "p_C0", "stop_d", "gam_idx", "n_vial", "vial_life"]
P = {k: i for i, k in enumerate(PNAMES)}

# compartiments d'hôtes : S, IH, UH, IC, IB, UB  (x 3 âges)
H_HIV = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
H_HCV = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
# aiguilles : N00, NH, NC, NHC
N_H = np.array([0, 1, 0, 1], dtype=np.int64)
N_C = np.array([0, 0, 1, 1], dtype=np.int64)
ADULT = 1

# bornes de récence (jours) : VIH [0-30), [30-100), [100-130), >=130
HIV_BINS = np.array([30.0, 100.0, 130.0])
# VHC [0-60), [60-180), >=180
HCV_BINS = np.array([60.0, 180.0])

ROUTE_INDEX, ROUTE_NEEDLE, ROUTE_SEX, ROUTE_EXT, ROUTE_VIAL, ROUTE_PRE = 0, 1, 2, 3, 4, 5
T_STOP_DAYS = 28.0        # 31 décembre 2014 -> 28 janvier 2015
INDEX_AGE_DAYS = 3650.0   # le cas index est infecté de longue date
KIND_ROOT, KIND_HOST, KIND_NEEDLE = 0, 1, 2


@njit(cache=True)
def _comp(hiv, hcv):
    return hiv + 3 * hcv


@njit(cache=True)
def _needle_comp(nh, nc):
    return nh + 2 * nc


@njit(cache=True)
def _pany(hv, hc, nh, nc, beta_H, beta_C, om_A, om_U, om_C):
    """Probabilité qu'une injection (hôte hv/hc, matériel nh/nc) change quelque chose."""
    q = 1.0
    if hv == 0 and nh == 1:
        q *= 1.0 - beta_H
    if hc == 0 and nc == 1:
        q *= 1.0 - beta_C
    if hv > 0 and nh == 0:
        q *= 1.0 - (om_A if hv == 1 else om_U)
    if hc == 1 and nc == 0:
        q *= 1.0 - om_C
    return 1.0 - q


@njit(cache=True)
def _draw(hv, hc, nh, nc, beta_H, beta_C, om_A, om_U, om_C):
    """Tirages indépendants conditionnés à au moins un succès.
    Retourne (infection VIH, infection VHC, contamination VIH, contamination VHC)."""
    probs = np.zeros(4)
    kinds = np.zeros(4, np.int64)
    K = 0
    if hv == 0 and nh == 1:
        probs[K] = beta_H
        kinds[K] = 0
        K += 1
    if hc == 0 and nc == 1:
        probs[K] = beta_C
        kinds[K] = 1
        K += 1
    if hv > 0 and nh == 0:
        probs[K] = om_A if hv == 1 else om_U
        kinds[K] = 2
        K += 1
    if hc == 1 and nc == 0:
        probs[K] = om_C
        kinds[K] = 3
        K += 1
    out = np.zeros(4, np.bool_)
    qq = 1.0
    for kk in range(K):
        qq *= 1.0 - probs[kk]
    w = np.random.random() * (1.0 - qq)
    acc = 0.0
    run = 1.0
    first = K - 1
    for kk in range(K):
        acc += run * probs[kk]
        if w < acc:
            first = kk
            break
        run *= 1.0 - probs[kk]
    out[kinds[first]] = True
    for kk in range(first + 1, K):
        if np.random.random() < probs[kk]:
            out[kinds[kk]] = True
    return out


@njit(cache=True)
def _move(members, gcount, gpos, h_comp, hid, g, g2, NA):
    if g2 == g:
        return
    last = members[g, gcount[g] - 1]
    members[g, gpos[hid]] = last
    gpos[last] = gpos[hid]
    gcount[g] -= 1
    members[g2, gcount[g2]] = hid
    gpos[hid] = gcount[g2]
    gcount[g2] += 1
    h_comp[hid] = g2 // NA


@njit(cache=True)
def simulate(par, n_cls, gamc, cls_age, idx_cls, n_needles, n_vials, scenario, seed,
             pool_on=True, sex_on=True, max_lin=300000, max_events=5000000):
    """Simulation IBM.

    n_cls : effectifs par classe ; cls_age : classe d'âge de chaque classe ;
    gamc : taux d'injection par classe (par jour). idx_cls : classe du cas index
    (patient fréquent, -1 si aucune). n_vials : nombre de flacons multidoses
    (scénario 4). pool_on : transmission par les seringues partagées (les
    flacons restent actifs) ; sex_on : voie sexuelle.
    Retourne (ok, lignées..., tableaux des hôtes...)."""
    np.random.seed(seed)
    NA = n_cls.shape[0]
    NG = 6 * NA
    beta_H, beta_C = par[0], par[1]
    sig_H, sig_C = par[2], par[3]
    om_A, om_U, om_C = par[4], par[5], par[6]
    t_max, P_i, pi = par[10], par[11], par[12]
    b_SI, b_SU, theta = par[13], par[14], par[15]
    tau_C = par[16]
    sig_VH, sig_VC = par[19], par[20]
    p_C0 = par[23]
    r_V = 1.0 / par[27] if par[27] > 0 else 0.0     # remplacement des flacons
    t_stop = t_max - par[24]          # arrêt des injections (arrestation)
    doneH = False
    doneC = False
    adult = np.zeros(NA, np.bool_)
    for k in range(NA):
        adult[k] = cls_age[k] == 1

    n_host = 0
    for k in range(NA):
        n_host += n_cls[k]
    # ---- hôtes
    h_age = np.empty(n_host, np.int64)        # classe (âge = classe % 3)
    h_comp = np.zeros(n_host, np.int64)
    h_lin = -np.ones((2, n_host), np.int64)
    members = -np.ones((NG, n_host), np.int64)   # groupe = comp * NA + classe
    gcount = np.zeros(NG, np.int64)
    gpos = np.zeros(n_host, np.int64)
    i = 0
    for k in range(NA):
        for j in range(n_cls[k]):
            h_age[i] = k
            members[k, gcount[k]] = i
            gpos[i] = gcount[k]
            gcount[k] += 1
            i += 1
    # ---- aiguilles du lot
    n_lin = -np.ones((2, n_needles), np.int64)
    nmem = -np.ones((4, n_needles), np.int64)
    ncount = np.zeros(4, np.int64)
    npos = np.zeros(n_needles, np.int64)
    for j in range(n_needles):
        nmem[0, j] = j
        npos[j] = j
    ncount[0] = n_needles
    # ---- flacons multidoses (scénario 4)
    nV = max(n_vials, 1)
    v_lin = -np.ones((2, nV), np.int64)
    vmem = -np.ones((4, nV), np.int64)
    vcount = np.zeros(4, np.int64)
    vpos = np.zeros(nV, np.int64)
    for j in range(n_vials):
        vmem[0, j] = j
        vpos[j] = j
    vcount[0] = n_vials
    # ---- lignées
    L_parent = -np.ones(max_lin, np.int64)
    L_birth = np.zeros(max_lin)
    L_end = np.full(max_lin, np.inf)
    L_kind = np.zeros(max_lin, np.int64)
    L_owner = -np.ones(max_lin, np.int64)
    L_age = -np.ones(max_lin, np.int64)
    L_route = np.zeros(max_lin, np.int64)
    L_virus = np.zeros(max_lin, np.int64)
    L_src = -np.ones(max_lin, np.int64)
    nL = 0
    root = np.zeros(2, np.int64)
    for v in range(2):
        root[v] = nL
        L_kind[nL] = KIND_ROOT
        L_virus[nL] = v
        nL += 1

    t = min(0.0, tau_C)
    # ---- VHC déjà présent (fraction p_C0 de la population, infection ancienne)
    if p_C0 > 0.0:
        for hid in range(n_host):
            if np.random.random() < p_C0:
                k = h_age[hid]
                h_lin[1, hid] = nL
                L_parent[nL] = root[1]
                L_birth[nL] = t - INDEX_AGE_DAYS
                L_kind[nL] = KIND_HOST
                L_owner[nL] = hid
                L_age[nL] = cls_age[k]
                L_route[nL] = ROUTE_PRE
                L_virus[nL] = 1
                nL += 1
                _move(members, gcount, gpos, h_comp, hid, k, 3 * NA + k, NA)

    n_ev = 0
    ok = True
    R_inj = np.zeros((NG, 4))
    R_ext = np.zeros(NG)
    R_vial = np.zeros((NG, 4))
    while True:
        # ---- cas index (scénarios 3 et 4), traités à leur date exacte
        if (scenario == 3 or scenario == 4) and ((not doneH and t >= 0.0) or (not doneC and t >= tau_C)):
            v = 0 if (not doneH and t >= 0.0) else 1
            if v == 0:
                doneH = True
            else:
                doneC = True
            cand = 0
            for c in range(6):
                if (v == 0 and H_HIV[c] == 0) or (v == 1 and H_HCV[c] == 0):
                    for k in range(NA):
                        if (v == 0 and idx_cls >= 0 and k == idx_cls) or \
                           ((v == 1 or idx_cls < 0) and adult[k] and k != idx_cls):
                            cand += gcount[c * NA + k]
            if cand > 0:
                w = np.random.randint(cand)
                g = -1
                for c in range(6):
                    if g >= 0:
                        break
                    if (v == 0 and H_HIV[c] == 0) or (v == 1 and H_HCV[c] == 0):
                        for k in range(NA):
                            if (v == 0 and idx_cls >= 0 and k == idx_cls) or \
                               ((v == 1 or idx_cls < 0) and adult[k] and k != idx_cls):
                                gg = c * NA + k
                                if w < gcount[gg]:
                                    g = gg
                                    break
                                w -= gcount[gg]
                hid = members[g, np.random.randint(gcount[g])]
                c = g // NA
                k = g % NA
                if v == 0:
                    ng2 = _comp(2, H_HCV[c]) * NA + k
                else:
                    ng2 = _comp(H_HIV[c], 1) * NA + k
                h_lin[v, hid] = nL
                L_parent[nL] = root[v]
                L_birth[nL] = t - INDEX_AGE_DAYS
                L_kind[nL] = KIND_HOST
                L_owner[nL] = hid
                L_age[nL] = cls_age[k]
                L_route[nL] = ROUTE_INDEX
                L_virus[nL] = v
                nL += 1
                _move(members, gcount, gpos, h_comp, hid, g, ng2, NA)
            continue
        inj_on = pool_on and (t < t_stop)
        introH = (t >= 0.0) and (t < P_i) and (scenario == 1 or scenario == 2)
        introC = (t >= tau_C) and (t < tau_C + P_i) and (scenario == 1 or scenario == 2)
        in_intro = (introH or introC) and (t < t_stop)
        t_next = np.inf
        for tb in (0.0, P_i, tau_C, tau_C + P_i, t_stop):
            if tb > t and tb < t_next:
                t_next = tb
        # ---------------- taux
        tot = 0.0
        pool_fac = 1.0
        if scenario == 1 and in_intro:
            pool_fac = 1.0 - pi
        if scenario == 4:
            pool_fac = 1.0 - pi
        for g in range(NG):
            c = g // NA
            k = g % NA
            ng = gcount[g]
            hv = H_HIV[c]
            hc = H_HCV[c]
            for n in range(4):
                R_inj[g, n] = 0.0
                if ng == 0 or ncount[n] == 0 or not inj_on:
                    continue
                p_any = _pany(hv, hc, N_H[n], N_C[n], beta_H, beta_C, om_A, om_U, om_C)
                if p_any > 0.0:
                    r = gamc[k] * ng * ncount[n] / n_needles * p_any * pool_fac
                    R_inj[g, n] = r
                    tot += r
            R_ext[g] = 0.0
            if scenario == 1 and in_intro and ng > 0:
                q = 1.0
                if hv == 0 and introH:
                    q *= 1.0 - beta_H
                if hc == 0 and introC:
                    q *= 1.0 - beta_C
                r = gamc[k] * ng * pi * (1.0 - q)
                R_ext[g] = r
                tot += r
            for n in range(4):
                R_vial[g, n] = 0.0
                if scenario != 4 or t >= t_stop or ng == 0 or vcount[n] == 0:
                    continue
                p_any = _pany(hv, hc, N_H[n], N_C[n], beta_H, beta_C, om_A, om_U, om_C)
                if p_any > 0.0:
                    r = gamc[k] * ng * pi * vcount[n] / n_vials * p_any
                    R_vial[g, n] = r
                    tot += r
        # sexuel (adultes)
        A2 = 0
        C2 = 0
        rec2 = 0
        for k in range(NA):
            if adult[k]:
                A2 += gcount[1 * NA + k] + gcount[4 * NA + k]
                C2 += gcount[2 * NA + k] + gcount[5 * NA + k]
                rec2 += gcount[0 * NA + k] + gcount[3 * NA + k]
        R_sexA = 0.0
        R_sexC = 0.0
        if sex_on:
            R_sexA = b_SI * A2 * rec2
            R_sexC = b_SU * C2 * rec2
        tot += R_sexA + R_sexC
        # progression aiguë -> latente
        nacute = 0
        for k in range(NA):
            nacute += gcount[1 * NA + k] + gcount[4 * NA + k]
        R_prog = theta * nacute
        tot += R_prog
        # décontamination (lot, puis flacon)
        R_clH = sig_H * (ncount[1] + ncount[3])
        R_clC = sig_C * (ncount[2] + ncount[3])
        R_vrep = r_V * (vcount[1] + vcount[2] + vcount[3])
        tot += R_clH + R_clC + R_vrep
        # importations (scénario 2)
        R_impH = 0.0
        R_impC = 0.0
        if scenario == 2:
            if introH and rec2 > 0:
                R_impH = pi / 365.0
            if introC:
                R_impC = pi / 365.0
        tot += R_impH + R_impC

        # ---------------- temps
        if tot <= 0.0:
            if t_next < t_max:
                t = t_next
                continue
            break
        dt = -np.log(np.random.random()) / tot
        if t + dt > t_max:
            break
        if t + dt > t_next:
            t = t_next
            continue
        t += dt
        n_ev += 1
        if n_ev > max_events or nL > max_lin - 10:
            ok = False
            break

        # ---------------- choix de l'événement
        u = np.random.random() * tot
        done = False
        # injections avec le lot de seringues
        for g in range(NG):
            if done:
                break
            for n in range(4):
                if u < R_inj[g, n]:
                    hid = members[g, np.random.randint(gcount[g])]
                    nid = nmem[n, np.random.randint(ncount[n])]
                    c = g // NA
                    k = g % NA
                    hv = H_HIV[c]
                    hc = H_HCV[c]
                    s = _draw(hv, hc, N_H[n], N_C[n], beta_H, beta_C, om_A, om_U, om_C)
                    new_hv, new_hc, new_nh, new_nc = hv, hc, N_H[n], N_C[n]
                    for vv in range(2):
                        if s[vv]:
                            if vv == 0:
                                new_hv = 1
                            else:
                                new_hc = 1
                            h_lin[vv, hid] = nL
                            L_parent[nL] = n_lin[vv, nid]
                            L_birth[nL] = t
                            L_kind[nL] = KIND_HOST
                            L_owner[nL] = hid
                            L_age[nL] = cls_age[k]
                            L_route[nL] = ROUTE_NEEDLE
                            L_virus[nL] = vv
                            L_src[nL] = nid
                            nL += 1
                        if s[2 + vv]:
                            if vv == 0:
                                new_nh = 1
                            else:
                                new_nc = 1
                            n_lin[vv, nid] = nL
                            L_parent[nL] = h_lin[vv, hid]
                            L_birth[nL] = t
                            L_kind[nL] = KIND_NEEDLE
                            L_owner[nL] = nid
                            L_virus[nL] = vv
                            L_src[nL] = hid
                            nL += 1
                    _move(members, gcount, gpos, h_comp, hid, g, _comp(new_hv, new_hc) * NA + k, NA)
                    nn2 = _needle_comp(new_nh, new_nc)
                    if nn2 != n:
                        last = nmem[n, ncount[n] - 1]
                        nmem[n, npos[nid]] = last
                        npos[last] = npos[nid]
                        ncount[n] -= 1
                        nmem[nn2, ncount[nn2]] = nid
                        npos[nid] = ncount[nn2]
                        ncount[nn2] += 1
                    done = True
                    break
                u -= R_inj[g, n]
        if done:
            continue
        # injections avec un flacon multidose (scénario 4)
        for g in range(NG):
            if done:
                break
            for n in range(4):
                if u < R_vial[g, n]:
                    hid = members[g, np.random.randint(gcount[g])]
                    vid = vmem[n, np.random.randint(vcount[n])]
                    c = g // NA
                    k = g % NA
                    hv = H_HIV[c]
                    hc = H_HCV[c]
                    s = _draw(hv, hc, N_H[n], N_C[n], beta_H, beta_C, om_A, om_U, om_C)
                    new_hv, new_hc, new_nh, new_nc = hv, hc, N_H[n], N_C[n]
                    for vv in range(2):
                        if s[vv]:
                            if vv == 0:
                                new_hv = 1
                            else:
                                new_hc = 1
                            h_lin[vv, hid] = nL
                            L_parent[nL] = v_lin[vv, vid]
                            L_birth[nL] = t
                            L_kind[nL] = KIND_HOST
                            L_owner[nL] = hid
                            L_age[nL] = cls_age[k]
                            L_route[nL] = ROUTE_VIAL
                            L_virus[nL] = vv
                            L_src[nL] = -2 - vid
                            nL += 1
                        if s[2 + vv]:
                            if vv == 0:
                                new_nh = 1
                            else:
                                new_nc = 1
                            v_lin[vv, vid] = nL
                            L_parent[nL] = h_lin[vv, hid]
                            L_birth[nL] = t
                            L_kind[nL] = KIND_NEEDLE
                            L_owner[nL] = n_needles + vid
                            L_virus[nL] = vv
                            L_src[nL] = hid
                            nL += 1
                    _move(members, gcount, gpos, h_comp, hid, g, _comp(new_hv, new_hc) * NA + k, NA)
                    nn2 = _needle_comp(new_nh, new_nc)
                    if nn2 != n:
                        last = vmem[n, vcount[n] - 1]
                        vmem[n, vpos[vid]] = last
                        vpos[last] = vpos[vid]
                        vcount[n] -= 1
                        vmem[nn2, vcount[nn2]] = vid
                        vpos[vid] = vcount[nn2]
                        vcount[nn2] += 1
                    done = True
                    break
                u -= R_vial[g, n]
        if done:
            continue
        # injections avec la source contaminée (scénario 1)
        for g in range(NG):
            if u < R_ext[g]:
                hid = members[g, np.random.randint(gcount[g])]
                c = g // NA
                k = g % NA
                hv = H_HIV[c]
                hc = H_HCV[c]
                pH = beta_H if (hv == 0 and introH) else 0.0
                pC = beta_C if (hc == 0 and introC) else 0.0
                p_any = 1.0 - (1.0 - pH) * (1.0 - pC)
                w = np.random.random() * p_any
                sH = False
                sC = False
                if w < pH:
                    sH = True
                    if np.random.random() < pC:
                        sC = True
                else:
                    sC = True
                new_hv, new_hc = hv, hc
                for vv in range(2):
                    if (vv == 0 and sH) or (vv == 1 and sC):
                        if vv == 0:
                            new_hv = 1
                        else:
                            new_hc = 1
                        h_lin[vv, hid] = nL
                        L_parent[nL] = root[vv]
                        L_birth[nL] = t
                        L_kind[nL] = KIND_HOST
                        L_owner[nL] = hid
                        L_age[nL] = cls_age[k]
                        L_route[nL] = ROUTE_EXT
                        L_virus[nL] = vv
                        nL += 1
                _move(members, gcount, gpos, h_comp, hid, g, _comp(new_hv, new_hc) * NA + k, NA)
                done = True
                break
            u -= R_ext[g]
        if done:
            continue
        # sexuel
        if u < R_sexA + R_sexC:
            stage = 1 if u < R_sexA else 2
            w = np.random.randint(rec2)
            g = -1
            for c in (0, 3):
                if g >= 0:
                    break
                for k in range(NA):
                    if adult[k]:
                        gg = c * NA + k
                        if w < gcount[gg]:
                            g = gg
                            break
                        w -= gcount[gg]
            hid = members[g, np.random.randint(gcount[g])]
            nd = A2 if stage == 1 else C2
            w = np.random.randint(nd)
            gd = -1
            for c in ((1, 4) if stage == 1 else (2, 5)):
                if gd >= 0:
                    break
                for k in range(NA):
                    if adult[k]:
                        gg = c * NA + k
                        if w < gcount[gg]:
                            gd = gg
                            break
                        w -= gcount[gg]
            donor = members[gd, np.random.randint(gcount[gd])]
            k = g % NA
            h_lin[0, hid] = nL
            L_parent[nL] = h_lin[0, donor]
            L_birth[nL] = t
            L_kind[nL] = KIND_HOST
            L_owner[nL] = hid
            L_age[nL] = cls_age[k]
            L_route[nL] = ROUTE_SEX
            L_virus[nL] = 0
            L_src[nL] = donor
            nL += 1
            _move(members, gcount, gpos, h_comp, hid, g, _comp(1, H_HCV[g // NA]) * NA + k, NA)
            continue
        u -= R_sexA + R_sexC
        # progression
        if u < R_prog:
            w = np.random.randint(nacute)
            g = -1
            for c in (1, 4):
                if g >= 0:
                    break
                for k in range(NA):
                    gg = c * NA + k
                    if w < gcount[gg]:
                        g = gg
                        break
                    w -= gcount[gg]
            hid = members[g, np.random.randint(gcount[g])]
            _move(members, gcount, gpos, h_comp, hid, g, g + NA, NA)
            continue
        u -= R_prog
        # décontamination d'une seringue
        if u < R_clH + R_clC:
            if u < R_clH:
                v = 0
                c1, c2 = 1, 3
            else:
                v = 1
                c1, c2 = 2, 3
            if np.random.random() * (ncount[c1] + ncount[c2]) < ncount[c1]:
                n = c1
            else:
                n = c2
            nid = nmem[n, np.random.randint(ncount[n])]
            L_end[n_lin[v, nid]] = t
            n_lin[v, nid] = -1
            nn2 = n - 1 if v == 0 else n - 2
            last = nmem[n, ncount[n] - 1]
            nmem[n, npos[nid]] = last
            npos[last] = npos[nid]
            ncount[n] -= 1
            nmem[nn2, ncount[nn2]] = nid
            npos[nid] = ncount[nn2]
            ncount[nn2] += 1
            continue
        u -= R_clH + R_clC
        # remplacement d'un flacon contaminé (flacon neuf, propre)
        if u < R_vrep:
            w = np.random.randint(vcount[1] + vcount[2] + vcount[3])
            n = 1
            while w >= vcount[n]:
                w -= vcount[n]
                n += 1
            vid = vmem[n, w]
            for vv in range(2):
                if v_lin[vv, vid] >= 0:
                    L_end[v_lin[vv, vid]] = t
                    v_lin[vv, vid] = -1
            last = vmem[n, vcount[n] - 1]
            vmem[n, vpos[vid]] = last
            vpos[last] = vpos[vid]
            vcount[n] -= 1
            vmem[0, vcount[0]] = vid
            vpos[vid] = vcount[0]
            vcount[0] += 1
            continue
        u -= R_vrep
        # importations (scénario 2)
        if u < R_impH:
            w = np.random.randint(rec2)
            g = -1
            for c in (0, 3):
                if g >= 0:
                    break
                for k in range(NA):
                    if adult[k]:
                        gg = c * NA + k
                        if w < gcount[gg]:
                            g = gg
                            break
                        w -= gcount[gg]
            vv = 0
            ng2 = _comp(1, H_HCV[g // NA]) * NA + g % NA
        else:
            nn = 0
            for c in (0, 1, 2):
                for k in range(NA):
                    if adult[k]:
                        nn += gcount[c * NA + k]
            if nn == 0:
                continue
            w = np.random.randint(nn)
            g = -1
            for c in (0, 1, 2):
                if g >= 0:
                    break
                for k in range(NA):
                    if adult[k]:
                        gg = c * NA + k
                        if w < gcount[gg]:
                            g = gg
                            break
                        w -= gcount[gg]
            vv = 1
            ng2 = _comp(H_HIV[g // NA], 1) * NA + g % NA
        hid = members[g, np.random.randint(gcount[g])]
        h_lin[vv, hid] = nL
        L_parent[nL] = root[vv]
        L_birth[nL] = t
        L_kind[nL] = KIND_HOST
        L_owner[nL] = hid
        L_age[nL] = cls_age[g % NA]
        L_route[nL] = ROUTE_INDEX
        L_virus[nL] = vv
        nL += 1
        _move(members, gcount, gpos, h_comp, hid, g, ng2, NA)

    return (ok, nL, L_parent[:nL].copy(), L_birth[:nL].copy(), L_end[:nL].copy(),
            L_kind[:nL].copy(), L_owner[:nL].copy(), L_age[:nL].copy(),
            L_route[:nL].copy(), L_virus[:nL].copy(), L_src[:nL].copy(),
            h_age, h_comp, h_lin, gcount, ncount, n_ev)


@njit(cache=True)
def summary_counts(test_time, h_age, h_comp, h_lin, L_birth):
    """Effectifs par âge, statut et récence évalués à la date de test de
    chaque individu (test_time, un par hôte).

    Retourne un tableau (3, 12) :
      col 0-3  : VIH+ par récence  [0-30), [30-100), [100-130), >=130 jours
      col 4-6  : VIH+VHC+ par récence VHC [0-60), [60-180), >=180
      col 7    : VIH seul
      col 8    : co-infectés
      col 9    : VHC seul
      col 10   : VIH total
      col 11   : VHC total
    """
    out = np.zeros((3, 12))
    for i in range(h_age.shape[0]):
        a = h_age[i]
        T = test_time[i]
        hv = h_lin[0, i] >= 0 and L_birth[h_lin[0, i]] <= T
        hc = h_lin[1, i] >= 0 and L_birth[h_lin[1, i]] <= T
        if hv:
            d = T - L_birth[h_lin[0, i]]
            if d < 30.0:
                out[a, 0] += 1
            elif d < 100.0:
                out[a, 1] += 1
            elif d < 130.0:
                out[a, 2] += 1
            else:
                out[a, 3] += 1
            out[a, 10] += 1
        if hc:
            out[a, 11] += 1
        if hv and hc:
            d = T - L_birth[h_lin[1, i]]
            if d < 60.0:
                out[a, 4] += 1
            elif d < 180.0:
                out[a, 5] += 1
            else:
                out[a, 6] += 1
            out[a, 8] += 1
        elif hv:
            out[a, 7] += 1
        elif hc:
            out[a, 9] += 1
    return out


@njit(cache=True)
def genealogy_branches(L_parent, L_birth, tip_lin, tip_time):
    """Généalogie d'un échantillon de lignées (tip_lin) échantillonnées aux
    temps tip_time. Chaque lignée continue après avoir transmis : une
    transmission vers une lignée utile est un noeud à la date de naissance de
    la lignée fille. Retourne (longueurs des branches en jours, nombre de
    feuilles sous chaque branche, date du MRCA)."""
    nL = L_parent.shape[0]
    n = tip_lin.shape[0]
    useful = np.zeros(nL, np.bool_)
    tip_of = -np.ones(nL, np.int64)
    for i in range(n):
        x = tip_lin[i]
        tip_of[x] = i
        while x >= 0 and not useful[x]:
            useful[x] = True
            x = L_parent[x]
    nch = np.zeros(nL + 1, np.int64)
    for x in range(nL):
        if useful[x] and L_parent[x] >= 0:
            nch[L_parent[x] + 1] += 1
    off = np.cumsum(nch)
    fill = off[:-1].copy()
    kids = np.empty(off[-1], np.int64)
    for x in range(nL):
        if useful[x] and L_parent[x] >= 0:
            p = L_parent[x]
            kids[fill[p]] = x
            fill[p] += 1
    sub_size = np.zeros(nL, np.int64)
    sub_top = np.zeros(nL)
    blen = np.zeros(2 * n)
    bk = np.zeros(2 * n, np.int64)
    mt = np.zeros(n)
    nb = 0
    nm = 0
    tmrca = np.inf
    for x in range(nL - 1, -1, -1):
        if not useful[x]:
            continue
        m = off[x + 1] - off[x] + (1 if tip_of[x] >= 0 else 0)
        et = np.empty(m)
        es = np.empty(m, np.int64)
        etop = np.empty(m)
        j = 0
        for q in range(off[x], off[x + 1]):
            c = kids[q]
            et[j] = L_birth[c]
            es[j] = sub_size[c]
            etop[j] = sub_top[c]
            j += 1
        if tip_of[x] >= 0:
            et[j] = tip_time[tip_of[x]]
            es[j] = 1
            etop[j] = tip_time[tip_of[x]]
        order = np.argsort(-et)
        rs = 0
        rt = 0.0
        for q in order:
            if rs == 0:
                rs = es[q]
                rt = etop[q]
            else:
                blen[nb] = etop[q] - et[q]
                bk[nb] = es[q]
                nb += 1
                blen[nb] = rt - et[q]
                bk[nb] = rs
                nb += 1
                rs += es[q]
                rt = et[q]
                mt[nm] = et[q]
                nm += 1
                if et[q] < tmrca:
                    tmrca = et[q]
        sub_size[x] = rs
        sub_top[x] = rt
    return blen[:nb], bk[:nb], tmrca, mt[:nm]


def ltt_peak(merge_times, t_end, window=30):
    """Date (jours) du maximum du nombre de noeuds internes, lissé sur
    `window` jours : analogue du pic du graphe LTT non cumulé."""
    t0 = int(np.floor(merge_times.min()))
    t1 = int(np.ceil(t_end)) + 1
    cnt = np.histogram(merge_times, bins=np.arange(t0, t1 + 1))[0]
    # somme glissante en entiers (pas de division) : les ex aequo sont exacts et
    # argmax prend le premier sur toutes les machines, contrairement à une
    # moyenne en flottants dont les arrondis dépendent du processeur
    sm = np.convolve(cnt.astype(np.int64), np.ones(window, np.int64), mode="same")
    return t0 + int(np.argmax(sm)) + 0.5


def load_test_offsets():
    """Distribution des dates de test (jours avant la dernière séquence)."""
    import json
    from .paths import DATA
    d = json.load(open(DATA / "test_dates.json"))
    off = np.array(d["days_before_last"], float)
    w = np.array(d["counts"], float)
    return off, w / w.sum()


_TEST = None
N_SEQ = 199               # séquences env du cluster épidémique
SEQ_LEN = None            # longueur effective de l'alignement (data/observed_sequence_stats.json)


class Sim:
    """Enveloppe Python : résultats d'une simulation IBM.

    Le nombre de seringues est le paramètre n_syr (arrondi à l'entier).
    Chaque individu reçoit une date de test tirée dans la distribution des
    dates de prélèvement, et son statut et sa récence sont évalués à cette date.
    """

    def __init__(self, par, n_age, scenario, seed, pool_on=True, sex_on=True):
        global _TEST
        if _TEST is None:
            _TEST = load_test_offsets()
        par = np.asarray(par, float)
        n_needles = max(1, int(round(par[P["n_syr"]])))
        n_age = np.asarray(n_age, np.int64)
        gam = par[P["g1"]:P["g3"] + 1]
        f_hi, kappa = par[P["f_hi"]], par[P["kappa"]]
        if f_hi > 0 and kappa > 1:
            hi = np.round(f_hi * n_age).astype(np.int64)
            n_cls = np.concatenate([n_age - hi, hi])
            gamc = np.concatenate([gam, kappa * gam])
        else:
            n_cls, gamc = n_age.copy(), gam.copy()
        cls_age = np.arange(len(n_cls)) % 3
        idx_cls = -1
        if par[P["gam_idx"]] > 0:            # cas index patient fréquent (classe à part)
            n_cls[1] -= 1
            n_cls = np.append(n_cls, 1)
            gamc = np.append(gamc, par[P["gam_idx"]])
            cls_age = np.append(cls_age, 1)
            idx_cls = len(n_cls) - 1
        n_vials = max(1, int(round(par[P["n_vial"]]))) if int(scenario) == 4 else 0
        self.n_vials = n_vials
        r = simulate(par, n_cls.astype(np.int64), gamc.astype(float), cls_age.astype(np.int64),
                     idx_cls, n_needles, n_vials, int(scenario), int(seed), pool_on, sex_on)
        (self.ok, self.nL, self.parent, self.birth, self.end, self.kind,
         self.owner, self.age, self.route, self.virus, self.src,
         self.h_cls, self.h_comp, self.h_lin, self.gcount, self.ncount,
         self.n_ev) = r
        self.h_age = cls_age[self.h_cls]
        self.par = par
        self.n_needles = n_needles
        self.t_max = par[P["t_max"]]
        self.rng = np.random.default_rng(seed + 7_000_000_000)
        off, w = _TEST
        self.test_time = self.t_max - self.rng.choice(off, size=len(self.h_age), p=w)
        self.counts = summary_counts(self.test_time, self.h_age, self.h_comp,
                                     self.h_lin, self.birth)
        self.children = None

    def hiv_positive_at_test(self):
        hl = self.h_lin[0]
        pos = np.where(hl >= 0)[0]
        return pos[self.birth[hl[pos]] <= self.test_time[pos]]

    def sample_tips(self, n=N_SEQ, virus=0):
        """Lignées VIH de n individus positifs à leur date de test."""
        pos = self.hiv_positive_at_test()
        k = min(n, len(pos))
        pick = self.rng.choice(pos, size=k, replace=False) if k > 0 else pos[:0]
        return self.h_lin[virus][pick].astype(np.int64), self.test_time[pick].astype(float)

    def genealogy(self, n=N_SEQ):
        tl, tt = self.sample_tips(n)
        if len(tl) < 2:
            return None
        blen, bk, tmrca, mt = genealogy_branches(self.parent, self.birth, tl, tt)
        return dict(blen=blen, k=bk, n=len(tl), tmrca=tmrca, merge_times=mt,
                    tip_lin=tl, tip_time=tt)

    def phylo_stats(self, n=N_SEQ):
        """(tMRCA, date du pic de branchements), en jours avant la dernière
        séquence (28 janvier 2015)."""
        g = self.genealogy(n)
        if g is None or len(g["merge_times"]) == 0:
            return None, g
        return np.array([self.t_max - g["tmrca"],
                         self.t_max - ltt_peak(g["merge_times"], self.t_max)]), g

    def sequence_stats(self, mu, seq_len, n=N_SEQ):
        """Mutations sur la généalogie (sites infinis, horloge stricte mu en
        substitutions/site/an) : sites polymorphes S, singletons repliés,
        diversité nucléotidique moyenne par paire (en différences)."""
        g = self.genealogy(n)
        if g is None:
            return np.array([0.0, 0.0, 0.0]), g
        lam = mu / 365.0 * seq_len * g["blen"]
        m = self.rng.poisson(np.maximum(lam, 0.0))
        nn = g["n"]
        k = g["k"]
        S = m.sum()
        single = m[(k == 1) | (k == nn - 1)].sum()
        pi = (m * k * (nn - k)).sum() / (nn * (nn - 1) / 2)
        return np.array([S, single, pi], float), g

    def build_children(self):
        ch = [[] for _ in range(self.nL)]
        for i in range(self.nL):
            p = self.parent[i]
            if p >= 0:
                ch[p].append(i)
        self.children = ch
        return ch
