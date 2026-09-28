"""Lecture des données (data/Roka_data.txt, data/study_design.json).

Roka_data.txt : la colonne Time donne la borne basse (jours) de la classe de
récence. VIH : 0 = [0-30), 30 = [30-100), 100 = [100-130), 130 = >=130 jours.
VHC (co-infectés) : 0 = [0-60), 60 = [60-180), 180 = >=180 jours.
"""
import json
import numpy as np
from .paths import DATA

HIV_ROWS = [0, 30, 100, 130]
HCV_ROWS = [0, 60, 180]


def load_design():
    return json.load(open(DATA / "study_design.json"))


def load_recency():
    d = np.genfromtxt(DATA / "Roka_data.txt", names=True)
    t = d["Time"].astype(int)
    hiv = np.array([[d[c][t == r][0] for r in HIV_ROWS] for c in ("hiv15", "hiv65", "hivOld")], float)
    hcv = np.array([[d[c][t == r][0] for r in HCV_ROWS] for c in ("hcv15", "hcv65", "hcvOld")], float)
    return hiv, hcv
