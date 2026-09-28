"""Conversion entre temps du modèle (jours avant la dernière séquence) et dates."""
import datetime as dt
import json
from .paths import DATA

_D = json.load(open(DATA / "study_design.json"))
LAST = dt.date.fromisoformat(_D["last_sequence_date"])


def dec(d):
    y0, y1 = dt.date(d.year, 1, 1), dt.date(d.year + 1, 1, 1)
    return d.year + (d - y0).days / (y1 - y0).days


T_LAST = dec(LAST)


def ym(s):
    """'2013-09' -> année décimale au 15 du mois."""
    y, m = map(int, s.split("-"))
    return dec(dt.date(y, m, 15))


def todate(x):
    y = int(x)
    d0, d1 = dt.date(y, 1, 1), dt.date(y + 1, 1, 1)
    return d0 + dt.timedelta(days=(x - y) * (d1 - d0).days)


def years_before_to_date(yb):
    return todate(T_LAST - yb)


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def month_label(d):
    return f"{MONTHS[d.month - 1]} {d.year}"


def rouet():
    return json.load(open(DATA / "rouet2018_beast.json"))
