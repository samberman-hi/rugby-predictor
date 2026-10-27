"""Pre-match features, built in one chronological pass with no look-ahead.

For every match the features are recorded *before* the result updates any state. Scheduled
fixtures (no score) go through the same pass: they get features and count towards rest days and
activity, but they never update a rating.

The 20 model inputs (FEATURES), from the home side's point of view:

  ratings    elo_diff          Elo gap incl. home advantage
             wr_diff           World Rugby ranking-points gap (reconstructed)
             pts_exp_margin    attack/defence expected points margin
             pts_exp_total     attack/defence expected total points
             tries_exp_margin  attack/defence expected try margin (from try-scorer data)
  form       form5_diff        win rate over the last 5 tests, home minus away
             margin10_diff     mean margin over the last 10 tests (capped at +-40)
             momentum_diff     Elo change over the previous 365 days
  schedule   rest_home         days since the home side's previous test (capped at 120)
             rest_away         same for the away side
             tests12m_diff     tests played in the previous 365 days
  venue      venue             +1 home side at home, 0 neutral, -1 away side at home
             travel_diff       km travelled to the venue (away minus home), thousands
             tz_diff           time-zone shift in hours (away minus home)
  matchup    h2h_share         home side's win share in the last 10 meetings, shrunk to 0.5
             h2h_margin        mean margin in the last 5 meetings (capped)
             tier_gap          tier-1 flag, home minus away
  context    comp              competition type (categorical)
             offseason_tour    +1 if the away side is touring at the end of its own season
                               (northern side south in Jun-Aug, southern side north in Oct-Dec),
                               -1 if it's the home side
             lions_window      -1 if the home side is a home nation in a Lions-tour June/July
                               (best players away), +1 if the away side is
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import teams
from .ratings import AttackDefence, Elo, WorldRugby

FEATURES = ["elo_diff", "wr_diff", "pts_exp_margin", "pts_exp_total", "tries_exp_margin",
            "form5_diff", "margin10_diff", "momentum_diff", "rest_home", "rest_away",
            "tests12m_diff", "venue", "travel_diff", "tz_diff", "h2h_share", "h2h_margin",
            "tier_gap", "comp", "offseason_tour", "lions_window"]

COMPS = ["test", "warmup", "six_nations", "rugby_championship", "nations_championship",
         "tier2_comp", "rwc_qual", "rwc_pool", "rwc_knockout"]

HOME_NATIONS = {"ENG", "IRE", "SCO", "WAL"}
LIONS_YEARS = {1989, 1993, 1997, 2001, 2005, 2009, 2013, 2017, 2021, 2025, 2029}


def km(a, b):
    """Great-circle distance between two team home locations (codes), in km."""
    (_, la1, lo1), (_, la2, lo2) = teams.TEAMS[a], teams.TEAMS[b]
    p1, p2, dl = math.radians(la1), math.radians(la2), math.radians(lo2 - lo1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return 6371.0 * math.acos(max(-1.0, min(1.0, c)))


def tz_hours(a, b):
    lo1, lo2 = teams.TEAMS[a][2], teams.TEAMS[b][2]
    d = abs(lo1 - lo2) % 360
    return min(d, 360 - d) / 15.0


def offseason(team, host, month):
    """Touring away from home at the end of the team's own domestic season."""
    if team == host:
        return 0
    south = teams.TEAMS[team][1] < 0
    host_south = teams.TEAMS[host][1] < 0
    if not south and host_south and month in (6, 7, 8):
        return 1
    if south and not host_south and month in (10, 11, 12):
        return 1
    return 0


def comp_label(row):
    if row.comp == "rwc":
        return "rwc_knockout" if row.knockout else "rwc_pool"
    return row.comp


class History:
    """Rolling per-team and per-pair records."""

    def __init__(self):
        self.results = defaultdict(lambda: deque(maxlen=10))  # (date, margin, result)
        self.dates = defaultdict(list)                         # all appearance dates, incl. fixtures
        self.elo_hist = defaultdict(list)                       # (date, elo after match)
        self.h2h = defaultdict(lambda: deque(maxlen=10))        # pair -> (date, winner, margin, home)

    def form(self, team, n):
        r = list(self.results[team])[-n:]
        return np.mean([x[2] for x in r]) if r else 0.5

    def margin(self, team, n):
        r = list(self.results[team])[-n:]
        return np.mean([x[1] for x in r]) if r else 0.0

    def rest(self, team, date):
        prev = [d for d in self.dates[team] if d < date]
        return min((date - prev[-1]).days, 120) if prev else 120

    def tests_12m(self, team, date):
        return sum(1 for d in self.dates[team] if 0 < (date - d).days <= 365)

    def momentum(self, team, date, elo_now):
        past = [e for d, e in self.elo_hist[team] if (date - d).days >= 365]
        return elo_now - past[-1] if past else 0.0


def build(matches, fixtures=None, elo_params=None):
    """Return a DataFrame with one row per match/fixture: identifiers, FEATURES and targets."""
    df = matches.assign(played=True)
    if fixtures is not None and len(fixtures):
        df = pd.concat([df, fixtures.assign(played=False)], ignore_index=True)
    df = df.sort_values(["date", "played"], ascending=[True, False], kind="stable").reset_index(drop=True)

    elo, wr = Elo(**(elo_params or {})), WorldRugby()
    pts = AttackDefence()
    tries = AttackDefence(k=0.08, home=0.6, mu=2.5, k_mu=0.002, cap=8)
    hist = History()
    rows = []
    for r in df.itertuples():
        h, a, v, date = r.home, r.away, int(r.venue_country), r.date
        host = r.host if isinstance(r.host, str) else h
        pre = {}
        pre.update(elo.pre(h, a, v))
        pre.update(wr.pre(h, a, v))
        pre.update(pts.pre(h, a, v, "pts"))
        pre.update(tries.pre(h, a, v, "tries"))
        pair = tuple(sorted((h, a)))
        meetings = list(hist.h2h[pair])
        w = sum(1.0 if m[1] == h else 0.5 if m[1] is None else 0.0 for m in meetings)
        last5 = [m[2] if m[1] == h else -m[2] for m in meetings[-5:]]
        lions = 0
        if date.year in LIONS_YEARS and date.month in (6, 7):
            lions = (1 if a in HOME_NATIONS else 0) - (1 if h in HOME_NATIONS else 0)
        feat = dict(
            elo_diff=pre["elo_diff"],
            wr_diff=pre["wr_home"] - pre["wr_away"],
            pts_exp_margin=pre["pts_exp_margin"],
            pts_exp_total=pre["pts_exp_home"] + pre["pts_exp_away"],
            tries_exp_margin=pre["tries_exp_margin"],
            form5_diff=hist.form(h, 5) - hist.form(a, 5),
            margin10_diff=hist.margin(h, 10) - hist.margin(a, 10),
            momentum_diff=hist.momentum(h, date, pre["elo_home"]) - hist.momentum(a, date, pre["elo_away"]),
            rest_home=hist.rest(h, date),
            rest_away=hist.rest(a, date),
            tests12m_diff=hist.tests_12m(h, date) - hist.tests_12m(a, date),
            venue=v,
            travel_diff=(km(a, host) - km(h, host)) / 1000.0,
            tz_diff=tz_hours(a, host) - tz_hours(h, host),
            h2h_share=(w + 1.0) / (len(meetings) + 2.0),
            h2h_margin=float(np.mean(last5)) if last5 else 0.0,
            tier_gap=int(h in teams.TIER1) - int(a in teams.TIER1),
            comp=COMPS.index(comp_label(r)),
            offseason_tour=offseason(a, host, date.month) - offseason(h, host, date.month),
            lions_window=lions,
        )
        rows.append({"date": date, "home": h, "away": a, "comp_name": comp_label(r),
                     "played": r.played, "home_score": r.home_score, "away_score": r.away_score,
                     "elo_p": pre["elo_p"], "wr_gap": pre["wr_gap"],
                     "pts_exp_home": pre["pts_exp_home"], "pts_exp_away": pre["pts_exp_away"],
                     **feat})

        hist.dates[h].append(date)
        hist.dates[a].append(date)
        if not r.played:
            continue
        hs, as_ = r.home_score, r.away_score
        elo.update(h, a, v, hs, as_)
        wr.update(h, a, v, hs, as_, world_cup=r.comp == "rwc")
        pts.update(h, a, v, hs, as_)
        if not (pd.isna(r.home_tries) or pd.isna(r.away_tries)):
            tries.update(h, a, v, r.home_tries, r.away_tries)
        m = max(-40.0, min(40.0, hs - as_))
        res = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        hist.results[h].append((date, m, res))
        hist.results[a].append((date, -m, 1.0 - res))
        hist.elo_hist[h].append((date, elo.rating(h)))
        hist.elo_hist[a].append((date, elo.rating(a)))
        hist.h2h[pair].append((date, h if hs > as_ else a if as_ > hs else None, abs(m), h))

    out = pd.DataFrame(rows)
    out["margin"] = out.home_score - out.away_score
    out["result"] = np.sign(out.margin).map({1.0: 1.0, 0.0: 0.5, -1.0: 0.0})
    out.attrs["state"] = {"elo": elo, "wr": wr, "pts": pts, "tries": tries, "hist": hist}
    return out
