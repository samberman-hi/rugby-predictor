"""Predict upcoming or hypothetical men's tests with the model fitted on every played test.

    python -m rugby.predict Ireland "New Zealand"                   # in Dublin
    python -m rugby.predict "New Zealand" Ireland --at USA          # neutral venue
    python -m rugby.predict France "South Africa" --date 2026-11-13 --comp nations_championship
    python -m rugby.predict --fixtures --until 2026-11-30           # every scheduled test
"""
import argparse

import numpy as np
import pandas as pd

from . import data, features, teams
from .evaluation import TRAIN_FROM
from .model import NUMERIC, Ensemble


def fit(matches=None, fixtures=None):
    """Features for all played matches (+ fixtures) and an Ensemble fitted on the played ones."""
    matches = data.load_matches() if matches is None else matches
    feat = features.build(matches, fixtures)
    train = feat[feat.played & (feat.date >= TRAIN_FROM)]
    return feat, Ensemble().fit(train)


def custom_fixture(home, away, at=None, date=None, comp="test", knockout=False):
    h, a = teams.code_for(home), teams.code_for(away)
    host = teams.code_for(at) if at else h
    venue = 1 if host == h else -1 if host == a else 0
    return pd.DataFrame([dict(date=pd.Timestamp(date) if date else pd.Timestamp.today().normalize(),
                              home=h, away=a, home_score=np.nan, away_score=np.nan,
                              home_tries=np.nan, away_tries=np.nan, comp=comp, knockout=knockout,
                              city=None, venue="", page="custom", host=host, venue_country=venue)])


def explain(model, row):
    """Each feature's push on the logistic part, in log-odds (coef x standardised value)."""
    logit = model.parts[0]
    num, _ = logit._design(row)
    z = logit.scaler.transform(num)[0]
    coefs = logit.coefficients()
    contrib = pd.Series({f: coefs[f] * z[i] for i, f in enumerate(NUMERIC)})
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index)


def predict_rows(feat, model, rows):
    full = model.predict_full(rows).reset_index(drop=True)
    parts = pd.DataFrame(model.predict_parts(rows)).reset_index(drop=True)
    base = rows[["date", "home", "away", "comp_name", "venue", "elo_p"]].reset_index(drop=True)
    out = pd.concat([base, full, parts], axis=1)
    out["home_name"] = out.home.map(teams.name)
    out["away_name"] = out.away.map(teams.name)
    return out


def upcoming(until=None, after=None):
    fx = data.load_fixtures(after=after)
    if until:
        fx = fx[fx.date <= pd.Timestamp(until)]
    feat, model = fit(fixtures=fx)
    rows = feat[~feat.played]
    return predict_rows(feat, model, rows), model, feat


def fmt(r):
    venue = {1: "home", 0: "neutral", -1: "away side at home"}[int(r.venue)]
    return (f"{r.date:%a %d %b %Y}  {r.home_name} v {r.away_name} ({venue}, {r.comp_name})\n"
            f"  {r.home_name} {r.p_home:5.1%}   draw {r.p_draw:4.1%}   {r.away_name} {r.p_away:5.1%}\n"
            f"  margin {r.margin:+.1f} (sd {r.sd:.0f})   expected score {r.home_pts:.0f}-{r.away_pts:.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("home", nargs="?")
    ap.add_argument("away", nargs="?")
    ap.add_argument("--at", help="country the match is played in (default: the home side's)")
    ap.add_argument("--date")
    ap.add_argument("--comp", default="test", choices=[c for c in features.COMPS if not c.startswith("rwc_")] + ["rwc"])
    ap.add_argument("--knockout", action="store_true", help="World Cup knockout match")
    ap.add_argument("--fixtures", action="store_true", help="predict every scheduled test")
    ap.add_argument("--until", help="with --fixtures: last date to include")
    ap.add_argument("--tier1", action="store_true", help="with --fixtures: tier-1 v tier-1 only")
    ap.add_argument("--csv", help="write predictions to this CSV")
    args = ap.parse_args(argv)

    if args.fixtures:
        preds, _, _ = upcoming(until=args.until)
        if args.tier1:
            preds = preds[preds.home.isin(teams.TIER1) | preds.away.isin(teams.TIER1)]
        for r in preds.itertuples():
            print(fmt(r) + "\n")
        if args.csv:
            preds.to_csv(args.csv, index=False)
        return
    if not (args.home and args.away):
        ap.error("give two teams, or --fixtures")
    fx = custom_fixture(args.home, args.away, args.at, args.date, args.comp, args.knockout)
    feat, model = fit(fixtures=fx)
    rows = feat[~feat.played]
    r = next(predict_rows(feat, model, rows).itertuples())
    print(fmt(r))
    print(f"  models: " + ", ".join(f"{k.split(' (')[0]} {v[0]:.1%}"
                                    for k, v in model.predict_parts(rows).items()) + f", Elo {r.elo_p:.1%}")
    print("  biggest drivers vs an average test (logistic part, log-odds, + favours home):")
    for k, v in explain(model, rows).head(6).items():
        print(f"    {k:18s} {v:+.2f}   (value {rows.iloc[0][k]:+.2f})")


if __name__ == "__main__":
    main()
