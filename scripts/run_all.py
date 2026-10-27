"""Reproduce every number and figure: python scripts/run_all.py [--ablation]

Writes to figures/: summary.csv, backtest_predictions.csv, ratings_current.csv,
upcoming_predictions.csv, feature_importance.csv, (ablation.csv) and the PNG figures.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rugby import data, evaluation as ev, features, model as md, predict, teams  # noqa: E402

FIG = ROOT / "figures"
TODAY = pd.Timestamp("2026-10-26")

# reference palette (dataviz skill): slot 1 blue = our model, 2 orange = Elo, 3 aqua = World Rugby
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10, "lines.linewidth": 2,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})

GROUPS = {
    "ratings": ["elo_diff", "wr_diff", "pts_exp_margin", "pts_exp_total", "tries_exp_margin"],
    "form": ["form5_diff", "margin10_diff", "momentum_diff"],
    "schedule": ["rest_home", "rest_away", "tests12m_diff"],
    "venue & travel": ["venue", "travel_diff", "tz_diff"],
    "matchup": ["h2h_share", "h2h_margin", "tier_gap"],
    "context": ["comp", "offseason_tour", "lions_window"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_model_comparison(tables):
    order = ["Home-win base rate", "World Rugby rankings", "Elo (raw expectancy)", "Elo + logistic",
             "Logistic (20 features)", "Gradient boosting (20 features)",
             "Margin regression (20 features)", "Ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (label, t) in zip(axes, tables.items()):
        t = t.reindex(order[1:])  # base rate is far off the scale; reported in the table
        colors = [AQUA if "World" in m else ORANGE if "Elo" in m else BLUE if m == "Ensemble" else GRAY
                  for m in t.index]
        y = np.arange(len(t))[::-1]
        # dot plot, not bars: the axis doesn't start at zero, so bar lengths would mislead
        ax.scatter(t.log_loss, y, color=colors, s=70, zorder=3, edgecolors=SURFACE, linewidths=2)
        ax.set_xlim(t.log_loss.min() - 0.006, t.log_loss.max() + 0.008)
        for yi, v in zip(y, t.log_loss):
            ax.text(v + 0.0012, yi, f"{v:.3f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(y, t.index)
        ax.set_title(f"{label} ({int(t.matches.iloc[0])} tests)")
        ax.set_xlabel("log-loss, walk-forward 2008-2026 (lower is better)")
        ax.grid(axis="y", visible=False)
    save(fig, "01_model_comparison.png")


def fig_calibration(preds):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
    for col, color, label in [("Ensemble", BLUE, "Ensemble"), ("Elo (raw expectancy)", ORANGE, "Elo")]:
        c = ev.calibration(preds[col].to_numpy(), preds.result.to_numpy())
        ax.plot(c.pred, c.obs, color=color, marker="o", ms=5, label=label)
    ax.set_xlabel("predicted home-win probability")
    ax.set_ylabel("observed home-win rate (draw = ½)")
    ax.set_title("Calibration, 2008-2026")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "02_calibration.png")


def fig_by_year(preds):
    t1 = preds[ev.tier1_mask(preds)]
    rows = []
    for y, g in t1.groupby(t1.date.dt.year):
        rows.append({"year": y, **{c: ev.log_loss(g[c], g.result) for c in
                                   ["Ensemble", "Elo (raw expectancy)", "World Rugby rankings"]}})
    d = pd.DataFrame(rows).set_index("year")
    fig, ax = plt.subplots(figsize=(10, 4))
    for c, color, label in [("World Rugby rankings", AQUA, "World Rugby rankings"),
                            ("Elo (raw expectancy)", ORANGE, "Elo"), ("Ensemble", BLUE, "Ensemble")]:
        ax.plot(d.index, d[c], color=color, marker="o", ms=4, label=label)
    ax.set_title("Tier-1 v tier-1 log-loss by year (lower is better)")
    ax.set_xticks(d.index[::2])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, "03_logloss_by_year.png")
    return d


def permutation_importance(feat, split="2018-01-01", reps=5):
    """Fit on 1995-2017, then shuffle each feature in 2018-2026 and measure the log-loss rise."""
    played = feat[feat.played]
    train = played[(played.date >= ev.TRAIN_FROM) & (played.date < split)]
    test = played[played.date >= split]
    ens = md.Ensemble().fit(train)
    base = ev.log_loss(ens.predict(test), test.result)
    rng = np.random.default_rng(0)
    out = {}
    for f in features.FEATURES:
        rises = []
        for _ in range(reps):
            shuffled = test.copy()
            shuffled[f] = rng.permutation(shuffled[f].to_numpy())
            rises.append(ev.log_loss(ens.predict(shuffled), test.result) - base)
        out[f] = np.mean(rises)
    return pd.Series(out).sort_values(), base, ens


def fig_importance(imp):
    group_of = {f: g for g, fs in GROUPS.items() for f in fs}
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    y = np.arange(len(imp))
    ax.barh(y, imp.values * 1000, color=BLUE, height=0.6)
    ax.set_yticks(y, [f"{f}  ·  {group_of[f]}" for f in imp.index])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("log-loss increase when shuffled (×1000), 2018-2026 holdout")
    ax.set_title("Permutation importance of the 20 inputs")
    ax.grid(axis="y", visible=False)
    save(fig, "04_feature_importance.png")


def fig_elo_history(feat):
    """Elo relative to the tier-1 average each quarter. Raw Elo drifts upward over time (new
    nations enter low and lose points to the established sides), so raw levels mislead; only
    gaps matter for prediction."""
    hist = feat.attrs["state"]["hist"].elo_hist

    def quarterly(code):
        s = pd.DataFrame(hist[code], columns=["date", "elo"]).set_index("date").elo
        s = s[~s.index.duplicated(keep="last")]
        return s.resample("QE").last().ffill()

    q = pd.DataFrame({c: quarterly(c) for c in teams.TIER1}).ffill()
    q = q[q.index >= "2000"]
    rel = q.sub(q.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    picks = [("NZL", BLUE), ("RSA", "#008300"), ("IRE", ORANGE), ("FRA", "#6250d6"), ("ENG", "#e34948")]
    for code, color in picks:
        ax.plot(rel.index, rel[code], color=color, lw=1.8)
    # end labels, nudged apart so they never overlap
    ends = sorted(((rel[c].iloc[-1], c) for c, _ in picks))
    min_gap = (rel[[c for c, _ in picks]].max().max() - rel[[c for c, _ in picks]].min().min()) * 0.06
    placed = []
    for v, c in ends:
        y = max(v, placed[-1] + min_gap) if placed else v
        placed.append(y)
        ax.text(rel.index[-1] + pd.Timedelta(days=60), y, teams.name(c), color=INK2, fontsize=9, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Elo since 2000, relative to the tier-1 average (margin-adjusted, end of quarter)")
    ax.set_ylabel("Elo points above tier-1 average")
    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2028-09-01"))
    save(fig, "05_elo_history.png")


def fig_margin(preds):
    t1 = preds[ev.tier1_mask(preds)]
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(t1.pred_margin, t1.margin, s=9, color=BLUE, alpha=0.35, linewidths=0)
    lim = 60
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, lw=1, ls="--")
    ax.set_xlim(-35, 35); ax.set_ylim(-lim, lim)
    mae = (t1.pred_margin - t1.margin).abs().mean()
    ax.set_title(f"Tier-1 margins: predicted v actual (MAE {mae:.1f} pts)")
    ax.set_xlabel("predicted home margin"); ax.set_ylabel("actual home margin")
    save(fig, "06_margin.png")


def fig_upcoming(up):
    nc = up[up.comp_name == "nations_championship"].sort_values("date")
    if nc.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(nc) + 1.2))
    y = np.arange(len(nc))[::-1]
    ax.barh(y, nc.p_home, color=BLUE, height=0.62, label="home win")
    ax.barh(y, nc.p_draw, left=nc.p_home, color=GRAY, height=0.62, label="draw")
    ax.barh(y, nc.p_away, left=nc.p_home + nc.p_draw, color=ORANGE, height=0.62, label="away win")
    labels = [f"{r.date:%d %b}  {r.home_name} v {r.away_name}" for r in nc.itertuples()]
    ax.set_yticks(y, labels)
    for yi, r in zip(y, nc.itertuples()):
        ax.text(0.01, yi, f"{r.p_home:.0%}", va="center", color="white", fontsize=8.5, weight="bold")
        ax.text(0.99, yi, f"{r.p_away:.0%}", va="center", ha="right", color="white", fontsize=8.5, weight="bold")
        ax.text(1.02, yi, f"{r.home_pts:.0f}-{r.away_pts:.0f}", va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(visible=False)
    ax.set_title("Nations Championship, November 2026: win probabilities and expected score")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.04))
    save(fig, "07_nations_championship_nov2026.png")


def ablation(feat):
    """Walk-forward again with each feature group removed (logistic + boosting + margin)."""
    rows = []
    full = None
    for drop in [None, *GROUPS]:
        keep = [f for f in features.FEATURES if drop is None or f not in GROUPS[drop]]
        md.set_features(keep)
        p = ev.walk_forward(feat, models=[md.LogitModel(), md.BoostModel(), md.MarginModel(),
                                          md.EloLogit(), md.Ensemble()])
        t1 = ev.tier1_mask(p)
        r = {"dropped": drop or "(none)", "log_loss_all": ev.log_loss(p.Ensemble, p.result),
             "log_loss_tier1": ev.log_loss(p.Ensemble[t1], p.result[t1])}
        rows.append(r)
        log(f"ablation {r}")
    md.set_features(features.FEATURES)
    return pd.DataFrame(rows)


def main():
    FIG.mkdir(exist_ok=True)
    log("loading matches")
    matches = data.load_matches()
    fixtures = data.load_fixtures(after=TODAY)
    log(f"{len(matches)} played tests ({matches.date.min():%Y}-{matches.date.max():%d %b %Y}), "
        f"{len(fixtures)} fixtures")
    feat = features.build(matches)

    log("walk-forward backtest 2008-2026")
    preds = ev.walk_forward(feat)
    preds.to_csv(FIG / "backtest_predictions.csv", index=False)
    masks = {"All tests": None, "Tier-1 v tier-1": ev.tier1_mask(preds),
             "Involving a tier-1 side": preds.home.isin(teams.TIER1) | preds.away.isin(teams.TIER1),
             "Rugby World Cup": preds.comp_name.isin(["rwc_pool", "rwc_knockout"]),
             "2008-2016 (selection window)": preds.date < "2017-01-01",
             "2017-2026 (clean holdout)": preds.date >= "2017-01-01",
             "Holdout, tier-1 v tier-1": (preds.date >= "2017-01-01") & ev.tier1_mask(preds)}
    tables = {k: ev.score_table(preds, m) for k, m in masks.items()}
    summary = pd.concat(tables, names=["subset"]).round(4)
    summary.to_csv(FIG / "summary.csv")
    with pd.option_context("display.width", 160):
        print(summary)
    t1 = ev.tier1_mask(preds)
    log(f"margin MAE all {np.mean(np.abs(preds.pred_margin - preds.margin)):.2f}, "
        f"tier-1 {np.mean(np.abs(preds.pred_margin - preds.margin)[t1]):.2f}")

    fig_model_comparison({k: tables[k] for k in ["All tests", "Tier-1 v tier-1"]})
    fig_calibration(preds)
    by_year = fig_by_year(preds)
    by_year.round(4).to_csv(FIG / "tier1_logloss_by_year.csv")
    fig_margin(preds)

    log("permutation importance")
    imp, base, _ = permutation_importance(feat)
    imp.sort_values(ascending=False).to_csv(FIG / "feature_importance.csv", header=["logloss_rise"])
    fig_importance(imp)
    log(f"holdout 2018-2026 ensemble log-loss {base:.4f}")

    log("current ratings and upcoming fixtures")
    up, final_model, feat_fx = predict.upcoming(after=TODAY)
    state = feat_fx.attrs["state"]
    ratings = pd.DataFrame({"elo": pd.Series(state["elo"].r), "wr_points": pd.Series(state["wr"].r),
                            "attack": pd.Series(state["pts"].att), "defence": pd.Series(state["pts"].dfn)})
    ratings["team"] = ratings.index.map(teams.name)
    ratings = ratings.sort_values("elo", ascending=False).round(2)
    ratings.to_csv(FIG / "ratings_current.csv", index_label="code")
    print(ratings.head(15).to_string())
    fig_elo_history(feat_fx)
    up.round({c: 4 for c in up.select_dtypes("number").columns}).to_csv(
        FIG / "upcoming_predictions.csv", index=False)
    fig_upcoming(up)
    for r in up[up.comp_name == "nations_championship"].itertuples():
        print(predict.fmt(r))
    coefs = final_model.parts[0].coefficients().sort_values(key=np.abs, ascending=False)
    coefs.round(4).to_csv(FIG / "logistic_coefficients.csv", header=["coef_per_sd"])

    if "--ablation" in sys.argv:
        log("feature-group ablation (slow)")
        ablation(feat).round(4).to_csv(FIG / "ablation.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
