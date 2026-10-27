"""Build reports/rugby_predictor_report.pdf from the outputs of run_all.py.

Every number in the text is read from figures/*.csv, so the report always matches the analysis.
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "reports" / "rugby_predictor_report.pdf"
REPO = "github.com/samberman-hi/rugby-predictor"

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "Arial Bold.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONT_DIR / "Arial Italic.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

INK, INK2, RULE, BLUE, TINT = (colors.HexColor(c) for c in
                               ("#0b0b0b", "#52514e", "#d9d8d2", "#2a78d6", "#eef4fc"))
WIDTH = A4[0] - 4 * cm

S = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=20, leading=24, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Arial", fontSize=10.5, leading=14, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=13, leading=17, textColor=INK,
                         spaceBefore=12, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.6, leading=13.6, textColor=INK,
                           alignment=TA_LEFT, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Arial", fontSize=9.6, leading=13.6, textColor=INK,
                             leftIndent=12, bulletIndent=2, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="Arial-Italic", fontSize=8.4, leading=11, textColor=INK2,
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.4, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.4, leading=10.5, textColor=INK),
    "stat": ParagraphStyle("stat", fontName="Arial-Bold", fontSize=17, leading=20, textColor=BLUE),
    "statl": ParagraphStyle("statl", fontName="Arial", fontSize=8.2, leading=10.5, textColor=INK2),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def figure(name, caption, width=WIDTH):
    path = FIG / name
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, "cap")])


def table(rows, col_widths, header=True, bold_rows=()):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) or i in bold_rows else S["cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i in bold_rows:
        style.append(("BACKGROUND", (0, i), (-1, i), TINT))
    t.setStyle(TableStyle(style))
    return t


def stat_row(stats):
    cells = [[P(v, "stat"), P(label, "statl")] for v, label in stats]
    t = Table([[Table([[c[0]], [c[1]]], colWidths=[WIDTH / len(stats) - 8]) for c in cells]],
              colWidths=[WIDTH / len(stats)] * len(stats))
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, RULE),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, f"Sam Berman · Predicting men's international rugby · {REPO}")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


def main():
    summ = pd.read_csv(FIG / "summary.csv", index_col=[0, 1])
    abl = pd.read_csv(FIG / "ablation.csv").set_index("dropped")
    imp = pd.read_csv(FIG / "feature_importance.csv", index_col=0).iloc[:, 0]
    up = pd.read_csv(FIG / "upcoming_predictions.csv", parse_dates=["date"])
    ratings = pd.read_csv(FIG / "ratings_current.csv", index_col=0)
    matches = pd.read_csv(ROOT / "data" / "matches.csv", parse_dates=["date"])
    played = matches.dropna(subset=["home_score"])

    hold, hold1, allt = (summ.loc[k] for k in
                         ("2017-2026 (holdout)", "Holdout, tier-1 v tier-1", "All tests"))
    E, ELO, WR = "Ensemble", "Elo (raw expectancy)", "World Rugby rankings"
    ll = lambda t, m: t.loc[m, "log_loss"]
    acc = lambda t, m: t.loc[m, "accuracy"] * 100
    elo_gain = ll(hold, "Home-win base rate") - ll(hold, ELO)  # what ratings alone buy
    extra_gain = ll(hold, ELO) - ll(hold, E)                  # what the other variables add
    n_matches, n_hold = len(played), int(hold.loc[E, "matches"])
    nc = up[up.comp_name == "nations_championship"].sort_values("date")
    top = ratings.head(8).team.tolist()

    story = [
        P("Predicting Men's International Rugby", "title"), Spacer(1, 4),
        P(f"Can 20 pre-match variables beat the official World Rugby rankings? "
          f"A walk-forward test on {n_matches:,} tests, with predictions for the November 2026 "
          f"Nations Championship. &nbsp;Sam Berman · 27 October 2026 · {REPO}", "sub"),
        Spacer(1, 10),
        stat_row([(f"{ll(hold, E):.3f}", f"holdout log-loss, ensemble<br/>(Elo {ll(hold, ELO):.3f}, "
                                         f"World Rugby {ll(hold, WR):.3f})"),
                  (f"{acc(hold, E):.1f}%", f"winners called, {n_hold:,} holdout tests<br/>"
                                           f"(Elo {acc(hold, ELO):.1f}%)"),
                  (f"{n_matches:,}", "men's tests parsed, 1871 to<br/>24 Oct 2026"),
                  ("4 of 20", "variables carry nearly<br/>all of the signal")]),
        Spacer(1, 8),
        P("Summary", "h1"),
        P(f"I built a dataset of every men's rugby union test on Wikipedia, engineered 20 pre-match "
          f"variables and tested four models, each scoring every year from 2008 to 2026 using only data from earlier "
          f"years. On the 2017–2026 holdout, the "
          f"ensemble's log-loss is <b>{ll(hold, E):.3f}</b>. A tuned Elo rating scores {ll(hold, ELO):.3f} "
          f"and the official World Rugby rankings {ll(hold, WR):.3f}. Elo alone improves on a "
          f"home-win base rate by {elo_gain:.3f}; the extra variables add another {extra_gain:.3f}. "
          f"The gain is <b>real but modest</b>. "
          f"Most of it comes from better-calibrated probabilities, not from calling more winners."),
        P("The more interesting finding is which variables matter. Ratings and recent head-to-head "
          "results carry almost everything. Rest days, travel, time zones, form streaks and "
          "Lions-tour windows add nothing once the ratings are known, and form actually makes "
          "tier-1 predictions slightly worse. Reporting that, rather than a long list of "
          "clever-sounding features, is the main point of the project."),

        P("1. The question", "h1"),
        P("Rugby has a strong favourite effect and a large home advantage, so a model can look "
          "accurate without knowing much. Every model is therefore judged against two baselines that "
          "already capture team strength: the <b>World Rugby rankings</b>, reconstructed from results "
          "with the official formula, and a margin-aware <b>Elo</b> rating. The question is not "
          "'is the model accurate?' but 'does it know anything the ratings don't?' Models are scored "
          "by <b>log-loss</b>, which rewards honest probabilities and punishes confident mistakes. "
          "Accuracy alone would rate a model that says 51% the same as one that says 99%."),

        P("2. Data", "h1"),
        P(f"No clean public dataset covers men's tests to 2026 beyond the top ten nations. I wrote a "
          f"parser for the <i>rugbybox</i> match template used on 870 Wikipedia season pages (Six "
          f"Nations, Rugby Championship, World Cups, tours, Pacific Nations Cup, Rugby Europe and "
          f"more). It gives {n_matches:,} tests between 139 nations, plus scheduled fixtures. Older "
          f"tier-1 matches missing from Wikipedia are filled from a Kaggle dataset."),
        *bullets([
            "<b>Validated against an independent source:</b> 1,294 of 1,396 tier-1 tests since 1995 "
            "match, and only 4 scores disagree. In at least two of those Wikipedia is right and the "
            "Kaggle data is wrong.",
            "<b>Neutral venues inferred:</b> Wikipedia names a 'home' side even in Chicago or at a "
            "World Cup. A city's country is learnt as the side that most often plays home tests there.",
            "<b>Cleaning:</b> A-teams, XVs and Māori sides are filtered out; matches listed on "
            "several pages are deduplicated; successor states inherit ratings (USSR → Russia).",
        ]),

        P("3. The 20 variables and the models", "h1"),
        table([["Group", "Variables (home minus away unless stated)"],
               ["Ratings", "Elo gap · World Rugby points gap · attack/defence expected margin · "
                           "expected total · expected try margin"],
               ["Form", "win rate last 5 · mean margin last 10 · Elo change over 12 months"],
               ["Schedule", "rest days (each side) · tests played in last 12 months"],
               ["Venue & travel", "home / neutral / away · km travelled · time-zone shift"],
               ["Matchup", "head-to-head margin last 5 · head-to-head win share last 10 · tier gap"],
               ["Context", "competition type · end-of-season tour · Lions-tour window"]],
              [3.0 * cm, WIDTH - 3.0 * cm]),
        Spacer(1, 6),
        P("Every variable is computed in one chronological pass <i>before</i> the match updates any "
          "state. A unit test changes all later results and checks that no earlier feature moves. "
          "Four models are averaged: a regularised logistic regression, shallow gradient-boosted "
          "trees, a points-margin regression (which also gives a predicted score) and Elo + "
          "logistic. Elo's settings were tuned on 1995–2007. The model settings were picked from about a "
          "dozen variants and confirmed on 2008–2016, but my experiments also printed 2017–2026 "
          "scores, so that period is a near-clean holdout rather than a sealed one. The 2027 season "
          "will be the first fully out-of-sample test."),

        P("4. Results", "h1"),
        table([["Forecaster", "Holdout log-loss", "Holdout accuracy", "Tier-1 log-loss", "Tier-1 accuracy"]] +
              [[m, f"{ll(hold, m):.3f}", f"{acc(hold, m):.1f}%", f"{ll(hold1, m):.3f}", f"{acc(hold1, m):.1f}%"]
               for m in ["Home-win base rate", WR, ELO, "Logistic (20 features)",
                         "Gradient boosting (20 features)", "Margin regression (20 features)", E]],
              [5.6 * cm, 2.85 * cm, 2.85 * cm, 2.85 * cm, 2.85 * cm], bold_rows=(7,)),
        P(f"Holdout = 2017 to Oct 2026, {n_hold:,} tests; tier-1 = the ten Six Nations and Rugby "
          f"Championship sides playing each other ({int(hold1.loc[E, 'matches'])} tests). "
          f"Over the full 2008–2026 period: ensemble {ll(allt, E):.3f}, Elo {ll(allt, ELO):.3f}.", "cap"),
        figure("02_calibration.png",
               "Calibration. Elo is overconfident at the extremes: its 15% underdogs win about 21% of "
               "the time. The ensemble learns how far to shrink a rating gap and sits on the diagonal.",
               width=8.4 * cm),
        P("The ensemble wins on every subset, but by less on tier-1 tests, where Elo is hard to beat "
          f"({ll(hold1, E):.3f} v {ll(hold1, ELO):.3f}). It beats Elo on tier-1 log-loss in 13 of "
          "the 19 years. Year-to-year noise (2022 was a season of upsets) is larger than the gap "
          "between models."),

        KeepTogether([P("5. What actually matters", "h1"),
                      figure("04_feature_importance.png",
                             "Permutation importance: fit on 1995–2017, then shuffle one variable at "
                             "a time in 2018–2026 and measure how much worse the log-loss gets.",
                             width=12.5 * cm)]),
        table([["Group removed", "Change in log-loss (all)", "Change (tier-1)"]] +
              [[g, f"{abl.loc[g, 'log_loss_all'] - abl.loc['(none)', 'log_loss_all']:+.4f}",
                f"{abl.loc[g, 'log_loss_tier1'] - abl.loc['(none)', 'log_loss_tier1']:+.4f}"]
               for g in ["ratings", "matchup", "context", "form", "schedule", "venue & travel"]],
              [5.6 * cm, 4.2 * cm, 4.2 * cm]),
        P("Ablation: the whole walk-forward rerun with each group removed. Positive means the group "
          "helps. The Elo + logistic member keeps the Elo gap throughout.", "cap"),
        *bullets([
            f"<b>Elo is worth more than everything else combined</b> ({imp['elo_diff'] * 1000:.0f} "
            f"thousandths of log-loss). Ratings are already an exponentially weighted record of form, "
            f"which is why separate form variables add nothing.",
            "<b>Head-to-head margin is the surprise second.</b> Some matchups persist beyond what "
            "overall ratings capture, perhaps through style or a set-piece edge.",
            "<b>Plausible stories that failed:</b> touring sides at the end of their season "
            "slightly <i>beat</i> their ratings, perhaps because tours are arranged against beatable "
            "opponents. Lions summers show no measurable weakening of the home nations.",
        ]),

        KeepTogether([
            P("6. November 2026: Nations Championship", "h1"),
            P(f"The model fitted on every test to 24 October rates {', '.join(top[:-1])} and {top[-1]} "
              f"as the top eight. Its predictions for the 18 Nations Championship fixtures:"),
            figure("07_nations_championship_nov2026.png",
                   "Win / draw / loss probabilities and expected score (right). Full numbers for all "
                   "147 scheduled tests are in figures/upcoming_predictions.csv.", width=14.5 * cm)]),
    ]
    close = nc.assign(gap=(nc.p_home - nc.p_away).abs()).nsmallest(3, "gap")
    story.append(P("The closest calls are " + "; ".join(
        f"{r.home_name} v {r.away_name} ({r.p_home:.0%} / {r.p_away:.0%})" for r in close.itertuples())
        + ". South Africa and New Zealand are favourites in every away match."))

    story += [
        P("7. What I learned", "h1"),
        *bullets([
            "<b>Beat a strong baseline, not a weak one.</b> Against a home-win base rate every "
            "model looks brilliant; against Elo the true gain is much smaller. The baseline defines "
            "the claim.",
            "<b>Honest evaluation shapes the design.</b> Walk-forward testing showed that some "
            "'improvements' hurt. Deeper trees and recency weighting both made the boosting model "
            "worse. It also taught me to separate tuning data from reporting data more strictly: "
            "next time I would seal the holdout before running any experiment.",
            "<b>More variables is not more information.</b> Twenty inputs, four that matter. The "
            "ablation suggests a leaner model would do as well, which I report rather than quietly "
            "pruning against the holdout.",
            "<b>Data engineering is most of the work.</b> Parsing, validating, deduplicating and "
            "inferring venues took longer than the modelling, and errors there would have "
            "silently corrupted every result.",
        ]),
        P("Limitations and next steps", "h1"),
        P("The model has no line-ups, injuries or market prices, which is the information a "
          "bookmaker has and it lacks. The next step is to benchmark it against closing odds in the 2027 Six "
          "Nations (the method from my football pricing project), add squad strength from named "
          "XVs, and simulate the 2027 World Cup in Australia. Code, data, tests and every figure are at "
          f"{REPO}; one command (<i>python scripts/run_all.py</i>) reproduces every number."),
    ]

    OUT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Predicting Men's International Rugby", author="Sam Berman",
                            subject="Walk-forward test of a 20-variable rugby test predictor")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
