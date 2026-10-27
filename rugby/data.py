"""Parse {{rugbybox}} templates from Wikipedia wikitext into a clean table of men's test matches.

Output columns (one row per match, deduplicated across pages):
    date, home, away                       canonical team codes (rugby.teams)
    home_score, away_score                 NaN for scheduled fixtures
    home_tries, away_tries                 NaN when the box has no scorer detail
    comp                                   six_nations | rugby_championship | nations_championship |
                                           rwc | rwc_qual | warmup | tier2_comp | test
    knockout                               RWC quarter-final or later
    city, venue_country, venue             venue_country: 1 home side at home, 0 neutral, -1 away side at home
    page                                   source Wikipedia page
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import teams

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wikitext.json"
KAGGLE = ROOT / "data" / "raw" / "kaggle_internationals.csv"
CACHE = ROOT / "data" / "matches.csv"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9

# Host nations, so RWC matches at an otherwise unknown city get the right venue country.
RWC_HOSTS = {1987: "NZL", 1991: "ENG", 1995: "RSA", 1999: "WAL", 2003: "AUS", 2007: "FRA",
             2011: "NZL", 2015: "ENG", 2019: "JPN", 2023: "FRA", 2027: "AUS"}

SENIOR_TEMPLATES = {"ru", "ru-rt", "rut", "ru-t"}
BAD_NAME = re.compile(r"\bXV\b|\bA\b|Māori|Maori|President|Develop|Select|Invitation", re.I)
SCORE = re.compile(r"(\d{1,3})\s*(?:–|—|-|−|&ndash;|&mdash;|\{\{ndash\}\})\s*(\d{1,3})")


# Cities that host neutral tests but rarely (or never) a home test, so ownership can't be learnt.
CITY_COUNTRY = {"Wakefield": "ENG", "Ealing": "ENG", "London": "ENG", "Bourg-en-Bresse": "FRA",
                "Chambery": "FRA", "Chambéry": "FRA", "Angouleme (France)": "FRA",
                "Angoulême": "FRA", "Leganés": "ESP", "Gosford": "AUS", "Chicago": "USA",
                "Washington, D.C.": "USA", "Las Vegas": "USA", "Tokyo": "JPN", "Yokohama": "JPN",
                "Hong Kong": "HKG", "Kowloon": "HKG"}


# ---------------------------------------------------------------- template parsing helpers

def iter_templates(text, name="rugbybox"):
    """Yield (start, body) for each {{name ...}} template, handling nested braces."""
    pat = re.compile(r"\{\{\s*" + name + r"(?:\s*collapsible2?)?\s*[|\n]", re.I)
    for m in pat.finditer(text):
        i, depth = m.start() + 2, 1
        while depth and i < len(text):
            if text.startswith("{{", i):
                depth += 1
                i += 2
            elif text.startswith("}}", i):
                depth -= 1
                i += 2
            else:
                i += 1
        yield m.start(), text[m.start() + 2:i - 2]


def split_params(body):
    """Split a template body on top-level pipes into {key: value}."""
    parts, depth, link, cur = [], 0, 0, []
    i = 0
    while i < len(body):
        two = body[i:i + 2]
        if two == "{{":
            depth += 1; cur.append(two); i += 2; continue
        if two == "}}":
            depth -= 1; cur.append(two); i += 2; continue
        if two == "[[":
            link += 1; cur.append(two); i += 2; continue
        if two == "]]":
            link -= 1; cur.append(two); i += 2; continue
        if body[i] == "|" and depth == 0 and link == 0:
            parts.append("".join(cur)); cur = []; i += 1; continue
        cur.append(body[i]); i += 1
    parts.append("".join(cur))
    out = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def strip_refs(s):
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    return re.sub(r"<!--.*?-->", "", s, flags=re.S)


def parse_team(value):
    """Return a canonical team code if the field names a senior national side, else None."""
    v = strip_refs(value)
    m = re.search(r"\{\{\s*([A-Za-z-]+)\s*\|\s*([^|}]+)((?:\|[^}]*)?)\}\}", v)
    if m:
        tmpl, code, rest = m.group(1).lower(), m.group(2).strip(), m.group(3)
        if tmpl not in SENIOR_TEMPLATES:
            return None
        nm = re.search(r"name\s*=\s*([^|}]+)", rest)
        if nm and BAD_NAME.search(nm.group(1)):
            return None
        return teams.canonical(code)
    m = re.search(r"\[\[([^\]|]+?) national rugby union team(?:\|[^\]]*)?\]\]", v)
    if m and not BAD_NAME.search(m.group(1)) and "women" not in m.group(1).lower():
        return teams.canonical(m.group(1))
    return None


def parse_date(value, page_year):
    v = re.sub(r"'''|\[\[|\]\]", "", strip_refs(value)).strip()
    m = re.search(r"start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})", v, re.I)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", v)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\.?,?\s*(\d{4})?", v)
    if m and m.group(2).lower() in MONTHS:
        year = int(m.group(3)) if m.group(3) else page_year
        return pd.Timestamp(year, MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s*(\d{4})?", v)
    if m and m.group(1).lower() in MONTHS:
        year = int(m.group(3)) if m.group(3) else page_year
        return pd.Timestamp(year, MONTHS[m.group(1).lower()], int(m.group(2)))
    return None


def parse_score(value):
    v = strip_refs(value)
    if re.search(r"cancel|postpone|abandon|forfeit|walkover|void|w/o", v, re.I):
        return "void"
    v = re.sub(r"\[\[[^\]|]*\|", "", v)  # keep link text of [[target|19–20]]
    m = SCORE.search(v)
    return (int(m.group(1)), int(m.group(2))) if m else None


def count_tries(value):
    """Count tries in a try1/try2 field. 'Name (2) 5' c, 16' m' counts 2; 'Smith, Jones' counts 2."""
    v = strip_refs(value)
    v = re.sub(r"\{\{\s*(?:sfn|efn|refn|nowrap)[^}]*\}\}", "", v, flags=re.I)
    v = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", v)
    v = re.sub(r"'''|''", "", v)
    segments = re.split(r"<br\s*/?>|\n|\{\{\s*small\s*\|", v)
    n = 0
    for seg in segments:
        seg = seg.strip(" ,;}")
        if not re.search(r"[A-Za-zÀ-ÿ]", seg):
            continue
        has_minutes = re.search(r"\d+\s*(?:'|′|&prime;|\+)", seg)
        pieces = [seg] if has_minutes else [p for p in seg.split(",") if p.strip()]
        for p in pieces:
            k = re.search(r"\((\d)\)", p)
            n += int(k.group(1)) if k else 1
    return n


def classify(title, heading):
    t = title.lower()
    if "rugby world cup" in t:
        if re.search(r"qualif|repechage|qualification", t):
            return "rwc_qual", False
        if "warm-up" in t or "warm up" in t:
            return "warmup", False
        ko = bool(re.search(r"knockout|final|quarter|semi|bronze|third", t + " " + heading.lower())) \
            and "pool" not in heading.lower()
        return "rwc", ko
    if re.search(r"asian|asia rugby|south american", t):
        return "tier2_comp", False
    if re.search(r"six nations|five nations|home nations", t):
        return "six_nations", False
    if re.search(r"^\d{4} rugby championship|tri nations|bledisloe", t):
        return "rugby_championship", False
    if "nations championship" in t and int(title[:4]) >= 2026:
        return "nations_championship", False
    if re.search(r"pacific nations|rugby europe|european nations|americas rugby|asia|africa cup|"
                 r"south american|nations cup|fira|oceania|cup of nations|pacific tri|tbilisi|"
                 r"pacific challenge", t):
        return "tier2_comp", False
    return "test", False


def city_of(stadium):
    s = strip_refs(stadium)
    links = [l for l in re.findall(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]", s)
             if teams.canonical(l[1] or l[0]) is None]  # drop trailing country links
    if len(links) >= 2:
        target, text = links[-1]
        return (text or target).split(",")[0].strip()
    plain = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", s)
    plain = re.sub(r"\{\{[^}]*\}\}|<[^>]+>", "", plain).strip()
    if "," in plain:
        return plain.split(",")[-1].strip() or None
    return None


# ---------------------------------------------------------------- page -> rows

def parse_page(title, text):
    year_m = re.match(r"(\d{4})", title)
    page_year = int(year_m.group(1)) if year_m else None
    headings = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^=+\s*(.*?)\s*=+\s*$", text, re.M)]
    rows = []
    for start, body in iter_templates(text):
        p = split_params(body)
        # older boxes use home/away, newer ones team1/team2
        home = parse_team(p.get("home") or p.get("team1", ""))
        away = parse_team(p.get("away") or p.get("team2", ""))
        if not home or not away or home == away:
            continue
        date = parse_date(p.get("date", ""), page_year) if page_year or p.get("date") else None
        if date is None:
            continue
        score = parse_score(p.get("score", ""))
        if score == "void":
            continue
        heading = next((h for pos, h in reversed(headings) if pos < start), "")
        comp, ko = classify(title, heading)
        detail = any(p.get(k, "").strip() for k in ("try1", "try2", "pen1", "pen2", "con1", "con2"))
        rows.append(dict(
            date=date, home=home, away=away,
            home_score=score[0] if score else np.nan, away_score=score[1] if score else np.nan,
            home_tries=count_tries(p.get("try1", "")) if detail and score else np.nan,
            away_tries=count_tries(p.get("try2", "")) if detail and score else np.nan,
            comp=comp, knockout=ko, city=city_of(p.get("stadium", "")),
            venue=re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", strip_refs(p.get("stadium", "")))[:80],
            page=title,
        ))
    return rows


COMP_PRIORITY = ["rwc", "six_nations", "rugby_championship", "nations_championship", "rwc_qual",
                 "tier2_comp", "warmup", "test"]


def dedupe(df):
    """The same match often appears on several pages (tour page + season page). Keep one row,
    preferring the most specific competition and a row with try detail. Dates can differ by a
    day between pages (time zones), so matches with the same pair within 3 days are merged."""
    df = df.copy()
    df["pair"] = [tuple(sorted(x)) for x in zip(df.home, df.away)]
    df["prio"] = df.comp.map(COMP_PRIORITY.index) + df.home_tries.isna() * 0.5
    df = df.sort_values(["pair", "date", "prio"]).reset_index(drop=True)
    keep, last = [], {}
    for i, r in df.iterrows():
        prev = last.get(r.pair)
        if prev is not None and (r.date - df.at[prev, "date"]).days <= 3:
            if df.at[prev, "prio"] > r.prio:  # better row for the same match
                keep[keep.index(prev)] = i
                last[r.pair] = i
            elif pd.isna(df.at[prev, "home_score"]) and not pd.isna(r.home_score):
                keep[keep.index(prev)] = i
                last[r.pair] = i
            continue
        keep.append(i)
        last[r.pair] = i
    return df.loc[keep].drop(columns=["pair", "prio"]).sort_values("date").reset_index(drop=True)


def infer_venues(df):
    """A city's country is the team that most often plays *home* tests there outside World Cups.
    venue_country: 1 = listed home side at home, -1 = listed away side at home, 0 = neutral."""
    played = df[df.comp != "rwc"].dropna(subset=["city"])
    owner = played.groupby("city").home.agg(lambda s: s.value_counts().index[0]).to_dict()
    owner.update(CITY_COUNTRY)
    vc, host = [], []
    for r in df.itertuples():
        h = owner.get(r.city)
        if h is None and r.comp == "rwc":
            h = RWC_HOSTS.get(r.date.year)
        if h is None:
            h = r.home  # unknown city: assume a genuine home match
        host.append(h)
        vc.append(1 if h == r.home else -1 if h == r.away else 0)
    df = df.copy()
    df["host"] = host
    df["venue_country"] = vc
    return df


def kaggle_fill(wiki):
    """Tier-1 matches from the Kaggle dataset that Wikipedia's season pages don't have."""
    if not KAGGLE.exists():
        return wiki
    k = pd.read_csv(KAGGLE, parse_dates=["date"])
    k["home"] = k.home_team.map(teams.canonical)
    k["away"] = k.away_team.map(teams.canonical)
    seen = {}
    for r in wiki.itertuples():
        seen.setdefault(tuple(sorted((r.home, r.away))), []).append(
            (r.date, {(r.home, r.home_score), (r.away, r.away_score)}))
    rows = []
    for r in k.itertuples():
        score = {(r.home, r.home_score), (r.away, r.away_score)}
        # same match: within 3 days, or same score within 60 days (Kaggle has some wrong dates)
        if any(abs((d - r.date).days) <= 3 or (abs((d - r.date).days) <= 60 and s == score)
               for d, s in seen.get(tuple(sorted((r.home, r.away))), [])):
            continue
        comp, ko = classify(f"{r.date.year} {r.competition}", "")
        rows.append(dict(date=r.date, home=r.home, away=r.away, home_score=r.home_score,
                         away_score=r.away_score, home_tries=np.nan, away_tries=np.nan,
                         comp=comp, knockout=ko, city=r.city, venue=f"{r.stadium}, {r.city}",
                         page="kaggle: " + str(r.competition)))
    return pd.concat([wiki, pd.DataFrame(rows)], ignore_index=True).sort_values("date", kind="stable")


@lru_cache(maxsize=1)
def _load_all(rebuild=False):
    if CACHE.exists() and not rebuild and (not RAW.exists() or CACHE.stat().st_mtime > RAW.stat().st_mtime):
        return pd.read_csv(CACHE, parse_dates=["date"], keep_default_na=True)
    pages = json.loads(RAW.read_text())
    rows = [r for t, txt in pages.items() for r in parse_page(t, txt)]
    wiki = dedupe(pd.DataFrame(rows)).dropna(subset=["date"])
    df = infer_venues(kaggle_fill(wiki).reset_index(drop=True))
    df.to_csv(CACHE, index=False)
    return df


def load_matches(rebuild=False):
    """All played men's tests (with scores), oldest first."""
    df = _load_all(rebuild)
    return df.dropna(subset=["home_score"]).reset_index(drop=True)


def load_fixtures(after=None, rebuild=False):
    """Scheduled tests with no score yet (dated after `after`, default today)."""
    df = _load_all(rebuild)
    after = pd.Timestamp(after) if after is not None else pd.Timestamp.today().normalize()
    fx = df[df.home_score.isna() & (df.date >= after)]
    return fx.reset_index(drop=True)
