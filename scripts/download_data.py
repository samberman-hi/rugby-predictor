"""Download the raw wikitext of every Wikipedia page that holds men's international match boxes.

Wikipedia season pages (Six Nations, Rugby Championship, mid-year / end-of-year tests,
Rugby World Cup pools, tours, tier-2 competitions) record each match in a {{rugbybox}}
template. We list every article that uses that template, keep the ones whose titles look like
international rugby, and save their wikitext to data/raw/wikitext.json. Parsing happens in
rugby/data.py, which keeps only matches where both sides are senior national teams.

Uses the public MediaWiki API (50 pages per request), so the full download is ~20 requests.
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "wikitext.json"
# Lyle Begbie's Kaggle dataset of tier-1 results 1871-2024, mirrored on GitHub. Used only to fill
# tier-1 matches Wikipedia's season pages miss (mostly before 2005).
KAGGLE_URL = ("https://raw.githubusercontent.com/kyejones25-alt/rugby-sql-analysis/main/"
              "data/internationals.csv")
KAGGLE_OUT = OUT.parent / "kaggle_internationals.csv"
HEADERS = {"User-Agent": "rugby-predictor/0.1 (educational sports-modelling project)"}

INCLUDE = re.compile(
    r"Six Nations|Five Nations|Home Nations|Tri Nations|^\d{4} Rugby Championship|"
    r"rugby union internationals|rugby union tests?|tour of|Rugby World Cup|Pacific Nations Cup|"
    r"Nations Cup|Rugby Europe|European Nations Cup|Americas Rugby Championship|"
    r"Asian? Rugby Championship|Asian Five Nations|Africa Cup|South American Rugby Championship|"
    r"Nations Championship|Bledisloe|FORU Oceania|Cup of Nations|Pacific Tri-Nations|"
    r"Autumn Nations|FIRA|Tbilisi Cup|Pacific Challenge|mid-year|end-of-year|June|July|November",
    re.I,
)
EXCLUDE = re.compile(
    r"women|sevens|7s|Laurie O'Reilly|Pacific Four|Under|U-?\d\d|Oxford|Cambridge|Barbarians|"
    r"Lions|Varsity|Junior|Youth|Schools|club|league|\"B\"|Super Cup",
    re.I,
)


def get(session, params, timeout=60):
    """GET with backoff: Wikipedia answers bursts with 429 and a Retry-After header."""
    for attempt in range(6):
        r = session.get(API, params=params, timeout=timeout)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        wait = float(r.headers.get("Retry-After", 5 * 2 ** attempt))
        print(f"  rate-limited, waiting {wait:.0f}s")
        time.sleep(wait)
    r.raise_for_status()


def list_pages(session):
    titles, cont = [], {}
    while True:
        data = get(session, {"action": "query", "list": "embeddedin",
                             "eititle": "Template:Rugbybox", "einamespace": 0,
                             "eilimit": 500, "format": "json", **cont})
        titles += [p["title"] for p in data["query"]["embeddedin"]]
        if "continue" not in data:
            return titles
        cont = data["continue"]
        time.sleep(1)


def fetch_wikitext(session, titles):
    pages = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = get(session, {"action": "query", "prop": "revisions", "rvprop": "content",
                             "rvslots": "main", "titles": "|".join(batch),
                             "format": "json", "formatversion": 2})
        for p in data["query"]["pages"]:
            if "revisions" in p:
                pages[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
        print(f"  {min(i + 50, len(titles))}/{len(titles)} pages", end="\r")
        time.sleep(1.5)
    print()
    return pages


def main(force=False):
    session = requests.Session()
    session.headers.update(HEADERS)
    if not KAGGLE_OUT.exists() or force:
        r = session.get(KAGGLE_URL, timeout=60)
        r.raise_for_status()
        KAGGLE_OUT.parent.mkdir(parents=True, exist_ok=True)
        KAGGLE_OUT.write_bytes(r.content)
        print(f"saved Kaggle tier-1 results -> {KAGGLE_OUT}")
    if OUT.exists() and not force:
        print(f"{OUT} exists (use --force to re-download)")
        return
    titles = list_pages(session)
    keep = sorted(t for t in titles if INCLUDE.search(t) and not EXCLUDE.search(t))
    print(f"{len(titles)} pages use {{{{rugbybox}}}}; {len(keep)} look international")
    pages = fetch_wikitext(session, keep)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pages))
    print(f"saved {len(pages)} pages, {OUT.stat().st_size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
