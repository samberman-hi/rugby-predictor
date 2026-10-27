import pandas as pd

from rugby import data

BOX = """
=== Pool A ===
{{rugbybox
|date = 18 October 2025
|home = (1 BP) {{ru-rt|IRE}}
|score = 13&ndash;23
|away = {{ru|NZL}}
|try1 = [[James Lowe|Lowe]] 21' c
|try2 = [[Will Jordan|Jordan]] (2) 5' c, 16' m<br />Penalty try 70'
|stadium = [[Aviva Stadium]], [[Dublin]]<ref>{{cite web|url=x}}</ref>
}}
{{Rugbybox collapsible
|date = 2026-11-07
|team1 = {{Ru-rt|WAL}}
|score =
|team2 = {{ru|JAP}}
|stadium = [[Millennium Stadium]], [[Cardiff]]
}}
{{rugbybox
|date = 1 July 2025
|home = {{Ru-rt|JPN|name=Japan XV}}
|score = 10–12
|away = {{ruA|AUS}}
}}
"""


def test_parse_page_keeps_senior_tests_and_fixtures():
    rows = data.parse_page("2025 end-of-year rugby union internationals", BOX)
    assert len(rows) == 2  # the Japan XV v Australia A box is dropped
    played, fixture = rows
    assert (played["home"], played["away"]) == ("IRE", "NZL")
    assert (played["home_score"], played["away_score"]) == (13, 23)
    assert (played["home_tries"], played["away_tries"]) == (1, 3)
    assert played["city"] == "Dublin"
    assert (fixture["home"], fixture["away"]) == ("WAL", "JPN")  # team1/team2, JAP alias
    assert fixture["date"] == pd.Timestamp("2026-11-07")
    assert pd.isna(fixture["home_score"])


def test_scores_and_dates():
    assert data.parse_score("33 - 11") == (33, 11)
    assert data.parse_score("[[2019 Rugby World Cup knockout stage#Wales|19–20]]") == (19, 20)
    assert data.parse_score("Cancelled") == "void"
    assert data.parse_date("{{start date|2019|10|20|df=y}}", None) == pd.Timestamp("2019-10-20")
    assert data.parse_date("June 14, 2003", None) == pd.Timestamp("2003-06-14")
    assert data.parse_date("14 June", 2003) == pd.Timestamp("2003-06-14")


def test_try_counting():
    assert data.count_tries("Smith, Jones (2)") == 3
    assert data.count_tries("Smith (2) 5', 16'<br />Brown 40'") == 3
    assert data.count_tries("") == 0


def test_classify():
    assert data.classify("2023 Rugby World Cup knockout stage", "Final") == ("rwc", True)
    assert data.classify("2023 Rugby World Cup Pool B", "Pool B") == ("rwc", False)
    assert data.classify("2014 Asian Five Nations", "")[0] == "tier2_comp"
    assert data.classify("2018 South American Six Nations", "")[0] == "tier2_comp"
    assert data.classify("2026 Nations Championship Northern Hemisphere Series", "")[0] == "nations_championship"


def test_dedupe_merges_cross_page_duplicates():
    base = dict(home="IRE", away="NZL", home_score=13.0, away_score=23.0, knockout=False,
                city="Dublin", venue="", home_tries=float("nan"), away_tries=float("nan"))
    df = pd.DataFrame([
        dict(base, date=pd.Timestamp("2025-10-18"), comp="test", page="tour page"),
        dict(base, date=pd.Timestamp("2025-10-19"), comp="six_nations", page="season page"),
        dict(base, date=pd.Timestamp("2026-02-01"), comp="six_nations", page="next year"),
    ])
    out = data.dedupe(df)
    assert len(out) == 2
    assert out.iloc[0].page == "season page"  # more specific competition wins


def test_infer_venues():
    rows = [dict(date=pd.Timestamp(f"20{y:02d}-06-01"), home="IRE", away="NZL", comp="test", city="Dublin")
            for y in range(10, 14)]
    rows.append(dict(date=pd.Timestamp("2016-11-05"), home="NZL", away="IRE", comp="test", city="Chicago"))
    rows.append(dict(date=pd.Timestamp("2018-11-17"), home="NZL", away="IRE", comp="test", city="Dublin"))
    out = data.infer_venues(pd.DataFrame(rows))
    assert list(out.venue_country) == [1, 1, 1, 1, 0, -1]  # Chicago is neutral
