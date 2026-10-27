"""Three sequential rating systems, each updated one match at a time.

Elo            margin-of-victory Elo (FiveThirtyEight-style multiplier with autocorrelation fix)
WorldRugby     World Rugby's official ranking formula, reconstructed from results
AttackDefence  opponent-adjusted points (or tries) scored and conceded, updated like Elo

All three expose `pre(home, away, venue)` (the pre-match view) and `update(...)` (post-match).
`venue` is +1 when the listed home side is at home, 0 neutral, -1 when the away side is at home.
"""
import math
from collections import defaultdict

# Original members at the founding of the international game keep a strong prior; everyone else
# starts lower with a provisional (larger) K until they have played enough tests.
FOUNDERS = {"ENG", "SCO", "IRE", "WAL", "FRA", "NZL", "AUS", "RSA"}


class Elo:
    def __init__(self, k=36.0, home=120.0, new_team=1300.0, founder=1500.0, provisional=10):
        self.k, self.home, self.new_team, self.founder = k, home, new_team, founder
        self.provisional = provisional
        self.r = {}
        self.n = defaultdict(int)

    def rating(self, team):
        if team not in self.r:
            self.r[team] = self.founder if team in FOUNDERS else self.new_team
        return self.r[team]

    def diff(self, home, away, venue):
        return self.rating(home) - self.rating(away) + self.home * venue

    def expected(self, home, away, venue):
        return 1.0 / (1.0 + 10 ** (-self.diff(home, away, venue) / 400.0))

    def pre(self, home, away, venue):
        return {"elo_home": self.rating(home), "elo_away": self.rating(away),
                "elo_diff": self.diff(home, away, venue), "elo_p": self.expected(home, away, venue)}

    def update(self, home, away, venue, hs, as_):
        d = self.diff(home, away, venue)
        e = 1.0 / (1.0 + 10 ** (-d / 400.0))
        s = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        margin = abs(hs - as_)
        # Bigger wins move ratings more (log margin), damped when the favourite wins big so
        # ratings don't inflate (FiveThirtyEight's autocorrelation correction).
        winner_d = d if hs > as_ else -d
        mult = math.log(margin + 1.0) * 2.2 / (max(winner_d, -400) * 0.001 + 2.2) if margin else 1.0
        for team, sign in ((home, 1), (away, -1)):
            k = self.k * (2.0 if self.n[team] < self.provisional else 1.0)
            self.r[team] = self.rating(team) + sign * k * mult * (s - e)
            self.n[team] += 1


class WorldRugby:
    """World Rugby's ranking formula: gap D = R_home + 3*home - R_away, capped at +-10;
    expected E = 0.5 + D/20; exchange = 2(S - E), x1.5 for a margin over 15, x2 at a World Cup.
    It's a linear Elo with no margin information, the official yardstick to beat."""

    def __init__(self, new_team=40.0, founder=70.0):
        self.new_team, self.founder = new_team, founder
        self.r = {}

    def rating(self, team):
        if team not in self.r:
            self.r[team] = self.founder if team in FOUNDERS else self.new_team
        return self.r[team]

    def gap(self, home, away, venue):
        d = self.rating(home) - self.rating(away) + 3.0 * venue
        return max(-10.0, min(10.0, d))

    def pre(self, home, away, venue):
        return {"wr_home": self.rating(home), "wr_away": self.rating(away),
                "wr_gap": self.gap(home, away, venue)}

    def update(self, home, away, venue, hs, as_, world_cup=False):
        e = 0.5 + self.gap(home, away, venue) / 20.0
        s = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        x = 2.0 * (s - e) * (1.5 if abs(hs - as_) > 15 else 1.0) * (2.0 if world_cup else 1.0)
        self.r[home] = self.rating(home) + x
        self.r[away] = self.rating(away) - x


class AttackDefence:
    """Expected points for = mu + attack_self - defence_opp +- half the home advantage.
    After each match, attack and defence move by k times the (capped) surprise."""

    def __init__(self, k=0.08, home=4.0, mu=18.0, k_mu=0.002, cap=40.0):
        self.k, self.home, self.mu, self.k_mu, self.cap = k, home, mu, k_mu, cap
        self.att = defaultdict(float)
        self.dfn = defaultdict(float)

    def expected(self, home, away, venue):
        h = self.mu + self.att[home] - self.dfn[away] + self.home / 2 * venue
        a = self.mu + self.att[away] - self.dfn[home] - self.home / 2 * venue
        return max(h, 0.0), max(a, 0.0)

    def pre(self, home, away, venue, prefix="pts"):
        h, a = self.expected(home, away, venue)
        return {f"{prefix}_exp_home": h, f"{prefix}_exp_away": a, f"{prefix}_exp_margin": h - a,
                f"{prefix}_att_home": self.att[home], f"{prefix}_def_home": self.dfn[home],
                f"{prefix}_att_away": self.att[away], f"{prefix}_def_away": self.dfn[away]}

    def update(self, home, away, venue, hs, as_):
        eh, ea = self.expected(home, away, venue)
        rh = max(-self.cap, min(self.cap, hs - eh))
        ra = max(-self.cap, min(self.cap, as_ - ea))
        self.att[home] += self.k * rh
        self.dfn[away] -= self.k * rh
        self.att[away] += self.k * ra
        self.dfn[home] -= self.k * ra
        self.mu += self.k_mu * (rh + ra) / 2
