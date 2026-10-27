import pytest

from rugby.ratings import AttackDefence, Elo, WorldRugby


def test_world_rugby_exchange_matches_published_examples():
    # World Rugby's explainer: equal teams, home side wins -> home gains 1 - 3/10 = 0.7.
    wr = WorldRugby()
    wr.r = {"A": 80.0, "B": 80.0}
    wr.update("A", "B", 1, 20, 10)
    assert wr.r["A"] == pytest.approx(80.7)
    assert wr.r["B"] == pytest.approx(79.3)
    # A win by more than 15 is worth 1.5x, a World Cup match doubles it.
    wr.r = {"A": 80.0, "B": 80.0}
    wr.update("A", "B", 0, 30, 10, world_cup=True)
    assert wr.r["A"] == pytest.approx(80.0 + 1.0 * 1.5 * 2)
    # Gap capped at 10: a side 15 points better gains nothing for a win.
    wr.r = {"A": 95.0, "B": 80.0}
    wr.update("A", "B", 0, 20, 10)
    assert wr.r["A"] == pytest.approx(95.0)


def test_elo_is_zero_sum_after_provisional_period_and_rewards_upsets():
    elo = Elo(provisional=0)
    elo.r = {"A": 1600.0, "B": 1400.0}
    elo.update("A", "B", 0, 10, 20)  # upset
    assert elo.r["A"] + elo.r["B"] == pytest.approx(3000.0)
    assert elo.r["B"] > 1400.0 + 20
    big, small = Elo(provisional=0), Elo(provisional=0)
    big.r = {"A": 1500.0, "B": 1500.0}
    small.r = {"A": 1500.0, "B": 1500.0}
    big.update("A", "B", 0, 50, 0)
    small.update("A", "B", 0, 11, 10)
    assert big.r["A"] > small.r["A"]  # margin of victory matters


def test_elo_home_advantage():
    elo = Elo()
    elo.r = {"A": 1500.0, "B": 1500.0}
    assert elo.expected("A", "B", 1) > 0.5 > elo.expected("A", "B", -1)
    assert elo.expected("A", "B", 0) == pytest.approx(0.5)


def test_attack_defence_learns_strengths():
    ad = AttackDefence()
    for _ in range(300):
        ad.update("A", "B", 0, 35, 10)
    h, a = ad.expected("A", "B", 0)
    assert h - a == pytest.approx(25, abs=3)
