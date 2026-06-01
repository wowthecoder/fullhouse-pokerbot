import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bots.my_bots import complete_v2 as bot


@pytest.fixture(autouse=True)
def reset_models():
    bot._reset_opponent_models()
    bot.EQUITY_RANGE_CACHE.clear()
    yield
    bot._reset_opponent_models()
    bot.EQUITY_RANGE_CACHE.clear()


def player(seat, stack=10_000, folded=False, all_in=False, bet=0):
    return {
        "seat": seat,
        "bot_id": "hero" if seat == 0 else f"villain_{seat}",
        "stack": stack,
        "state": "folded" if folded else "all_in" if all_in else "active",
        "is_folded": folded,
        "is_all_in": all_in,
        "bet_this_street": bet,
        "hole_cards": None,
    }


def make_state(
    *,
    hand_id="hand",
    seat=0,
    n_players=2,
    cards=("Ah", "Kh"),
    board=("Qh", "Jh", "Th"),
    street="flop",
    current_bet=0,
    amount_owed=0,
    min_raise_to=100,
    can_check=True,
    pot=1_000,
    action_log=(),
):
    players = [player(idx) for idx in range(n_players)]
    blind_log = [
        {"seat": 0, "action": "small_blind", "amount": 50},
        {"seat": 1, "action": "big_blind", "amount": 100},
    ]
    return {
        "type": "action_request",
        "hand_id": hand_id,
        "street": street,
        "seat_to_act": seat,
        "pot": pot,
        "community_cards": list(board),
        "current_bet": current_bet,
        "min_raise_to": min_raise_to,
        "amount_owed": amount_owed,
        "can_check": can_check,
        "your_cards": list(cards),
        "your_stack": 10_000,
        "your_bet_this_street": 0,
        "players": players,
        "action_log": blind_log + list(action_log),
    }


def weighted_range(*items):
    return bot._make_weighted_range(list(items))


def test_estimate_equity_exact_river_win():
    rng = random.Random(1)
    villain_range = weighted_range(("Ac", "Ad", 1.0), ("2s", "2d", 1.0))

    equity = bot._estimate_equity(
        ["Ah", "Kh"],
        ["Qh", "Jh", "Th", "2c", "3d"],
        villain_range,
        rng=rng,
    )

    assert equity == pytest.approx(1.0)


def test_estimate_equity_exact_river_tie_uses_half_credit():
    rng = random.Random(1)
    villain_range = weighted_range(("4c", "5d", 1.0))

    equity = bot._estimate_equity(
        ["2c", "3d"],
        ["As", "Ks", "Qs", "Js", "Ts"],
        villain_range,
        rng=rng,
    )

    assert equity == pytest.approx(0.5)


def test_estimate_equity_skips_dead_card_combos():
    rng = random.Random(1)
    villain_range = weighted_range(("Ah", "Ac", 1000.0), ("2s", "2d", 1.0))

    equity = bot._estimate_equity(
        ["Ah", "Kh"],
        ["Qh", "Jh", "Th", "2c", "3d"],
        villain_range,
        rng=rng,
    )

    assert equity == pytest.approx(1.0)


def test_estimate_equity_is_deterministic_with_seeded_rng():
    villain_range = weighted_range(("As", "Ad", 1.0), ("2s", "2d", 2.0), ("9c", "8c", 1.0))

    first = bot._estimate_equity(
        ["Ah", "Kh"],
        ["Qh", "Jh", "2c"],
        villain_range,
        samples=80,
        rng=random.Random(42),
    )
    second = bot._estimate_equity(
        ["Ah", "Kh"],
        ["Qh", "Jh", "2c"],
        villain_range,
        samples=80,
        rng=random.Random(42),
    )

    assert first == second


def test_facing_postflop_bet_calls_when_equity_clears_pot_odds(monkeypatch):
    state = make_state(current_bet=250, amount_owed=250, can_check=False, pot=1_000)
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.30)

    assert bot._facing_postflop_bet(state, info, board) == {"action": "call"}


def test_facing_postflop_bet_folds_when_equity_misses_threshold(monkeypatch):
    state = make_state(current_bet=700, amount_owed=700, can_check=False, pot=1_000)
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.20)

    assert bot._facing_postflop_bet(state, info, board) == {"action": "fold"}


def test_facing_postflop_bet_is_not_rescued_by_hand_tier_when_equity_is_low(monkeypatch):
    state = make_state(
        cards=("As", "Kh"),
        board=("Ad", "7c", "2h", "Td", "9s"),
        street="river",
        current_bet=900,
        amount_owed=900,
        can_check=False,
        pot=1_000,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "raise", "amount": 900, "street": "river"},
        ],
    )
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.28)

    assert info["strength"] == "tptk_plus"
    assert bot._facing_postflop_bet(state, info, board) == {"action": "fold"}


def test_positive_ev_draw_semi_bluff_raises(monkeypatch):
    state = make_state(
        hand_id="semi-bluff",
        cards=("Ks", "Jh"),
        board=("Qd", "Ts", "2c"),
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
        ],
    )
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.42)
    monkeypatch.setattr(bot, "_fold_equity_estimate", lambda state, bet_fraction, board, role: 0.62)

    action = bot.decide(state)

    assert action["action"] == "raise"


def test_negative_ev_draw_semi_bluff_checks(monkeypatch):
    state = make_state(
        hand_id="no-semi-bluff",
        cards=("Ks", "Jh"),
        board=("Qd", "Ts", "2c"),
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
        ],
    )
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.10)
    monkeypatch.setattr(bot, "_fold_equity_estimate", lambda state, bet_fraction, board, role: 0.05)

    assert bot.decide(state) == {"action": "check"}
