import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bots.my_bots import complete_v2 as bot


@pytest.fixture(autouse=True)
def reset_models():
    bot._reset_opponent_models()
    yield
    bot._reset_opponent_models()


def player(seat, stack=10_000, folded=False, bet=0):
    return {
        "seat": seat,
        "bot_id": "hero" if seat == 0 else f"villain_{seat}",
        "stack": stack,
        "state": "folded" if folded else "active",
        "is_folded": folded,
        "is_all_in": False,
        "bet_this_street": bet,
        "hole_cards": None,
    }


def make_preflop_state(
    *,
    hand_id="preflop",
    seat=0,
    cards=("As", "Ah"),
    players=None,
    blinds=(1, 2),
    current_bet=0,
    amount_owed=0,
    min_raise_to=100,
    your_stack=10_000,
    your_bet=0,
    action_log=(),
):
    if players is None:
        players = [player(idx, stack=your_stack if idx == seat else 10_000) for idx in range(6)]
    blind_log = [
        {"seat": blinds[0], "action": "small_blind", "amount": 50, "street": "preflop"},
        {"seat": blinds[1], "action": "big_blind", "amount": 100, "street": "preflop"},
    ]
    return {
        "type": "action_request",
        "hand_id": hand_id,
        "street": "preflop",
        "seat_to_act": seat,
        "pot": 1_350,
        "community_cards": [],
        "current_bet": current_bet,
        "min_raise_to": min_raise_to,
        "amount_owed": amount_owed,
        "can_check": amount_owed == 0,
        "your_cards": list(cards),
        "your_stack": your_stack,
        "your_bet_this_street": your_bet,
        "players": players,
        "action_log": blind_log + list(action_log),
    }


def test_hero_4bet_uses_chart_size_at_30bb():
    state = make_preflop_state(
        hand_id="chart-4bet-30bb",
        seat=0,
        cards=("As", "Ah"),
        current_bet=900,
        amount_owed=650,
        min_raise_to=1_550,
        your_stack=2_750,
        your_bet=250,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 2, "action": "raise", "amount": 900, "street": "preflop"},
        ],
    )

    assert bot.decide(state) == {"action": "raise", "amount": 2_500}


def test_hero_4bet_jams_at_25bb():
    state = make_preflop_state(
        hand_id="chart-4bet-25bb",
        seat=0,
        cards=("As", "Ah"),
        current_bet=900,
        amount_owed=650,
        min_raise_to=1_550,
        your_stack=2_250,
        your_bet=250,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 2, "action": "raise", "amount": 900, "street": "preflop"},
        ],
    )

    assert bot.decide(state) == {"action": "all_in"}


def test_bb_folds_dominated_offsuit_defense_vs_button_open():
    state = make_preflop_state(
        hand_id="bb-dominated-offsuit",
        seat=2,
        cards=("Ah", "8d"),
        current_bet=250,
        amount_owed=150,
        min_raise_to=400,
        your_bet=100,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
        ],
    )

    assert bot.decide(state) == {"action": "fold"}


def test_bb_keeps_suited_defense_vs_button_open():
    state = make_preflop_state(
        hand_id="bb-suited-defense",
        seat=2,
        cards=("Kh", "7h"),
        current_bet=250,
        amount_owed=150,
        min_raise_to=400,
        your_bet=100,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
        ],
    )

    assert bot.decide(state) == {"action": "call"}


def test_three_handed_preflop_uses_live_button_not_sixmax_lj():
    players = [
        player(0, bet=50),
        player(1, bet=100),
        player(2),
        player(3, folded=True),
        player(4, folded=True),
        player(5, folded=True),
    ]
    state = make_preflop_state(
        hand_id="three-handed-button",
        seat=2,
        cards=("Kh", "9d"),
        players=players,
        blinds=(0, 1),
        current_bet=100,
        amount_owed=100,
        min_raise_to=200,
    )

    situation = bot._parse_preflop_situation(state)

    assert situation["hero_pos"] == "BTN"
    assert bot.decide(state) == {"action": "raise", "amount": 250}


def test_heads_up_preflop_uses_only_blind_positions():
    players = [
        player(0, bet=50),
        player(1, bet=100),
        player(2, folded=True),
        player(3, folded=True),
        player(4, folded=True),
        player(5, folded=True),
    ]
    state = make_preflop_state(
        hand_id="heads-up-blinds",
        seat=0,
        cards=("Kh", "9d"),
        players=players,
        blinds=(0, 1),
        current_bet=100,
        amount_owed=50,
        min_raise_to=200,
        your_bet=50,
    )

    assert bot._preflop_positions(state) == {0: "SB", 1: "BB"}
    assert bot._parse_preflop_situation(state)["hero_pos"] == "SB"
