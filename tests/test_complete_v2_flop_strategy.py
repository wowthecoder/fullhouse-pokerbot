import os
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
    hand_id="flop-hand",
    seat=0,
    n_players=2,
    cards=("Ah", "Kh"),
    board=("As", "7d", "2c"),
    pot=1_000,
    action_log=(),
):
    blind_log = [
        {"seat": 0, "action": "small_blind", "amount": 50, "street": "preflop"},
        {"seat": 1, "action": "big_blind", "amount": 100, "street": "preflop"},
    ]
    return {
        "type": "action_request",
        "hand_id": hand_id,
        "street": "flop",
        "seat_to_act": seat,
        "pot": pot,
        "community_cards": list(board),
        "current_bet": 0,
        "min_raise_to": 100,
        "amount_owed": 0,
        "can_check": True,
        "your_cards": list(cards),
        "your_stack": 10_000,
        "your_bet_this_street": 0,
        "players": [player(idx) for idx in range(n_players)],
        "action_log": blind_log + list(action_log),
    }


def pfa_log(hero=0, villain=1):
    return [
        {"seat": hero, "action": "raise", "amount": 250, "street": "preflop"},
        {"seat": villain, "action": "call", "amount": 250, "street": "preflop"},
    ]


def threebet_pfa_log(hero=0, villain=1):
    return [
        {"seat": villain, "action": "raise", "amount": 250, "street": "preflop"},
        {"seat": hero, "action": "raise", "amount": 800, "street": "preflop"},
        {"seat": villain, "action": "call", "amount": 800, "street": "preflop"},
    ]


def bb_call_log():
    return [
        {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
        {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
    ]


@pytest.mark.parametrize(
    ("board", "bucket"),
    [
        (("Ah", "7h", "2h"), "monotone"),
        (("Ks", "Kd", "2c"), "paired_high"),
        (("7s", "7d", "2c"), "paired_low"),
        (("As", "7d", "2c"), "A_high_dry"),
        (("Ks", "7d", "2c"), "K_high_dry"),
        (("Qs", "Js", "9d"), "QJT_connected"),
        (("6h", "5c", "4s"), "low_connected_rainbow"),
        (("6h", "5h", "4s"), "low_connected_twotone"),
        (("9s", "7s", "5d"), "two_tone_dynamic"),
        (("As", "Kd", "7c"), "broadway_heavy"),
        (("Qs", "7d", "2c"), "static_high"),
        (("9s", "8d", "6c"), "dynamic_other"),
    ],
)
def test_flop_bucket_classifies_named_textures(board, bucket):
    assert bot._flop_bucket(bot._board_class(board)) == bucket


def test_dry_ace_overfolder_can_face_small_air_cbet_without_good_bluff(monkeypatch):
    state = make_state(
        hand_id="overfolder-air",
        cards=("Kh", "Qd"),
        board=("As", "7d", "2c"),
        action_log=pfa_log(),
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "overfolder")
    monkeypatch.setattr(bot, "_flop_mixed_frequency_allows", lambda state, bucket, role, frequency: True)
    monkeypatch.setattr(bot, "_good_flop_bluff", lambda info, board, role: False)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.10)

    assert bot.decide(state) == {"action": "raise", "amount": 250}


def test_mid_frequency_connected_board_checks_medium_showdown(monkeypatch):
    state = make_state(
        hand_id="connected-medium-check",
        cards=("9h", "2c"),
        board=("Qs", "Js", "9d"),
        action_log=pfa_log(),
    )
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: None)

    assert bot.decide(state) == {"action": "check"}


def test_mid_frequency_connected_board_bets_polar_value_larger(monkeypatch):
    state = make_state(
        hand_id="connected-value-bet",
        cards=("Ah", "Kd"),
        board=("Qs", "Js", "Tc"),
        action_log=pfa_log(),
    )
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: None)

    assert bot.decide(state) == {"action": "raise", "amount": 600}


def test_strong_blocker_top_set_checks_back_on_dry_ace(monkeypatch):
    state = make_state(
        hand_id="top-set-check",
        cards=("As", "Ad"),
        board=("Ah", "7c", "2d"),
        action_log=pfa_log(),
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "unknown")
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.95)

    assert bot.decide(state) == {"action": "check"}


def test_multiway_checks_air_but_bets_strong_value(monkeypatch):
    air = make_state(
        hand_id="multiway-air",
        n_players=3,
        cards=("4h", "2d"),
        board=("Qs", "Js", "9d"),
        action_log=pfa_log(hero=0, villain=1),
    )
    value = make_state(
        hand_id="multiway-value",
        n_players=3,
        cards=("Ah", "Kd"),
        board=("Qs", "Js", "Tc"),
        action_log=pfa_log(hero=0, villain=1),
    )
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: None)

    assert bot.decide(air) == {"action": "check"}
    assert bot.decide(value) == {"action": "raise", "amount": 670}


def test_bb_donks_654_rainbow_but_tightens_dangerous_low_boards(monkeypatch):
    donk = make_state(
        hand_id="bb-donk-rainbow",
        seat=1,
        cards=("7h", "3d"),
        board=("6h", "5c", "4s"),
        action_log=bb_call_log(),
    )
    dangerous = make_state(
        hand_id="bb-check-twotone",
        seat=1,
        cards=("Ah", "Kd"),
        board=("6h", "5h", "4s"),
        action_log=bb_call_log(),
    )
    ace_high = make_state(
        hand_id="bb-check-axx",
        seat=1,
        cards=("As", "Ad"),
        board=("Ah", "7c", "2d"),
        action_log=bb_call_log(),
    )
    monkeypatch.setattr(bot, "_flop_mixed_frequency_allows", lambda state, bucket, role, frequency: True)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: None)

    assert bot.decide(donk) == {"action": "raise", "amount": 670}
    assert bot.decide(dangerous) == {"action": "check"}
    assert bot.decide(ace_high) == {"action": "check"}


def test_threebet_pot_static_small_bet_dynamic_uses_larger_polar_size(monkeypatch):
    static = make_state(
        hand_id="threebet-static",
        cards=("Kh", "Qd"),
        board=("As", "7d", "2c"),
        action_log=threebet_pfa_log(),
    )
    dynamic = make_state(
        hand_id="threebet-dynamic",
        cards=("Ah", "Kd"),
        board=("Qs", "Js", "Tc"),
        action_log=threebet_pfa_log(),
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "overfolder")
    monkeypatch.setattr(bot, "_flop_mixed_frequency_allows", lambda state, bucket, role, frequency: True)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: None)

    assert bot.decide(static) == {"action": "raise", "amount": 250}
    assert bot._postflop_size(dynamic, bot._board_class(dynamic["community_cards"]), bot._hand_info(dynamic), "IP_PFA") == 0.60
    assert bot.decide(dynamic) == {"action": "raise", "amount": 600}


def test_calling_station_checks_air_and_faces_larger_value_bet(monkeypatch):
    air = make_state(
        hand_id="station-air",
        cards=("Kh", "Qd"),
        board=("As", "7d", "2c"),
        action_log=pfa_log(),
    )
    value = make_state(
        hand_id="station-value",
        cards=("Ks", "Qd"),
        board=("Kh", "7c", "2d"),
        action_log=pfa_log(),
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "calling_station")
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: None)

    assert bot.decide(air) == {"action": "check"}
    assert bot.decide(value) == {"action": "raise", "amount": 750}
