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


def player(seat, stack=10_000, folded=False, all_in=False):
    return {
        "seat": seat,
        "bot_id": "hero" if seat == 0 else f"villain_{seat}",
        "stack": stack,
        "state": "folded" if folded else "all_in" if all_in else "active",
        "is_folded": folded,
        "is_all_in": all_in,
        "bet_this_street": 0,
        "hole_cards": None,
    }


def make_turn_state(
    *,
    hand_id="turn-hand",
    seat=0,
    cards=("Qh", "Jd"),
    board=("Ks", "7d", "2c", "3h"),
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
        "street": "turn",
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
        "players": [player(0), player(1)],
        "action_log": blind_log + list(action_log),
    }


def pfa_cbet_called_log(hero=0, villain=1):
    return [
        {"seat": hero, "action": "raise", "amount": 250, "street": "preflop"},
        {"seat": villain, "action": "call", "amount": 250, "street": "preflop"},
        {"seat": hero, "action": "raise", "amount": 330, "street": "flop"},
        {"seat": villain, "action": "call", "amount": 330, "street": "flop"},
    ]


def checked_through_log(pfa=0, caller=1):
    return [
        {"seat": pfa, "action": "raise", "amount": 250, "street": "preflop"},
        {"seat": caller, "action": "call", "amount": 250, "street": "preflop"},
        {"seat": caller, "action": "check", "street": "flop"},
        {"seat": pfa, "action": "check", "street": "flop"},
    ]


def test_classify_turn_low_blank_does_not_reuse_high_flop_as_scare():
    effect = bot.classify_turn_card(("Ks", "7d", "2c"), "3h", "IP_PFA")

    assert effect["overcard_to_flop"] is False
    assert effect["improves_PFA_range"] is False
    assert bot._turn_good_for_pfa_barrel(effect, {"strength": "air"}, "unknown") is False


def test_classify_turn_ace_overcard_improves_pfa_range():
    effect = bot.classify_turn_card(("Ks", "7d", "2c"), "Ah", "IP_PFA")

    assert effect["overcard_to_flop"] is True
    assert effect["improves_PFA_range"] is True


def test_classify_turn_flush_texture_changes():
    completes = bot.classify_turn_card(("Kh", "7h", "2c"), "Ah", "IP_PFA")
    adds = bot.classify_turn_card(("Ks", "7d", "2c"), "As", "IP_PFA")

    assert completes["completes_flush"] is True
    assert adds["adds_flush_draw"] is True


def test_classify_turn_connected_runout_marks_straight_or_caller_gain():
    effect = bot.classify_turn_card(("9s", "8d", "6c"), "7h", "IP_PFA")

    assert effect["completes_straight"] is True
    assert effect["improves_caller_range"] is True


def test_pfa_air_checks_low_blank_after_cbet(monkeypatch):
    state = make_turn_state(
        hand_id="blank-turn-check",
        cards=("Qh", "Jd"),
        board=("Ks", "7d", "2c", "3h"),
        action_log=pfa_cbet_called_log(),
    )
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.35)
    monkeypatch.setattr(bot, "_should_semi_bluff", lambda *args, **kwargs: True)

    assert bot.decide(state) == {"action": "check"}


def test_pfa_can_barrel_ace_overcard_after_cbet(monkeypatch):
    state = make_turn_state(
        hand_id="ace-turn-barrel",
        cards=("Qh", "Jd"),
        board=("Ks", "7d", "2c", "Ah"),
        action_log=pfa_cbet_called_log(),
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "unknown")
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.35)
    monkeypatch.setattr(bot, "_should_semi_bluff", lambda *args, **kwargs: True)

    assert bot.decide(state) == {"action": "raise", "amount": 670}


def test_calling_station_suppresses_pure_turn_bluff(monkeypatch):
    state = make_turn_state(
        hand_id="station-no-turn-bluff",
        cards=("9h", "8d"),
        board=("Ks", "7d", "2c", "Ah"),
        action_log=pfa_cbet_called_log(),
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "calling_station")
    monkeypatch.setattr(bot, "_should_semi_bluff", lambda *args, **kwargs: True)

    assert bot.decide(state) == {"action": "check"}


def test_overfolder_can_face_favorable_turn_barrel_from_fold_equity(monkeypatch):
    state = make_turn_state(
        hand_id="overfolder-turn-barrel",
        cards=("9h", "8d"),
        board=("Ks", "7d", "2c", "Ah"),
        action_log=pfa_cbet_called_log(),
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "overfolder")
    monkeypatch.setattr(bot, "_should_semi_bluff", lambda *args, **kwargs: False)
    monkeypatch.setattr(bot, "_fold_equity_estimate", lambda *args, **kwargs: 0.60)

    assert bot.decide(state) == {"action": "raise", "amount": 670}


def test_oop_probe_fires_after_checked_through_caller_favorable_turn(monkeypatch):
    state = make_turn_state(
        hand_id="oop-probe-low-turn",
        seat=1,
        cards=("8c", "4s"),
        board=("8s", "6d", "2c", "7h"),
        action_log=checked_through_log(),
    )
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: None)

    assert bot.decide(state) == {"action": "raise", "amount": 670}
