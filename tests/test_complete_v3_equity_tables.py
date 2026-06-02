import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bots.my_bots import complete_v3 as bot
from tools import generate_complete_equity_tables as generator


@pytest.fixture(autouse=True)
def reset_tables():
    old_tables = bot.COMPLETE_EQUITY_TABLES
    bot.COMPLETE_EQUITY_TABLES = {}
    bot.EQUITY_RESULT_CACHE.clear()
    yield
    bot.COMPLETE_EQUITY_TABLES = old_tables
    bot.EQUITY_RESULT_CACHE.clear()


def player(seat, folded=False):
    return {
        "seat": seat,
        "bot_id": "hero" if seat == 0 else f"villain_{seat}",
        "stack": 10_000,
        "state": "folded" if folded else "active",
        "is_folded": folded,
        "is_all_in": False,
        "bet_this_street": 0,
        "hole_cards": None,
    }


def forbid_equity_estimate(*args, **kwargs):
    raise AssertionError("simple flop c-bet should not estimate equity")


def flop_state():
    return {
        "type": "action_request",
        "hand_id": "lookup-flop",
        "street": "flop",
        "seat_to_act": 0,
        "pot": 1_000,
        "community_cards": ["As", "7d", "2c"],
        "current_bet": 0,
        "min_raise_to": 100,
        "amount_owed": 0,
        "can_check": True,
        "your_cards": ["Ah", "Kh"],
        "your_stack": 10_000,
        "your_bet_this_street": 0,
        "players": [player(0), player(1)],
        "action_log": [],
    }


def river_state():
    return {
        "type": "action_request",
        "hand_id": "lookup-river",
        "street": "river",
        "seat_to_act": 0,
        "pot": 1_000,
        "community_cards": ["As", "7d", "2c", "Kc", "9h"],
        "current_bet": 0,
        "min_raise_to": 100,
        "amount_owed": 0,
        "can_check": True,
        "your_cards": ["Ah", "Kh"],
        "your_stack": 5_000,
        "your_bet_this_street": 0,
        "players": [player(0), player(1)],
        "action_log": [],
    }


def turn_state():
    return {
        "type": "action_request",
        "hand_id": "lookup-turn",
        "street": "turn",
        "seat_to_act": 0,
        "pot": 1_000,
        "community_cards": ["As", "7d", "2c", "Kc"],
        "current_bet": 0,
        "min_raise_to": 100,
        "amount_owed": 0,
        "can_check": True,
        "your_cards": ["Ah", "Kh"],
        "your_stack": 3_000,
        "your_bet_this_street": 0,
        "players": [player(0), player(1)],
        "action_log": [],
    }


def preflop_state():
    return {
        "type": "action_request",
        "hand_id": "deep-fivebet",
        "street": "preflop",
        "seat_to_act": 0,
        "pot": 6_000,
        "community_cards": [],
        "current_bet": 5_500,
        "min_raise_to": 8_000,
        "amount_owed": 3_000,
        "can_check": False,
        "your_cards": ["Ah", "Kd"],
        "your_stack": 10_000,
        "your_bet_this_street": 2_500,
        "players": [player(0), player(1)],
        "action_log": [
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "raise", "amount": 800, "street": "preflop"},
            {"seat": 0, "action": "raise", "amount": 2_500, "street": "preflop"},
            {"seat": 1, "action": "raise", "amount": 5_500, "street": "preflop"},
        ],
    }


def committing_fourbet_tree_state():
    return {
        "type": "action_request",
        "hand_id": "committing-fourbet-tree",
        "street": "preflop",
        "seat_to_act": 0,
        "pot": 7_200,
        "community_cards": [],
        "current_bet": 3_600,
        "min_raise_to": 6_000,
        "amount_owed": 2_400,
        "can_check": False,
        "your_cards": ["Ah", "Kh"],
        "your_stack": 10_800,
        "your_bet_this_street": 1_200,
        "players": [
            {**player(0), "stack": 10_800, "bet_this_street": 1_200},
            {**player(1), "stack": 8_400, "bet_this_street": 3_600},
        ],
        "action_log": [
            {"seat": 1, "action": "raise", "amount": 300, "street": "preflop"},
            {"seat": 0, "action": "raise", "amount": 1_200, "street": "preflop"},
            {"seat": 1, "action": "raise", "amount": 3_600, "street": "preflop"},
        ],
    }


def committed_fivebet_jam_state():
    state = committing_fourbet_tree_state()
    state.update(
        {
            "hand_id": "committed-fivebet-jam",
            "pot": 22_000,
            "current_bet": 10_950,
            "min_raise_to": 10_950,
            "amount_owed": 1_950,
            "your_stack": 1_950,
            "your_bet_this_street": 9_000,
            "action_log": [
                {"seat": 1, "action": "raise", "amount": 300, "street": "preflop"},
                {"seat": 0, "action": "raise", "amount": 1_200, "street": "preflop"},
                {"seat": 1, "action": "raise", "amount": 3_600, "street": "preflop"},
                {"seat": 0, "action": "raise", "amount": 9_000, "street": "preflop"},
                {"seat": 1, "action": "all_in", "amount": 10_950, "street": "preflop"},
            ],
            "players": [
                {**player(0), "stack": 1_950, "bet_this_street": 9_000},
                {**player(1), "stack": 0, "bet_this_street": 10_950, "is_all_in": True},
            ],
        }
    )
    return state


def set_effective_stack_bb(state, stack_bb):
    stack = int(stack_bb * 100)
    state["your_stack"] = stack
    state["your_bet_this_street"] = 0
    for player_state in state["players"]:
        player_state["stack"] = stack
        player_state["bet_this_street"] = 0


def table_with_flop_cell(equity=0.61, realized=0.55):
    return {
        "version": 1,
        "preflop": {},
        "flop": {
            "cells": {
                "top_pair_top_kicker|A_high_dry|random|IP_PFA": {
                    "equity": equity,
                    "realized_equity": realized,
                    "samples": 25,
                }
            }
        },
    }


def table_with_turn_cell(state, equity=0.52, realized=0.46):
    board = bot._board_class(state["community_cards"])
    info = bot._strategic_hand_info(state["your_cards"], state["community_cards"])
    role = "IP_PFA"
    effect = bot.classify_turn_card(state["community_cards"][:3], state["community_cards"][3], role)
    key = "|".join(
        (
            bot._turn_lookup_hand_class(info),
            bot._turn_lookup_draw_class(info),
            bot._turn_lookup_board_texture(board, effect),
            bot._turn_lookup_card_bucket(effect),
            "random",
            role,
            bot._turn_lookup_spr_bucket(state),
        )
    )
    return {
        "version": 1,
        "preflop": {},
        "flop": {},
        "turn": {
            "key_fields": [
                "hand_class",
                "draw_class",
                "board_texture",
                "turn_card_bucket",
                "range_bucket",
                "role",
                "spr_bucket",
            ],
            "cells": {
                key: {
                    "equity": equity,
                    "realized_equity": realized,
                    "samples": 25,
                }
            },
        },
    }


def test_generator_quick_outputs_required_preflop_and_flop_shapes(tmp_path):
    tables = generator.main(["--quick", "--out", str(tmp_path), "--seed", "7"])

    assert (tmp_path / "equity_tables.json").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert len(tables["preflop"]["heads_up"]) == 169
    assert "random" in tables["preflop"]["heads_up"]["AKo"]
    assert "2" in tables["preflop"]["multiway_random"]["AKo"]
    assert any(key.endswith("|IP_PFA") for key in tables["flop"]["cells"])
    assert tables["turn"]["cells"]
    assert tables["turn"]["key_fields"] == [
        "hand_class",
        "draw_class",
        "board_texture",
        "turn_card_bucket",
        "range_bucket",
        "role",
        "spr_bucket",
    ]
    assert len(tables["river_policy"]["table"]) == 86400
    assert "hand_class" in tables["river_policy"]["key_fields"]


def test_flop_lookup_short_circuits_live_range_build(monkeypatch):
    state = flop_state()
    bot.COMPLETE_EQUITY_TABLES = table_with_flop_cell()
    monkeypatch.setattr(bot, "_postflop_role", lambda state: "IP_PFA")
    monkeypatch.setattr(bot, "_equity_lookup_range_bucket", lambda state: "random")
    monkeypatch.setattr(
        bot,
        "_build_heads_up_villain_range",
        lambda state, board_cards: (_ for _ in ()).throw(AssertionError("range builder should not run")),
    )

    assert bot._estimate_state_equity(state) == pytest.approx(0.61)


def test_missing_lookup_keeps_existing_estimator_fallback(monkeypatch):
    state = flop_state()
    villain_range = bot._make_weighted_range([("Ac", "Ad", 1.0)])
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_estimate_equity", lambda *args, **kwargs: 0.42)

    assert bot._estimate_state_equity(state) == pytest.approx(0.42)


def test_turn_lookup_short_circuits_live_range_build(monkeypatch):
    state = turn_state()
    bot.COMPLETE_EQUITY_TABLES = table_with_turn_cell(state)
    monkeypatch.setattr(bot, "_postflop_role", lambda state: "IP_PFA")
    monkeypatch.setattr(bot, "_equity_lookup_range_bucket", lambda state: "random")
    monkeypatch.setattr(
        bot,
        "_build_heads_up_villain_range",
        lambda state, board_cards: (_ for _ in ()).throw(AssertionError("range builder should not run")),
    )

    assert bot._estimate_state_equity(state) == pytest.approx(0.52)


def test_missing_turn_lookup_keeps_existing_estimator_fallback(monkeypatch):
    state = turn_state()
    villain_range = bot._make_weighted_range([("Ac", "Ad", 1.0)])
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_estimate_equity", lambda *args, **kwargs: 0.43)

    assert bot._estimate_state_equity(state) == pytest.approx(0.43)


def test_lookup_realized_equity_factor_uses_table(monkeypatch):
    state = flop_state()
    info = bot._strategic_hand_info(state["your_cards"], state["community_cards"])
    bot.COMPLETE_EQUITY_TABLES = table_with_flop_cell(equity=0.50, realized=0.40)
    monkeypatch.setattr(bot, "_postflop_role", lambda state: "IP_PFA")
    monkeypatch.setattr(bot, "_equity_lookup_range_bucket", lambda state: "random")

    assert bot._equity_realization_factor(state, info) == pytest.approx(0.80)


def test_turn_lookup_realized_equity_factor_uses_table(monkeypatch):
    state = turn_state()
    info = bot._strategic_hand_info(state["your_cards"], state["community_cards"])
    bot.COMPLETE_EQUITY_TABLES = table_with_turn_cell(state, equity=0.50, realized=0.40)
    monkeypatch.setattr(bot, "_postflop_role", lambda state: "IP_PFA")
    monkeypatch.setattr(bot, "_equity_lookup_range_bucket", lambda state: "random")

    assert bot._equity_realization_factor(state, info) == pytest.approx(0.80)


def test_simple_flop_cbet_bets_top_pair_on_dry_range_board(monkeypatch):
    state = flop_state()
    board = bot._board_class(state["community_cards"])
    info = bot._strategic_hand_info(state["your_cards"], state["community_cards"])
    monkeypatch.setattr(bot, "_estimate_state_equity", forbid_equity_estimate)

    assert bot._flop_decision(state, info, board, "IP_PFA") == {"action": "raise", "amount": 330}


def test_simple_flop_cbet_range_bets_q_high_dry_without_equity(monkeypatch):
    state = flop_state()
    state.update(
        {
            "hand_id": "simple-q-high-cbet",
            "your_cards": ["Th", "9h"],
            "community_cards": ["Qs", "7d", "2c"],
        }
    )
    board = bot._board_class(state["community_cards"])
    info = bot._strategic_hand_info(state["your_cards"], state["community_cards"])
    monkeypatch.setattr(bot, "stable_mix", lambda state, tag, frequency: tag == "simple_range_cbet")
    monkeypatch.setattr(bot, "_estimate_state_equity", forbid_equity_estimate)

    assert bot._flop_bucket(board) == "static_high"
    assert bot._flop_decision(state, info, board, "OOP_PFA") == {"action": "raise", "amount": 330}


def test_simple_flop_cbet_bets_good_draw_for_half_pot_without_equity(monkeypatch):
    state = flop_state()
    state.update(
        {
            "hand_id": "simple-draw-cbet",
            "your_cards": ["Ah", "Qh"],
            "community_cards": ["Kh", "Jh", "2c"],
        }
    )
    board = bot._board_class(state["community_cards"])
    info = bot._strategic_hand_info(state["your_cards"], state["community_cards"])
    monkeypatch.setattr(bot, "_estimate_state_equity", forbid_equity_estimate)

    assert info["strength"] == "strong_draw"
    assert bot._flop_decision(state, info, board, "IP_PFA") == {"action": "raise", "amount": 500}


def test_fivebet_continue_is_depth_aware():
    state = preflop_state()
    situation = {"type": "hero_4bet_faces_5bet", "hero_pos": "BTN"}

    set_effective_stack_bb(state, 150)
    assert bot._missing_source_fallback("AKo", situation, state) == {"action": "fold"}
    assert bot._missing_source_fallback("QQ", situation, state) == {"action": "fold"}
    assert bot._missing_source_fallback("KK", situation, state) == {"action": "all_in"}

    set_effective_stack_bb(state, 100)
    assert bot._missing_source_fallback("QQ", situation, state) == {"action": "all_in"}
    assert bot._missing_source_fallback("AKs", situation, state) == {"action": "all_in"}
    assert bot._missing_source_fallback("AKo", situation, state) == {"action": "fold"}

    set_effective_stack_bb(state, 70)
    assert bot._missing_source_fallback("AKo", situation, state) == {"action": "all_in"}

    set_effective_stack_bb(state, 45)
    assert bot._missing_source_fallback("JJ", situation, state) == {"action": "all_in"}
    assert bot._missing_source_fallback("TT", situation, state) == {"action": "fold"}

    set_effective_stack_bb(state, 25)
    assert bot._missing_source_fallback("TT", situation, state) == {"action": "all_in"}
    assert bot._missing_source_fallback("AQs", situation, state) == {"action": "all_in"}


def test_deep_fivebet_keeps_aa_kk_despite_low_equity_table_row():
    state = preflop_state()
    set_effective_stack_bb(state, 150)
    situation = {"type": "hero_4bet_faces_5bet", "hero_pos": "BTN"}
    bot.COMPLETE_EQUITY_TABLES = {
        "preflop": {
            "heads_up": {
                "KK": {"premium_aa_kk": 0.24},
            }
        }
    }

    assert bot._missing_source_fallback("KK", situation, state) == {"action": "all_in"}


def test_committing_preflop_raise_guard_jams_or_folds_instead_of_stranding():
    state = committing_fourbet_tree_state()
    situation = {"type": "threebet_or_fourbet_tree", "hero_pos": "BB"}

    assert bot._missing_source_fallback("AKs", situation, state) == {"action": "all_in"}
    assert bot._missing_source_fallback("AKo", situation, state) == {"action": "fold"}


def test_committed_preflop_value_range_does_not_fold_tiny_remaining_jam():
    state = committed_fivebet_jam_state()
    situation = {"type": "hero_4bet_faces_5bet", "hero_pos": "BB"}

    assert bot._missing_source_fallback("AKo", situation, state) == {"action": "all_in"}


def test_postflop_commitment_guard_calls_showdown_value_but_not_air():
    state = flop_state()
    state.update({"amount_owed": 800, "can_check": False, "pot": 5_000, "your_stack": 1_000})
    board = bot._board_class(state["community_cards"])
    medium_pair = {"strength": "medium_pair", "draw_power": 0.0}
    air = {"strength": "air", "draw_power": 0.0}

    assert bot._postflop_commitment_call_guard(
        state, medium_pair, board, equity=0.05, realized=0.05, required=0.14,
        near_nut=False, reverse_danger=False, opponents=1,
    ) == {"action": "call"}
    assert bot._postflop_commitment_call_guard(
        state, air, board, equity=0.05, realized=0.05, required=0.14,
        near_nut=False, reverse_danger=False, opponents=1,
    ) is None


def test_river_policy_lookup_replaces_live_ev_scorer(monkeypatch):
    state = river_state()
    board = bot._board_class(state["community_cards"])
    info = bot._strategic_hand_info(state["your_cards"], state["community_cards"])
    role = "IP_PFA"
    context = bot._river_context(state, info, board, role)
    key = bot._river_policy_key_from_context(state, info, board, role, context, villain_label="unknown")
    bot.COMPLETE_EQUITY_TABLES = {
        "river_policy": {
            "table": {
                key: {
                    "action": "bet",
                    "fraction": 0.67,
                    "ev": 0.8,
                    "check_ev": 0.5,
                }
            }
        }
    }
    assert bot._river_decision(state, info, board, role) == {"action": "raise", "amount": 670}
