import os
import random
import sys
import hashlib

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
    your_stack=10_000,
    action_log=(),
):
    players = [player(idx, stack=your_stack if idx == seat else 10_000) for idx in range(n_players)]
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
        "your_stack": your_stack,
        "your_bet_this_street": 0,
        "players": players,
        "action_log": blind_log + list(action_log),
    }


def weighted_range(*items):
    return bot._make_weighted_range(list(items))


def test_stable_mix_uses_blake2_hand_street_tag_and_seat_key():
    state = make_state(hand_id="mix-hand", street="flop", seat=0)
    tag = "cbet_A_high_dry"
    key = "mix-hand|flop|cbet_A_high_dry|0"
    digest = hashlib.blake2b(key.encode(), digest_size=4).digest()
    expected_roll = int.from_bytes(digest, "big") / 2**32

    assert bot._stable_mix_roll(state, tag) == pytest.approx(expected_roll)
    assert bot.stable_mix(state, tag, expected_roll + 1e-12) is True
    assert bot.stable_mix(state, tag, expected_roll - 1e-12) is False


def test_flop_mixed_frequency_uses_stable_mix_tag(monkeypatch):
    state = make_state(hand_id="mix-flop", street="flop")
    calls = []

    def fake_stable_mix(state_arg, tag, frequency):
        calls.append((state_arg, tag, frequency))
        return True

    monkeypatch.setattr(bot, "stable_mix", fake_stable_mix)

    assert bot._flop_mixed_frequency_allows(state, "A_high_dry", "IP_PFA", 0.80) is True
    assert calls == [(state, "flop_A_high_dry_IP_PFA", 0.80)]


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


def test_estimate_equity_timeout_uses_rule_based_fallback_when_too_few_samples(monkeypatch):
    villain_range = weighted_range(("As", "Ad", 1.0), ("2s", "2d", 2.0), ("9c", "8c", 1.0))
    times = iter([0.0, 2.0])
    monkeypatch.setattr(bot._time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(bot, "_rule_based_equity_estimate", lambda hero, board, villain_range: 0.37)

    equity = bot._estimate_equity(
        ["Ah", "Kh"],
        ["Qh", "Jh", "2c"],
        villain_range,
        samples=200,
        rng=random.Random(42),
    )

    assert equity == pytest.approx(0.37)


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


def test_facing_bet_alpha_and_mdf_use_pot_before_bet():
    state = make_state(current_bet=100, amount_owed=100, can_check=False, pot=200)

    result = bot._facing_bet_alpha_mdf(state)

    assert result["pot_before_bet"] == 100
    assert result["bet"] == 100
    assert result["alpha"] == pytest.approx(0.5)
    assert result["mdf"] == pytest.approx(0.5)


def test_tight_bettor_margin_folds_borderline_equity(monkeypatch):
    state = make_state(
        cards=("Kh", "Qd"),
        board=("Ks", "7d", "2c"),
        current_bet=250,
        amount_owed=250,
        can_check=False,
        pot=1_000,
    )
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.24)
    monkeypatch.setattr(bot, "_equity_realization_factor", lambda state, info: 1.0)
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "nit")

    assert bot._facing_postflop_bet(state, info, board) == {"action": "fold"}


def test_aggressive_bettor_margin_allows_thin_call(monkeypatch):
    state = make_state(
        cards=("Kh", "Qd"),
        board=("Ks", "7d", "2c"),
        current_bet=250,
        amount_owed=250,
        can_check=False,
        pot=1_000,
    )
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.197)
    monkeypatch.setattr(bot, "_equity_realization_factor", lambda state, info: 1.0)
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "aggressive")

    assert bot._facing_postflop_bet(state, info, board) == {"action": "call"}


def test_mdf_guardrail_rescues_unknown_bluffcatcher(monkeypatch):
    state = make_state(
        cards=("Kh", "Qd"),
        board=("Ks", "7d", "2c"),
        current_bet=100,
        amount_owed=100,
        can_check=False,
        pot=200,
    )
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.30)
    monkeypatch.setattr(bot, "_equity_realization_factor", lambda state, info: 1.0)
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "unknown")

    assert bot._facing_postflop_bet(state, info, board) == {"action": "call"}


def test_mdf_guardrail_does_not_rescue_weak_low_equity_hand(monkeypatch):
    state = make_state(
        cards=("4h", "2d"),
        board=("Ks", "7d", "2c"),
        current_bet=100,
        amount_owed=100,
        can_check=False,
        pot=200,
    )
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.18)
    monkeypatch.setattr(bot, "_equity_realization_factor", lambda state, info: 1.0)
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "unknown")

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


def test_river_value_hand_prefers_large_positive_ev_bet(monkeypatch):
    state = make_state(
        hand_id="river-value",
        cards=("Kh", "Qs"),
        board=("Ah", "Jc", "Tc", "2d", "3s"),
        street="river",
        pot=1_000,
    )
    villain_range = weighted_range(("Ac", "Qd", 100.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    action = bot.decide(state)

    assert action["action"] == "raise"
    assert action["amount"] >= 670


def test_river_nutted_hand_overbets_capped_range(monkeypatch):
    state = make_state(
        hand_id="river-nutted-overbet",
        cards=("Kh", "Qs"),
        board=("Ah", "Jc", "Tc", "2d", "3s"),
        street="river",
        pot=1_000,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
            {"seat": 0, "action": "raise", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "call", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "check", "street": "turn"},
            {"seat": 0, "action": "check", "street": "turn"},
        ],
    )
    villain_range = weighted_range(("Ac", "Qd", 100.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_calling_range", lambda villain_range, board, bet_fraction, model: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    action = bot.decide(state)

    assert action == {"action": "raise", "amount": 1_250}


def test_river_context_detects_draws_capped_range_and_showdown(monkeypatch):
    state = make_state(
        hand_id="river-context",
        cards=("Qs", "Ad"),
        board=("Qh", "Jh", "9c", "2d", "2s"),
        street="river",
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
            {"seat": 0, "action": "raise", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "call", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "check", "street": "turn"},
            {"seat": 0, "action": "check", "street": "turn"},
        ],
    )
    board = bot._board_class(state["community_cards"])
    info = bot._hand_info(state)
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "unknown")

    context = bot._river_context(state, info, board, "IP_PFA")

    assert context["missed_frontdoor_flush"] is True
    assert context["missed_straight_draw"] is True
    assert context["completed_draw"] is False
    assert context["paired_board"] is True
    assert context["range_capped_villain"] is True
    assert context["showdown_value"] is True

    completed = make_state(cards=("As", "Kd"), board=("Qh", "Jh", "9c", "2d", "3h"), street="river")
    completed_context = bot._river_context(
        completed,
        bot._hand_info(completed),
        bot._board_class(completed["community_cards"]),
        "IP_CALLER",
    )
    assert completed_context["completed_frontdoor_flush"] is True
    assert completed_context["completed_draw"] is True


def test_river_thin_value_checks_when_calling_range_is_too_strong(monkeypatch):
    state = make_state(
        hand_id="river-thin-check",
        cards=("As", "Kh"),
        board=("Ah", "7c", "2d", "9s", "Tc"),
        street="river",
        pot=1_000,
    )
    villain_range = weighted_range(("7h", "7d", 50.0), ("2c", "2s", 50.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    assert bot.decide(state) == {"action": "check"}


def test_river_showdown_ev_can_make_medium_pair_check(monkeypatch):
    state = make_state(
        hand_id="river-showdown-check",
        cards=("9s", "9d"),
        board=("Ah", "Kc", "7d", "2s", "3c"),
        street="river",
        pot=1_000,
    )
    villain_range = weighted_range(("8c", "8d", 60.0), ("Ac", "Qd", 40.0))
    calling_range = weighted_range(("Ac", "Qd", 40.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_calling_range", lambda villain_range, board, bet_fraction, model: calling_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    assert bot.decide(state) == {"action": "check"}


def test_river_top_pair_thin_values_station_capped_range(monkeypatch):
    state = make_state(
        hand_id="river-thin-station",
        cards=("As", "Jd"),
        board=("Ah", "7c", "2d", "9s", "Tc"),
        street="river",
        pot=1_000,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
            {"seat": 0, "action": "raise", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "call", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "check", "street": "turn"},
            {"seat": 0, "action": "check", "street": "turn"},
        ],
    )
    villain_range = weighted_range(("Ac", "8d", 70.0), ("7h", "6h", 30.0))
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "calling_station")
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    action = bot.decide(state)

    assert action["action"] == "raise"
    assert action["amount"] <= 330


def test_river_overpair_values_dry_runout(monkeypatch):
    state = make_state(
        hand_id="river-overpair-dry",
        cards=("Ks", "Kd"),
        board=("Qh", "7c", "2d", "9s", "3c"),
        street="river",
        pot=1_000,
    )
    villain_range = weighted_range(("Qd", "Jd", 100.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    action = bot.decide(state)

    assert action["action"] == "raise"
    assert 600 <= action["amount"] <= 800


def test_river_overpair_checks_bad_completed_draw_runout(monkeypatch):
    state = make_state(
        hand_id="river-overpair-bad",
        cards=("Ks", "Kd"),
        board=("Qh", "7h", "2c", "9d", "3h"),
        street="river",
        pot=1_000,
    )
    villain_range = weighted_range(("Ah", "Th", 100.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    assert bot.decide(state) == {"action": "check"}


def test_river_nut_blocker_bluff_bets_when_fold_probability_is_high(monkeypatch):
    state = make_state(
        hand_id="river-blocker-bluff",
        cards=("Ah", "Kd"),
        board=("Qh", "7h", "2h", "Tc", "3s"),
        street="river",
        pot=1_000,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
        ],
    )
    villain_range = weighted_range(("Ac", "Ad", 100.0))
    calling_range = weighted_range(("Ac", "Ad", 4.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_calling_range", lambda villain_range, board, bet_fraction, model: calling_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    action = bot.decide(state)

    assert action["action"] == "raise"


def test_river_bluff_frequency_is_seeded_and_size_aware():
    state = make_state(
        hand_id="river-bluff-frequency",
        cards=("Ah", "Kd"),
        board=("Qh", "7h", "2h", "Tc", "3s"),
        street="river",
        pot=1_000,
    )
    context = {
        "hand_class": "bluff_candidate",
        "hero_blocks_value": True,
        "hero_blocks_bluffs": False,
        "range_capped_villain": True,
        "missed_draw_density": 0.35,
    }
    candidate = {"kind": "bet", "fraction": 0.67, "target": 670, "cost": 670, "all_in": False}

    first = bot._river_bluff_frequency_allows(state, candidate, context, 0.44, 0.10)
    second = bot._river_bluff_frequency_allows(state, candidate, context, 0.44, 0.10)

    assert first is second
    assert bot._river_bluff_frequency_allows(state, candidate, context, 0.20, 0.10) is False
    assert bot._river_candidate_permitted({**candidate, "fraction": 0.25}, context) is False


def test_river_bluff_checks_when_station_calling_range_continues(monkeypatch):
    state = make_state(
        hand_id="river-station-check",
        cards=("Ah", "Kd"),
        board=("Qh", "7h", "2h", "Tc", "3s"),
        street="river",
        pot=1_000,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
        ],
    )
    villain_range = weighted_range(("Ac", "Ad", 100.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_calling_range", lambda villain_range, board, bet_fraction, model: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    assert bot.decide(state) == {"action": "check"}


def test_river_raise_heavy_node_penalizes_bet_fold_lines(monkeypatch):
    state = make_state(
        hand_id="river-raise-heavy",
        cards=("As", "Kh"),
        board=("Ah", "7c", "2d", "9s", "Tc"),
        street="river",
        pot=1_000,
    )
    villain_range = weighted_range(("Ac", "Qd", 100.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_calling_range", lambda villain_range, board, bet_fraction, model: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.95)

    assert bot.decide(state) == {"action": "check"}


def test_river_low_spr_can_choose_all_in(monkeypatch):
    state = make_state(
        hand_id="river-low-spr",
        cards=("Kh", "Qs"),
        board=("Ah", "Jc", "Tc", "2d", "3s"),
        street="river",
        pot=1_000,
        your_stack=900,
    )
    villain_range = weighted_range(("Ac", "Qd", 100.0))
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: villain_range)
    monkeypatch.setattr(bot, "_river_calling_range", lambda villain_range, board, bet_fraction, model: villain_range)
    monkeypatch.setattr(bot, "_river_raise_probability", lambda state, bet_fraction, board, model: 0.0)

    assert bot.decide(state) == {"action": "all_in"}


def test_river_passive_large_bet_folds_one_pair_without_major_blocker(monkeypatch):
    state = make_state(
        hand_id="river-passive-big-bet",
        cards=("Kh", "Qd"),
        board=("Ks", "7d", "2c", "9h", "3s"),
        street="river",
        current_bet=900,
        amount_owed=900,
        can_check=False,
        pot=1_900,
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "tight_passive")
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.55)

    assert bot.decide(state) == {"action": "fold"}


def test_river_aggressive_missed_draw_can_bluffcatch(monkeypatch):
    state = make_state(
        hand_id="river-aggro-bluffcatch",
        cards=("Qs", "Ad"),
        board=("Qh", "Jh", "9c", "2d", "2s"),
        street="river",
        current_bet=500,
        amount_owed=500,
        can_check=False,
        pot=1_500,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
            {"seat": 0, "action": "raise", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "call", "amount": 350, "street": "flop"},
            {"seat": 1, "action": "check", "street": "turn"},
            {"seat": 0, "action": "check", "street": "turn"},
            {"seat": 1, "action": "raise", "amount": 500, "street": "river"},
        ],
    )
    monkeypatch.setattr(bot, "_opponent_label", lambda state: "aggressive")
    monkeypatch.setattr(bot, "_estimate_state_equity", lambda state, samples=None: 0.18)

    assert bot.decide(state) == {"action": "call"}


def test_close_river_evs_use_seeded_stable_selection():
    state = make_state(
        hand_id="river-close",
        cards=("As", "Kh"),
        board=("Ah", "7c", "2d", "9s", "Tc"),
        street="river",
        pot=1_000,
    )
    scored = [
        {"candidate": {"kind": "check", "target": 0}, "ev": 0.0},
        {"candidate": {"kind": "bet", "target": 330}, "ev": 9.0},
        {"candidate": {"kind": "bet", "target": 670}, "ev": 11.0},
    ]

    first = bot._choose_close_river_action(state, scored)
    second = bot._choose_close_river_action(state, scored)

    assert first == second
    assert first in scored


def test_river_multiway_preserves_existing_fallback_bet(monkeypatch):
    state = make_state(
        hand_id="river-multiway-fallback",
        n_players=3,
        cards=("Kh", "Qs"),
        board=("Ah", "Jc", "Tc", "2d", "3s"),
        street="river",
        pot=1_000,
    )
    monkeypatch.setattr(bot, "_build_heads_up_villain_range", lambda state, board_cards: None)

    action = bot.decide(state)

    assert action["action"] == "raise"
