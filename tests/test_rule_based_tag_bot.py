import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bots.rule_based_TAG import blackrain79 as tag_bot


VALID_ACTIONS = {"fold", "check", "call", "raise", "all_in"}


@pytest.fixture(autouse=True)
def reset_hand_state():
    tag_bot.HAND_STATE.clear()
    yield
    tag_bot.HAND_STATE.clear()


def player(seat, bot_id=None, stack=10_000, folded=False, all_in=False, bet=0):
    return {
        "seat": seat,
        "bot_id": bot_id or ("hero" if seat == 0 else f"villain_{seat}"),
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
    street="preflop",
    seat=0,
    n_players=6,
    cards=("As", "Kh"),
    board=(),
    pot=None,
    current_bet=None,
    min_raise_to=None,
    amount_owed=None,
    can_check=None,
    your_stack=10_000,
    your_bet_this_street=None,
    small_blind_seat=None,
    big_blind_seat=None,
    small_blind=50,
    big_blind=100,
    folded_seats=(),
    all_in_seats=(),
    action_log=(),
    match_action_log=(),
    bot_ids=None,
):
    if small_blind_seat is None:
        small_blind_seat = 0 if n_players == 2 else 1
    if big_blind_seat is None:
        big_blind_seat = 1 if n_players == 2 else 2

    if street == "preflop":
        current_bet = big_blind if current_bet is None else current_bet
        min_raise_to = current_bet + big_blind if min_raise_to is None else min_raise_to
        if your_bet_this_street is None:
            if seat == small_blind_seat:
                your_bet_this_street = small_blind
            elif seat == big_blind_seat:
                your_bet_this_street = big_blind
            else:
                your_bet_this_street = 0
        default_owed = max(0, current_bet - your_bet_this_street)
        amount_owed = default_owed if amount_owed is None else amount_owed
        can_check = amount_owed == 0 if can_check is None else can_check
        pot = small_blind + big_blind if pot is None else pot
    else:
        current_bet = 0 if current_bet is None else current_bet
        min_raise_to = current_bet + big_blind if min_raise_to is None else min_raise_to
        your_bet_this_street = 0 if your_bet_this_street is None else your_bet_this_street
        amount_owed = max(0, current_bet - your_bet_this_street) if amount_owed is None else amount_owed
        can_check = amount_owed == 0 if can_check is None else can_check
        pot = 600 if pot is None else pot

    folded_seats = set(folded_seats)
    all_in_seats = set(all_in_seats)
    bot_ids = bot_ids or {0: "hero"}
    players = []
    for idx in range(n_players):
        bet = 0
        if street == "preflop":
            if idx == small_blind_seat:
                bet = small_blind
            elif idx == big_blind_seat:
                bet = big_blind
        if idx == seat:
            bet = your_bet_this_street
        elif not can_check and street != "preflop" and idx != seat and current_bet:
            bet = current_bet
        players.append(
            player(
                idx,
                bot_id=bot_ids.get(idx, f"villain_{idx}"),
                stack=your_stack if idx == seat else 10_000,
                folded=idx in folded_seats,
                all_in=idx in all_in_seats,
                bet=bet,
            )
        )

    blind_log = [
        {"seat": small_blind_seat, "action": "small_blind", "amount": small_blind},
        {"seat": big_blind_seat, "action": "big_blind", "amount": big_blind},
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
        "your_bet_this_street": your_bet_this_street,
        "players": players,
        "action_log": blind_log + list(action_log),
        "match_action_log": list(match_action_log),
    }


def mark_history(hand_id, **flags):
    tag_bot.HAND_STATE[hand_id] = {
        "raised_preflop": False,
        "flop_bet": False,
        "turn_bet": False,
        **flags,
    }


def opponent_log(bot_id, actions):
    return [{"bot_id": bot_id, "action": action} for action in actions]


def station_log(bot_id="villain_1"):
    return opponent_log(bot_id, ["call"] * 15 + ["check"] * 5)


def tight_weak_log(bot_id="villain_1"):
    return opponent_log(bot_id, ["fold"] * 11 + ["check"] * 9)


def aggressive_log(bot_id="villain_1"):
    return opponent_log(bot_id, ["raise"] * 7 + ["check"] * 13)


def assert_raise(action, amount=None):
    assert action["action"] == "raise"
    if amount is not None:
        assert action["amount"] == amount


def assert_legal(action):
    assert action["action"] in VALID_ACTIONS


def test_combo_normalizes_rank_order_and_suitedness():
    assert tag_bot._combo(["Kh", "As"]) == "AKo"
    assert tag_bot._combo(["2s", "2h"]) == "22"
    assert tag_bot._combo(["Td", "9d"]) == "T9s"


@pytest.mark.parametrize(
    ("seat", "expected"),
    [
        (1, "SB"),
        (2, "BB"),
        (3, "EP"),
        (4, "MP"),
        (5, "CO"),
        (0, "BTN"),
    ],
)
def test_position_labels_standard_six_max_table(seat, expected):
    assert tag_bot._position(make_state(seat=seat)) == expected


@pytest.mark.parametrize(
    ("position", "in_range", "out_of_range"),
    [
        ("EP", "AQo", "AJo"),
        ("MP", "JTs", "A9o"),
        ("CO", "65s", "K9o"),
        ("BTN", "54s", "Q8s"),
        ("SB", "AJo", "65s"),
        ("BB", "76s", "Q8s"),
    ],
)
def test_open_ranges_match_tag_position_width(position, in_range, out_of_range):
    open_range = tag_bot._open_range_for(position)
    assert in_range in open_range
    assert out_of_range not in open_range


@pytest.mark.parametrize(
    ("cards", "seat", "expected_amount"),
    [
        (("As", "Ah"), 3, 300),
        (("Ad", "Qd"), 3, 300),
        (("9s", "9h"), 3, 300),
        (("6s", "5s"), 5, 300),
        (("5s", "4s"), 0, 300),
    ],
)
def test_preflop_first_in_playable_hands_open_raise_to_three_bb(cards, seat, expected_amount):
    action = tag_bot.decide(make_state(cards=cards, seat=seat))
    assert_raise(action, expected_amount)


@pytest.mark.parametrize(
    ("cards", "seat"),
    [
        (("Ks", "9h"), 3),
        (("Qs", "8s"), 3),
        (("6s", "5s"), 1),
        (("Ad", "9c"), 3),
    ],
)
def test_preflop_first_in_unplayable_hands_fold_without_limping(cards, seat):
    action = tag_bot.decide(make_state(cards=cards, seat=seat))
    assert action["action"] == "fold"


def test_preflop_big_blind_checks_free_option_instead_of_limping():
    action = tag_bot.decide(make_state(cards=("7s", "2h"), seat=2, amount_owed=0, can_check=True))
    assert action["action"] == "check"


@pytest.mark.parametrize("cards", [("As", "Ah"), ("Qs", "Qh"), ("Td", "Tc"), ("As", "Kh"), ("Ad", "Qh")])
def test_preflop_value_hands_three_bet_to_three_times_open(cards):
    state = make_state(
        cards=cards,
        seat=0,
        current_bet=300,
        amount_owed=300,
        min_raise_to=500,
        pot=450,
        action_log=[{"seat": 3, "action": "raise", "amount": 300}],
    )
    assert_raise(tag_bot.decide(state), 900)


@pytest.mark.parametrize("cards", [("Js", "Jh"), ("Ts", "Th"), ("Ad", "Qh")])
def test_preflop_flats_jj_tt_aq_against_ep_open_at_reasonable_price(cards):
    state = make_state(
        cards=cards,
        seat=0,
        current_bet=300,
        amount_owed=300,
        min_raise_to=500,
        pot=800,
        action_log=[{"seat": 3, "action": "raise", "amount": 300}],
    )
    assert tag_bot.decide(state)["action"] == "call"


@pytest.mark.parametrize("cards", [("As", "Jh"), ("Td", "Ah"), ("As", "9s"), ("Ks", "Qh"), ("8s", "8d"), ("Js", "Ts")])
def test_preflop_expands_three_bet_range_versus_late_steals(cards):
    state = make_state(
        cards=cards,
        seat=2,
        current_bet=300,
        amount_owed=200,
        min_raise_to=500,
        pot=450,
        action_log=[{"seat": 0, "action": "raise", "amount": 300}],
    )
    assert_raise(tag_bot.decide(state), 900)


@pytest.mark.parametrize("cards", [("6s", "6h"), ("9s", "8s"), ("Js", "Ts"), ("Kc", "Qd")])
def test_preflop_flats_speculative_hands_in_position_for_reasonable_price(cards):
    state = make_state(
        cards=cards,
        seat=0,
        current_bet=300,
        amount_owed=300,
        min_raise_to=500,
        pot=900,
        action_log=[{"seat": 3, "action": "raise", "amount": 300}],
    )
    assert tag_bot.decide(state)["action"] == "call"


def test_preflop_folds_speculative_hands_when_price_is_too_high():
    state = make_state(
        cards=("6s", "6h"),
        seat=0,
        current_bet=1_500,
        amount_owed=1_500,
        min_raise_to=2_500,
        pot=2_000,
        action_log=[{"seat": 3, "action": "raise", "amount": 1_500}],
    )
    assert tag_bot.decide(state)["action"] == "fold"


def test_preflop_big_blind_defends_playable_range_at_discount():
    state = make_state(
        cards=("Qs", "Ts"),
        seat=2,
        current_bet=200,
        amount_owed=100,
        min_raise_to=300,
        pot=600,
        action_log=[{"seat": 0, "action": "raise", "amount": 200}],
    )
    assert tag_bot.decide(state)["action"] == "call"


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        (["As", "2d", "3h", "4c", "5s"], "straight"),
        (["As", "Kd", "Qh", "Jc", "Ts"], "straight"),
        (["Ah", "Kh", "Qh", "Jh", "Th"], "straight flush"),
        (["7s", "7d", "7h", "Kc", "Ks"], "full house"),
        (["2h", "7h", "9h", "Jh", "Kh"], "flush"),
        (["9s", "9d", "9h", "2c", "5d"], "three of a kind"),
        (["Ks", "Kd", "2h", "2c", "7d"], "two pair"),
    ],
)
def test_hand_type_classifies_showdown_hands(cards, expected):
    assert tag_bot._hand_type(cards) == expected


@pytest.mark.parametrize(
    ("board", "texture"),
    [
        (["Ks", "7d", "2c"], "dry"),
        (["Jd", "Ts", "9s"], "wet"),
        (["As", "Kd", "2h"], "wet"),
        (["8s", "6d", "5c"], "wet"),
        (["Qh", "7h", "2c"], "wet"),
    ],
)
def test_board_texture_dry_wet_classification(board, texture):
    assert tag_bot._board_texture(board) == texture


@pytest.mark.parametrize(
    ("cards", "board", "expected"),
    [
        (("As", "Kh"), ("Ks", "7d", "2c"), {"strength": "tptk_plus", "top_pair": True, "tptk": True}),
        (("Ks", "8h"), ("Kd", "7c", "2s"), {"strength": "top_pair", "top_pair": True, "tptk": False}),
        (("Qs", "Qh"), ("Jd", "7c", "2s"), {"strength": "tptk_plus", "overpair": True}),
        (("7s", "7h"), ("Kd", "7c", "2s"), {"strength": "monster", "set": True}),
        (("Ah", "Qh"), ("7h", "2h", "Kd"), {"strength": "strong_draw", "flush_draw": True}),
        (("8s", "7d"), ("6h", "5c", "2s"), {"strength": "strong_draw", "oesd": True}),
        (("8s", "4d"), ("6h", "5c", "2s"), {"strength": "weak_equity", "gutshot": True}),
        (("Ah", "Qh"), ("7h", "2h", "Kd", "Kh", "3c"), {"flush_draw": False}),
    ],
)
def test_hand_info_classifies_strength_draws_and_edge_cases(cards, board, expected):
    info = tag_bot._hand_info(make_state(street="flop", cards=cards, board=board))
    for key, value in expected.items():
        assert info[key] == value


def test_opponent_type_unknown_until_enough_actions():
    state = make_state(n_players=2, match_action_log=opponent_log("villain_1", ["call"] * 19))
    assert tag_bot._opponent_type(state) == "unknown"


@pytest.mark.parametrize(
    ("logs", "expected"),
    [
        (station_log(), "calling_station"),
        (tight_weak_log(), "tight_weak"),
        (aggressive_log(), "aggressive_reg"),
    ],
)
def test_opponent_type_profiles_single_opponent(logs, expected):
    assert tag_bot._opponent_type(make_state(n_players=2, match_action_log=logs)) == expected


def test_opponent_type_multiway_calling_station_takes_precedence():
    logs = station_log("villain_1") + aggressive_log("villain_2")
    state = make_state(n_players=3, match_action_log=logs, bot_ids={0: "hero", 1: "villain_1", 2: "villain_2"})
    assert tag_bot._opponent_type(state) == "calling_station"


def test_opponent_type_multiway_all_tight_weak_enables_pressure():
    logs = tight_weak_log("villain_1") + tight_weak_log("villain_2")
    state = make_state(n_players=3, match_action_log=logs, bot_ids={0: "hero", 1: "villain_1", 2: "villain_2"})
    assert tag_bot._opponent_type(state) == "tight_weak"


@pytest.mark.parametrize(
    ("cards", "board"),
    [
        (("Kh", "8d"), ("Ks", "7d", "2c")),
        (("8s", "7d"), ("6h", "5c", "2s")),
        (("Ah", "Qd"), ("7s", "3d", "2c")),
    ],
)
def test_flop_preflop_aggressor_cbet_heads_up_with_pair_draw_or_overcards(cards, board):
    hand_id = "flop-cbet"
    mark_history(hand_id, raised_preflop=True)
    action = tag_bot.decide(make_state(hand_id=hand_id, street="flop", n_players=2, cards=cards, board=board, pot=800))
    assert_raise(action, 400)


def test_flop_preflop_aggressor_bluffs_any_air_on_dry_board_versus_tight_weak():
    hand_id = "flop-tight-weak"
    mark_history(hand_id, raised_preflop=True)
    state = make_state(
        hand_id=hand_id,
        street="flop",
        n_players=2,
        cards=("9h", "4d"),
        board=("Ks", "7d", "2c"),
        pot=800,
        match_action_log=tight_weak_log(),
    )
    assert_raise(tag_bot.decide(state), 400)


def test_flop_skips_air_cbet_on_wet_board():
    hand_id = "flop-wet-air"
    mark_history(hand_id, raised_preflop=True)
    state = make_state(hand_id=hand_id, street="flop", n_players=2, cards=("4h", "2d"), board=("Js", "Ts", "9d"), pot=800)
    assert tag_bot.decide(state)["action"] == "check"


def test_flop_never_bluffs_calling_station():
    hand_id = "flop-station"
    mark_history(hand_id, raised_preflop=True)
    state = make_state(
        hand_id=hand_id,
        street="flop",
        n_players=2,
        cards=("Ah", "Qd"),
        board=("7s", "3d", "2c"),
        pot=800,
        match_action_log=station_log(),
    )
    assert tag_bot.decide(state)["action"] == "check"


def test_flop_multiway_checks_weak_hands_and_bets_strong_draws():
    weak_state = make_state(street="flop", n_players=3, cards=("Ah", "Qd"), board=("7s", "3d", "2c"), pot=900)
    assert tag_bot.decide(weak_state)["action"] == "check"

    draw_state = make_state(street="flop", n_players=3, cards=("8s", "7d"), board=("6h", "5c", "2s"), pot=900)
    assert_raise(tag_bot.decide(draw_state), 540)


def test_flop_fast_plays_tptk_plus_and_monsters_for_value():
    tptk = make_state(street="flop", n_players=2, cards=("Ah", "Kd"), board=("Ks", "7d", "2c"), pot=800)
    assert_raise(tag_bot.decide(tptk), 560)

    monster = make_state(street="flop", n_players=2, cards=("7h", "7d"), board=("Ks", "7s", "2c"), pot=800)
    assert_raise(tag_bot.decide(monster), 560)


@pytest.mark.parametrize(
    ("cards", "board", "owed", "pot", "expected"),
    [
        (("7h", "7d"), ("Ks", "7s", "2c"), 200, 800, "raise"),
        (("Kh", "8d"), ("Ks", "7s", "2c"), 200, 800, "call"),
        (("Ah", "Qh"), ("7h", "2h", "Kd"), 200, 800, "call"),
        (("Ah", "Qh"), ("7h", "2h", "Kd"), 500, 800, "fold"),
        (("8s", "4d"), ("6h", "5c", "2s"), 200, 800, "fold"),
        (("9s", "4d"), ("Kh", "7c", "2s"), 200, 800, "fold"),
    ],
)
def test_flop_facing_bet_raises_value_calls_reasonable_equity_and_folds_air(cards, board, owed, pot, expected):
    state = make_state(
        street="flop",
        n_players=2,
        cards=cards,
        board=board,
        pot=pot,
        current_bet=owed,
        amount_owed=owed,
        min_raise_to=owed * 2,
        can_check=False,
    )
    action = tag_bot.decide(state)
    assert action["action"] == expected


@pytest.mark.parametrize(
    ("cards", "board"),
    [
        (("Kh", "8d"), ("Ks", "7d", "2c", "4h")),
        (("8s", "7d"), ("6h", "5c", "2s", "Qd")),
    ],
)
def test_turn_double_barrels_top_pair_or_strong_draw_after_flop_bet(cards, board):
    hand_id = "turn-value"
    mark_history(hand_id, raised_preflop=True, flop_bet=True)
    action = tag_bot.decide(make_state(hand_id=hand_id, street="turn", n_players=2, cards=cards, board=board, pot=1_000))
    assert_raise(action, 660)


def test_turn_bluffs_only_dry_board_scare_card_tight_weak_heads_up_after_flop_bet():
    hand_id = "turn-bluff"
    mark_history(hand_id, raised_preflop=True, flop_bet=True)
    state = make_state(
        hand_id=hand_id,
        street="turn",
        n_players=2,
        cards=("Qh", "Jd"),
        board=("7s", "3d", "2c", "Ah"),
        pot=1_000,
        match_action_log=tight_weak_log(),
    )
    assert_raise(tag_bot.decide(state), 660)


@pytest.mark.parametrize(
    ("board", "logs", "history"),
    [
        (("7s", "3d", "2c", "4h"), tight_weak_log(), {"flop_bet": True}),
        (("7s", "3d", "2c", "Ah"), station_log(), {"flop_bet": True}),
        (("7s", "3d", "2c", "Ah"), tight_weak_log(), {"flop_bet": False}),
    ],
)
def test_turn_gives_up_when_required_bluff_conditions_are_missing(board, logs, history):
    hand_id = "turn-give-up"
    mark_history(hand_id, raised_preflop=True, **history)
    state = make_state(
        hand_id=hand_id,
        street="turn",
        n_players=2,
        cards=("Qh", "Jd"),
        board=board,
        pot=1_000,
        match_action_log=logs,
    )
    assert tag_bot.decide(state)["action"] == "check"


@pytest.mark.parametrize(
    ("cards", "board", "expected_amount"),
    [
        (("7h", "7d"), ("Ks", "7s", "2c", "4h", "9d"), 1_200),
        (("Ah", "Kd"), ("Ks", "7s", "2c", "4h", "9d"), 1_200),
    ],
)
def test_river_value_bets_monsters_and_tptk_plus(cards, board, expected_amount):
    action = tag_bot.decide(make_state(street="river", n_players=2, cards=cards, board=board, pot=1_500))
    assert_raise(action, expected_amount)


def test_river_thin_values_top_pair_against_station_only_on_dry_boards():
    dry = make_state(
        street="river",
        n_players=2,
        cards=("Kh", "8d"),
        board=("Ks", "7d", "2c", "4h", "9s"),
        pot=1_500,
        match_action_log=station_log(),
    )
    assert_raise(tag_bot.decide(dry), 900)

    wet = make_state(
        street="river",
        n_players=2,
        cards=("Kh", "8d"),
        board=("Ks", "Qs", "Js", "4h", "9s"),
        pot=1_500,
        match_action_log=station_log(),
    )
    assert tag_bot.decide(wet)["action"] == "check"


@pytest.mark.parametrize(
    ("cards", "board"),
    [
        (("Ah", "Qh"), ("7h", "2h", "Kd", "4s", "9c")),
        (("8h", "8d"), ("Ks", "7s", "2c", "4h", "9d")),
        (("Qh", "Jd"), ("7s", "3d", "2c", "4h", "9c")),
    ],
)
def test_river_checks_missed_draws_medium_pairs_and_air(cards, board):
    state = make_state(street="river", n_players=2, cards=cards, board=board, pot=1_500)
    assert tag_bot.decide(state)["action"] == "check"


def test_river_facing_bet_folds_weak_pair_on_completed_draw_board():
    state = make_state(
        street="river",
        n_players=2,
        cards=("8h", "8d"),
        board=("Ks", "Qs", "Js", "Ts", "2c"),
        pot=2_000,
        current_bet=900,
        amount_owed=900,
        min_raise_to=1_800,
        can_check=False,
    )
    assert tag_bot.decide(state)["action"] == "fold"


def test_defensive_warmup_and_malformed_states_return_legal_fallbacks():
    assert tag_bot.decide({"type": "warmup"}) == {"action": "check"}
    assert tag_bot.decide({"can_check": True}) == {"action": "check"}
    assert tag_bot.decide({}) == {"action": "fold"}


def test_short_stack_premium_returns_all_in_when_calling_costs_stack():
    state = make_state(
        cards=("Qs", "Qh"),
        seat=0,
        current_bet=100,
        min_raise_to=200,
        amount_owed=100,
        your_stack=80,
        your_bet_this_street=20,
        action_log=(),
    )
    assert tag_bot.decide(state)["action"] == "all_in"


def test_raise_to_respects_minimum_and_stack_cap():
    state = make_state(street="flop", n_players=2, pot=1_000, min_raise_to=700, your_stack=550)
    action = tag_bot._raise_to(state, 400)
    assert_raise(action, 550)
    assert action["amount"] <= state["your_stack"] + state["your_bet_this_street"]

    deep = make_state(street="flop", n_players=2, pot=1_000, min_raise_to=700, your_stack=5_000)
    assert_raise(tag_bot._raise_to(deep, 400), 700)


def test_hand_state_records_aggression_and_prunes_old_hand_ids():
    preflop = make_state(hand_id="record-pre", street="preflop", cards=("As", "Ah"), seat=3)
    tag_bot.decide(preflop)
    assert tag_bot.HAND_STATE["record-pre"]["raised_preflop"] is True

    flop = make_state(hand_id="record-flop", street="flop", n_players=2, cards=("Ah", "Kd"), board=("Ks", "7d", "2c"))
    tag_bot.decide(flop)
    assert tag_bot.HAND_STATE["record-flop"]["flop_bet"] is True

    turn = make_state(hand_id="record-turn", street="turn", n_players=2, cards=("Ah", "Kd"), board=("Ks", "7d", "2c", "4h"))
    tag_bot.decide(turn)
    assert tag_bot.HAND_STATE["record-turn"]["turn_bet"] is True

    check = make_state(hand_id="record-check", street="river", n_players=2, cards=("Qh", "Jd"), board=("Ks", "7d", "2c", "4h", "9s"))
    tag_bot.decide(check)
    assert tag_bot.HAND_STATE["record-check"] == {"raised_preflop": False, "flop_bet": False, "turn_bet": False}

    for idx in range(45):
        tag_bot.HAND_STATE[f"old-{idx:02d}"] = {"raised_preflop": False, "flop_bet": False, "turn_bet": False}
    tag_bot._street_state({"hand_id": "newest"})
    assert len(tag_bot.HAND_STATE) <= 20


def test_decide_returns_legal_action_for_validator_edge_state_shape():
    state = make_state(
        hand_id="edge",
        street="river",
        n_players=2,
        cards=("2c", "3d"),
        board=("7s", "Td", "2h", "Kc", "5d"),
        pot=4_000,
        current_bet=2_000,
        amount_owed=2_000,
        min_raise_to=4_000,
        can_check=False,
        your_stack=6_000,
    )
    assert_legal(tag_bot.decide(state))
