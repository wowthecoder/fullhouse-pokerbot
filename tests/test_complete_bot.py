import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bots.my_bots import complete as bot


@pytest.fixture(autouse=True)
def reset_models():
    bot._reset_opponent_models()
    yield
    bot._reset_opponent_models()


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
    n_players=6,
    cards=("As", "Kh"),
    current_bet=None,
    min_raise_to=None,
    amount_owed=None,
    can_check=None,
    your_stack=10_000,
    your_bet_this_street=None,
    small_blind_seat=None,
    big_blind_seat=None,
    action_log=(),
    street="preflop",
):
    small_blind = 50
    big_blind = 100
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
        amount_owed = max(0, current_bet - your_bet_this_street) if amount_owed is None else amount_owed
        can_check = amount_owed == 0 if can_check is None else can_check
    else:
        current_bet = 0 if current_bet is None else current_bet
        min_raise_to = 100 if min_raise_to is None else min_raise_to
        your_bet_this_street = 0 if your_bet_this_street is None else your_bet_this_street
        amount_owed = max(0, current_bet - your_bet_this_street) if amount_owed is None else amount_owed
        can_check = amount_owed == 0 if can_check is None else can_check

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
        players.append(player(idx, stack=your_stack if idx == seat else 10_000, bet=bet))

    blind_log = [
        {"seat": small_blind_seat, "action": "small_blind", "amount": small_blind},
        {"seat": big_blind_seat, "action": "big_blind", "amount": big_blind},
    ]
    return {
        "type": "action_request",
        "hand_id": hand_id,
        "street": street,
        "seat_to_act": seat,
        "pot": 150,
        "community_cards": [] if street == "preflop" else ["As", "7d", "2c"],
        "current_bet": current_bet,
        "min_raise_to": min_raise_to,
        "amount_owed": amount_owed,
        "can_check": can_check,
        "your_cards": list(cards),
        "your_stack": your_stack,
        "your_bet_this_street": your_bet_this_street,
        "players": players,
        "action_log": blind_log + list(action_log),
    }


def assert_raise(action, amount):
    assert action["action"] == "raise"
    assert action["amount"] == amount


def seed_model(seat, profile):
    model = bot._model_for(f"villain_{seat}", seat)
    model.hands = 80
    if profile == "nit":
        model.vpip_hands = 10
        model.pfr_hands = 8
        model.threebet_opps = 30
        model.threebets = 1
    elif profile == "tight_passive":
        model.vpip_hands = 16
        model.pfr_hands = 6
        model.postflop_calls = 16
        model.postflop_bets_raises = 1
        model.postflop_actions = 20
    elif profile == "station":
        model.vpip_hands = 38
        model.pfr_hands = 8
        model.postflop_calls = 24
        model.postflop_bets_raises = 2
        model.postflop_actions = 34
        model.saw_flop_hands = 32
        model.showdown_hands = 18
        model.cb_opps = 12
        model.cb_folds = 1
        model.faced_threebet = 12
        model.folded_to_threebet = 1
    elif profile == "overfolder":
        model.vpip_hands = 18
        model.pfr_hands = 14
        model.steal_opps = 20
        model.folded_to_steal = 18
        model.threebet_opps = 30
        model.threebets = 1
        model.cb_opps = 12
        model.cb_folds = 10
    return model


def test_combo_normalizes_rank_order_pairs_and_suitedness():
    assert bot._combo(["Kh", "As"]) == "AKo"
    assert bot._combo(["Td", "9d"]) == "T9s"
    assert bot._combo(["2s", "2h"]) == "22"


@pytest.mark.parametrize(
    ("seat", "expected"),
    [(0, "BTN"), (1, "SB"), (2, "BB"), (3, "LJ"), (4, "HJ"), (5, "CO")],
)
def test_position_labels_match_engine_six_max_blind_order(seat, expected):
    assert bot._position(make_state(seat=seat)) == expected


@pytest.mark.parametrize(
    ("seat", "cards", "amount"),
    [
        (3, ("Ad", "Qh"), 250),
        (4, ("5s", "5h"), 250),
        (5, ("4s", "4c"), 250),
        (0, ("5s", "4s"), 250),
        (1, ("Ad", "Qh"), 300),
    ],
)
def test_unopened_chart_raises_to_position_sizing(seat, cards, amount):
    action = bot.decide(make_state(seat=seat, cards=cards))
    assert_raise(action, amount)


@pytest.mark.parametrize(
    ("seat", "cards"),
    [
        (3, ("Ad", "9c")),
        (4, ("4s", "4h")),
        (5, ("Ks", "2s")),
        (0, ("7s", "2h")),
        (1, ("Ks", "2h")),
    ],
)
def test_unopened_unlisted_hands_fold_or_check(seat, cards):
    action = bot.decide(make_state(seat=seat, cards=cards))
    assert action["action"] in {"fold", "check"}


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        (("Ah", "Kd"), "call"),
        (("Ah", "Qh"), "call"),
        (("7s", "2h"), "fold"),
    ],
)
def test_sb_complete_uses_page_six_limp_raise_fold_first_actions(cards, expected):
    action = bot.decide(make_state(seat=1, cards=cards))
    assert action["action"] == expected


def test_sb_after_open_faces_bb_threebet_uses_suffix_action():
    state = make_state(
        seat=1,
        cards=("Ah", "Kd"),
        current_bet=900,
        amount_owed=600,
        min_raise_to=1_500,
        action_log=[
            {"seat": 1, "action": "call", "amount": 50},
            {"seat": 2, "action": "raise", "amount": 900},
        ],
    )
    assert_raise(bot.decide(state), 2250)


def test_bb_vs_sb_limp_raises_chart_hands_and_checks_others():
    raise_state = make_state(
        seat=2,
        cards=("Ah", "Qd"),
        amount_owed=0,
        can_check=True,
        action_log=[{"seat": 1, "action": "call", "amount": 100}],
    )
    assert_raise(bot.decide(raise_state), 350)

    check_state = make_state(
        seat=2,
        cards=("7s", "2h"),
        amount_owed=0,
        can_check=True,
        action_log=[{"seat": 1, "action": "call", "amount": 100}],
    )
    assert bot.decide(check_state)["action"] == "check"


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        (("As", "Ah"), "raise"),
        (("Ad", "9d"), "call"),
        (("7s", "2h"), "fold"),
    ],
)
def test_bb_vs_sb_raise_uses_threebet_call_fold_chart(cards, expected):
    state = make_state(
        seat=2,
        cards=cards,
        current_bet=300,
        amount_owed=200,
        min_raise_to=500,
        action_log=[{"seat": 1, "action": "raise", "amount": 300}],
    )
    action = bot.decide(state)
    assert action["action"] == expected
    if expected == "raise":
        assert action["amount"] == 1200


@pytest.mark.parametrize(
    ("seat", "cards", "opener_seat", "open_amount", "expected", "amount"),
    [
        (4, ("As", "Ah"), 3, 250, "raise", 875),
        (0, ("Ad", "Jd"), 3, 250, "call", None),
        (1, ("Ts", "9s"), 5, 250, "raise", 1000),
        (2, ("Qs", "Ts"), 3, 250, "call", None),
        (2, ("7s", "2h"), 3, 250, "fold", None),
    ],
)
def test_facing_single_open_uses_ip_and_oop_charts(seat, cards, opener_seat, open_amount, expected, amount):
    state = make_state(
        seat=seat,
        cards=cards,
        current_bet=open_amount,
        amount_owed=open_amount - (100 if seat == 2 else 50 if seat == 1 else 0),
        min_raise_to=open_amount + 100,
        action_log=[{"seat": opener_seat, "action": "raise", "amount": open_amount}],
    )
    action = bot.decide(state)
    assert action["action"] == expected
    if amount is not None:
        assert action["amount"] == amount


def test_missing_threebet_tree_uses_conservative_fallback():
    strong = make_state(
        seat=3,
        cards=("As", "Ah"),
        current_bet=1_000,
        amount_owed=1_000,
        min_raise_to=1_700,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 1_000},
        ],
    )
    assert bot.decide(strong)["action"] == "raise"

    weak = make_state(
        seat=3,
        cards=("Ad", "9c"),
        current_bet=1_000,
        amount_owed=750,
        min_raise_to=1_700,
        your_bet_this_street=250,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 1_000},
        ],
    )
    assert bot.decide(weak)["action"] == "fold"


def test_rangeconverter_lj_open_vs_hj_threebet_fourbets_calls_and_folds():
    fourbet = make_state(
        seat=3,
        cards=("As", "Ah"),
        current_bet=900,
        amount_owed=650,
        min_raise_to=1_550,
        your_bet_this_street=250,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
        ],
    )
    assert_raise(bot.decide(fourbet), 2360)

    call = make_state(
        seat=3,
        cards=("7s", "6s"),
        current_bet=900,
        amount_owed=650,
        min_raise_to=1_550,
        your_bet_this_street=250,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
        ],
    )
    assert bot.decide(call)["action"] == "call"

    fold = make_state(
        seat=3,
        cards=("Ks", "2h"),
        current_bet=900,
        amount_owed=650,
        min_raise_to=1_550,
        your_bet_this_street=250,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
        ],
    )
    assert bot.decide(fold)["action"] == "fold"


def test_rangeconverter_mixed_cells_are_deterministic():
    state = make_state(
        hand_id="mixed-cell",
        seat=3,
        cards=("Ad", "Ts"),
        current_bet=900,
        amount_owed=650,
        min_raise_to=1_550,
        your_bet_this_street=250,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
        ],
    )
    first = bot.decide(state)
    second = bot.decide(state)
    assert first == second
    assert first["action"] in {"fold", "raise"}


def test_rangeconverter_btn_open_vs_bb_threebet_uses_embedded_chart():
    state = make_state(
        seat=0,
        cards=("As", "Ah"),
        current_bet=1_000,
        amount_owed=750,
        min_raise_to=1_750,
        your_bet_this_street=250,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250},
            {"seat": 2, "action": "raise", "amount": 1_000},
        ],
    )
    assert_raise(bot.decide(state), 2500)


def test_hero_fourbet_faces_fivebet_jams_safe_range_and_folds_weak():
    strong = make_state(
        seat=3,
        cards=("As", "Kh"),
        current_bet=6_000,
        amount_owed=3_700,
        min_raise_to=9_700,
        your_bet_this_street=2_300,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
            {"seat": 3, "action": "raise", "amount": 2_300},
            {"seat": 4, "action": "raise", "amount": 6_000},
        ],
    )
    assert bot.decide(strong)["action"] == "all_in"

    weak = make_state(
        seat=3,
        cards=("Ad", "Qs"),
        current_bet=6_000,
        amount_owed=3_700,
        min_raise_to=9_700,
        your_bet_this_street=2_300,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
            {"seat": 3, "action": "raise", "amount": 2_300},
            {"seat": 4, "action": "raise", "amount": 6_000},
        ],
    )
    assert bot.decide(weak)["action"] == "fold"


def test_open_plus_callers_squeezes_with_position_sizing():
    state = make_state(
        seat=0,
        cards=("Ks", "Qs"),
        current_bet=250,
        amount_owed=250,
        min_raise_to=350,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "call", "amount": 250},
        ],
    )
    assert_raise(bot.decide(state), 1125)


def test_raise_and_threebet_before_hero_cold_fourbets_only_safe_range():
    strong = make_state(
        seat=0,
        cards=("Qs", "Qh"),
        current_bet=900,
        amount_owed=900,
        min_raise_to=1_550,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
        ],
    )
    assert_raise(bot.decide(strong), 2300)

    weak = make_state(
        seat=0,
        cards=("Js", "Jh"),
        current_bet=900,
        amount_owed=900,
        min_raise_to=1_550,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
        ],
    )
    assert bot.decide(weak)["action"] == "fold"


def test_limpers_no_raise_iso_and_overlimp_policy():
    iso = make_state(
        seat=0,
        cards=("Ad", "Th"),
        current_bet=100,
        amount_owed=100,
        min_raise_to=200,
        action_log=[{"seat": 3, "action": "call", "amount": 100}],
    )
    assert_raise(bot.decide(iso), 400)

    overlimp = make_state(
        seat=0,
        cards=("6s", "5s"),
        current_bet=100,
        amount_owed=100,
        min_raise_to=200,
        action_log=[{"seat": 3, "action": "call", "amount": 100}],
    )
    assert bot.decide(overlimp)["action"] == "call"


def test_postflop_top_pair_bets_and_continues():
    assert bot.decide(make_state(street="flop", can_check=True))["action"] == "raise"
    assert bot.decide(make_state(street="flop", current_bet=500, amount_owed=500, can_check=False))["action"] == "call"


def test_postflop_air_checks_and_folds():
    check = make_state(street="flop", cards=("4h", "2d"), can_check=True)
    assert bot.decide(check)["action"] == "check"

    fold = make_state(street="flop", cards=("4h", "2d"), current_bet=500, amount_owed=500, can_check=False)
    assert bot.decide(fold)["action"] == "fold"


def test_opponent_model_tracks_hands_vpip_pfr_and_threebet_opportunities():
    state = make_state(
        hand_id="model-3bet",
        seat=0,
        current_bet=900,
        amount_owed=900,
        min_raise_to=1_500,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
            {"seat": 5, "action": "fold", "amount": 0},
        ],
    )

    bot.decide(state)
    profiles = bot._opponent_profiles()

    opener = profiles["villain_3"]
    threebettor = profiles["villain_4"]
    folder = profiles["villain_5"]

    assert opener["hands"] == 1
    assert opener["vpip"] == 1
    assert opener["pfr"] == 1
    assert threebettor["threebet_opps"] == 1
    assert threebettor["threebet_pct"] == 1
    assert folder["threebet_opps"] == 0


def test_opponent_model_tracks_fold_to_steal_and_fold_to_threebet():
    steal = make_state(
        hand_id="model-steal",
        seat=0,
        action_log=[
            {"seat": 5, "action": "raise", "amount": 250},
            {"seat": 1, "action": "fold", "amount": 0},
            {"seat": 2, "action": "fold", "amount": 0},
        ],
    )
    bot.decide(steal)

    threebet = make_state(
        hand_id="model-fold-3bet",
        seat=0,
        current_bet=900,
        amount_owed=900,
        min_raise_to=1_500,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "raise", "amount": 900},
            {"seat": 3, "action": "fold", "amount": 0},
        ],
    )
    bot.decide(threebet)

    profiles = bot._opponent_profiles()
    assert profiles["villain_1"]["steal_opps"] == 1
    assert profiles["villain_1"]["fold_to_steal"] == 1
    assert profiles["villain_2"]["fold_to_steal"] == 1
    assert profiles["villain_3"]["faced_threebet"] == 1
    assert profiles["villain_3"]["fold_to_threebet"] == 1


def test_opponent_model_does_not_double_count_repeated_state():
    state = make_state(
        hand_id="repeat-model",
        seat=0,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "fold", "amount": 0},
        ],
    )
    bot.decide(state)
    bot.decide(state)

    profiles = bot._opponent_profiles()
    assert profiles["villain_3"]["hands"] == 1
    assert profiles["villain_3"]["pfr"] == 1


def test_observed_history_keeps_preflop_and_flop_actions_separate_without_log_streets():
    preflop = make_state(
        hand_id="street-history",
        seat=0,
        current_bet=250,
        amount_owed=250,
        min_raise_to=350,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "call", "amount": 250},
        ],
    )
    bot.decide(preflop)

    flop = make_state(
        hand_id="street-history",
        seat=0,
        street="flop",
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "call", "amount": 250},
            {"seat": 3, "action": "check", "amount": 0},
        ],
    )
    bot.decide(flop)

    assert bot._last_preflop_raiser(flop) == 3
    assert [entry["action"] for entry in bot._postflop_actions(flop, "flop")] == ["check"]


def test_unknown_opponent_keeps_baseline_preflop_fold():
    assert bot.decide(make_state(seat=0, cards=("Qs", "2s")))["action"] == "fold"


def test_overfolding_blinds_unlock_named_button_steal_expansion():
    seed_model(1, "overfolder")
    seed_model(2, "overfolder")

    action = bot.decide(make_state(seat=0, cards=("Qs", "2s")))

    assert_raise(action, 250)


def test_nit_opener_makes_marginal_broadway_fold():
    seed_model(3, "nit")
    state = make_state(
        seat=4,
        cards=("Ks", "Qh"),
        current_bet=250,
        amount_owed=250,
        min_raise_to=350,
        action_log=[{"seat": 3, "action": "raise", "amount": 250}],
    )

    assert bot.decide(state)["action"] == "fold"


def test_calling_station_opener_removes_light_threebet_bluff():
    seed_model(3, "station")
    state = make_state(
        seat=4,
        cards=("As", "5s"),
        current_bet=250,
        amount_owed=250,
        min_raise_to=350,
        action_log=[{"seat": 3, "action": "raise", "amount": 250}],
    )

    assert bot.decide(state)["action"] != "raise"


def test_station_flop_checks_air_but_bets_value_larger():
    seed_model(1, "station")
    action_log = [
        {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
        {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
    ]
    air = make_state(street="flop", n_players=2, seat=0, cards=("Ks", "Qh"), action_log=action_log)
    air["pot"] = 1_000
    assert bot.decide(air)["action"] == "check"

    value = make_state(street="flop", n_players=2, seat=0, cards=("As", "Kh"), action_log=action_log)
    value["pot"] = 1_000
    assert_raise(bot.decide(value), 750)


def test_overfolder_faces_selected_flop_cbet_with_equity():
    seed_model(1, "overfolder")
    state = make_state(
        hand_id="overfolder-cbet",
        street="flop",
        n_players=2,
        seat=0,
        cards=("Ks", "Jh"),
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
        ],
    )
    state["community_cards"] = ["Qd", "Ts", "2c"]
    state["pot"] = 1_000

    assert bot.decide(state)["action"] == "raise"


def test_passive_big_river_bet_folds_one_pair_without_blocker():
    seed_model(1, "tight_passive")
    state = make_state(
        street="river",
        n_players=2,
        seat=0,
        cards=("As", "Kh"),
        current_bet=900,
        amount_owed=900,
        min_raise_to=1_800,
        can_check=False,
        action_log=[
            {"seat": 0, "action": "raise", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "call", "amount": 250, "street": "preflop"},
            {"seat": 1, "action": "raise", "amount": 900, "street": "river"},
        ],
    )
    state["community_cards"] = ["Ad", "7c", "2h", "Td", "9s"]
    state["pot"] = 1_000

    assert bot.decide(state)["action"] == "fold"


def test_raise_to_emits_all_in_when_stack_capped():
    state = make_state(street="flop", current_bet=500, min_raise_to=1_000, your_stack=1_000)
    assert bot._raise_to(state, 1_200) == {"action": "all_in"}


def test_opponent_model_tracks_postflop_aggression_and_fold_to_cbet():
    preflop = make_state(
        hand_id="model-cbet",
        seat=0,
        current_bet=250,
        amount_owed=250,
        min_raise_to=350,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "call", "amount": 250},
        ],
    )
    bot.decide(preflop)

    flop = make_state(
        hand_id="model-cbet",
        seat=0,
        street="flop",
        current_bet=400,
        amount_owed=400,
        min_raise_to=800,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "call", "amount": 250},
            {"seat": 3, "action": "raise", "amount": 400},
            {"seat": 4, "action": "fold", "amount": 0},
        ],
    )
    bot.decide(flop)

    profiles = bot._opponent_profiles()
    aggressor = profiles["villain_3"]
    caller = profiles["villain_4"]

    assert aggressor["af"] == 1
    assert caller["saw_flop_hands"] == 1
    assert caller["cbet_opps"] == 1
    assert caller["fold_to_cbet"] == 1
