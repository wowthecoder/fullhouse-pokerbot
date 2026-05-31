import importlib.util
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sandbox.match import _prepare_bot_mount


ROOT = os.path.join(os.path.dirname(__file__), "..")
BOT_PATH = os.path.join(ROOT, "bots", "my_bots", "g5-bot.py")


def load_bot():
    spec = importlib.util.spec_from_file_location("g5_bot_under_test", BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    hand_id="g5-hand",
    seat=0,
    cards=("As", "Ah"),
    street="preflop",
    current_bet=None,
    min_raise_to=None,
    amount_owed=None,
    can_check=None,
    your_bet_this_street=None,
    action_log=(),
):
    small_blind = 50
    big_blind = 100
    small_blind_seat = 1
    big_blind_seat = 2
    if street == "preflop":
        current_bet = big_blind if current_bet is None else current_bet
        min_raise_to = current_bet + big_blind if min_raise_to is None else min_raise_to
        if your_bet_this_street is None:
            your_bet_this_street = small_blind if seat == small_blind_seat else big_blind if seat == big_blind_seat else 0
        amount_owed = max(0, current_bet - your_bet_this_street) if amount_owed is None else amount_owed
    else:
        current_bet = 0 if current_bet is None else current_bet
        min_raise_to = 100 if min_raise_to is None else min_raise_to
        your_bet_this_street = 0 if your_bet_this_street is None else your_bet_this_street
        amount_owed = max(0, current_bet - your_bet_this_street) if amount_owed is None else amount_owed
    can_check = amount_owed == 0 if can_check is None else can_check
    players = []
    for idx in range(6):
        bet = small_blind if idx == small_blind_seat else big_blind if idx == big_blind_seat else 0
        if idx == seat:
            bet = your_bet_this_street
        players.append(player(idx, bet=bet))
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
        "your_stack": 10_000,
        "your_bet_this_street": your_bet_this_street,
        "players": players,
        "action_log": blind_log + list(action_log),
    }


def test_g5_data_assets_load():
    bot = load_bot()
    assert bot.G5_PREFLOP_CHARTS["rfi"]["LJ"]["AA"]["raise"] == 100
    assert bot.G5_PRIORS["base"]["players"] > 0
    assert len(bot.G5_PREFLOP_EQUITY) == bot.HOLE_INDEX_SIZE * bot.HOLE_INDEX_SIZE


def test_preflop_chart_rfi_and_mixed_cells_are_deterministic():
    bot = load_bot()
    state = make_state(hand_id="g5-rfi", seat=3, cards=("As", "Ah"))
    assert bot.decide(state) == {"action": "raise", "amount": 250}

    mixed = make_state(hand_id="g5-mixed", seat=3, cards=("Ad", "Ts"))
    first = bot.decide(mixed)
    second = bot.decide(mixed)
    assert first == second
    assert first["action"] in {"fold", "check", "raise"}


def test_preflop_chart_facing_open_uses_generated_chart():
    bot = load_bot()
    state = make_state(
        seat=0,
        cards=("As", "Ah"),
        current_bet=250,
        amount_owed=250,
        min_raise_to=350,
        action_log=[{"seat": 3, "action": "raise", "amount": 250}],
    )
    assert bot.decide(state) == {"action": "raise", "amount": 875}


def test_flat_log_street_inference_keeps_preflop_actions():
    bot = load_bot()
    preflop = make_state(
        hand_id="g5-streets",
        seat=0,
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "call", "amount": 250},
        ],
    )
    bot.decide(preflop)
    flop = make_state(
        hand_id="g5-streets",
        seat=0,
        street="flop",
        action_log=[
            {"seat": 3, "action": "raise", "amount": 250},
            {"seat": 4, "action": "call", "amount": 250},
            {"seat": 3, "action": "check", "amount": 0},
        ],
    )
    actions = [a for a in bot.actions_for_current_hand(flop) if a["action"] not in {"small_blind", "big_blind"}]
    assert [a["street"] for a in actions] == ["preflop", "preflop", "flop"]


def test_single_file_mount_copies_g5_data_for_local_matches():
    mount_src, cleanup_dir = _prepare_bot_mount(BOT_PATH)
    try:
        assert os.path.isfile(os.path.join(mount_src, "bot.py"))
        assert os.path.isfile(os.path.join(mount_src, "data", "preflop_charts_100bb.json"))
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
