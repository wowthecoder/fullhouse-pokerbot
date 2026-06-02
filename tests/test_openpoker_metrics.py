import json

from engine.openpoker_metrics import OpenPokerMetricsTracker


PLAYERS = [
    {"seat": 0, "name": "btn", "stack": 10_000},
    {"seat": 1, "name": "sb", "stack": 10_000},
    {"seat": 2, "name": "bb", "stack": 10_000},
    {"seat": 3, "name": "utg", "stack": 10_000},
    {"seat": 4, "name": "hj", "stack": 10_000},
    {"seat": 5, "name": "hero", "stack": 10_000},
]


def make_tracker(tmp_path):
    tracker = OpenPokerMetricsTracker(
        tmp_path / "openpoker_metrics.json",
        ws_url="wss://example.test/ws",
        buy_in=2000,
        bot_path="bots/my_bots/complete_v2.py",
    )
    tracker.on_connected({"type": "connected", "name": "hero"})
    tracker.on_table_joined({"type": "table_joined", "table_id": "t1", "seat": 5, "players": PLAYERS})
    return tracker


def hero_state(street="preflop", pot=150, owed=0, stack=10_000, bet=0, board=None):
    return {
        "hand_id": "h",
        "street": street,
        "seat_to_act": 5,
        "pot": pot,
        "amount_owed": owed,
        "can_check": owed == 0,
        "community_cards": board or [],
        "your_cards": ["As", "Kh"],
        "your_stack": stack,
        "your_bet_this_street": bet,
    }


def turn_msg(hand_id="h", actions=None):
    return {
        "type": "your_turn",
        "hand_id": hand_id,
        "valid_actions": actions
        or [
            {"action": "fold"},
            {"action": "call", "amount": 0},
            {"action": "raise", "min": 300, "max": 10_000},
            {"action": "all_in"},
        ],
    }


def start_hand(tracker, hand_id, hero_seat=5):
    tracker.on_hand_start({
        "type": "hand_start",
        "hand_id": hand_id,
        "seat": hero_seat,
        "dealer_seat": 0,
        "blinds": {"small_blind": 50, "big_blind": 100},
    })


def test_openpoker_metrics_cover_preflop_cbet_big_pot_latency_and_json(tmp_path):
    tracker = make_tracker(tmp_path)

    start_hand(tracker, "steal")
    tracker.on_player_action({"seat": 3, "name": "utg", "street": "preflop", "action": "fold"})
    tracker.on_player_action({"seat": 4, "name": "hj", "street": "preflop", "action": "fold"})
    tracker.on_hero_decision(
        hero_state(pot=150),
        {"action": "raise", "amount": 250},
        {"action": "raise", "amount": 300},
        turn_msg("steal"),
        latency_ms=12,
    )
    tracker.on_player_action({"seat": 0, "name": "btn", "street": "preflop", "action": "fold"})
    tracker.on_player_action({"seat": 1, "name": "sb", "street": "preflop", "action": "fold"})
    tracker.on_player_action({"seat": 2, "name": "bb", "street": "preflop", "action": "fold"})
    tracker.on_hand_result({"type": "hand_result", "hand_id": "steal", "pot": 450, "winners": [{"seat": 5, "amount": 450}], "showdown": False})

    start_hand(tracker, "cbet")
    tracker.on_player_action({"seat": 3, "name": "utg", "street": "preflop", "action": "fold"})
    tracker.on_player_action({"seat": 4, "name": "hj", "street": "preflop", "action": "fold"})
    tracker.on_hero_decision(hero_state(pot=150), {"action": "raise", "amount": 300}, {"action": "raise", "amount": 300}, turn_msg("cbet"), 20)
    tracker.on_player_action({"seat": 0, "name": "btn", "street": "preflop", "action": "fold"})
    tracker.on_player_action({"seat": 1, "name": "sb", "street": "preflop", "action": "fold"})
    tracker.on_player_action({"seat": 2, "name": "bb", "street": "preflop", "action": "call", "contribution_delta": 200})
    tracker.on_community_cards({"type": "community_cards", "street": "flop", "cards": ["Qs", "7d", "2c"]})
    tracker.on_hero_decision(
        hero_state(street="flop", pot=650, stack=9700, board=["Qs", "7d", "2c"]),
        {"action": "raise", "amount": 600},
        {"action": "raise", "amount": 600},
        turn_msg("cbet", [{"action": "check"}, {"action": "raise", "min": 200, "max": 9700}]),
        latency_ms=40,
    )
    tracker.on_player_action({"seat": 2, "name": "bb", "street": "flop", "action": "fold"})
    tracker.on_hand_result({"type": "hand_result", "hand_id": "cbet", "pot": 5500, "winners": [{"seat": 5, "amount": 1250}], "showdown": False})

    snapshot = tracker.snapshot()
    metrics = snapshot["metrics"]

    assert snapshot["summary"]["hands"] == 2
    assert snapshot["summary"]["chip_delta"] == 500
    assert metrics["Core performance metrics"]["bb_per_100"] == 250.0
    assert metrics["Preflop style metrics"]["vpip"] == 100.0
    assert metrics["Preflop style metrics"]["pfr"] == 100.0
    assert metrics["Preflop style metrics"]["steal_attempt"] == 100.0
    assert metrics["Postflop aggression metrics"]["continuation_bet_flop"] == 100.0
    assert metrics["Engineering reliability metrics"]["raw_action_changed_count"] == 1
    assert metrics["Engineering reliability metrics"]["bet_below_min_attempts"] == 1
    assert metrics["Engineering reliability metrics"]["latency_by_street"]["flop"]["max_ms"] == 40.0
    assert metrics["Engineering reliability metrics"]["latency_by_branch"]["raise_available"]["count"] == 3
    assert metrics["Big pot / stack-off metrics"]["net_result_pots_gt_50bb"] == 350.0
    assert metrics["Bet sizing metrics"]["flop_bet_size_pct_pot"] == 0.923
    assert metrics["Street by street EV"]["ev_after_seeing_flop"] == 350.0

    written = json.loads((tmp_path / "openpoker_metrics.json").read_text())
    assert written["schema_version"] == 1
    assert written["recent_hands"][-1]["hand_id"] == "cbet"


def test_openpoker_metrics_cover_fold_to_three_bet_and_pot_odds(tmp_path):
    tracker = make_tracker(tmp_path)

    start_hand(tracker, "fold_to_3bet")
    tracker.on_player_action({"seat": 3, "name": "utg", "street": "preflop", "action": "fold"})
    tracker.on_player_action({"seat": 4, "name": "hj", "street": "preflop", "action": "fold"})
    tracker.on_hero_decision(hero_state(pot=150), {"action": "raise", "amount": 300}, {"action": "raise", "amount": 300}, turn_msg("fold_to_3bet"), 10)
    tracker.on_player_action({"seat": 0, "name": "btn", "street": "preflop", "action": "raise", "amount": 900, "contribution_delta": 900})
    tracker.on_hero_decision(
        hero_state(pot=1350, owed=600, stack=9700, bet=300),
        {"action": "fold"},
        {"action": "fold"},
        turn_msg("fold_to_3bet", [{"action": "fold"}, {"action": "call", "amount": 600}, {"action": "raise", "min": 1800, "max": 9700}]),
        14,
    )
    tracker.on_hand_result({"type": "hand_result", "hand_id": "fold_to_3bet", "pot": 1350, "winners": [{"seat": 0, "amount": 1350}], "showdown": False})

    metrics = tracker.snapshot()["metrics"]

    assert metrics["Preflop style metrics"]["fold_to_three_bet"] == 100.0
    assert metrics["Pot odds / call quality metrics"]["average_pot_odds_faced_when_folding"] == 0.308
