import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.metrics import aggregate_match_metrics


BOTS = ["A", "B", "C", "D", "E", "F"]


def _events(include_flop=False):
    events = [
        {"type": "blind", "street": "preflop", "seat": 1, "bot_id": "B", "action": "small_blind", "amount": 50},
        {"type": "blind", "street": "preflop", "seat": 2, "bot_id": "C", "action": "big_blind", "amount": 100},
        {"type": "street_start", "street": "preflop", "community_cards": []},
    ]
    if include_flop:
        events.append({"type": "street_start", "street": "flop", "community_cards": ["As", "7d", "2c"]})
    return events


def _decision(seat, bot_id, street, action, amount=0, latency=5, illegal=False, error=None):
    return {
        "seat": seat,
        "bot_id": bot_id,
        "street": street,
        "raw_action": action,
        "raw_amount": amount,
        "action": action,
        "amount": amount,
        "latency_ms": latency,
        "illegal_action": illegal,
        "runner_error": error,
        "timeout": error == "timeout",
        "crash": bool(error and error != "timeout"),
    }


def _hand(decisions, finals=None, showdown=False, revealed=None, strengths=None, include_flop=False):
    starts = {bid: 10_000 for bid in BOTS}
    finals = finals or starts
    return {
        "starting_stacks": starts,
        "final_stacks": finals,
        "events": _events(include_flop=include_flop),
        "decision_log": decisions,
        "showdown": showdown,
        "revealed_cards": revealed or {},
        "hand_strengths": strengths or {},
        "winners": [{"bot_id": "F", "seat": 5, "amount": 1000}] if showdown else [{"bot_id": "F", "seat": 5, "amount": 150}],
    }


def _bot(metrics, bot_id):
    return next(b for b in metrics["bots"] if b["bot_id"] == bot_id)


def test_preflop_metrics_cover_three_bet_fold_and_steal():
    hand_three_bet = _hand([
        _decision(3, "D", "preflop", "fold"),
        _decision(4, "E", "preflop", "raise", 300),
        _decision(5, "F", "preflop", "raise", 900),
        _decision(0, "A", "preflop", "fold"),
        _decision(1, "B", "preflop", "fold"),
        _decision(2, "C", "preflop", "fold"),
        _decision(4, "E", "preflop", "fold"),
    ])
    hand_steal = _hand([
        _decision(3, "D", "preflop", "fold"),
        _decision(4, "E", "preflop", "fold"),
        _decision(5, "F", "preflop", "raise", 300),
        _decision(0, "A", "preflop", "fold"),
        _decision(1, "B", "preflop", "fold"),
        _decision(2, "C", "preflop", "fold"),
    ])

    metrics = aggregate_match_metrics({"bot_ids": BOTS, "hands": [hand_three_bet, hand_steal], "bot_errors": {}})

    f_preflop = _bot(metrics, "F")["groups"]["Preflop style metrics"]
    c_preflop = _bot(metrics, "C")["groups"]["Preflop style metrics"]
    e_preflop = _bot(metrics, "E")["groups"]["Preflop style metrics"]

    assert f_preflop["three_bet"] == 100.0
    assert f_preflop["steal_attempt"] == 100.0
    assert c_preflop["fold_bb_to_steal"] == 100.0
    assert e_preflop["fold_to_three_bet"] == 100.0


def test_postflop_showdown_position_and_reliability_metrics():
    hand = _hand(
        [
            _decision(3, "D", "preflop", "fold", latency=1),
            _decision(4, "E", "preflop", "raise", 300, latency=2),
            _decision(5, "F", "preflop", "call", latency=3),
            _decision(0, "A", "preflop", "fold", latency=4, illegal=True),
            _decision(1, "B", "preflop", "fold", latency=5, error="timeout"),
            _decision(2, "C", "preflop", "fold", latency=6, error="exception"),
            _decision(4, "E", "flop", "raise", 600, latency=7),
            _decision(5, "F", "flop", "call", latency=8),
            _decision(4, "E", "turn", "check", latency=9),
            _decision(5, "F", "turn", "check", latency=10),
            _decision(4, "E", "river", "check", latency=11),
            _decision(5, "F", "river", "check", latency=12),
        ],
        finals={"A": 10_000, "B": 10_000, "C": 10_000, "D": 10_000, "E": 9_500, "F": 10_500},
        showdown=True,
        revealed={"E": ["Ah", "Kd"], "F": ["2s", "2h"]},
        strengths={"E": "High Card", "F": "Pair"},
        include_flop=True,
    )

    metrics = aggregate_match_metrics({"bot_ids": BOTS, "hands": [hand], "bot_errors": {"C": ["exception"]}})

    e = _bot(metrics, "E")
    f = _bot(metrics, "F")
    a_reliability = _bot(metrics, "A")["groups"]["Engineering reliability metrics"]
    b_reliability = _bot(metrics, "B")["groups"]["Engineering reliability metrics"]
    c_reliability = _bot(metrics, "C")["groups"]["Engineering reliability metrics"]

    assert e["groups"]["Core performance metrics"]["ev_per_hand"] == -500.0
    assert e["groups"]["Postflop aggression metrics"]["continuation_bet_flop"] == 100.0
    assert e["groups"]["Postflop aggression metrics"]["showdown_observed_bluff_frequency"] == 100.0
    assert f["groups"]["Showdown and hand-quality metrics"]["wonsd"] == 100.0
    assert f["groups"]["Position metrics"]["by_position"]["CO"]["bb_per_100"] == 500.0
    assert a_reliability["illegal_action_rate"] == 100.0
    assert b_reliability["timeout_rate"] == 100.0
    assert c_reliability["crash_rate"] == 100.0
    assert c_reliability["bot_error_count"] == 1
