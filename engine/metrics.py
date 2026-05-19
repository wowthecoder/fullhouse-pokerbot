"""
Analytics helpers for local Fullhouse demo runs.

The functions in this module consume the public match result shape returned by
``sandbox.match.run_match``. They do not need bot code or live engine objects,
which keeps the demo routes thin and makes the statistics easy to test.
"""

from __future__ import annotations

import math
from collections import defaultdict

from engine.game import BIG_BLIND


POSITION_LABELS_6MAX = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
LATE_POSITIONS = {"CO", "BTN", "SB"}
WEAK_SHOWDOWN_HANDS = {"High Card", "Pair"}


def aggregate_match_metrics(results):
    """Return grouped per-bot metrics for one or more match results."""
    if isinstance(results, dict):
        results = [results]

    stats = defaultdict(_new_bot_stats)
    match_count = 0

    for result in results:
        match_count += 1
        for bid in result.get("bot_ids", []):
            stats[bid]["bot_id"] = bid

        for hand in result.get("hands", []):
            _consume_hand(stats, hand)

        for bid, errors in result.get("bot_errors", {}).items():
            stats[bid]["bot_id"] = bid
            stats[bid]["bot_error_count"] += len(errors or [])

    bots = [_finalize_bot_metrics(s) for s in stats.values()]
    bots.sort(key=lambda b: (-b["chip_delta"], b["bot_id"]))

    return {
        "summary": {
            "matches": match_count,
            "hands": sum(len(r.get("hands", [])) for r in results),
            "big_blind": BIG_BLIND,
        },
        "bots": bots,
    }


def _new_bot_stats():
    return {
        "bot_id": "",
        "hands": 0,
        "chip_delta": 0,
        "hand_bb_deltas": [],
        "positions": defaultdict(_new_position_stats),
        "vpip": 0,
        "pfr": 0,
        "three_bet": 0,
        "three_bet_opp": 0,
        "fold_to_three_bet": 0,
        "fold_to_three_bet_opp": 0,
        "limp": 0,
        "steal_attempt": 0,
        "steal_opp": 0,
        "fold_bb_to_steal": 0,
        "fold_bb_to_steal_opp": 0,
        "postflop_aggressive": 0,
        "postflop_calls": 0,
        "postflop_opportunities": 0,
        "cbet": defaultdict(int),
        "cbet_opp": defaultdict(int),
        "fold_to_cbet": 0,
        "fold_to_cbet_opp": 0,
        "bluff": 0,
        "bluff_opp": 0,
        "saw_flop": 0,
        "went_to_showdown": 0,
        "won_at_showdown": 0,
        "showdown_winnings": 0,
        "non_showdown_winnings": 0,
        "decisions": 0,
        "illegal_actions": 0,
        "timeouts": 0,
        "crashes": 0,
        "latencies": [],
        "bot_error_count": 0,
    }


def _new_position_stats():
    return {
        "hands": 0,
        "chip_delta": 0,
        "bb_deltas": [],
        "vpip": 0,
        "pfr": 0,
    }


def _consume_hand(stats, hand):
    starts = hand.get("starting_stacks") or {}
    finals = hand.get("final_stacks") or {}
    positions = _position_by_bot(hand)
    decisions = hand.get("decision_log") or []
    showdown = bool(hand.get("showdown"))
    revealed = hand.get("revealed_cards") or {}
    winners = {w.get("bot_id") for w in hand.get("winners", [])}
    strengths = hand.get("hand_strengths") or {}

    bot_ids = set(starts) | set(finals) | {d.get("bot_id") for d in decisions if d.get("bot_id")}
    for bid in bot_ids:
        stats[bid]["bot_id"] = bid

    preflop = [d for d in decisions if d.get("street") == "preflop"]
    postflop = [d for d in decisions if d.get("street") in ("flop", "turn", "river")]
    preflop_by_bot = _group_by_bot(preflop)
    postflop_by_bot = _group_by_bot(postflop)
    preflop_aggressor = _last_aggressor(preflop)
    saw_flop = _hand_reached_street(hand, "flop")
    folded_preflop = {d.get("bot_id") for d in preflop if d.get("action") == "fold"}

    _consume_preflop(stats, hand, preflop, preflop_by_bot, positions)
    _consume_postflop(stats, hand, postflop, postflop_by_bot, preflop_aggressor, strengths)
    _consume_reliability(stats, decisions)

    for bid in bot_ids:
        if bid not in starts and bid not in finals:
            continue
        start = starts.get(bid, finals.get(bid, 0))
        final = finals.get(bid, start)
        delta = final - start
        pos = positions.get(bid, "UNK")

        stats[bid]["hands"] += 1
        stats[bid]["chip_delta"] += delta
        stats[bid]["hand_bb_deltas"].append(delta / BIG_BLIND)

        pst = stats[bid]["positions"][pos]
        pst["hands"] += 1
        pst["chip_delta"] += delta
        pst["bb_deltas"].append(delta / BIG_BLIND)

        if saw_flop and bid not in folded_preflop:
            stats[bid]["saw_flop"] += 1
            if showdown and bid in revealed:
                stats[bid]["went_to_showdown"] += 1

        if showdown and bid in revealed:
            stats[bid]["showdown_winnings"] += delta
            if bid in winners or delta > 0:
                stats[bid]["won_at_showdown"] += 1
        elif not showdown:
            stats[bid]["non_showdown_winnings"] += delta


def _consume_preflop(stats, hand, preflop, preflop_by_bot, positions):
    seen_raise = False
    first_raiser = None
    three_bettor = None
    acted_after_three_bet = set()
    steal_opener = None
    steal_target_bb = None
    vpip_bots = set()
    pfr_bots = set()

    for d in preflop:
        bid = d.get("bot_id")
        action = d.get("action")
        pos = positions.get(bid, "UNK")
        aggressive = action in ("raise", "all_in")

        if bid is None:
            continue

        if not seen_raise and pos in LATE_POSITIONS:
            stats[bid]["steal_opp"] += 1

        if action in ("call", "raise", "all_in"):
            vpip_bots.add(bid)

        if seen_raise and first_raiser != bid:
            stats[bid]["three_bet_opp"] += 1

        if aggressive:
            pfr_bots.add(bid)
            if not seen_raise:
                first_raiser = bid
                if pos in LATE_POSITIONS:
                    stats[bid]["steal_attempt"] += 1
                    steal_opener = bid
                    steal_target_bb = _bot_in_position(positions, "BB")
            elif first_raiser != bid:
                stats[bid]["three_bet"] += 1
                three_bettor = bid
                if first_raiser:
                    stats[first_raiser]["fold_to_three_bet_opp"] += 1
            seen_raise = True
            continue

        if three_bettor and bid == first_raiser and bid not in acted_after_three_bet:
            acted_after_three_bet.add(bid)
            if action == "fold":
                stats[bid]["fold_to_three_bet"] += 1

        if steal_opener and bid == steal_target_bb:
            stats[bid]["fold_bb_to_steal_opp"] += 1
            if action == "fold":
                stats[bid]["fold_bb_to_steal"] += 1

    for bid, actions in preflop_by_bot.items():
        first_action = actions[0].get("action") if actions else None
        had_raise_before = False
        for d in preflop:
            if d.get("bot_id") == bid:
                break
            if d.get("action") in ("raise", "all_in"):
                had_raise_before = True
        if first_action == "call" and not had_raise_before:
            stats[bid]["limp"] += 1

    for bid in vpip_bots:
        pos = positions.get(bid, "UNK")
        stats[bid]["vpip"] += 1
        stats[bid]["positions"][pos]["vpip"] += 1

    for bid in pfr_bots:
        pos = positions.get(bid, "UNK")
        stats[bid]["pfr"] += 1
        stats[bid]["positions"][pos]["pfr"] += 1


def _consume_postflop(stats, hand, postflop, postflop_by_bot, preflop_aggressor, strengths):
    by_street = defaultdict(list)
    for d in postflop:
        by_street[d.get("street")].append(d)
        bid = d.get("bot_id")
        action = d.get("action")
        if bid is None:
            continue
        stats[bid]["postflop_opportunities"] += 1
        if action in ("raise", "all_in"):
            stats[bid]["postflop_aggressive"] += 1
        elif action == "call":
            stats[bid]["postflop_calls"] += 1

    for street in ("flop", "turn", "river"):
        street_actions = by_street.get(street, [])
        if not street_actions or not preflop_aggressor:
            continue

        aggressor_actions = [d for d in street_actions if d.get("bot_id") == preflop_aggressor]
        if aggressor_actions:
            stats[preflop_aggressor]["cbet_opp"][street] += 1
            first_aggressor_action = aggressor_actions[0]
            if first_aggressor_action.get("action") in ("raise", "all_in"):
                stats[preflop_aggressor]["cbet"][street] += 1
                _consume_fold_to_cbet(stats, street_actions, first_aggressor_action)

    if hand.get("showdown"):
        for bid, actions in postflop_by_bot.items():
            aggressive_actions = [d for d in actions if d.get("action") in ("raise", "all_in")]
            if not aggressive_actions or bid not in strengths:
                continue
            stats[bid]["bluff_opp"] += len(aggressive_actions)
            if strengths.get(bid) in WEAK_SHOWDOWN_HANDS:
                stats[bid]["bluff"] += len(aggressive_actions)


def _consume_fold_to_cbet(stats, street_actions, cbet_action):
    cbetter = cbet_action.get("bot_id")
    seen_cbet = False
    faced = set()
    for d in street_actions:
        if d is cbet_action:
            seen_cbet = True
            continue
        if not seen_cbet or d.get("bot_id") == cbetter:
            continue
        bid = d.get("bot_id")
        if bid in faced:
            continue
        faced.add(bid)
        stats[bid]["fold_to_cbet_opp"] += 1
        if d.get("action") == "fold":
            stats[bid]["fold_to_cbet"] += 1


def _consume_reliability(stats, decisions):
    for d in decisions:
        bid = d.get("bot_id")
        if bid is None:
            continue
        stats[bid]["decisions"] += 1
        if d.get("illegal_action"):
            stats[bid]["illegal_actions"] += 1
        if d.get("timeout"):
            stats[bid]["timeouts"] += 1
        if d.get("crash"):
            stats[bid]["crashes"] += 1
        latency = d.get("latency_ms")
        if isinstance(latency, (int, float)):
            stats[bid]["latencies"].append(float(latency))


def _finalize_bot_metrics(s):
    bb100, low, high = _bb100_ci(s["hand_bb_deltas"])
    hands = s["hands"]
    decisions = s["decisions"]
    showdowns = s["went_to_showdown"]

    positions = {}
    for pos in sorted(s["positions"]):
        pst = s["positions"][pos]
        positions[pos] = {
            "hands": pst["hands"],
            "bb_per_100": _rate(sum(pst["bb_deltas"]) * 100, pst["hands"]),
            "vpip": _pct(pst["vpip"], pst["hands"]),
            "pfr": _pct(pst["pfr"], pst["hands"]),
            "chip_delta": pst["chip_delta"],
        }

    return {
        "bot_id": s["bot_id"],
        "hands": hands,
        "chip_delta": s["chip_delta"],
        "groups": {
            "Core performance metrics": {
                "bb_per_100": bb100,
                "bb_per_100_ci_low": low,
                "bb_per_100_ci_high": high,
                "ev_per_hand": _rate(s["chip_delta"], hands),
            },
            "Preflop style metrics": {
                "vpip": _pct(s["vpip"], hands),
                "pfr": _pct(s["pfr"], hands),
                "three_bet": _pct(s["three_bet"], s["three_bet_opp"]),
                "fold_to_three_bet": _pct(s["fold_to_three_bet"], s["fold_to_three_bet_opp"]),
                "limp": _pct(s["limp"], hands),
                "steal_attempt": _pct(s["steal_attempt"], s["steal_opp"]),
                "fold_bb_to_steal": _pct(s["fold_bb_to_steal"], s["fold_bb_to_steal_opp"]),
                "open_raise_by_position": {
                    pos: _pct(pst["pfr"], pst["hands"])
                    for pos, pst in sorted(s["positions"].items())
                },
            },
            "Postflop aggression metrics": {
                "aggression_factor": _ratio(s["postflop_aggressive"], s["postflop_calls"]),
                "aggression_frequency": _pct(s["postflop_aggressive"], s["postflop_opportunities"]),
                "continuation_bet_flop": _pct(s["cbet"]["flop"], s["cbet_opp"]["flop"]),
                "continuation_bet_turn": _pct(s["cbet"]["turn"], s["cbet_opp"]["turn"]),
                "continuation_bet_river": _pct(s["cbet"]["river"], s["cbet_opp"]["river"]),
                "fold_to_continuation_bet": _pct(s["fold_to_cbet"], s["fold_to_cbet_opp"]),
                "showdown_observed_bluff_frequency": _pct(s["bluff"], s["bluff_opp"]),
            },
            "Showdown and hand-quality metrics": {
                "wtsd": _pct(s["went_to_showdown"], s["saw_flop"]),
                "wonsd": _pct(s["won_at_showdown"], showdowns),
                "showdown_win_rate": _pct(s["won_at_showdown"], showdowns),
                "non_showdown_winnings": s["non_showdown_winnings"],
                "showdown_winnings": s["showdown_winnings"],
            },
            "Position metrics": {
                "by_position": positions,
                "blind_loss_rate": _blind_loss_rate(positions),
                "button_win_rate": positions.get("BTN", {}).get("bb_per_100"),
            },
            "Engineering reliability metrics": {
                "crash_rate": _pct(s["crashes"], decisions),
                "illegal_action_rate": _pct(s["illegal_actions"], decisions),
                "timeout_rate": _pct(s["timeouts"], decisions),
                "average_decision_latency_ms": _mean(s["latencies"]),
                "p95_decision_latency_ms": _percentile(s["latencies"], 95),
                "p99_decision_latency_ms": _percentile(s["latencies"], 99),
                "bot_error_count": s["bot_error_count"],
                "decision_count": decisions,
            },
        },
    }


def _position_by_bot(hand):
    seat_to_bot = {}
    for event in hand.get("events", []):
        if event.get("bot_id") is not None and event.get("seat") is not None:
            seat_to_bot[event["seat"]] = event["bot_id"]
    for d in hand.get("decision_log") or []:
        if d.get("bot_id") is not None and d.get("seat") is not None:
            seat_to_bot[d["seat"]] = d["bot_id"]

    blind_events = [e for e in hand.get("events", []) if e.get("type") == "blind"]
    sb = next((e.get("seat") for e in blind_events if e.get("action") == "small_blind"), None)
    if sb is None or not seat_to_bot:
        return {bid: "UNK" for bid in set(seat_to_bot.values())}

    n = len(seat_to_bot)
    dealer = sb if n == 2 else (sb - 1) % n
    labels = POSITION_LABELS_6MAX if n == 6 else _generic_position_labels(n)
    positions = {}
    for offset in range(n):
        seat = (dealer + offset) % n
        bid = seat_to_bot.get(seat)
        if bid is not None:
            positions[bid] = labels[offset] if offset < len(labels) else "UNK"
    return positions


def _generic_position_labels(n):
    if n == 2:
        return ["BTN", "BB"]
    labels = ["BTN", "SB", "BB"]
    middle = ["UTG", "UTG+1", "MP", "LJ", "HJ", "CO"]
    return labels + middle[: max(0, n - 3)]


def _group_by_bot(actions):
    grouped = defaultdict(list)
    for action in actions:
        if action.get("bot_id") is not None:
            grouped[action["bot_id"]].append(action)
    return grouped


def _last_aggressor(actions):
    aggressor = None
    for action in actions:
        if action.get("action") in ("raise", "all_in"):
            aggressor = action.get("bot_id")
    return aggressor


def _hand_reached_street(hand, street):
    return any(e.get("type") == "street_start" and e.get("street") == street for e in hand.get("events", []))


def _bot_in_position(positions, target):
    for bid, pos in positions.items():
        if pos == target:
            return bid
    return None


def _bb100_ci(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    bb100 = mean * 100
    if n == 1:
        return _round(bb100), _round(bb100), _round(bb100)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = math.sqrt(variance) / math.sqrt(n)
    margin = 1.96 * stderr * 100
    return _round(bb100), _round(bb100 - margin), _round(bb100 + margin)


def _blind_loss_rate(positions):
    blind_values = []
    for pos in ("SB", "BB"):
        value = positions.get(pos, {}).get("bb_per_100")
        if value is not None:
            blind_values.append(value)
    if not blind_values:
        return None
    return _round(sum(blind_values) / len(blind_values))


def _pct(num, den):
    return _round((num / den) * 100) if den else None


def _ratio(num, den):
    return _round(num / den) if den else None


def _rate(num, den):
    return _round(num / den) if den else 0.0


def _mean(values):
    return _round(sum(values) / len(values)) if values else None


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return _round(ordered[lo])
    return _round(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo))


def _round(value):
    if value is None:
        return None
    return round(float(value), 3)
