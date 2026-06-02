"""Live OpenPoker metrics captured from wrapper-visible events.

This module intentionally depends only on websocket payloads and wrapper state.
It does not import or instrument bot strategy modules.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_RECENT_HAND_LIMIT = 100
POSITION_LABELS_6MAX = ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
LATE_POSITIONS = {"CO", "BTN", "SB"}
STREETS = ("preflop", "flop", "turn", "river")
POSTFLOP_STREETS = ("flop", "turn", "river")


@dataclass
class HandRecord:
    hand_id: str
    table_id: str | None
    dealer_seat: int | None
    hero_seat: int | None
    hero_name: str
    blinds: dict[str, float]
    players: dict[int, dict[str, Any]]
    starting_stacks: dict[int, float]
    positions: dict[int, str]
    hole_cards: list[str] = field(default_factory=list)
    community_cards: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    hero_decisions: list[dict[str, Any]] = field(default_factory=list)
    street_bets: dict[int, float] = field(default_factory=dict)
    total_invested: dict[int, float] = field(default_factory=dict)
    pot: float = 0.0
    final_stacks: dict[int, float] = field(default_factory=dict)
    winners: list[dict[str, Any]] = field(default_factory=list)
    showdown: bool | None = None
    ending_street: str = "preflop"
    delta: float | None = None


class OpenPokerMetricsTracker:
    def __init__(
        self,
        metrics_file: str | Path,
        *,
        ws_url: str,
        buy_in: float,
        bot_path: str | Path,
        recent_hand_limit: int = DEFAULT_RECENT_HAND_LIMIT,
    ) -> None:
        self.metrics_file = Path(metrics_file)
        self.recent_hands = deque(maxlen=recent_hand_limit)
        self.hands: list[dict[str, Any]] = []
        self.current_hand: HandRecord | None = None
        self.players: dict[int, dict[str, Any]] = {}
        self.hero_name: str = "hero"
        self.hero_seat: int | None = None
        self.table_id: str | None = None
        self.big_blind = 20.0
        self.session = {
            "ws_url": ws_url,
            "bot_path": str(bot_path),
            "hero_name": None,
            "table_id": None,
            "buy_in": buy_in,
            "big_blind": self.big_blind,
        }
        self.notes = [
            "Reliable-only OpenPoker metrics: hidden-card equity, true bluff frequency, "
            "estimated-equity call quality, hero-call efficiency, and stack-off hand class "
            "are omitted unless OpenPoker exposes enough data in future payloads."
        ]

    def on_connected(self, msg: dict[str, Any]) -> None:
        name = msg.get("name")
        if name:
            self.hero_name = str(name)
            self.session["hero_name"] = self.hero_name

    def on_table_joined(self, msg: dict[str, Any]) -> None:
        self.table_id = msg.get("table_id")
        self.session["table_id"] = self.table_id
        self.hero_seat = _parse_int(msg.get("seat"), self.hero_seat)
        self._set_players(msg.get("players") or [])

    def on_hand_start(self, msg: dict[str, Any], players: dict[int, dict[str, Any]] | None = None) -> None:
        if players:
            self._set_players(players.values())

        blinds = msg.get("blinds") or {}
        self.big_blind = _parse_float(blinds.get("big_blind"), self.big_blind) or self.big_blind
        self.session["big_blind"] = self.big_blind

        dealer = _parse_int(msg.get("dealer_seat"))
        self.hero_seat = _parse_int(msg.get("seat"), self.hero_seat)
        hand_id = str(msg.get("hand_id") or f"hand_{len(self.hands) + 1}")
        starting = {seat: _parse_float(player.get("stack"), 0.0) or 0.0 for seat, player in self.players.items()}
        positions = _positions_by_seat(sorted(self.players), dealer)
        self.current_hand = HandRecord(
            hand_id=hand_id,
            table_id=self.table_id,
            dealer_seat=dealer,
            hero_seat=self.hero_seat,
            hero_name=self.hero_name,
            blinds={"small_blind": _parse_float(blinds.get("small_blind"), 0.0) or 0.0, "big_blind": self.big_blind},
            players={seat: dict(player) for seat, player in self.players.items()},
            starting_stacks=starting,
            positions=positions,
        )
        self._record_blinds()

    def on_hole_cards(self, msg: dict[str, Any]) -> None:
        if self.current_hand is not None:
            self.current_hand.hole_cards = list(msg.get("cards") or [])

    def on_community_cards(self, msg: dict[str, Any]) -> None:
        hand = self.current_hand
        if hand is None:
            return
        street = msg.get("street")
        cards = list(msg.get("cards") or [])
        if len(cards) == 1 and hand.community_cards:
            hand.community_cards.extend(cards)
        elif cards:
            hand.community_cards = cards
        if street in POSTFLOP_STREETS:
            hand.ending_street = str(street)
            hand.street_bets = {}

    def on_player_action(self, msg: dict[str, Any]) -> None:
        hand = self.current_hand
        if hand is None:
            return
        seat = _parse_int(msg.get("seat"))
        if seat is None:
            return
        action = _normalize_action(msg.get("action"))
        street = _street_from_msg(msg, hand)
        if street != hand.ending_street and street in STREETS:
            hand.street_bets = {}
            hand.ending_street = street
        amount = _parse_float(msg.get("amount"), 0.0) or 0.0
        contribution = self._contribution_from_action(hand, seat, action, amount, msg)
        self._append_action(hand, seat, action, street, amount, contribution, msg)

        stack = _parse_float(msg.get("stack"), None)
        if stack is None:
            stack = _parse_float(msg.get("stack_after"), None)
        if stack is not None:
            self.players.setdefault(seat, {"seat": seat})["stack"] = stack

    def on_hero_decision(
        self,
        fullhouse_state: dict[str, Any],
        raw_action: dict[str, Any],
        clamped_action: dict[str, Any],
        turn_msg: dict[str, Any],
        latency_ms: float,
    ) -> None:
        hand = self.current_hand
        if hand is None:
            return
        seat = _parse_int(fullhouse_state.get("seat_to_act"), self.hero_seat)
        if seat is None:
            return
        self.hero_seat = seat
        street = str(fullhouse_state.get("street") or _street_from_board(fullhouse_state.get("community_cards") or []))
        if street != hand.ending_street and street in STREETS:
            hand.street_bets = {}
            hand.ending_street = street

        raw_name = _normalize_action(raw_action.get("action"))
        action = _normalize_action(clamped_action.get("action"))
        amount = _parse_float(clamped_action.get("amount"), 0.0) or 0.0
        amount_owed = _parse_float(fullhouse_state.get("amount_owed"), 0.0) or 0.0
        pot = _parse_float(fullhouse_state.get("pot"), hand.pot) or hand.pot
        valid_actions = _valid_action_map(turn_msg)
        contribution = self._hero_contribution(hand, seat, action, amount, fullhouse_state)
        branches = _decision_branches(fullhouse_state, valid_actions)
        raw_amount = _parse_float(raw_action.get("amount"), None)
        min_raise = _parse_float(valid_actions.get("raise", {}).get("min"), None)
        max_raise = _parse_float(valid_actions.get("raise", {}).get("max"), None)
        illegal_low = raw_name == "raise" and raw_amount is not None and min_raise is not None and raw_amount < min_raise
        illegal_high = raw_name == "raise" and raw_amount is not None and max_raise is not None and raw_amount > max_raise

        decision = {
            "seat": seat,
            "bot_id": self.hero_name,
            "street": street,
            "raw_action": raw_name,
            "raw_amount": raw_amount,
            "action": action,
            "amount": amount,
            "pot": pot,
            "amount_owed": amount_owed,
            "required_equity": _required_equity(amount_owed, pot) if amount_owed > 0 else None,
            "latency_ms": round(float(latency_ms), 3),
            "branches": branches,
            "raw_action_changed": raw_name != action or (raw_amount is not None and action == "raise" and raw_amount != amount),
            "bet_below_min_attempt": bool(illegal_low),
            "bet_above_max_or_stack_attempt": bool(illegal_high),
            "valid_actions": sorted(valid_actions),
            "position": hand.positions.get(seat, "UNK"),
            "community_cards": list(fullhouse_state.get("community_cards") or []),
            "hole_cards": list(fullhouse_state.get("your_cards") or []),
            "stack": _parse_float(fullhouse_state.get("your_stack"), None),
        }
        hand.hero_decisions.append(decision)
        self._append_action(hand, seat, action, street, amount, contribution, decision, from_hero_decision=True)

    def on_hand_result(self, msg: dict[str, Any]) -> dict[str, Any]:
        hand = self.current_hand
        if hand is None:
            return self.write_snapshot()

        self._apply_final_stacks(hand, msg)
        hand.winners = list(msg.get("winners") or [])
        hand.pot = _parse_float(msg.get("pot"), hand.pot) or hand.pot
        hand.showdown = _showdown_from_result(msg)
        hand.ending_street = str(msg.get("street") or _street_from_board(msg.get("community_cards") or hand.community_cards))
        hand.delta = self._hero_delta(hand)

        finalized = self._compact_hand(hand)
        self.hands.append(finalized)
        self.recent_hands.append(finalized)
        self.current_hand = None
        return self.write_snapshot()

    def write_snapshot(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        _atomic_write_json(self.metrics_file, snapshot)
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "session": dict(self.session),
            "summary": self._summary(),
            "metrics": _aggregate_hero_metrics(self.hands, self.big_blind),
            "recent_hands": list(self.recent_hands),
            "notes": list(self.notes),
        }

    def _set_players(self, players: Any) -> None:
        for player in players:
            seat = _parse_int(player.get("seat") if isinstance(player, dict) else None)
            if seat is None:
                continue
            previous = self.players.get(seat, {})
            self.players[seat] = {
                "seat": seat,
                "name": player.get("name") or previous.get("name") or f"seat_{seat}",
                "bot_id": player.get("name") or previous.get("bot_id") or f"seat_{seat}",
                "stack": _parse_float(player.get("stack"), previous.get("stack", 0.0)) or 0.0,
            }

    def _record_blinds(self) -> None:
        hand = self.current_hand
        if hand is None or hand.dealer_seat is None:
            return
        seats = sorted(hand.players)
        if not seats:
            return
        sb_seat = hand.dealer_seat if len(seats) == 2 else _next_seat(seats, hand.dealer_seat)
        bb_seat = _next_seat(seats, sb_seat) if sb_seat is not None else None
        if sb_seat is not None:
            self._append_action(hand, sb_seat, "small_blind", "preflop", hand.blinds["small_blind"], hand.blinds["small_blind"], {})
        if bb_seat is not None:
            self._append_action(hand, bb_seat, "big_blind", "preflop", hand.blinds["big_blind"], hand.blinds["big_blind"], {})

    def _contribution_from_action(
        self,
        hand: HandRecord,
        seat: int,
        action: str,
        amount: float,
        msg: dict[str, Any],
    ) -> float:
        contribution = _parse_float(msg.get("contribution_delta"), 0.0) or 0.0
        if contribution <= 0:
            before = _parse_float(msg.get("stack_before"), None)
            after = _parse_float(msg.get("stack_after"), _parse_float(msg.get("stack"), None))
            if before is not None and after is not None:
                contribution = max(0.0, before - after)
        if contribution <= 0:
            previous_bet = hand.street_bets.get(seat, 0.0)
            if action in {"raise", "bet"} and amount > 0:
                contribution = max(0.0, amount - previous_bet)
            elif action == "call":
                contribution = max(0.0, max(hand.street_bets.values() or [0.0]) - previous_bet)
            elif action == "all_in":
                contribution = amount
        return contribution

    def _hero_contribution(
        self,
        hand: HandRecord,
        seat: int,
        action: str,
        amount: float,
        fullhouse_state: dict[str, Any],
    ) -> float:
        previous_bet = hand.street_bets.get(seat, _parse_float(fullhouse_state.get("your_bet_this_street"), 0.0) or 0.0)
        if action == "call":
            return _parse_float(fullhouse_state.get("amount_owed"), 0.0) or 0.0
        if action in {"raise", "bet"}:
            return max(0.0, amount - previous_bet)
        if action == "all_in":
            return _parse_float(fullhouse_state.get("your_stack"), 0.0) or 0.0
        return 0.0

    def _append_action(
        self,
        hand: HandRecord,
        seat: int,
        action: str,
        street: str,
        amount: float,
        contribution: float,
        source: dict[str, Any],
        *,
        from_hero_decision: bool = False,
    ) -> None:
        if action in {"call", "raise", "bet", "all_in", "small_blind", "big_blind"}:
            hand.street_bets[seat] = hand.street_bets.get(seat, 0.0) + contribution
            if action in {"raise", "bet"} and amount > 0:
                hand.street_bets[seat] = max(hand.street_bets[seat], amount)
            hand.total_invested[seat] = hand.total_invested.get(seat, 0.0) + contribution
        hand.pot += contribution
        entry = {
            "seat": seat,
            "bot_id": _bot_id_for_seat(hand, seat),
            "street": street,
            "action": action,
            "amount": _round(amount),
            "contribution": _round(contribution),
            "pot": _round(_parse_float(source.get("pot"), hand.pot) or hand.pot),
            "position": hand.positions.get(seat, "UNK"),
            "from_hero_decision": from_hero_decision,
        }
        if from_hero_decision:
            entry["amount_owed"] = source.get("amount_owed")
            entry["required_equity"] = source.get("required_equity")
        hand.actions.append(entry)

    def _apply_final_stacks(self, hand: HandRecord, msg: dict[str, Any]) -> None:
        players = msg.get("players")
        if isinstance(players, list):
            for player in players:
                seat = _parse_int(player.get("seat"))
                stack = _parse_float(player.get("stack"), None)
                if seat is not None and stack is not None:
                    hand.final_stacks[seat] = stack
                    self.players.setdefault(seat, {"seat": seat})["stack"] = stack
        for key in ("stacks", "final_stacks", "player_stacks"):
            stacks = msg.get(key)
            if isinstance(stacks, dict):
                for raw_key, value in stacks.items():
                    seat = self._seat_from_key(hand, raw_key)
                    stack = _parse_float(value, None)
                    if seat is not None and stack is not None:
                        hand.final_stacks[seat] = stack
                        self.players.setdefault(seat, {"seat": seat})["stack"] = stack

    def _seat_from_key(self, hand: HandRecord, raw_key: Any) -> int | None:
        seat = _parse_int(raw_key)
        if seat is not None:
            return seat
        key = str(raw_key)
        for candidate, player in hand.players.items():
            if key in {str(player.get("name")), str(player.get("bot_id"))}:
                return candidate
        if key == self.hero_name:
            return self.hero_seat
        return None

    def _hero_delta(self, hand: HandRecord) -> float | None:
        seat = hand.hero_seat
        if seat is None:
            return None
        if seat in hand.final_stacks and seat in hand.starting_stacks:
            return hand.final_stacks[seat] - hand.starting_stacks[seat]
        awards = _award_total_for_hero(hand)
        if awards is not None:
            return awards - hand.total_invested.get(seat, 0.0)
        return None

    def _compact_hand(self, hand: HandRecord) -> dict[str, Any]:
        hero_actions = [a for a in hand.actions if a.get("seat") == hand.hero_seat]
        return {
            "hand_id": hand.hand_id,
            "table_id": hand.table_id,
            "hero_seat": hand.hero_seat,
            "hero_position": hand.positions.get(hand.hero_seat, "UNK"),
            "big_blind": self.big_blind,
            "pot": _round(hand.pot),
            "ending_street": hand.ending_street,
            "showdown": hand.showdown,
            "delta": _round(hand.delta) if hand.delta is not None else None,
            "hole_cards": list(hand.hole_cards),
            "community_cards": list(hand.community_cards),
            "positions": {str(seat): pos for seat, pos in hand.positions.items()},
            "hero_total_invested": _round(hand.total_invested.get(hand.hero_seat, 0.0)),
            "winners": hand.winners,
            "actions": hand.actions[-80:],
            "hero_actions": hero_actions,
            "hero_decisions": hand.hero_decisions,
        }

    def _summary(self) -> dict[str, Any]:
        known_deltas = [h["delta"] for h in self.hands if h.get("delta") is not None]
        return {
            "hands": len(self.hands),
            "hands_with_known_delta": len(known_deltas),
            "first_hand_id": self.hands[0]["hand_id"] if self.hands else None,
            "last_hand_id": self.hands[-1]["hand_id"] if self.hands else None,
            "chip_delta": _round(sum(known_deltas)),
        }


def _aggregate_hero_metrics(hands: list[dict[str, Any]], big_blind: float) -> dict[str, Any]:
    known = [h for h in hands if h.get("delta") is not None]
    deltas = [float(h["delta"]) for h in known]
    bb_deltas = [d / big_blind for d in deltas if big_blind]
    bb100, ci_low, ci_high = _bb100_ci(bb_deltas)
    hand_count = len(hands)

    preflop = _preflop_metrics(hands)
    postflop = _postflop_metrics(hands)
    showdown = _showdown_metrics(hands)
    position = _position_metrics(hands, big_blind)
    reliability = _reliability_metrics(hands)
    big_pot = _big_pot_metrics(hands, big_blind)
    pot_odds = _pot_odds_metrics(hands)
    sizing = _bet_sizing_metrics(hands)
    street_ev = _street_ev_metrics(hands)

    return {
        "Core performance metrics": {
            "hands": hand_count,
            "hands_with_known_delta": len(known),
            "chip_delta": _round(sum(deltas)),
            "bb_per_100": bb100,
            "bb_per_100_ci_low": ci_low,
            "bb_per_100_ci_high": ci_high,
            "ev_per_hand": _rate(sum(deltas), len(known)),
        },
        "Preflop style metrics": preflop,
        "Postflop aggression metrics": postflop,
        "Showdown and hand-quality metrics": showdown,
        "Position metrics": position,
        "Engineering reliability metrics": reliability,
        "Big pot / stack-off metrics": big_pot,
        "Pot odds / call quality metrics": pot_odds,
        "Bet sizing metrics": sizing,
        "Street by street EV": street_ev,
    }


def _preflop_metrics(hands: list[dict[str, Any]]) -> dict[str, Any]:
    hands_n = len(hands)
    counts = Counter()
    pos_hands = Counter()
    pos_vpip = Counter()
    pos_pfr = Counter()
    for hand in hands:
        pos = hand.get("hero_position") or "UNK"
        pos_hands[pos] += 1
        actions = _actions(hand, "preflop")
        hero_actions = [a for a in actions if a.get("seat") == hand.get("hero_seat")]
        voluntary = [a for a in hero_actions if a.get("action") in {"call", "raise", "bet", "all_in"}]
        aggressive = [a for a in hero_actions if a.get("action") in {"raise", "bet", "all_in"}]
        if voluntary:
            counts["vpip"] += 1
            pos_vpip[pos] += 1
        if aggressive:
            counts["pfr"] += 1
            pos_pfr[pos] += 1
        if _hero_limped(actions, hand):
            counts["limp"] += 1
        three_opp, three_bet = _hero_three_bet(actions, hand)
        counts["three_bet_opp"] += three_opp
        counts["three_bet"] += three_bet
        f3b_opp, f3b = _hero_fold_to_three_bet(actions, hand)
        counts["fold_to_three_bet_opp"] += f3b_opp
        counts["fold_to_three_bet"] += f3b
        steal_opp, steal = _hero_steal(actions, hand)
        counts["steal_opp"] += steal_opp
        counts["steal_attempt"] += steal
        fbb_opp, fbb = _hero_fold_bb_to_steal(actions, hand)
        counts["fold_bb_to_steal_opp"] += fbb_opp
        counts["fold_bb_to_steal"] += fbb

    return {
        "vpip": _pct(counts["vpip"], hands_n),
        "pfr": _pct(counts["pfr"], hands_n),
        "three_bet": _pct(counts["three_bet"], counts["three_bet_opp"]),
        "fold_to_three_bet": _pct(counts["fold_to_three_bet"], counts["fold_to_three_bet_opp"]),
        "limp": _pct(counts["limp"], hands_n),
        "steal_attempt": _pct(counts["steal_attempt"], counts["steal_opp"]),
        "fold_bb_to_steal": _pct(counts["fold_bb_to_steal"], counts["fold_bb_to_steal_opp"]),
        "open_raise_by_position": {pos: _pct(pos_pfr[pos], pos_hands[pos]) for pos in sorted(pos_hands)},
        "vpip_by_position": {pos: _pct(pos_vpip[pos], pos_hands[pos]) for pos in sorted(pos_hands)},
        "pfr_by_position": {pos: _pct(pos_pfr[pos], pos_hands[pos]) for pos in sorted(pos_hands)},
    }


def _postflop_metrics(hands: list[dict[str, Any]]) -> dict[str, Any]:
    aggressive = calls = opportunities = 0
    cbet = Counter()
    cbet_opp = Counter()
    fold_to_cbet = fold_to_cbet_opp = 0
    for hand in hands:
        pfa = _preflop_aggressor(hand)
        hero_seat = hand.get("hero_seat")
        for action in _hero_actions(hand):
            if action.get("street") not in POSTFLOP_STREETS:
                continue
            opportunities += 1
            if action.get("action") in {"bet", "raise", "all_in"}:
                aggressive += 1
            elif action.get("action") == "call":
                calls += 1

        for street in POSTFLOP_STREETS:
            street_actions = _actions(hand, street)
            if not street_actions or pfa is None:
                continue
            first_aggr = next((a for a in street_actions if a.get("seat") == pfa), None)
            if pfa == hero_seat and first_aggr:
                cbet_opp[street] += 1
                if first_aggr.get("action") in {"bet", "raise", "all_in"}:
                    cbet[street] += 1
            elif pfa != hero_seat and first_aggr and first_aggr.get("action") in {"bet", "raise", "all_in"}:
                response = next((a for a in street_actions if a.get("seat") == hero_seat and street_actions.index(a) > street_actions.index(first_aggr)), None)
                if response:
                    fold_to_cbet_opp += 1
                    if response.get("action") == "fold":
                        fold_to_cbet += 1

    return {
        "aggression_factor": _ratio(aggressive, calls),
        "aggression_frequency": _pct(aggressive, opportunities),
        "continuation_bet_flop": _pct(cbet["flop"], cbet_opp["flop"]),
        "continuation_bet_turn": _pct(cbet["turn"], cbet_opp["turn"]),
        "continuation_bet_river": _pct(cbet["river"], cbet_opp["river"]),
        "fold_to_continuation_bet": _pct(fold_to_cbet, fold_to_cbet_opp),
    }


def _showdown_metrics(hands: list[dict[str, Any]]) -> dict[str, Any]:
    saw_flop = [h for h in hands if _saw_flop(h)]
    showdowns = [h for h in saw_flop if h.get("showdown") is True]
    wins = [h for h in showdowns if _hero_won(h)]
    showdown_winnings = [h.get("delta") for h in showdowns if h.get("delta") is not None]
    non_showdown_winnings = [h.get("delta") for h in hands if h.get("showdown") is False and h.get("delta") is not None]
    return {
        "wtsd": _pct(len(showdowns), len(saw_flop)),
        "wonsd": _pct(len(wins), len(showdowns)),
        "showdown_win_rate": _pct(len(wins), len(showdowns)),
        "non_showdown_winnings": _round(sum(non_showdown_winnings)) if non_showdown_winnings else None,
        "showdown_winnings": _round(sum(showdown_winnings)) if showdown_winnings else None,
    }


def _position_metrics(hands: list[dict[str, Any]], big_blind: float) -> dict[str, Any]:
    by_pos: dict[str, dict[str, Any]] = {}
    for pos in sorted({h.get("hero_position") or "UNK" for h in hands}):
        ph = [h for h in hands if (h.get("hero_position") or "UNK") == pos]
        known = [h for h in ph if h.get("delta") is not None]
        vpip = sum(1 for h in ph if any(a.get("action") in {"call", "raise", "bet", "all_in"} for a in _actions(h, "preflop") if a.get("seat") == h.get("hero_seat")))
        pfr = sum(1 for h in ph if any(a.get("action") in {"raise", "bet", "all_in"} for a in _actions(h, "preflop") if a.get("seat") == h.get("hero_seat")))
        by_pos[pos] = {
            "hands": len(ph),
            "bb_per_100": _rate(sum(h["delta"] / big_blind for h in known) * 100, len(known)) if big_blind else None,
            "vpip": _pct(vpip, len(ph)),
            "pfr": _pct(pfr, len(ph)),
            "chip_delta": _round(sum(h["delta"] for h in known)),
        }
    blind_values = [by_pos[pos]["bb_per_100"] for pos in ("SB", "BB") if pos in by_pos and by_pos[pos]["bb_per_100"] is not None]
    return {
        "by_position": by_pos,
        "blind_loss_rate": _round(sum(blind_values) / len(blind_values)) if blind_values else None,
        "button_win_rate": by_pos.get("BTN", {}).get("bb_per_100"),
    }


def _reliability_metrics(hands: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [d for h in hands for d in h.get("hero_decisions", [])]
    latencies = [float(d["latency_ms"]) for d in decisions if isinstance(d.get("latency_ms"), (int, float))]
    by_street = defaultdict(list)
    by_branch = defaultdict(list)
    raw_changed = below_min = above_max = 0
    for d in decisions:
        latency = d.get("latency_ms")
        if isinstance(latency, (int, float)):
            by_street[d.get("street") or "unknown"].append(float(latency))
            for branch in d.get("branches") or []:
                by_branch[branch].append(float(latency))
        raw_changed += int(bool(d.get("raw_action_changed")))
        below_min += int(bool(d.get("bet_below_min_attempt")))
        above_max += int(bool(d.get("bet_above_max_or_stack_attempt")))
    return {
        "decision_count": len(decisions),
        "average_latency_ms": _mean(latencies),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "max_latency_ms": _round(max(latencies)) if latencies else None,
        "latency_by_street": {street: _latency_summary(values) for street, values in sorted(by_street.items())},
        "latency_by_branch": {branch: _latency_summary(values) for branch, values in sorted(by_branch.items())},
        "raw_action_changed_count": raw_changed,
        "bet_below_min_attempts": below_min,
        "bet_above_max_or_stack_attempts": above_max,
    }


def _big_pot_metrics(hands: list[dict[str, Any]], big_blind: float) -> dict[str, Any]:
    known = [h for h in hands if h.get("delta") is not None]
    pots_50 = [h for h in known if big_blind and h.get("pot", 0) / big_blind > 50]
    pots_100 = [h for h in known if big_blind and h.get("pot", 0) / big_blind > 100]
    all_in_by_street = Counter(a.get("street") for h in hands for a in _hero_actions(h) if a.get("action") == "all_in")
    committed_then_folded = 0
    large_loss_final_street = Counter()
    for h in hands:
        hero_actions = _hero_actions(h)
        if hero_actions and hero_actions[-1].get("action") == "fold":
            stack = _first_decision_stack(h)
            if stack and h.get("hero_total_invested", 0) / stack >= 0.4:
                committed_then_folded += 1
        if h.get("delta") is not None and h["delta"] < 0 and big_blind and h.get("pot", 0) / big_blind > 50:
            final_investment = next((a for a in reversed(hero_actions) if a.get("contribution", 0) > 0), None)
            large_loss_final_street[(final_investment or {}).get("street", h.get("ending_street"))] += 1
    return {
        "net_result_pots_gt_50bb": _round(sum(h["delta"] for h in pots_50)) if pots_50 else None,
        "net_result_pots_gt_100bb": _round(sum(h["delta"] for h in pots_100)) if pots_100 else None,
        "net_result_all_in_pots": _round(sum(h["delta"] for h in known if any(a.get("action") == "all_in" for a in _hero_actions(h)))) or None,
        "largest_10_wins": sorted([_round(h["delta"]) for h in known if h["delta"] > 0], reverse=True)[:10],
        "largest_10_losses": sorted([_round(h["delta"]) for h in known if h["delta"] < 0])[:10],
        "all_in_frequency_by_street": {street: _pct(all_in_by_street[street], len(hands)) for street in STREETS},
        "committed_then_folded_count": committed_then_folded,
        "large_pot_loss_by_street_of_final_investment": dict(sorted(large_loss_final_street.items())),
    }


def _pot_odds_metrics(hands: list[dict[str, Any]]) -> dict[str, Any]:
    calls = []
    folds = []
    required_on_calls = []
    river_calls = []
    for h in hands:
        for d in h.get("hero_decisions", []):
            owed = _parse_float(d.get("amount_owed"), 0.0) or 0.0
            if owed <= 0:
                continue
            pot_odds = _required_equity(owed, _parse_float(d.get("pot"), 0.0) or 0.0)
            if d.get("action") == "call":
                calls.append(pot_odds)
                required_on_calls.append(pot_odds)
                if d.get("street") == "river" and h.get("delta") is not None:
                    river_calls.append(h["delta"] > 0)
            elif d.get("action") == "fold":
                folds.append(pot_odds)
    return {
        "average_pot_odds_faced_when_calling": _mean(calls),
        "average_pot_odds_faced_when_folding": _mean(folds),
        "average_required_equity_on_call": _mean(required_on_calls),
        "river_call_win_rate": _pct(sum(1 for won in river_calls if won), len(river_calls)),
    }


def _bet_sizing_metrics(hands: list[dict[str, Any]]) -> dict[str, Any]:
    open_by_pos = defaultdict(list)
    threebet_ip = []
    threebet_oop = []
    fourbet_sizes = []
    street_pct = defaultdict(list)
    overbets = min_clicks = raise_fold_large = 0
    aggressive_count = 0
    for h in hands:
        preflop_actions = _actions(h, "preflop")
        hero_seat = h.get("hero_seat")
        pos = h.get("hero_position") or "UNK"
        prior_raises = 0
        hero_large_raise = False
        for action in preflop_actions:
            if action.get("action") in {"raise", "bet", "all_in"}:
                if action.get("seat") == hero_seat:
                    if prior_raises == 0:
                        open_by_pos[pos].append(action.get("amount", 0))
                    elif prior_raises == 1:
                        (threebet_ip if pos in {"BTN", "CO"} else threebet_oop).append(action.get("amount", 0))
                    elif prior_raises >= 2:
                        fourbet_sizes.append(action.get("amount", 0))
                    hero_large_raise = action.get("contribution", 0) >= 0.25 * max(1.0, _first_decision_stack(h) or 0.0)
                prior_raises += 1
            elif action.get("seat") == hero_seat and action.get("action") == "fold" and hero_large_raise:
                raise_fold_large += 1

        for action in _hero_actions(h):
            if action.get("street") in POSTFLOP_STREETS and action.get("action") in {"bet", "raise", "all_in"}:
                aggressive_count += 1
                pot = _parse_float(action.get("pot"), 0.0) or 0.0
                amount = _parse_float(action.get("amount"), 0.0) or 0.0
                if pot > 0:
                    pct = amount / pot
                    street_pct[action["street"]].append(pct)
                    if pct > 1.0:
                        overbets += 1
                if amount <= max(1.0, (pot * 0.1 if pot else 1.0)):
                    min_clicks += 1
    return {
        "average_open_size_by_position": {pos: _mean(values) for pos, values in sorted(open_by_pos.items())},
        "three_bet_size_ip": _mean(threebet_ip),
        "three_bet_size_oop": _mean(threebet_oop),
        "four_bet_size_distribution": _distribution(fourbet_sizes),
        "flop_bet_size_pct_pot": _mean(street_pct["flop"]),
        "turn_bet_size_pct_pot": _mean(street_pct["turn"]),
        "river_bet_size_pct_pot": _mean(street_pct["river"]),
        "overbet_frequency": _pct(overbets, aggressive_count),
        "min_click_raise_frequency": _pct(min_clicks, aggressive_count),
        "raise_fold_after_large_raise_count": raise_fold_large,
    }


def _street_ev_metrics(hands: list[dict[str, Any]]) -> dict[str, Any]:
    known = [h for h in hands if h.get("delta") is not None]
    by_end = {street: [h["delta"] for h in known if h.get("ending_street") == street] for street in STREETS}
    pfa = [h["delta"] for h in known if _preflop_aggressor(h) == h.get("hero_seat")]
    caller = [h["delta"] for h in known if _hero_was_preflop_caller(h)]
    three_bet = [h["delta"] for h in known if _preflop_raise_count(h) == 2]
    four_bet = [h["delta"] for h in known if _preflop_raise_count(h) >= 3]
    single = [h["delta"] for h in known if _preflop_raise_count(h) == 1]
    return {
        "preflop_ev": _mean(by_end["preflop"]),
        "flop_ev": _mean(by_end["flop"]),
        "turn_ev": _mean(by_end["turn"]),
        "river_ev": _mean(by_end["river"]),
        "ev_after_seeing_flop": _mean([h["delta"] for h in known if _saw_flop(h)]),
        "ev_as_preflop_aggressor": _mean(pfa),
        "ev_as_preflop_caller": _mean(caller),
        "ev_in_single_raised_pots": _mean(single),
        "ev_in_three_bet_pots": _mean(three_bet),
        "ev_in_four_bet_pots": _mean(four_bet),
    }


def _actions(hand: dict[str, Any], street: str | None = None) -> list[dict[str, Any]]:
    actions = list(hand.get("actions") or [])
    if street is not None:
        actions = [a for a in actions if a.get("street") == street]
    return [a for a in actions if a.get("action") not in {"small_blind", "big_blind"}]


def _hero_actions(hand: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for a in _actions(hand) if a.get("seat") == hand.get("hero_seat")]


def _hero_limped(actions: list[dict[str, Any]], hand: dict[str, Any]) -> bool:
    seen_raise = False
    for action in actions:
        if action.get("seat") == hand.get("hero_seat"):
            return action.get("action") == "call" and not seen_raise
        if action.get("action") in {"raise", "bet", "all_in"}:
            seen_raise = True
    return False


def _hero_three_bet(actions: list[dict[str, Any]], hand: dict[str, Any]) -> tuple[int, int]:
    seen_raise = False
    for action in actions:
        if action.get("seat") == hand.get("hero_seat"):
            if seen_raise:
                return 1, int(action.get("action") in {"raise", "bet", "all_in"})
            return 0, 0
        if action.get("action") in {"raise", "bet", "all_in"}:
            seen_raise = True
    return 0, 0


def _hero_fold_to_three_bet(actions: list[dict[str, Any]], hand: dict[str, Any]) -> tuple[int, int]:
    hero = hand.get("hero_seat")
    hero_opened = False
    faced_reraise = False
    for action in actions:
        if action.get("seat") == hero:
            if faced_reraise:
                return 1, int(action.get("action") == "fold")
            if action.get("action") in {"raise", "bet", "all_in"} and not hero_opened:
                hero_opened = True
            continue
        if hero_opened and action.get("action") in {"raise", "bet", "all_in"}:
            faced_reraise = True
    return 0, 0


def _hero_steal(actions: list[dict[str, Any]], hand: dict[str, Any]) -> tuple[int, int]:
    if hand.get("hero_position") not in LATE_POSITIONS:
        return 0, 0
    for action in actions:
        if action.get("seat") == hand.get("hero_seat"):
            return 1, int(action.get("action") in {"raise", "bet", "all_in"})
        if action.get("action") not in {"fold"}:
            return 0, 0
    return 0, 0


def _hero_fold_bb_to_steal(actions: list[dict[str, Any]], hand: dict[str, Any]) -> tuple[int, int]:
    if hand.get("hero_position") != "BB":
        return 0, 0
    saw_late_raise = False
    for action in actions:
        if action.get("seat") == hand.get("hero_seat"):
            if saw_late_raise:
                return 1, int(action.get("action") == "fold")
            return 0, 0
        if action.get("position") in LATE_POSITIONS and action.get("action") in {"raise", "bet", "all_in"}:
            saw_late_raise = True
    return 0, 0


def _preflop_aggressor(hand: dict[str, Any]) -> int | None:
    aggressor = None
    for action in _actions(hand, "preflop"):
        if action.get("action") in {"raise", "bet", "all_in"}:
            aggressor = action.get("seat")
    return aggressor


def _preflop_raise_count(hand: dict[str, Any]) -> int:
    return sum(1 for action in _actions(hand, "preflop") if action.get("action") in {"raise", "bet", "all_in"})


def _hero_was_preflop_caller(hand: dict[str, Any]) -> bool:
    return any(a.get("seat") == hand.get("hero_seat") and a.get("action") == "call" for a in _actions(hand, "preflop"))


def _saw_flop(hand: dict[str, Any]) -> bool:
    return bool(hand.get("community_cards")) or hand.get("ending_street") in POSTFLOP_STREETS


def _hero_won(hand: dict[str, Any]) -> bool:
    hero = hand.get("hero_name")
    hero_seat = hand.get("hero_seat")
    return any(w.get("seat") == hero_seat or w.get("bot_id") == hero or w.get("name") == hero for w in hand.get("winners") or [])


def _award_total_for_hero(hand: HandRecord) -> float | None:
    total = 0.0
    found = False
    for winner in hand.winners:
        if winner.get("seat") == hand.hero_seat or winner.get("bot_id") == hand.hero_name or winner.get("name") == hand.hero_name:
            amount = _parse_float(winner.get("amount"), 0.0) or 0.0
            total += amount
            found = True
    if not found and hand.winners:
        return 0.0
    return total if found or hand.winners else None


def _first_decision_stack(hand: dict[str, Any]) -> float | None:
    for decision in hand.get("hero_decisions") or []:
        stack = _parse_float(decision.get("stack"), None)
        if stack is not None:
            return stack
    return None


def _decision_branches(fullhouse_state: dict[str, Any], valid_actions: dict[str, dict[str, Any]]) -> list[str]:
    branches = [str(fullhouse_state.get("street") or "unknown")]
    if fullhouse_state.get("amount_owed", 0):
        branches.append("facing_call")
    if fullhouse_state.get("can_check"):
        branches.append("can_check")
    if "raise" in valid_actions:
        branches.append("raise_available")
    if "all_in" in valid_actions:
        branches.append("all_in_available")
    return branches


def _positions_by_seat(seats: list[int], dealer: int | None) -> dict[int, str]:
    if not seats or dealer is None:
        return {seat: "UNK" for seat in seats}
    labels = POSITION_LABELS_6MAX if len(seats) == 6 else _generic_position_labels(len(seats))
    ordered = []
    current = dealer
    for _ in seats:
        if current not in seats:
            current = seats[0]
        ordered.append(current)
        current = _next_seat(seats, current)
        if current is None:
            break
    return {seat: labels[index] if index < len(labels) else "UNK" for index, seat in enumerate(ordered)}


def _generic_position_labels(n: int) -> list[str]:
    if n == 2:
        return ["BTN", "BB"]
    labels = ["BTN", "SB", "BB"]
    middle = ["UTG", "UTG+1", "MP", "LJ", "HJ", "CO"]
    return labels + middle[: max(0, n - 3)]


def _next_seat(seats: list[int], seat: int | None) -> int | None:
    if seat is None or not seats:
        return None
    if seat not in seats:
        return seats[0]
    return seats[(seats.index(seat) + 1) % len(seats)]


def _bot_id_for_seat(hand: HandRecord, seat: int) -> str:
    if seat == hand.hero_seat:
        return hand.hero_name
    player = hand.players.get(seat, {})
    return str(player.get("bot_id") or player.get("name") or f"seat_{seat}")


def _street_from_msg(msg: dict[str, Any], hand: HandRecord) -> str:
    street = msg.get("street")
    if street in STREETS:
        return str(street)
    return hand.ending_street


def _street_from_board(cards: list[str]) -> str:
    if len(cards) >= 5:
        return "river"
    if len(cards) == 4:
        return "turn"
    if len(cards) >= 3:
        return "flop"
    return "preflop"


def _showdown_from_result(msg: dict[str, Any]) -> bool | None:
    if "showdown" in msg:
        return bool(msg.get("showdown"))
    if msg.get("revealed_cards") or msg.get("hand_strengths"):
        return True
    return None


def _valid_action_map(msg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["action"]: item for item in msg.get("valid_actions") or [] if "action" in item}


def _normalize_action(value: Any) -> str:
    action = str(value or "fold").lower()
    if action == "bet":
        return "raise"
    return action


def _required_equity(amount_to_call: float, pot: float) -> float | None:
    denom = pot + amount_to_call
    if denom <= 0:
        return None
    return amount_to_call / denom


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "average": _mean(values),
        "min": _round(min(values)) if values else None,
        "max": _round(max(values)) if values else None,
    }


def _latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "average_ms": _mean(values),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "max_ms": _round(max(values)) if values else None,
    }


def _bb100_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    n = len(values)
    if n == 0:
        return None, None, None
    mean = sum(values) / n
    bb100 = mean * 100
    if n == 1:
        rounded = _round(bb100)
        return rounded, rounded, rounded
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    stderr = math.sqrt(variance) / math.sqrt(n)
    margin = 1.96 * stderr * 100
    return _round(bb100), _round(bb100 - margin), _round(bb100 + margin)


def _pct(num: float, den: float) -> float | None:
    return _round((num / den) * 100) if den else None


def _ratio(num: float, den: float) -> float | None:
    return _round(num / den) if den else None


def _rate(num: float, den: float) -> float | None:
    return _round(num / den) if den else None


def _mean(values: list[float]) -> float | None:
    return _round(sum(values) / len(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
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


def _round(value: Any) -> float:
    return round(float(value), 3)


def _parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)
