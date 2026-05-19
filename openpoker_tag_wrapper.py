"""Run the Fullhouse TAG bot on Open Poker without changing bots/.

Usage:
    OPENPOKER_API_KEY=... python openpoker_tag_wrapper.py

This adapter connects to Open Poker, keeps enough table state to build a
Fullhouse-style game_state dict, calls bots/rule_based_TAG/bot.py, then clamps
the returned action to Open Poker's valid_actions for the current turn.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import uuid
import websockets
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FULLHOUSE_BOT_PATH = ROOT / "bots" / "rule_based_TAG" / "bot.py"
DEFAULT_WS_URL = "wss://openpoker.ai/ws"
DEFAULT_BUY_IN = 2000.0
load_dotenv()

def load_fullhouse_bot(path: Path):
    spec = importlib.util.spec_from_file_location("fullhouse_rule_based_tag", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load bot from {path}")
    spec.loader.exec_module(module)
    if not hasattr(module, "decide"):
        raise RuntimeError(f"{path} does not define decide(state)")
    return module


@dataclass
class OpenPokerState:
    agent_name: str | None = None
    table_id: str | None = None
    hero_seat: int | None = None
    dealer_seat: int | None = None
    hand_id: str | None = None
    street: str = "preflop"
    hole_cards: list[str] = field(default_factory=list)
    community_cards: list[str] = field(default_factory=list)
    players: dict[int, dict[str, Any]] = field(default_factory=dict)
    blinds: dict[str, float] = field(default_factory=lambda: {"small_blind": 10.0, "big_blind": 20.0})
    action_log: list[dict[str, Any]] = field(default_factory=list)
    match_action_log: list[dict[str, Any]] = field(default_factory=list)
    street_bets: dict[int, float] = field(default_factory=dict)
    current_bet: float = 0.0
    last_table_seq: int | None = None

    def update_seq(self, msg: dict[str, Any]) -> None:
        seq = msg.get("table_seq")
        if isinstance(seq, int):
            self.last_table_seq = seq

    def set_players(self, players: list[dict[str, Any]]) -> None:
        for player in players:
            seat = int(player["seat"])
            previous = self.players.get(seat, {})
            self.players[seat] = {
                "seat": seat,
                "bot_id": player.get("name") or previous.get("bot_id") or f"seat_{seat}",
                "name": player.get("name") or previous.get("name") or f"seat_{seat}",
                "stack": float(player.get("stack", previous.get("stack", 0.0))),
                "is_active": True,
                "is_folded": bool(previous.get("is_folded", False)),
                "is_all_in": bool(previous.get("is_all_in", False)),
                "bet_this_street": float(self.street_bets.get(seat, previous.get("bet_this_street", 0.0))),
                "hole_cards": None,
            }

    def start_hand(self, msg: dict[str, Any]) -> None:
        self.hand_id = msg.get("hand_id")
        self.hero_seat = int(msg.get("seat", self.hero_seat if self.hero_seat is not None else 0))
        self.dealer_seat = int(msg.get("dealer_seat", self.dealer_seat if self.dealer_seat is not None else 0))
        self.street = "preflop"
        self.hole_cards = []
        self.community_cards = []
        self.action_log = []
        self.street_bets = {}
        self.current_bet = 0.0
        self.blinds = msg.get("blinds") or self.blinds
        for player in self.players.values():
            player["is_folded"] = False
            player["is_all_in"] = False
            player["bet_this_street"] = 0.0

        small_blind = float(self.blinds.get("small_blind", 10.0))
        big_blind = float(self.blinds.get("big_blind", 20.0))
        sb_seat = self._small_blind_seat()
        bb_seat = self._big_blind_seat()
        if sb_seat is not None:
            self._record_blind(sb_seat, "small_blind", small_blind)
        if bb_seat is not None:
            self._record_blind(bb_seat, "big_blind", big_blind)

    def _small_blind_seat(self) -> int | None:
        if self.dealer_seat is None or not self.players:
            return None
        seats = self._ordered_seats()
        if len(seats) == 2:
            return self.dealer_seat
        return self._next_seat(self.dealer_seat)

    def _big_blind_seat(self) -> int | None:
        if self.dealer_seat is None or not self.players:
            return None
        seats = self._ordered_seats()
        if len(seats) == 2:
            return self._next_seat(self.dealer_seat)
        small_blind = self._small_blind_seat()
        return self._next_seat(small_blind) if small_blind is not None else None

    def _ordered_seats(self) -> list[int]:
        return sorted(self.players)

    def _next_seat(self, seat: int) -> int | None:
        seats = self._ordered_seats()
        if not seats:
            return None
        if seat not in seats:
            return seats[0]
        return seats[(seats.index(seat) + 1) % len(seats)]

    def _record_blind(self, seat: int, action: str, amount: float) -> None:
        self.street_bets[seat] = self.street_bets.get(seat, 0.0) + amount
        self.current_bet = max(self.current_bet, self.street_bets[seat])
        self.action_log.append({"seat": seat, "action": action, "amount": int(amount)})
        if seat in self.players:
            self.players[seat]["bet_this_street"] = self.street_bets[seat]

    def set_hole_cards(self, cards: list[str]) -> None:
        self.hole_cards = list(cards)

    def set_community_cards(self, msg: dict[str, Any]) -> None:
        street = msg.get("street")
        if street in {"flop", "turn", "river"}:
            self.street = street
        cards = list(msg.get("cards") or [])
        if len(cards) == 1 and self.community_cards:
            self.community_cards.extend(cards)
        elif cards:
            self.community_cards = cards
        self.street_bets = {}
        self.current_bet = 0.0
        for player in self.players.values():
            player["bet_this_street"] = 0.0

    def record_player_action(self, msg: dict[str, Any]) -> None:
        seat = int(msg.get("seat", -1))
        action = str(msg.get("action", "check"))
        amount = float(msg.get("amount") or 0.0)
        street = msg.get("street")
        if street in {"preflop", "flop", "turn", "river"} and street != self.street:
            self.street = street
            self.street_bets = {}
            self.current_bet = 0.0

        contribution = float(msg.get("contribution_delta") or 0.0)
        if contribution <= 0 and action in {"call", "raise", "all_in"}:
            before = float(msg.get("stack_before") or 0.0)
            after = float(msg.get("stack_after") or msg.get("stack") or before)
            contribution = max(0.0, before - after)
        if contribution <= 0 and action == "raise" and amount > 0:
            contribution = max(0.0, amount - self.street_bets.get(seat, 0.0))
        if contribution <= 0 and action == "call":
            contribution = max(0.0, self.current_bet - self.street_bets.get(seat, 0.0))

        if action in {"call", "raise", "all_in"}:
            self.street_bets[seat] = self.street_bets.get(seat, 0.0) + contribution
            if action == "raise" and amount > 0:
                self.street_bets[seat] = max(self.street_bets[seat], amount)
            self.current_bet = max(self.current_bet, self.street_bets[seat])

        if seat in self.players:
            self.players[seat]["stack"] = float(msg.get("stack", self.players[seat].get("stack", 0.0)))
            self.players[seat]["bet_this_street"] = self.street_bets.get(seat, 0.0)
            if action == "fold":
                self.players[seat]["is_folded"] = True
            if action == "all_in":
                self.players[seat]["is_all_in"] = True

        logged = {"seat": seat, "action": action, "amount": int(amount or self.street_bets.get(seat, 0.0))}
        self.action_log.append(logged)
        self.match_action_log.append({
            "hand_num": 0,
            "seat": seat,
            "bot_id": msg.get("name") or self.players.get(seat, {}).get("bot_id", f"seat_{seat}"),
            "action": action,
            "amount": logged["amount"],
        })
        if len(self.match_action_log) > 200:
            self.match_action_log = self.match_action_log[-200:]

    def apply_turn_snapshot(self, msg: dict[str, Any]) -> None:
        self.update_seq(msg)
        if msg.get("hand_id"):
            self.hand_id = msg["hand_id"]
        if msg.get("community_cards") is not None:
            self.community_cards = list(msg.get("community_cards") or [])
            self.street = street_from_board(self.community_cards)
        self.set_players(msg.get("players") or [])

        hero_stack = self.players.get(self.hero_seat or -1, {}).get("stack", 0.0)
        for action in msg.get("valid_actions") or []:
            if action.get("action") == "call":
                owed = float(action.get("amount") or 0.0)
                hero_bet = self.street_bets.get(self.hero_seat or -1, 0.0)
                self.current_bet = max(self.current_bet, hero_bet + owed)
                if self.hero_seat in self.players:
                    self.players[self.hero_seat]["stack"] = hero_stack

    def fullhouse_state(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.apply_turn_snapshot(msg)
        valid_actions = valid_action_map(msg)
        can_check = "check" in valid_actions
        amount_owed = 0.0
        if "call" in valid_actions:
            amount_owed = float(valid_actions["call"].get("amount") or 0.0)

        hero_seat = self.hero_seat if self.hero_seat is not None else int(msg.get("seat", 0))
        hero = self.players.get(hero_seat, {})
        min_raise = float(valid_actions.get("raise", {}).get("min") or msg.get("min_raise") or self.current_bet)
        big_blind = float(self.blinds.get("big_blind", 20.0))

        return {
            "type": "action_request",
            "hand_id": self.hand_id or msg.get("hand_id") or "",
            "street": self.street,
            "seat_to_act": hero_seat,
            "pot": int(float(msg.get("pot", 0.0))),
            "community_cards": list(self.community_cards),
            "current_bet": int(self.current_bet),
            "min_raise_to": int(max(min_raise, self.current_bet + big_blind)),
            "amount_owed": int(amount_owed),
            "can_check": can_check,
            "your_cards": list(self.hole_cards),
            "your_stack": int(float(hero.get("stack", 0.0))),
            "your_bet_this_street": int(self.street_bets.get(hero_seat, 0.0)),
            "players": self.fullhouse_players(),
            "action_log": list(self.action_log),
            "match_action_log": list(self.match_action_log),
        }

    def fullhouse_players(self) -> list[dict[str, Any]]:
        if not self.players and self.hero_seat is not None:
            self.players[self.hero_seat] = {
                "seat": self.hero_seat,
                "bot_id": self.agent_name or "hero",
                "name": self.agent_name or "hero",
                "stack": 0.0,
                "is_folded": False,
                "is_all_in": False,
                "bet_this_street": 0.0,
                "hole_cards": None,
            }
        players = []
        for seat in sorted(self.players):
            player = dict(self.players[seat])
            player["stack"] = int(float(player.get("stack", 0.0)))
            player["bet_this_street"] = int(self.street_bets.get(seat, player.get("bet_this_street", 0.0)))
            player["is_active"] = not player.get("is_folded", False) and not player.get("is_all_in", False)
            player["bot_id"] = player.get("bot_id") or player.get("name") or f"seat_{seat}"
            player["hole_cards"] = None
            players.append(player)
        return players


def street_from_board(cards: list[str]) -> str:
    if len(cards) >= 5:
        return "river"
    if len(cards) == 4:
        return "turn"
    if len(cards) >= 3:
        return "flop"
    return "preflop"


def valid_action_map(msg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["action"]: item for item in msg.get("valid_actions") or [] if "action" in item}


def clamp_openpoker_action(raw_action: dict[str, Any], turn_msg: dict[str, Any]) -> dict[str, Any]:
    actions = valid_action_map(turn_msg)
    requested = str(raw_action.get("action", "fold")).lower()

    if requested == "raise" and "raise" in actions:
        raise_info = actions["raise"]
        amount = float(raw_action.get("amount") or raise_info.get("min") or 0.0)
        amount = max(float(raise_info.get("min", amount)), amount)
        amount = min(float(raise_info.get("max", amount)), amount)
        return {"action": "raise", "amount": amount}

    if requested == "all_in":
        if "all_in" in actions:
            return {"action": "all_in"}
        if "raise" in actions:
            return {"action": "raise", "amount": float(actions["raise"].get("max") or actions["raise"].get("min") or 0.0)}

    if requested in actions:
        return {"action": requested}

    for fallback in ("check", "call", "fold"):
        if fallback in actions:
            return {"action": fallback}
    return {"action": "fold"}


async def send_action(ws, turn_msg: dict[str, Any], action: dict[str, Any]) -> None:
    payload = {
        "type": "action",
        "hand_id": turn_msg.get("hand_id"),
        "action": action["action"],
        "client_action_id": str(uuid.uuid4()),
        "turn_token": turn_msg.get("turn_token"),
    }
    if action["action"] == "raise":
        payload["amount"] = action["amount"]
    await ws.send(json.dumps(payload))


async def connect(api_key: str, ws_url: str, buy_in: float, once: bool = False) -> None:
    bot = load_fullhouse_bot(FULLHOUSE_BOT_PATH)
    state = OpenPokerState()
    headers = {"Authorization": f"Bearer {api_key.strip()}"}

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))
        await ws.send(json.dumps({"type": "set_auto_rebuy", "enabled": True}))
        print(f"Connected to {ws_url}; joined lobby with buy-in {buy_in:g}.")

        async for raw in ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")
            state.update_seq(msg)

            if msg_type == "connected":
                state.agent_name = msg.get("name")
                print(f"Connected as {state.agent_name}")
            elif msg_type == "lobby_joined":
                print(f"Joined lobby; position {msg.get('position')}")
            elif msg_type == "table_joined":
                state.table_id = msg.get("table_id")
                state.hero_seat = int(msg.get("seat", 0))
                state.set_players(msg.get("players") or [])
                print(f"Seated at table {state.table_id}, seat {state.hero_seat}")
            elif msg_type == "hand_start":
                state.start_hand(msg)
                print(f"Hand {state.hand_id} started")
            elif msg_type == "hole_cards":
                state.set_hole_cards(msg.get("cards") or [])
            elif msg_type == "community_cards":
                state.set_community_cards(msg)
            elif msg_type == "player_action":
                state.record_player_action(msg)
            elif msg_type == "your_turn":
                fullhouse_state = state.fullhouse_state(msg)
                raw_decision = bot.decide(fullhouse_state)
                decision = clamp_openpoker_action(raw_decision, msg)
                await send_action(ws, msg, decision)
                print(f"{fullhouse_state['street']}: {raw_decision} -> {decision}")
            elif msg_type == "action_ack":
                print(f"Action accepted: {msg.get('client_action_id')}")
            elif msg_type == "action_rejected":
                print(f"Action rejected: {msg.get('reason')} {msg.get('details') or ''}")
            elif msg_type == "hand_result":
                print(f"Hand result: pot {msg.get('pot')}, winners {msg.get('winners')}")
                if once:
                    await ws.send(json.dumps({"type": "leave_table"}))
                    return
            elif msg_type == "busted":
                await ws.send(json.dumps({"type": "rebuy", "amount": 1500}))
            elif msg_type in {"table_closed", "season_ended"}:
                await ws.send(json.dumps({"type": "join_lobby", "buy_in": buy_in}))
            elif msg_type == "error":
                print(f"Open Poker error: {msg}")
            else:
                print(f"Received: {msg_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Poker wrapper for the Fullhouse TAG bot")
    parser.add_argument("--api-key", default=os.environ.get("OPENPOKER_API_KEY"), help="Open Poker API key")
    parser.add_argument("--ws-url", default=os.environ.get("OPENPOKER_WS_URL", DEFAULT_WS_URL))
    parser.add_argument("--buy-in", type=float, default=float(os.environ.get("OPENPOKER_BUY_IN", DEFAULT_BUY_IN)))
    parser.add_argument("--once", action="store_true", help="Leave after one completed hand")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set OPENPOKER_API_KEY or pass --api-key.")
    asyncio.run(connect(args.api_key, args.ws_url, args.buy_in, once=args.once))
