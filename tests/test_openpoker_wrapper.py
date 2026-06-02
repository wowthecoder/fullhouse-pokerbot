import json
import asyncio

import openpoker_wrapper
from openpoker_wrapper import connect, leave_table_and_rejoin_lobby


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


def test_rebuy_error_recovery_leaves_table_then_rejoins_lobby():
    ws = FakeWebSocket()

    asyncio.run(leave_table_and_rejoin_lobby(ws, buy_in=2000))

    assert ws.sent == [
        {"type": "leave_table"},
        {"type": "join_lobby", "buy_in": 2000},
    ]


class FakeStreamingWebSocket(FakeWebSocket):
    def __init__(self, messages):
        super().__init__()
        self.messages = [json.dumps(message) for message in messages]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


class FakeBot:
    @staticmethod
    def decide(_state):
        return {"action": "raise", "amount": 300}


def test_connect_writes_metrics_snapshot_on_hand_result(monkeypatch, tmp_path):
    messages = [
        {"type": "connected", "name": "hero"},
        {
            "type": "table_joined",
            "table_id": "table-1",
            "seat": 5,
            "players": [
                {"seat": 0, "name": "btn", "stack": 10_000},
                {"seat": 1, "name": "sb", "stack": 10_000},
                {"seat": 2, "name": "bb", "stack": 10_000},
                {"seat": 3, "name": "utg", "stack": 10_000},
                {"seat": 4, "name": "hj", "stack": 10_000},
                {"seat": 5, "name": "hero", "stack": 10_000},
            ],
        },
        {
            "type": "hand_start",
            "hand_id": "live-1",
            "seat": 5,
            "dealer_seat": 0,
            "blinds": {"small_blind": 50, "big_blind": 100},
        },
        {"type": "hole_cards", "cards": ["As", "Kh"]},
        {
            "type": "your_turn",
            "hand_id": "live-1",
            "players": [
                {"seat": 0, "name": "btn", "stack": 10_000},
                {"seat": 1, "name": "sb", "stack": 9_950},
                {"seat": 2, "name": "bb", "stack": 9_900},
                {"seat": 3, "name": "utg", "stack": 10_000},
                {"seat": 4, "name": "hj", "stack": 10_000},
                {"seat": 5, "name": "hero", "stack": 10_000},
            ],
            "pot": 150,
            "community_cards": [],
            "valid_actions": [
                {"action": "fold"},
                {"action": "call", "amount": 0},
                {"action": "raise", "min": 300, "max": 10_000},
            ],
            "turn_token": "tok",
        },
        {
            "type": "hand_result",
            "hand_id": "live-1",
            "pot": 450,
            "winners": [{"seat": 5, "amount": 450}],
            "showdown": False,
        },
    ]
    ws = FakeStreamingWebSocket(messages)

    monkeypatch.setattr(openpoker_wrapper, "load_fullhouse_bot", lambda _path: FakeBot())
    monkeypatch.setattr(openpoker_wrapper.websockets, "connect", lambda *_args, **_kwargs: ws)

    metrics_file = tmp_path / "metrics.json"
    asyncio.run(connect("key", "wss://example.test/ws", 2000, once=True, metrics_file=metrics_file))

    snapshot = json.loads(metrics_file.read_text())
    assert snapshot["summary"]["hands"] == 1
    assert snapshot["recent_hands"][0]["hand_id"] == "live-1"
    assert ws.sent[-1] == {"type": "leave_table"}
