import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import demo


def _fixture_result(bot_paths, match_id="fixture"):
    bot_ids = list(bot_paths.keys())
    return {
        "match_id": match_id,
        "bot_ids": bot_ids,
        "n_hands": 0,
        "duration_s": 0,
        "final_stacks": {bid: 10_000 for bid in bot_ids},
        "chip_delta": {bid: 0 for bid in bot_ids},
        "bot_errors": {bid: [] for bid in bot_ids},
        "hands": [],
    }


def test_configured_hands_route_returns_metrics(monkeypatch):
    calls = []

    def fake_run_match(match_id, bot_paths, n_hands=400, verbose=False, seed=None):
        calls.append((match_id, bot_paths, n_hands, verbose, seed))
        return _fixture_result(bot_paths, match_id)

    monkeypatch.setattr(demo, "run_match", fake_run_match)

    client = demo.app.test_client()
    res = client.post("/run/hands")
    data = res.get_json()

    assert res.status_code == 200
    assert calls[0][2] == demo.RUN_HANDS_COUNT
    assert data["mode"] == "hands"
    assert data["hands"] == demo.RUN_HANDS_COUNT
    assert "metrics" in data
    assert data["metrics"]["summary"]["matches"] == 1


def test_round_robin_route_uses_all_my_bot_plus_five_opponent_combos(monkeypatch):
    calls = []

    def fake_run_match(match_id, bot_paths, n_hands=400, verbose=False, seed=None):
        calls.append((match_id, bot_paths, n_hands, verbose, seed))
        return _fixture_result(bot_paths, match_id)

    monkeypatch.setattr(demo, "run_match", fake_run_match)

    client = demo.app.test_client()
    res = client.post("/run/round_robin")
    data = res.get_json()

    assert res.status_code == 200
    assert data["tables"] == 126
    assert len(calls) == 126
    assert all(call[2] == demo.ROUND_ROBIN_HANDS for call in calls)
    assert all(call[1].get("my_bot") == "bots/my_bot/bot.py" for call in calls)
    assert all("template" not in call[1] for call in calls)
