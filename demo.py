"""
Fullhouse Hackathon — Local Demo
Run: python demo.py
Open: http://localhost:5000

No Docker, Redis, or Supabase needed.
Runs real matches using the actual game engine and shows results live.
"""

import itertools
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, Response, jsonify, render_template

from engine.metrics import aggregate_match_metrics
from engine.tournament import compute_standings, select_finalists, swiss_pairing
from sandbox.match import run_match

app = Flask(__name__)

# ---------------------------------------------------------------------------
# State (in-memory for demo)
# ---------------------------------------------------------------------------

BOT_PATHS = {
    "The Aggressor": "bots/aggressor.py",
    "The Mathematician": "bots/mathematician.py",
    # "The Shark": "bots/shark.py",
    # "Loose passive": "bots/loose_passive.py",
    "Complete v1": "bots/my_bots/complete_v1.py",
    "Complete no exploit": "bots/my_bots/complete_no_opponent_model.py",
    "BlackRain79": "bots/my_bots/blackrain79.py",
    # "G5 Bot": "bots/my_bots/g5-bot.py",
    "Complete v2": "bots/my_bots/complete_v2.py",
}

MY_BOT_PATH = "bots/my_bots/complete_v2.py"
RUN_HANDS_COUNT = 500
ROUND_ROBIN_HANDS = 500

state = {
    "log": [],        # event log (SSE stream)
    "standings": [],  # current leaderboard
    "hands": [],      # last match hand history
    "metrics": None,
    "running": False,
    "round": 0,
}

log_lock = threading.Lock()
run_lock = threading.Lock()


def emit(msg, kind="info"):
    with log_lock:
        state["log"].append({"t": time.time(), "msg": msg, "kind": kind})


def _result_standings(results):
    all_results = []
    for result in results:
        for bid, delta in result.get("chip_delta", {}).items():
            all_results.append({
                "bot_id": bid,
                "bot_path": "",
                "chip_delta": delta,
            })
    return compute_standings(all_results)


def _run_with_guard(fn):
    if not run_lock.acquire(blocking=False):
        return jsonify({"error": "A run is already in progress"}), 409
    state["running"] = True
    try:
        return fn()
    finally:
        state["running"] = False
        run_lock.release()


def _round_robin_tables():
    bots_dir = Path("bots")
    opponents = {}
    for path in sorted(bots_dir.glob("*.py")):
        if path.name == "template.py":
            continue
        opponents[path.stem] = str(path)

    if len(opponents) < 5:
        raise ValueError("Need at least 5 top-level bots in bots/ for round robin")

    tables = []
    for combo in itertools.combinations(opponents.items(), 5):
        table = {"my_bot": MY_BOT_PATH}
        table.update(dict(combo))
        tables.append(table)
    return tables


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "demo.html",
        run_hands_count=RUN_HANDS_COUNT,
        round_robin_hands=ROUND_ROBIN_HANDS,
    )


@app.route("/stream")
def stream():
    """SSE endpoint — pushes log events to the browser in real time."""
    def generate():
        last = 0
        while True:
            with log_lock:
                new = state["log"][last:]
                last = len(state["log"])
            for entry in new:
                yield f"data: {json.dumps(entry)}\n\n"
            time.sleep(0.2)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/run/match", methods=["POST"])
def run_single_match():
    """Run one 6-player match and return updated standings."""
    def work():
        match_id = f"demo_{uuid.uuid4().hex[:8]}"

        emit(f"Starting match {match_id}...", "dim")
        t0 = time.time()

        result = run_match(match_id, BOT_PATHS, n_hands=150, verbose=False)
        elapsed = time.time() - t0

        emit(f"Match complete in {elapsed:.1f}s", "dim")

        all_results = [
            {"bot_id": bid, "bot_path": BOT_PATHS.get(bid, ""), "chip_delta": delta}
            for bid, delta in result["chip_delta"].items()
        ]
        new_standings = compute_standings(
            [{**s, "chip_delta": s["cumulative_delta"]} for s in state["standings"]]
            + all_results
        )
        state["standings"] = new_standings

        sorted_res = sorted(result["chip_delta"].items(), key=lambda x: -x[1])
        for bid, delta in sorted_res:
            sign = "+" if delta >= 0 else ""
            kind = "win" if delta > 0 else "err" if delta < 0 else "dim"
            emit(f"  {bid:22s} {sign}{delta:,}", kind)

        state["hands"] = result.get("hands", [])

        return jsonify({
            "standings": new_standings,
            "hands": result.get("hands", []),
        })

    return _run_with_guard(work)


@app.route("/run/tournament", methods=["POST"])
def run_tournament():
    """Run a 3-round Swiss tournament across all bots."""
    def work():
        bot_list = [
            {"bot_id": bid, "bot_path": path, "cumulative_delta": 0, "matches_played": 0}
            for bid, path in BOT_PATHS.items()
        ]

        state["standings"] = []
        all_results = []

        for rnd in range(1, 4):
            state["round"] = rnd
            emit(f"=== ROUND {rnd} ===", "bold")

            standings_for_pairing = compute_standings(all_results) if all_results else bot_list
            tables = swiss_pairing(standings_for_pairing, table_size=min(6, len(bot_list)))

            emit(f"  {len(tables)} table(s) this round", "dim")

            for t_idx, table in enumerate(tables):
                bot_paths_for_match = {b["bot_id"]: b["bot_path"] for b in table}
                match_id = f"t_r{rnd}_t{t_idx}"

                emit(f"  Table {t_idx + 1}: {', '.join(bot_paths_for_match.keys())}", "dim")

                result = run_match(match_id, bot_paths_for_match, n_hands=150)

                for bid, delta in result["chip_delta"].items():
                    all_results.append({
                        "bot_id": bid,
                        "bot_path": BOT_PATHS.get(bid, ""),
                        "chip_delta": delta,
                    })
                    sign = "+" if delta >= 0 else ""
                    kind = "win" if delta > 0 else "err" if delta < 0 else "dim"
                    emit(f"    {bid:22s} {sign}{delta:,}", kind)

        final_standings = compute_standings(all_results)
        state["standings"] = final_standings

        finalists = select_finalists(final_standings, n=3)
        emit("=== FINALISTS ===", "bold")
        for i, f in enumerate(finalists):
            emit(f"  #{i + 1} {f['bot_id']}  ({f['cumulative_delta']:+,})", "win")

        return jsonify({
            "standings": final_standings,
            "round": 3,
            "finalists": len(finalists),
            "hands": state.get("hands", []),
        })

    return _run_with_guard(work)


@app.route("/run/hands", methods=["POST"])
def run_hands():
    """Run one quiet configured-length table and return grouped bot metrics."""
    def work():
        match_id = f"h{RUN_HANDS_COUNT}_{uuid.uuid4().hex[:8]}"
        emit(f"Starting {RUN_HANDS_COUNT}-hand metrics run {match_id}...", "bold")
        t0 = time.time()

        result = run_match(match_id, BOT_PATHS, n_hands=RUN_HANDS_COUNT, verbose=False)
        metrics = aggregate_match_metrics(result)
        standings = _result_standings([result])

        elapsed = time.time() - t0
        emit(f"{RUN_HANDS_COUNT}-hand metrics run complete in {elapsed:.1f}s", "win")

        state["standings"] = standings
        state["metrics"] = metrics
        state["hands"] = []

        return jsonify({
            "mode": "hands",
            "hands": RUN_HANDS_COUNT,
            "standings": standings,
            "metrics": metrics,
        })

    return _run_with_guard(work)


@app.route("/run/round_robin", methods=["POST"])
def run_round_robin():
    """Run all my_bot + five-opponent configured-length combinations."""
    def work():
        try:
            tables = _round_robin_tables()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        emit(f"Starting round robin: {len(tables)} tables x {ROUND_ROBIN_HANDS} hands", "bold")
        t0 = time.time()
        results = []

        for idx, table in enumerate(tables, start=1):
            match_id = f"rr_{idx:03d}_{uuid.uuid4().hex[:6]}"
            result = run_match(match_id, table, n_hands=ROUND_ROBIN_HANDS, verbose=False)
            results.append(result)

            if idx == 1 or idx == len(tables) or idx % 10 == 0:
                emit(f"  Round robin progress: {idx}/{len(tables)} tables", "dim")

        metrics = aggregate_match_metrics(results)
        standings = _result_standings(results)
        elapsed = time.time() - t0

        emit(f"Round robin complete in {elapsed:.1f}s", "win")

        state["standings"] = standings
        state["metrics"] = metrics
        state["hands"] = []

        return jsonify({
            "mode": "round_robin",
            "tables": len(tables),
            "hands_per_table": ROUND_ROBIN_HANDS,
            "standings": standings,
            "metrics": metrics,
        })

    return _run_with_guard(work)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  FULLHOUSE HACKATHON — LOCAL DEMO")
    print("=" * 50)
    print("  Open:  http://localhost:5000")
    print("  Bots:  ", ", ".join(BOT_PATHS.keys()))
    print("=" * 50 + "\n")
    app.run(debug=False, threaded=True, port=5000)
