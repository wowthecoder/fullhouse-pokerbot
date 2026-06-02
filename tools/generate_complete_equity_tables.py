import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bots.my_bots import complete_v3 as bot  # noqa: E402


RANK_GRID = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
G5_SUITS = "shdc"
G5_DECK = [rank + suit for rank in bot.RANKS for suit in G5_SUITS]
G5_CARD_INDEX = {card: idx for idx, card in enumerate(G5_DECK)}
HOLE_INDEX_SIZE = 52 * 52
PREFLOP_EQUITY_PATH = os.path.join(ROOT, "bots", "my_bots", "g5_data", "preflop_equity_u8.bin")
TURN_HERO_BOARD_CACHE = {}
MAX_TURN_HERO_BOARD_CACHE = 512

HEADS_UP_RANGES = {
    "random": {"kind": "top_pct", "pct": 1.00},
    "open_20": {"kind": "top_pct", "pct": 0.20},
    "open_30": {"kind": "top_pct", "pct": 0.30},
    "steal_45": {"kind": "top_pct", "pct": 0.45},
    "threebet_8": {"kind": "top_pct", "pct": 0.08},
    "fourbet_5": {"kind": "top_pct", "pct": 0.05},
    "premium_aa_kk_qq_ak": {"kind": "classes", "classes": ["AA", "KK", "QQ", "AKs", "AKo"]},
    "premium_aa_kk": {"kind": "classes", "classes": ["AA", "KK"]},
}

FLOP_ROLES = ["IP_PFA", "OOP_PFA", "IP_CALLER", "OOP_CALLER", "BB_CALLER_OOP"]
FLOP_HERO_CLASSES = [
    "overpair",
    "top_pair_top_kicker",
    "top_pair_weak_kicker",
    "middle_pair",
    "underpair",
    "two_pair_plus",
    "nut_flush_draw",
    "non_nut_flush_draw",
    "oesd",
    "gutshot",
    "combo_draw",
    "air",
]
TURN_HAND_CLASSES = [
    "air",
    "weak_pair",
    "middle_pair",
    "top_pair_weak_kicker",
    "top_pair_good_kicker",
    "overpair",
    "two_pair_plus",
    "set_plus",
    "straight_plus",
    "flush_plus",
]
TURN_DRAW_CLASSES = [
    "none",
    "gutshot",
    "oesd",
    "flush_draw",
    "nut_flush_draw",
    "combo_draw",
]
TURN_BOARD_TEXTURES = [
    "dry",
    "paired",
    "two_tone",
    "monotone",
    "straighty",
    "flush_completed",
    "straight_completed",
    "very_wet",
]
TURN_CARD_BUCKETS = [
    "blank",
    "overcard",
    "pairs_board",
    "completes_flush",
    "completes_straight",
    "adds_flush_draw",
    "adds_straight_draw",
]
TURN_SPR_BUCKETS = ["low", "medium", "high"]
ROLE_REALIZATION = {
    "IP_PFA": 1.02,
    "OOP_PFA": 0.93,
    "IP_CALLER": 0.98,
    "OOP_CALLER": 0.88,
    "BB_CALLER_OOP": 0.86,
}
CLASS_REALIZATION = {
    "two_pair_plus": 1.04,
    "overpair": 0.98,
    "top_pair_top_kicker": 0.96,
    "top_pair_weak_kicker": 0.91,
    "middle_pair": 0.88,
    "underpair": 0.82,
    "combo_draw": 1.03,
    "nut_flush_draw": 1.01,
    "non_nut_flush_draw": 0.96,
    "oesd": 0.95,
    "gutshot": 0.88,
    "air": 0.80,
}
TURN_CLASS_REALIZATION = {
    "air": 0.78,
    "weak_pair": 0.83,
    "middle_pair": 0.87,
    "top_pair_weak_kicker": 0.90,
    "top_pair_good_kicker": 0.94,
    "overpair": 0.97,
    "two_pair_plus": 1.02,
    "set_plus": 1.04,
    "straight_plus": 1.04,
    "flush_plus": 1.04,
}
TURN_DRAW_REALIZATION = {
    "none": 1.0,
    "gutshot": 0.84,
    "oesd": 0.88,
    "flush_draw": 0.89,
    "nut_flush_draw": 0.92,
    "combo_draw": 0.95,
}
RIVER_HAND_CLASSES = [
    "nutted",
    "strong",
    "thin",
    "bluffcatcher",
    "bluff_candidate",
    "air",
]
RIVER_ROLES = ["IP_PFA", "OOP_PFA", "IP_CALLER", "OOP_CALLER", "NEUTRAL"]
RIVER_BOARD_BUCKETS = ["safe", "paired", "flushy", "straighty", "scary"]
RIVER_VILLAIN_LABELS = ["unknown", "nit", "tight_passive", "calling_station", "aggressive", "overfolder"]
RIVER_MISSED_DRAW_BUCKETS = ["low", "med", "high"]
RIVER_SPR_BUCKETS = ["tiny", "low", "mid", "high"]
RIVER_CANDIDATES = [
    ("check", 0.0),
    ("bet", 0.33),
    ("bet", 0.67),
    ("bet", 0.80),
]


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def hand_classes():
    result = []
    for row, hi in enumerate(RANK_GRID):
        for col, lo in enumerate(RANK_GRID):
            if row == col:
                result.append(hi + lo)
            elif row < col:
                result.append(hi + lo + "s")
            else:
                result.append(lo + hi + "o")
    return sorted(set(result), key=lambda hand: (-bot._combo_score(hand), hand))


def combos_by_class():
    grouped = {hand: [] for hand in hand_classes()}
    for c1, c2, hand, score in bot.ALL_HOLE_COMBOS:
        grouped.setdefault(hand, []).append((c1, c2, hand, score))
    return grouped


def range_combos(range_name):
    spec = HEADS_UP_RANGES[range_name]
    if spec["kind"] == "classes":
        classes = set(spec["classes"])
        return [(c1, c2, hand, score, 1.0) for c1, c2, hand, score in bot.ALL_HOLE_COMBOS if hand in classes]
    ordered = sorted(bot.ALL_HOLE_COMBOS, key=lambda item: item[3], reverse=True)
    keep = max(1, int(round(len(ordered) * spec["pct"])))
    return [(c1, c2, hand, score, 1.0) for c1, c2, hand, score in ordered[:keep]]


def canonical_hole_index(cards):
    c1 = G5_CARD_INDEX[cards[0]]
    c2 = G5_CARD_INDEX[cards[1]]
    if c1 > c2:
        c1, c2 = c2, c1
    return c1 * 52 + c2


def load_pairwise_preflop_equity():
    try:
        with open(PREFLOP_EQUITY_PATH, "rb") as handle:
            data = handle.read()
    except Exception:
        return b""
    expected = HOLE_INDEX_SIZE * HOLE_INDEX_SIZE
    return data if len(data) == expected else b""


def pairwise_preflop_equity(matrix, hero, villain):
    if not matrix:
        return None
    hero_idx = canonical_hole_index(hero)
    villain_idx = canonical_hole_index(villain)
    value = matrix[hero_idx * HOLE_INDEX_SIZE + villain_idx]
    return value / 255.0


def estimate_preflop_vs_range(hand, hero_combos, villain_combos, matrix):
    total = 0.0
    count = 0
    for c1, c2, _, _ in hero_combos:
        dead = {c1, c2}
        for v1, v2, _, _, weight in villain_combos:
            if v1 in dead or v2 in dead:
                continue
            equity = pairwise_preflop_equity(matrix, (c1, c2), (v1, v2))
            if equity is None:
                continue
            total += equity * weight
            count += weight
    return round(total / count, 4) if count else None


def estimate_multiway_random(hand, hero_combos, opponent_count, samples, rng):
    wins = 0.0
    trials = 0
    for _ in range(samples):
        c1, c2, _, _ = rng.choice(hero_combos)
        deck = [card for card in bot.FULL_DECK if card not in {c1, c2}]
        draw = rng.sample(deck, opponent_count * 2 + 5)
        opponents = [tuple(draw[idx * 2 : idx * 2 + 2]) for idx in range(opponent_count)]
        board = draw[opponent_count * 2 :]
        hero_rank = bot._eval7_rank([c1, c2] + board)
        ranks = [bot._eval7_rank(list(villain) + board) for villain in opponents]
        best = max([hero_rank] + ranks)
        if hero_rank == best:
            wins += 1.0 / (1 + sum(1 for rank in ranks if rank == best))
        trials += 1
    return round(wins / trials, 4) if trials else None


def build_preflop_tables(samples, rng):
    grouped = combos_by_class()
    matrix = load_pairwise_preflop_equity()
    heads_up = {}
    multiway = {}
    ranges = {name: range_combos(name) for name in HEADS_UP_RANGES}
    for hand, hero_combos in grouped.items():
        heads_up[hand] = {
            range_name: estimate_preflop_vs_range(hand, hero_combos, villain_combos, matrix)
            for range_name, villain_combos in ranges.items()
        }
        multiway[hand] = {
            str(opponents): estimate_multiway_random(hand, hero_combos, opponents, samples, rng)
            for opponents in (2, 3, 4)
        }
    return {"heads_up": heads_up, "multiway_random": multiway}


def group_flops_by_bucket():
    grouped = {}
    for flop in itertools.combinations(bot.FULL_DECK, 3):
        bucket = bot._flop_bucket(bot._board_class(flop))
        grouped.setdefault(bucket, []).append(flop)
    return grouped


def weighted_choice(items, rng):
    total = sum(item[-1] for item in items)
    if total <= 0:
        return None
    roll = rng.random() * total
    seen = 0.0
    for item in items:
        seen += item[-1]
        if seen >= roll:
            return item
    return items[-1]


def sample_hero_combo(hero_class, flop, rng):
    for _ in range(35):
        c1, c2, _, _ = rng.choice(bot.ALL_HOLE_COMBOS)
        if c1 in flop or c2 in flop:
            continue
        info = bot._strategic_hand_info((c1, c2), flop)
        if bot._equity_lookup_hero_class(info, (c1, c2), flop) == hero_class:
            return c1, c2
    return None


def estimate_flop_cell(hero_class, board_bucket, range_items, flops, samples, rng):
    wins = 0.0
    trials = 0
    attempts = 0
    max_attempts = max(80, samples * 8)
    while trials < samples and attempts < max_attempts:
        attempts += 1
        flop = tuple(rng.choice(flops))
        hero = sample_hero_combo(hero_class, flop, rng)
        if hero is None:
            continue
        dead = set(flop) | set(hero)
        legal_villains = [item for item in range_items if item[0] not in dead and item[1] not in dead]
        villain = weighted_choice(legal_villains, rng)
        if villain is None:
            continue
        v1, v2 = villain[0], villain[1]
        remaining = [card for card in bot.FULL_DECK if card not in dead and card not in {v1, v2}]
        turn_river = rng.sample(remaining, 2)
        board = list(flop) + turn_river
        hero_rank = bot._eval7_rank(list(hero) + board)
        villain_rank = bot._eval7_rank([v1, v2] + board)
        if hero_rank > villain_rank:
            wins += 1.0
        elif hero_rank == villain_rank:
            wins += 0.5
        trials += 1
    if trials < max(6, samples // 5):
        return None
    return round(wins / trials, 4), trials


def realized_equity(equity, hero_class, role):
    factor = ROLE_REALIZATION[role] * CLASS_REALIZATION[hero_class]
    return round(clamp(equity * factor, 0.0, 1.0), 4)


def build_flop_tables(samples, rng, quick=False):
    flops_by_bucket = group_flops_by_bucket()
    ranges = {name: range_combos(name) for name in HEADS_UP_RANGES}
    if quick:
        flops_by_bucket = {
            key: value
            for key, value in flops_by_bucket.items()
            if key in {"A_high_dry", "QJT_connected"}
        }
        ranges = {key: ranges[key] for key in ("random", "open_30")}
        hero_classes = ["top_pair_top_kicker", "oesd", "air"]
    else:
        hero_classes = FLOP_HERO_CLASSES
    cells = {}
    base_cells = {}
    for board_bucket, flops in flops_by_bucket.items():
        for hero_class in hero_classes:
            for range_name, range_items in ranges.items():
                estimate = estimate_flop_cell(hero_class, board_bucket, range_items, flops, samples, rng)
                if estimate is None:
                    continue
                equity, used_samples = estimate
                base_cells[(hero_class, board_bucket, range_name)] = {
                    "equity": equity,
                    "samples": used_samples,
                }
    for (hero_class, board_bucket, range_name), record in base_cells.items():
        for role in FLOP_ROLES:
            key = "|".join((hero_class, board_bucket, range_name, role))
            cells[key] = {
                "equity": record["equity"],
                "realized_equity": realized_equity(record["equity"], hero_class, role),
                "samples": record["samples"],
            }
    return {"cells": cells}


def turn_hand_class(info):
    made = info.get("made", 0)
    hand_type = info.get("type")
    if made >= bot.MADE_VALUE["flush"]:
        return "flush_plus"
    if made >= bot.MADE_VALUE["straight"]:
        return "straight_plus"
    if info.get("set") or info.get("trips") or made >= bot.MADE_VALUE["full house"]:
        return "set_plus"
    if made >= bot.MADE_VALUE["two pair"] or info.get("strong_two_pair"):
        return "two_pair_plus"
    if info.get("overpair"):
        return "overpair"
    if info.get("tptk"):
        return "top_pair_good_kicker"
    if info.get("top_pair"):
        return "top_pair_weak_kicker"
    if info.get("strength") == "medium_pair" or made == bot.MADE_VALUE["pair"]:
        return "middle_pair" if hand_type == "pair" else "weak_pair"
    if info.get("showdown"):
        return "weak_pair"
    return "air"


def turn_draw_class(info):
    if info.get("combo_draw"):
        return "combo_draw"
    if info.get("nut_flush_draw"):
        return "nut_flush_draw"
    if info.get("flush_draw"):
        return "flush_draw"
    if info.get("oesd"):
        return "oesd"
    if info.get("gutshot"):
        return "gutshot"
    return "none"


def turn_card_bucket_from_effect(effect):
    if effect.get("completes_flush"):
        return "completes_flush"
    if effect.get("completes_straight"):
        return "completes_straight"
    if effect.get("pairs_board"):
        return "pairs_board"
    if effect.get("overcard_to_flop"):
        return "overcard"
    if effect.get("adds_flush_draw"):
        return "adds_flush_draw"
    if effect.get("turn_dynamic") and not effect.get("flop_dynamic"):
        return "adds_straight_draw"
    return "blank"


def turn_board_texture(board, effect):
    if effect.get("completes_flush"):
        return "flush_completed"
    if effect.get("completes_straight"):
        return "straight_completed"
    if board.get("monotone"):
        return "monotone"
    if board.get("paired"):
        return "paired"
    if board.get("two_tone"):
        if board.get("straight_potential", 0) >= 4 or board.get("is_dynamic"):
            return "very_wet"
        return "two_tone"
    if board.get("straight_potential", 0) >= 4 or board.get("low_connected"):
        return "straighty"
    if board.get("is_dynamic"):
        return "very_wet"
    return "dry"


def group_turn_boards(rng, max_per_bucket=180):
    grouped = {}
    counts = {}
    for flop in itertools.combinations(bot.FULL_DECK, 3):
        dead = set(flop)
        for turn in bot.FULL_DECK:
            if turn in dead:
                continue
            board_cards = tuple(flop) + (turn,)
            effect = bot.classify_turn_card(flop, turn, "IP_PFA")
            board = bot._board_class(board_cards)
            key = (turn_board_texture(board, effect), turn_card_bucket_from_effect(effect))
            counts[key] = counts.get(key, 0) + 1
            bucket = grouped.setdefault(key, [])
            if len(bucket) < max_per_bucket:
                bucket.append(board_cards)
                continue
            replace_at = rng.randrange(counts[key])
            if replace_at < max_per_bucket:
                bucket[replace_at] = board_cards
    return grouped


def quick_turn_boards_by_bucket():
    grouped = {}
    examples = [
        ("As", "7d", "2c", "3h"),
        ("As", "7d", "2c", "7h"),
        ("As", "7d", "2c", "Kd"),
        ("9s", "8d", "5c", "7h"),
    ]
    for board_cards in examples:
        flop = board_cards[:3]
        turn = board_cards[3]
        effect = bot.classify_turn_card(flop, turn, "IP_PFA")
        board = bot._board_class(board_cards)
        key = (turn_board_texture(board, effect), turn_card_bucket_from_effect(effect))
        grouped.setdefault(key, []).append(board_cards)
    return grouped


def turn_hero_combo_groups(board_cards):
    board_cards = tuple(board_cards)
    cached = TURN_HERO_BOARD_CACHE.get(board_cards)
    if cached is not None:
        return cached
    groups = {}
    dead = set(board_cards)
    for c1, c2, _, _ in bot.ALL_HOLE_COMBOS:
        if c1 in dead or c2 in dead:
            continue
        info = bot._strategic_hand_info((c1, c2), board_cards)
        key = (turn_hand_class(info), turn_draw_class(info))
        groups.setdefault(key, []).append((c1, c2))
    if len(TURN_HERO_BOARD_CACHE) >= MAX_TURN_HERO_BOARD_CACHE:
        TURN_HERO_BOARD_CACHE.pop(next(iter(TURN_HERO_BOARD_CACHE)), None)
    TURN_HERO_BOARD_CACHE[board_cards] = groups
    return groups


def sample_turn_hero_combo(hand_class, draw_class, board_cards, rng):
    options = turn_hero_combo_groups(board_cards).get((hand_class, draw_class), [])
    return rng.choice(options) if options else None


def turn_bucket_has_hero_combo(hand_class, draw_class, boards):
    for board_cards in boards[: min(24, len(boards))]:
        if turn_hero_combo_groups(board_cards).get((hand_class, draw_class)):
            return True
    return False


def estimate_turn_cell(hand_class, draw_class, boards, range_items, samples, rng):
    wins = 0.0
    trials = 0
    attempts = 0
    max_attempts = max(100, samples * 10)
    while trials < samples and attempts < max_attempts:
        attempts += 1
        board_cards = tuple(rng.choice(boards))
        hero = sample_turn_hero_combo(hand_class, draw_class, board_cards, rng)
        if hero is None:
            continue
        dead = set(board_cards) | set(hero)
        legal_villains = [item for item in range_items if item[0] not in dead and item[1] not in dead]
        villain = weighted_choice(legal_villains, rng)
        if villain is None:
            continue
        v1, v2 = villain[0], villain[1]
        remaining = [card for card in bot.FULL_DECK if card not in dead and card not in {v1, v2}]
        if not remaining:
            continue
        river = rng.choice(remaining)
        final_board = list(board_cards) + [river]
        hero_rank = bot._eval7_rank(list(hero) + final_board)
        villain_rank = bot._eval7_rank([v1, v2] + final_board)
        if hero_rank > villain_rank:
            wins += 1.0
        elif hero_rank == villain_rank:
            wins += 0.5
        trials += 1
    if trials < max(6, samples // 5):
        return None
    return round(wins / trials, 4), trials


def turn_realized_equity(equity, hand_class, draw_class, role, spr_bucket):
    factor = ROLE_REALIZATION[role] * TURN_CLASS_REALIZATION[hand_class] * TURN_DRAW_REALIZATION[draw_class]
    if spr_bucket == "low" and hand_class in {
        "top_pair_good_kicker",
        "overpair",
        "two_pair_plus",
        "set_plus",
        "straight_plus",
        "flush_plus",
    }:
        factor += 0.04
    if spr_bucket == "high" and draw_class in {"gutshot", "oesd", "flush_draw", "nut_flush_draw", "combo_draw"}:
        factor += 0.03
    return round(clamp(equity * factor, 0.0, 1.0), 4)


def sample_legal_hole_combo(board_cards, rng):
    dead = set(board_cards)
    for _ in range(80):
        c1, c2, _, _ = rng.choice(bot.ALL_HOLE_COMBOS)
        if c1 not in dead and c2 not in dead:
            return c1, c2
    legal = [(c1, c2) for c1, c2, _, _ in bot.ALL_HOLE_COMBOS if c1 not in dead and c2 not in dead]
    return rng.choice(legal) if legal else None


def build_turn_tables(samples, rng, quick=False):
    TURN_HERO_BOARD_CACHE.clear()
    boards_by_bucket = quick_turn_boards_by_bucket() if quick else group_turn_boards(rng, max_per_bucket=90)
    ranges = {name: range_combos(name) for name in HEADS_UP_RANGES}
    if quick:
        ranges = {key: ranges[key] for key in ("random", "open_30")}

    cells = {}
    aggregate = {}
    for (board_texture, turn_card_bucket), boards in boards_by_bucket.items():
        trials_per_range = max(20, samples * (4 if quick else 8))
        for range_name, range_items in ranges.items():
            for _ in range(trials_per_range):
                board_cards = tuple(rng.choice(boards))
                hero = sample_legal_hole_combo(board_cards, rng)
                if hero is None:
                    continue
                dead = set(board_cards) | set(hero)
                legal_villains = [item for item in range_items if item[0] not in dead and item[1] not in dead]
                villain = weighted_choice(legal_villains, rng)
                if villain is None:
                    continue
                v1, v2 = villain[0], villain[1]
                remaining = [card for card in bot.FULL_DECK if card not in dead and card not in {v1, v2}]
                if not remaining:
                    continue
                river = rng.choice(remaining)
                final_board = list(board_cards) + [river]
                hero_rank = bot._eval7_rank(list(hero) + final_board)
                villain_rank = bot._eval7_rank([v1, v2] + final_board)
                score = 0.0
                if hero_rank > villain_rank:
                    score = 1.0
                elif hero_rank == villain_rank:
                    score = 0.5
                info = bot._strategic_hand_info(hero, board_cards)
                key = (
                    turn_hand_class(info),
                    turn_draw_class(info),
                    board_texture,
                    turn_card_bucket,
                    range_name,
                )
                total, count = aggregate.get(key, (0.0, 0))
                aggregate[key] = (total + score, count + 1)

    for (hand_class, draw_class, board_texture, turn_card_bucket, range_name), (total, count) in aggregate.items():
        if count <= 0:
            continue
        equity = round(total / count, 4)
        for role in FLOP_ROLES:
            for spr_bucket in TURN_SPR_BUCKETS:
                key = "|".join(
                    (
                        hand_class,
                        draw_class,
                        board_texture,
                        turn_card_bucket,
                        range_name,
                        role,
                        spr_bucket,
                    )
                )
                cells[key] = {
                    "equity": equity,
                    "realized_equity": turn_realized_equity(
                        equity,
                        hand_class,
                        draw_class,
                        role,
                        spr_bucket,
                    ),
                    "samples": count,
                }
    return {
        "key_fields": [
            "hand_class",
            "draw_class",
            "board_texture",
            "turn_card_bucket",
            "range_bucket",
            "role",
            "spr_bucket",
        ],
        "cells": cells,
    }


def river_base_showdown_equity(hand_class):
    return {
        "nutted": 0.92,
        "strong": 0.75,
        "thin": 0.57,
        "bluffcatcher": 0.38,
        "bluff_candidate": 0.14,
        "air": 0.08,
    }[hand_class]


def river_equity_when_called(hand_class, board_bucket, fraction):
    equity = {
        "nutted": 0.88,
        "strong": 0.67,
        "thin": 0.49,
        "bluffcatcher": 0.25,
        "bluff_candidate": 0.13,
        "air": 0.07,
    }[hand_class]
    if fraction >= 0.67:
        equity -= 0.04
    if fraction >= 0.80:
        equity -= 0.03
    if board_bucket in {"flushy", "straighty", "scary"} and hand_class in {"strong", "thin", "bluffcatcher"}:
        equity -= 0.06
    if board_bucket == "paired" and hand_class in {"nutted", "strong"}:
        equity += 0.02
    return clamp(equity, 0.02, 0.98)


def river_villain_continue_prob(hand_class, villain_label, board_bucket, fraction, range_capped):
    if fraction <= 0.33:
        probability = 0.58
    elif fraction <= 0.67:
        probability = 0.44
    else:
        probability = 0.35
    if villain_label == "calling_station":
        probability += 0.18
    elif villain_label in {"nit", "tight_passive"}:
        probability -= 0.08
    elif villain_label == "overfolder":
        probability -= 0.18
    elif villain_label == "aggressive":
        probability += 0.03
    if range_capped:
        probability -= 0.08
        if fraction >= 0.67:
            probability -= 0.07
    if board_bucket in {"flushy", "straighty", "scary"} and range_capped:
        probability -= 0.05
    if hand_class == "nutted":
        probability += 0.06
    return clamp(probability, 0.08, 0.85)


def river_villain_raise_prob(villain_label, board_bucket, fraction, hero_blocks_value, hero_blocks_bluffs):
    probability = 0.03
    if villain_label == "aggressive":
        probability += 0.05
    elif villain_label in {"nit", "tight_passive", "calling_station"}:
        probability -= 0.02
    if board_bucket in {"flushy", "straighty", "scary"}:
        probability += 0.03
    if fraction >= 0.80:
        probability -= 0.01
    if hero_blocks_value:
        probability -= 0.025
    if hero_blocks_bluffs:
        probability += 0.015
    return clamp(probability, 0.0, 0.15)


def river_candidate_allowed(
    hand_class,
    fraction,
    villain_label,
    board_bucket,
    range_capped,
    hero_blocks_value,
    hero_blocks_bluffs,
    missed_draw_bucket,
):
    if fraction == 0.0 or hand_class == "nutted":
        return True
    if hand_class == "strong":
        if board_bucket in {"flushy", "straighty", "scary"} and villain_label != "calling_station":
            return fraction <= 0.33
        return fraction <= 0.80
    if hand_class == "thin":
        if villain_label == "calling_station" and board_bucket == "safe":
            return fraction <= 0.67
        return fraction <= 0.33
    if hand_class == "bluffcatcher":
        return fraction <= 0.33 and range_capped and villain_label in {"calling_station", "tight_passive"}
    if hand_class == "bluff_candidate":
        if not hero_blocks_value or hero_blocks_bluffs or villain_label == "calling_station":
            return False
        if missed_draw_bucket == "low" and not range_capped:
            return False
        return fraction >= 0.67
    return False


def river_normalized_ev(
    hand_class,
    villain_label,
    board_bucket,
    fraction,
    range_capped,
    hero_blocks_value,
    hero_blocks_bluffs,
    missed_draw_bucket,
):
    if fraction == 0.0:
        return river_base_showdown_equity(hand_class)
    if not river_candidate_allowed(
        hand_class,
        fraction,
        villain_label,
        board_bucket,
        range_capped,
        hero_blocks_value,
        hero_blocks_bluffs,
        missed_draw_bucket,
    ):
        return -999.0
    continue_probability = river_villain_continue_prob(
        hand_class,
        villain_label,
        board_bucket,
        fraction,
        range_capped,
    )
    raise_probability = min(
        river_villain_raise_prob(villain_label, board_bucket, fraction, hero_blocks_value, hero_blocks_bluffs),
        continue_probability,
    )
    call_probability = max(0.0, continue_probability - raise_probability)
    fold_probability = max(0.0, 1.0 - continue_probability)
    equity = river_equity_when_called(hand_class, board_bucket, fraction)
    ev_when_called = equity * (1.0 + fraction) - (1.0 - equity) * fraction
    raise_ev = ev_when_called if hand_class == "nutted" else -fraction
    return fold_probability + call_probability * ev_when_called + raise_probability * raise_ev


def river_policy_key(
    hand_class,
    role,
    board_bucket,
    villain_label,
    range_capped,
    hero_blocks_value,
    hero_blocks_bluffs,
    missed_draw_bucket,
    spr_bucket,
):
    return "|".join(
        map(
            str,
            (
                hand_class,
                role,
                board_bucket,
                villain_label,
                range_capped,
                hero_blocks_value,
                hero_blocks_bluffs,
                missed_draw_bucket,
                spr_bucket,
            ),
        )
    )


def choose_river_policy_entry(
    hand_class,
    role,
    board_bucket,
    villain_label,
    range_capped,
    hero_blocks_value,
    hero_blocks_bluffs,
    missed_draw_bucket,
    spr_bucket,
):
    scored = []
    for action, fraction in RIVER_CANDIDATES:
        ev = river_normalized_ev(
            hand_class,
            villain_label,
            board_bucket,
            fraction,
            range_capped,
            hero_blocks_value,
            hero_blocks_bluffs,
            missed_draw_bucket,
        )
        if role.startswith("OOP") and hand_class in {"thin", "bluffcatcher"} and fraction > 0:
            ev -= 0.08
        if spr_bucket == "tiny" and hand_class in {"nutted", "strong"} and fraction >= 0.67:
            ev += 0.04
        scored.append((ev, action, fraction))
    best_ev, best_action, best_fraction = max(scored, key=lambda item: item[0])
    check_ev = next(ev for ev, action, _ in scored if action == "check")
    if best_action == "bet" and best_ev < check_ev + 0.025:
        best_action, best_fraction = "check", 0.0
    return {
        "action": best_action,
        "fraction": best_fraction,
        "ev": round(float(best_ev), 4),
        "check_ev": round(float(check_ev), 4),
    }


def build_river_policy_table():
    table = {}
    for values in itertools.product(
        RIVER_HAND_CLASSES,
        RIVER_ROLES,
        RIVER_BOARD_BUCKETS,
        RIVER_VILLAIN_LABELS,
        (0, 1),
        (0, 1),
        (0, 1),
        RIVER_MISSED_DRAW_BUCKETS,
        RIVER_SPR_BUCKETS,
    ):
        table[river_policy_key(*values)] = choose_river_policy_entry(*values)
    return {
        "version": 1,
        "key_fields": [
            "hand_class",
            "role",
            "board_bucket",
            "villain_label",
            "range_capped",
            "hero_blocks_value",
            "hero_blocks_bluffs",
            "missed_draw_bucket",
            "spr_bucket",
        ],
        "candidates": [{"action": action, "fraction": fraction} for action, fraction in RIVER_CANDIDATES],
        "table": table,
    }


def source_hash(path):
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except Exception:
        return None


def write_outputs(out_dir, tables, metadata):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "equity_tables.json"), "w", encoding="utf-8") as handle:
        json.dump(tables, handle, sort_keys=True, separators=(",", ":"))
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True, indent=2)


def load_existing_tables(out_dir):
    path = os.path.join(out_dir, "equity_tables.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            tables = json.load(handle)
    except Exception:
        tables = {"version": 1}
    if not isinstance(tables, dict):
        return {"version": 1}
    tables.setdefault("version", 1)
    return tables


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate complete_v3 equity lookup tables.")
    parser.add_argument("--out", default=os.path.join(ROOT, "bots", "my_bots", "complete_v3_data"))
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--multiway-samples", type=int, default=900)
    parser.add_argument("--flop-samples", type=int, default=80)
    parser.add_argument("--turn-samples", type=int, default=80)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--river-only", action="store_true", help="Preserve existing tables and update river_policy only.")
    parser.add_argument("--turn-only", action="store_true", help="Preserve existing tables and update turn only.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.multiway_samples = min(args.multiway_samples, 80)
        args.flop_samples = min(args.flop_samples, 18)
        args.turn_samples = min(args.turn_samples, 8)
    rng = random.Random(args.seed)
    river_policy = build_river_policy_table()
    if args.river_only or args.turn_only:
        tables = load_existing_tables(args.out)
        if args.river_only:
            tables["river_policy"] = river_policy
        if args.turn_only:
            tables["turn"] = build_turn_tables(args.turn_samples, rng, quick=args.quick)
    else:
        turn = build_turn_tables(args.turn_samples, rng, quick=args.quick)
        tables = {
            "version": 1,
            "preflop": build_preflop_tables(args.multiway_samples, rng),
            "flop": build_flop_tables(args.flop_samples, rng, quick=args.quick),
            "turn": turn,
            "river_policy": river_policy,
        }
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "multiway_samples": args.multiway_samples,
        "flop_samples": args.flop_samples,
        "turn_samples": args.turn_samples,
        "ranges": HEADS_UP_RANGES,
        "roles": FLOP_ROLES,
        "hero_classes": FLOP_HERO_CLASSES,
        "role_realization": ROLE_REALIZATION,
        "class_realization": CLASS_REALIZATION,
        "turn_only": args.turn_only,
        "turn_cells": len(tables.get("turn", {}).get("cells", {})),
        "river_only": args.river_only,
        "river_policy_entries": len(river_policy["table"]),
        "preflop_equity_source": PREFLOP_EQUITY_PATH,
        "preflop_equity_sha256": source_hash(PREFLOP_EQUITY_PATH),
    }
    write_outputs(args.out, tables, metadata)
    return tables


if __name__ == "__main__":
    main()
