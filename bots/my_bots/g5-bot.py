"""
G5-inspired Fullhouse bot, condensed into one self-contained file.

Core ideas implemented here:
  * 100bb preflop chart layer, using compact 6-max ranges.
  * Lightweight Bayesian opponent modelling from observed action frequencies.
  * Combo-range construction and range updates from opponent actions.
  * EV comparison between fold/check/call and bet/raise.
  * Time/depth-limited one-step search: hero action -> opponent fold/call/raise
    response -> rollout/showdown equity cutoff.

The implementation intentionally avoids file I/O, network, subprocesses, threads,
and dynamic code execution. It only needs the Fullhouse `decide(game_state)` API.
"""

import bisect
import itertools
import math
import random
import time
from collections import defaultdict

try:
    import eval7  # allowed by the Fullhouse README; pure-Python fallback below.
except Exception:  # pragma: no cover - fallback is here for robustness.
    eval7 = None

BOT_NAME = "RangeBayes100"
BOT_AVATAR = "robot_1"

# ---------------------------------------------------------------------------
# Card and hand utilities
# ---------------------------------------------------------------------------

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
VALUE_RANK = {v: r for r, v in RANK_VALUE.items()}
DECK = [r + s for r in RANKS for s in SUITS]
EVAL7_CARD_CACHE = {}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def stable_seed(*parts):
    """Deterministic process-independent seed; avoids Python's randomized hash()."""
    h = 2166136261
    for part in parts:
        for ch in str(part):
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
    return h or 1


def state_rng(state):
    return random.Random(
        stable_seed(
            state.get("hand_id", ""),
            state.get("street", ""),
            tuple(state.get("your_cards") or ()),
            tuple(state.get("community_cards") or ()),
            len(state.get("action_log") or ()),
        )
    )


def is_card(c):
    return isinstance(c, str) and len(c) == 2 and c[0] in RANK_VALUE and c[1] in SUITS


def clean_cards(cards):
    return [c for c in (cards or []) if is_card(c)]


def card_value(card):
    return RANK_VALUE[card[0]]


def normalize_hand(cards):
    """Return 169-grid key: AA, AKs, AKo, etc."""
    cards = clean_cards(cards)
    if len(cards) < 2:
        return ""
    c1, c2 = cards[0], cards[1]
    r1, r2 = card_value(c1), card_value(c2)
    s1, s2 = c1[1], c2[1]
    if r1 == r2:
        return VALUE_RANK[r1] + VALUE_RANK[r2]
    hi, lo = (r1, r2) if r1 > r2 else (r2, r1)
    return VALUE_RANK[hi] + VALUE_RANK[lo] + ("s" if s1 == s2 else "o")


def hand_playability_key(c1, c2):
    return normalize_hand([c1, c2])


def preflop_score_for_key(key):
    """0..1 rough preflop strength/playability score for heuristics and ranges."""
    if not key:
        return 0.0
    r1 = RANK_VALUE[key[0]]
    r2 = RANK_VALUE[key[1]]
    if len(key) == 2:  # pair
        return clamp(0.40 + (r1 - 2) / 12.0 * 0.55, 0.0, 1.0)
    hi, lo = max(r1, r2), min(r1, r2)
    suited = key.endswith("s")
    gap = hi - lo - 1
    score = 0.14 + (hi - 2) / 12.0 * 0.34 + (lo - 2) / 12.0 * 0.23
    if suited:
        score += 0.08
    if gap == 0:
        score += 0.08
    elif gap == 1:
        score += 0.055
    elif gap == 2:
        score += 0.025
    else:
        score -= min(0.16, gap * 0.028)
    if hi == 14 and suited and lo <= 5:
        score += 0.06  # wheel ace blocker/playability
    if hi >= 13 and lo >= 10:
        score += 0.05  # broadways
    if not suited and lo <= 7:
        score -= 0.05
    return clamp(score, 0.0, 1.0)


ALL_HOLE_COMBOS = []
for i, c1 in enumerate(DECK):
    for c2 in DECK[i + 1 :]:
        key = hand_playability_key(c1, c2)
        ALL_HOLE_COMBOS.append((c1, c2, key, preflop_score_for_key(key)))


def eval7_cards(cards):
    out = []
    for c in cards:
        obj = EVAL7_CARD_CACHE.get(c)
        if obj is None:
            obj = eval7.Card(c)
            EVAL7_CARD_CACHE[c] = obj
        out.append(obj)
    return out


def five_card_score(cards5):
    """Pure-Python 5-card score. Larger tuple is better."""
    vals = sorted([card_value(c) for c in cards5], reverse=True)
    suits = [c[1] for c in cards5]
    counts = defaultdict(int)
    for v in vals:
        counts[v] += 1
    unique = sorted(counts, reverse=True)
    straight_vals = set(unique)
    if 14 in straight_vals:
        straight_vals.add(1)
    straight_high = 0
    for high in range(14, 4, -1):
        if all(v in straight_vals for v in range(high - 4, high + 1)):
            straight_high = high
            break
    flush = len(set(suits)) == 1
    by_count = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if straight_high and flush:
        return (8, straight_high)
    if by_count[0][1] == 4:
        quad = by_count[0][0]
        kicker = max(v for v in vals if v != quad)
        return (7, quad, kicker)
    if by_count[0][1] == 3 and by_count[1][1] == 2:
        return (6, by_count[0][0], by_count[1][0])
    if flush:
        return (5,) + tuple(vals)
    if straight_high:
        return (4, straight_high)
    if by_count[0][1] == 3:
        trip = by_count[0][0]
        kickers = sorted([v for v in vals if v != trip], reverse=True)
        return (3, trip) + tuple(kickers)
    if by_count[0][1] == 2 and by_count[1][1] == 2:
        p1, p2 = sorted([by_count[0][0], by_count[1][0]], reverse=True)
        kicker = max(v for v in vals if v not in (p1, p2))
        return (2, p1, p2, kicker)
    if by_count[0][1] == 2:
        pair = by_count[0][0]
        kickers = sorted([v for v in vals if v != pair], reverse=True)
        return (1, pair) + tuple(kickers)
    return (0,) + tuple(vals)


def evaluate_cards(cards):
    """Comparable strength for 5-7 cards."""
    cards = clean_cards(cards)
    if eval7 is not None:
        return eval7.evaluate(eval7_cards(cards))
    if len(cards) < 5:
        return (0,)
    return max(five_card_score(c) for c in itertools.combinations(cards, 5))


def best_score_tuple(cards):
    """Always returns the pure tuple score, useful for made-hand categories."""
    cards = clean_cards(cards)
    if len(cards) < 5:
        return (0,)
    return max(five_card_score(c) for c in itertools.combinations(cards, 5))


def made_hand_level(hole, board):
    """
    Coarse 0..8 made hand bucket.
      0 high-card/air, 1 weak pair, 2 top pair, 3 overpair,
      4 two pair, 5 trips/set, 6 straight, 7 flush, 8 boat+.
    """
    hole = clean_cards(hole)
    board = clean_cards(board)
    if len(board) < 3 or len(hole) < 2:
        return 0
    score = best_score_tuple(hole + board)
    cat = score[0]
    if cat == 0:
        return 0
    if cat == 1:
        pair_rank = score[1]
        board_vals = [card_value(c) for c in board]
        hole_vals = [card_value(c) for c in hole]
        if pair_rank in hole_vals and board_vals:
            if pair_rank > max(board_vals):
                return 3  # overpair
            if pair_rank == max(board_vals):
                return 2  # top pair
        return 1
    if cat == 2:
        return 4
    if cat == 3:
        return 5
    if cat == 4:
        return 6
    if cat == 5:
        return 7
    return 8


def draw_flags(hole, board):
    hole = clean_cards(hole)
    board = clean_cards(board)
    cards = hole + board
    vals = {card_value(c) for c in cards}
    if 14 in vals:
        vals.add(1)
    suit_counts = defaultdict(int)
    for c in cards:
        suit_counts[c[1]] += 1
    board_suit_counts = defaultdict(int)
    for c in board:
        board_suit_counts[c[1]] += 1

    flush_draw = max(suit_counts.values() or [0]) >= 4 and len(board) < 5
    nut_flush_draw = False
    for s, cnt in suit_counts.items():
        if cnt >= 4 and ("A" + s) in hole:
            nut_flush_draw = True
    straight_outs = set()
    for high in range(5, 15):
        run = set(range(high - 4, high + 1))
        missing = [r for r in run if r not in vals]
        if len(missing) == 1:
            straight_outs.add(missing[0])
    straight_draw = bool(straight_outs) and len(board) < 5
    open_ended = len(straight_outs) >= 2
    board_high = max([card_value(c) for c in board] or [0])
    overcards = sum(1 for c in hole if card_value(c) > board_high)
    backdoor_flush = len(board) == 3 and max(suit_counts.values() or [0]) == 3
    return {
        "flush_draw": flush_draw,
        "nut_flush_draw": nut_flush_draw,
        "straight_draw": straight_draw,
        "open_ended": open_ended,
        "overcards": overcards,
        "backdoor_flush": backdoor_flush,
    }


def board_texture(board):
    board = clean_cards(board)
    if len(board) < 3:
        return {"wetness": 0.0, "paired": False, "monotone": False, "two_tone": False}
    vals = sorted([card_value(c) for c in board])
    suits = defaultdict(int)
    ranks = defaultdict(int)
    for c in board:
        suits[c[1]] += 1
        ranks[card_value(c)] += 1
    paired = max(ranks.values() or [1]) >= 2
    monotone = max(suits.values() or [0]) >= 3
    two_tone = max(suits.values() or [0]) == 2
    high_cards = sum(1 for v in vals if v >= 10)
    unique = sorted(set(vals))
    gaps = 0
    if len(unique) >= 2:
        gaps = max(0, max(unique) - min(unique) - (len(unique) - 1))
    connected = 1.0 if gaps <= 1 else 0.5 if gaps <= 3 else 0.0
    wet = 0.0
    wet += 0.34 if monotone else 0.18 if two_tone else 0.0
    wet += 0.25 * connected
    wet += 0.10 if paired else 0.0
    wet += 0.08 if high_cards >= 2 else 0.0
    return {
        "wetness": clamp(wet, 0.0, 1.0),
        "paired": paired,
        "monotone": monotone,
        "two_tone": two_tone,
    }


# ---------------------------------------------------------------------------
# 100bb preflop chart layer
# ---------------------------------------------------------------------------

# These are compact set-based charts derived from the provided 100bb 6-max ranges.
# Open sizes: 2.5bb except SB 3bb. IP 3bet ~3.5x, OOP 3bet ~4x.
RFI_RANGES = {
    "LJ": set("AA AKs AQs AJs ATs A9s A8s A7s A6s A5s A4s A3s AKo KK KQs KJs KTs K9s K8s AQo KQo QQ QJs QTs Q9s AJo KJo QJo JJ JTs J9s ATo TT T9s 99 88 77 66".split()),
    "HJ": set("AA AKs AQs AJs ATs A9s A8s A7s A6s A5s A4s A3s A2s AKo KK KQs KJs KTs K9s K8s K7s K6s AQo KQo QQ QJs QTs Q9s Q8s AJo KJo QJo JJ JTs J9s ATo KTo QTo TT T9s 99 98s 88 87s 77 76s 66 55".split()),
    "CO": set("AA AKs AQs AJs ATs A9s A8s A7s A6s A5s A4s A3s A2s AKo KK KQs KJs KTs K9s K8s K7s K6s K5s K4s K3s AQo KQo QQ QJs QTs Q9s Q8s Q7s Q6s AJo KJo QJo JJ JTs J9s J8s ATo KTo QTo JTo TT T9s T8s T7s A9o 99 98s 97s A8o 88 87s 77 76s 66 55 44 33".split()),
    "BTN": set("AA AKs AQs AJs ATs A9s A8s A7s A6s A5s A4s A3s A2s AKo KK KQs KJs KTs K9s K8s K7s K6s K5s K4s K3s K2s AQo KQo QQ QJs QTs Q9s Q8s Q7s Q6s Q5s Q4s Q3s AJo KJo QJo JJ JTs J9s J8s J7s J6s J5s J4s ATo KTo QTo JTo TT T9s T8s T7s T6s A9o K9o Q9o J9o T9o 99 98s 97s 96s A8o K8o T8o 98o 88 87s 86s 85s A7o 77 76s 75s A6o 66 65s 64s A5o 55 54s 53s A4o 44 33 22".split()),
}
SB_RAISE = set("AKs ATs A9s A8s A7s A5s KK KJs KTs K8s K5s K3s K2s AQo QQ QJs QTs Q5s Q4s Q3s Q2s AJo KJo JJ JTs J7s J6s J5s J4s T9s T6s T5s K9o Q9o J9o 96s A8o K8o T8o 98o A7o K7o A6o 65s 64s 54s 53s A4o 33 22".split())
SB_LIMP = set("AA AQs AJs A6s A4s A3s A2s AKo KQs K9s K7s K6s K4s KQo Q9s Q8s Q7s Q6s QJo J9s J8s J3s J2s ATo KTo QTo JTo TT T8s T7s T4s T3s A9o T9o 99 98s 97s 95s 94s Q8o J8o 88 87s 86s 85s 84s Q7o J7o T7o 97o 87o 77 76s 75s 74s K6o Q6o 86o 76o 66 63s A5o K5o Q5o 55 K4o 44 43s A3o A2o".split())

VS_OPEN = {
    "HJ_vs_LJ": {"3bet": set("AA AKs AQs AJs ATs A5s AKo KK KQs KJs KTs AQo KQo QQ QJs JJ TT 99".split()), "call": set()},
    "CO_vs_LJ": {"3bet": set("AA AKs AQs AJs ATs A5s AKo KK KQs KJs KTs AQo KQo QQ QJs JJ TT 99 88".split()), "call": set()},
    "CO_vs_HJ": {"3bet": set("AA AKs AQs AJs ATs A9s A5s A4s AKo KK KQs KJs KTs AQo KQo QQ QJs AJo JJ TT 99 88".split()), "call": set()},
    "BTN_vs_LJ": {"3bet": set("AA AKs AQs A9s A8s A4s A3s AKo KK K9s KQo QQ QJs AJo JJ T9s".split()), "call": set("AJs ATs A5s KQs KJs KTs AQo QTs JTs TT 99 88 77 76s 66 65s 55 54s".split())},
    "BTN_vs_HJ": {"3bet": set("AA AKs AQs A9s A8s A7s A4s A3s AKo KK KTs K9s K8s KQo QQ QTs Q9s AJo JJ T9s 66".split()), "call": set("AJs ATs A5s KQs KJs AQo QJs JTs TT 99 98s 88 87s 77 55 44".split())},
    "BTN_vs_CO": {"3bet": set("AA AKs AQs A8s A7s A6s A4s A3s AKo KK KQs K9s KQo QQ QJs Q9s AJo KJo QJo JJ JTs J9s ATo TT 55".split()), "call": set("AJs ATs A9s A5s KJs KTs AQo QTs T9s 99 98s 88 77 66".split())},
    "SB_vs_LJ": {"3bet": set("AA AKs AQs AJs ATs A5s AKo KK KQs KJs KTs AQo QQ QJs JJ TT 99".split()), "call": set()},
    "SB_vs_HJ": {"3bet": set("AA AKs AQs AJs ATs A5s AKo KK KQs KJs KTs AQo QQ QJs QTs JJ JTs TT 99 88 77".split()), "call": set()},
    "SB_vs_CO": {"3bet": set("AA AKs AQs AJs ATs A9s A5s AKo KK KQs KJs KTs AQo KQo QQ QJs QTs JJ JTs J9s TT T9s 99 88 77 66".split()), "call": set()},
    "SB_vs_BTN": {"3bet": set("AA AKs AQs AJs ATs A9s A8s A7s A5s A4s AKo KK KQs KJs KTs K9s AQo KQo QQ QJs QTs Q9s AJo KJo JJ JTs J9s TT T9s T8s 99 88 77 66 55".split()), "call": set()},
    "BB_vs_LJ": {"3bet": set("AA AKs AQs A5s A4s AKo KK KQs KJs QQ QJs JJ JTs 65s 54s".split()), "call": set("AJs ATs A9s A8s A7s A6s A3s A2s KTs K9s K8s K7s K6s K5s K4s K3s K2s AQo KQo QTs Q9s Q8s Q7s Q6s Q5s AJo KJo QJo J9s J8s ATo JTo TT T9s T8s T7s 99 98s 97s 96s 88 87s 86s 85s 77 76s 75s 74s 66 64s 63s 55 53s 44 43s 33 32s 22".split())},
    "BB_vs_HJ": {"3bet": set("AA AKs AQs A9s A5s A4s AKo KK KQs KJs KTs K5s QQ QJs QTs JJ JTs TT 65s 54s".split()), "call": set("AJs ATs A8s A7s A6s A3s A2s K9s K8s K7s K6s K4s K3s K2s AQo KQo Q9s Q8s Q7s Q6s Q5s AJo KJo QJo J9s J8s J7s ATo KTo QTo JTo T9s T8s T7s A9o 99 98s 97s 96s 88 87s 86s 85s 77 76s 75s 74s 66 64s 63s 55 53s 44 43s 33 22".split())},
    "BB_vs_CO": {"3bet": set("AA AKs AQs AJs A9s A5s A4s AKo KK KQs KJs KTs AQo QQ QJs QTs Q9s JJ JTs J9s TT T9s 99 65s 54s".split()), "call": set("ATs A8s A7s A6s A3s A2s K9s K8s K7s K6s K5s K4s K3s K2s KQo Q8s Q7s Q6s Q5s Q4s Q3s AJo KJo QJo J8s J7s J6s ATo KTo QTo JTo T8s T7s A9o T9o 98s 97s 96s A8o 88 87s 86s 85s 77 76s 75s 74s 66 64s 63s A5o 55 53s 52s 44 43s 33 22".split())},
    "BB_vs_BTN": {"3bet": set("AA AKs AQs AJs ATs A6s A5s A4s AKo KK KQs KJs KTs K9s AQo KQo QQ QJs QTs Q9s JJ JTs J9s J8s TT T9s T8s 99 98s 97s 88 87s 76s 65s 54s".split()), "call": set("A9s A8s A7s A3s A2s K8s K7s K6s K5s K4s K3s K2s Q8s Q7s Q6s Q5s Q4s Q3s Q2s AJo KJo QJo J7s J6s J5s J4s J3s J2s ATo KTo QTo JTo T7s T6s T5s T4s T3s T2s A9o K9o Q9o J9o T9o 96s 95s 94s A8o K8o Q8o J8o T8o 98o 86s 85s 84s A7o K7o 87o 77 75s 74s 73s A6o K6o 76o 66 64s 63s 62s A5o 65o 55 53s 52s A4o 54o 44 43s 42s A3o 33 32s 22".split())},
    "BB_vs_SB": {"3bet": set("AA AKs AQs AJs ATs A5s A4s AKo KK KQs KJs KTs AQo QQ QJs JJ J5s TT T5s 99 95s J8o 88 87s J7o T7o 76s A6o K6o Q6o 65s K5o 54s".split()), "call": set("A9s A8s A7s A6s A3s A2s K9s K8s K7s K6s K5s K4s K3s K2s KQo QTs Q9s Q8s Q7s Q6s Q5s Q4s Q3s Q2s AJo KJo QJo JTs J9s J8s J7s J6s J4s J3s J2s ATo KTo QTo JTo T9s T8s T7s T6s T4s T3s T2s A9o K9o Q9o J9o T9o 98s 97s 96s 94s 93s 92s A8o K8o Q8o T8o 98o 86s 85s 84s A7o K7o Q7o 97o 87o 77 75s 74s 73s 86o 76o 66 64s 63s 62s A5o 65o 55 53s 52s A4o 54o 44 43s 42s A3o 33 32s A2o 22".split())},
}
BB_VS_SB_LIMP_RAISE = set("AA AKs AQs AJs ATs A9s A8s A5s A4s A3s AKo KK KQs KJs KTs K9s K6s K5s AQo KQo QQ QJs QTs Q9s AJo KJo JJ JTs J9s J8s J2s ATo JTo TT T9s T8s T4s T3s T2s T9o 99 98s 97s 94s 93s 92s 88 87s 86s 84s J7o 77 76s 75s 74s 73s Q6o J6o T6o 96o 66 65s 64s 63s A5o K5o Q5o J5o T5o 95o 85o 75o 55 54s K4o Q4o 74o 44 33 32s".split())

PREMIUM_5BET = set("AA KK QQ AKs AKo".split())
VALUE_4BET = set("AA KK QQ JJ AKs AKo AQs".split())
BLUFF_4BET = set("A5s A4s A3s A2s KTs K9s QTs".split())
CALL_VS_3BET_IP = set("JJ TT 99 88 AQs AJs ATs KQs KJs QJs JTs T9s 98s 77 66 55".split())
CALL_VS_3BET_OOP = set("JJ TT 99 AQs AJs KQs QJs JTs 88 77".split())

# ---------------------------------------------------------------------------
# Opponent model: small Bayesian action-frequency tracker
# ---------------------------------------------------------------------------

class OpponentModel:
    def __init__(self):
        self.actions = defaultdict(lambda: defaultdict(int))
        self.total_actions = 0
        self.preflop_vpip = 0
        self.preflop_raises = 0
        self.response_fold = defaultdict(int)
        self.response_call = defaultdict(int)
        self.response_raise = defaultdict(int)

    def record(self, street, action):
        street = street if street in ("preflop", "flop", "turn", "river") else "unknown"
        action = normalize_action_name(action)
        if not action:
            return
        self.actions[street][action] += 1
        self.total_actions += 1
        if street == "preflop" and action in ("call", "raise", "bet", "all_in"):
            self.preflop_vpip += 1
            if action in ("raise", "bet", "all_in"):
                self.preflop_raises += 1
        if action == "fold":
            self.response_fold[street] += 1
        elif action == "call":
            self.response_call[street] += 1
        elif action in ("raise", "bet", "all_in"):
            self.response_raise[street] += 1

    def fold_rate(self, street):
        # Beta prior: most unknown players fold around 38-42% when facing pressure.
        f = self.response_fold[street]
        c = self.response_call[street]
        r = self.response_raise[street]
        return (4.0 + f) / (10.0 + f + c + r)

    def call_rate(self, street):
        f = self.response_fold[street]
        c = self.response_call[street]
        r = self.response_raise[street]
        return (4.5 + c) / (10.0 + f + c + r)

    def aggression(self, street):
        counts = self.actions[street]
        raises = counts.get("raise", 0) + counts.get("bet", 0) + counts.get("all_in", 0)
        passive = counts.get("call", 0) + counts.get("check", 0)
        return (2.0 + raises) / (8.0 + raises + passive)

    def looseness(self):
        # A rough VPIP proxy from actions, smoothed heavily.
        return clamp((8.0 + self.preflop_vpip) / (35.0 + self.total_actions), 0.15, 0.65)


OPPONENTS = defaultdict(OpponentModel)
PROCESSED_LOG_LENGTH = {}


def normalize_action_name(action):
    if action is None:
        return ""
    a = str(action).lower().strip()
    aliases = {
        "check": "check",
        "x": "check",
        "call": "call",
        "c": "call",
        "fold": "fold",
        "f": "fold",
        "raise": "raise",
        "r": "raise",
        "bet": "bet",
        "b": "bet",
        "allin": "all_in",
        "all_in": "all_in",
        "all-in": "all_in",
        "jam": "all_in",
        "shove": "all_in",
        "post": "post",
        "blind": "post",
    }
    return aliases.get(a, a)


def entry_get(entry, *names, default=None):
    if isinstance(entry, dict):
        for name in names:
            if name in entry:
                return entry.get(name)
        return default
    if isinstance(entry, (list, tuple)):
        # Very defensive tuple support: try common shapes such as
        # (street, seat, action, amount) or (seat, action, amount, street).
        for item in entry:
            if isinstance(item, dict):
                val = entry_get(item, *names, default=None)
                if val is not None:
                    return val
        wanted = set(names)
        if {"seat", "player", "player_seat", "actor", "seat_id"} & wanted:
            for item in entry:
                if isinstance(item, int):
                    return item
        if {"action", "type", "move"} & wanted:
            for item in entry:
                if isinstance(item, str) and normalize_action_name(item) in {
                    "fold",
                    "check",
                    "call",
                    "raise",
                    "bet",
                    "all_in",
                    "post",
                }:
                    return item
        if {"street", "round"} & wanted:
            for item in entry:
                if isinstance(item, str) and item.lower() in {"preflop", "flop", "turn", "river"}:
                    return item.lower()
        if {"amount", "bet", "to", "raise_to"} & wanted:
            ints = [item for item in entry if isinstance(item, int)]
            if ints:
                return ints[-1]
    return default


def player_id_for_seat(state, seat):
    for p in state.get("players") or []:
        if safe_int(p.get("seat"), -999) == seat:
            return str(p.get("bot_id", "seat_%s" % seat))
    return "seat_%s" % seat


def update_opponent_models(state):
    hand_id = str(state.get("hand_id", "unknown"))
    action_log = state.get("action_log") or []
    last_len = PROCESSED_LOG_LENGTH.get(hand_id, 0)
    hero_seat = safe_int(state.get("seat_to_act"), -1)
    for entry in action_log[last_len:]:
        seat = entry_get(entry, "seat", "player", "player_seat", "actor", "seat_id", default=None)
        if seat is None:
            continue
        seat = safe_int(seat, -999)
        if seat == hero_seat:
            continue
        action = entry_get(entry, "action", "type", "move", default="")
        street = entry_get(entry, "street", "round", default=state.get("street", "unknown"))
        if normalize_action_name(action) == "post":
            continue
        OPPONENTS[player_id_for_seat(state, seat)].record(street, action)
    PROCESSED_LOG_LENGTH[hand_id] = len(action_log)
    # Keep memory bounded over long matches.
    if len(PROCESSED_LOG_LENGTH) > 2000:
        for key in list(PROCESSED_LOG_LENGTH.keys())[:500]:
            PROCESSED_LOG_LENGTH.pop(key, None)


# ---------------------------------------------------------------------------
# Game-state interpretation
# ---------------------------------------------------------------------------

def active_players(state):
    players = []
    for p in state.get("players") or []:
        if p.get("is_active", True) and not p.get("is_folded", False):
            players.append(p)
    return players


def opponent_players(state):
    hero = safe_int(state.get("seat_to_act"), -1)
    return [p for p in active_players(state) if safe_int(p.get("seat"), -999) != hero]


def estimate_big_blind(state):
    if "big_blind" in state:
        bb = safe_int(state.get("big_blind"), 0)
        if bb > 0:
            return bb
    current_bet = safe_int(state.get("current_bet"), 0)
    min_raise_to = safe_int(state.get("min_raise_to"), 0)
    bets = sorted(
        [safe_int(p.get("bet_this_street"), 0) for p in (state.get("players") or []) if safe_int(p.get("bet_this_street"), 0) > 0]
    )
    if state.get("street") == "preflop" and bets:
        if len(bets) >= 3:
            # Usually [SB, BB, opener].
            if bets[0] * 2 <= bets[1] * 1.25:
                return max(1, bets[1])
            return max(1, bets[0])
        if len(bets) == 2:
            return max(1, bets[1])
        return max(1, bets[0])
    if current_bet > 0 and min_raise_to > current_bet:
        return max(1, min_raise_to - current_bet)
    if current_bet > 0:
        return max(1, current_bet)
    # Last-resort default. Most Fullhouse examples use 100-chip BB.
    return 100


def infer_blind_seats(state, bb):
    bets = []
    for p in state.get("players") or []:
        seat = safe_int(p.get("seat"), -999)
        bet = safe_int(p.get("bet_this_street"), 0)
        if seat != -999 and bet > 0:
            bets.append((bet, seat))
    if not bets:
        return None, None
    bets.sort()
    # If there is an opener/raiser, the blind bets are usually the two smallest.
    if len(bets) >= 2 and bets[0][0] <= bets[1][0] <= max(bb * 2, bets[0][0] * 3):
        sb_seat = bets[0][1]
        bb_seat = bets[1][1]
        return sb_seat, bb_seat
    # If only one forced bet remains visible, treat it as BB.
    return None, bets[0][1]


def seat_positions(state):
    players = state.get("players") or []
    seats = sorted([safe_int(p.get("seat"), -999) for p in players if safe_int(p.get("seat"), -999) != -999])
    if not seats:
        return {}
    bb = estimate_big_blind(state)
    _, bb_seat = infer_blind_seats(state, bb)
    if bb_seat not in seats:
        # Fallback: conventional 6-seat ring order. Imperfect but deterministic.
        bb_seat = seats[-1]
    n = len(seats)
    if n >= 6:
        names = ["LJ", "HJ", "CO", "BTN", "SB", "BB"]
    elif n == 5:
        names = ["HJ", "CO", "BTN", "SB", "BB"]
    elif n == 4:
        names = ["CO", "BTN", "SB", "BB"]
    elif n == 3:
        names = ["BTN", "SB", "BB"]
    else:
        names = ["SB", "BB"]
    idx = seats.index(bb_seat)
    order = seats[idx + 1 :] + seats[: idx + 1]  # first preflop after BB ... BB
    # Align length in unusual missing-seat cases.
    if len(order) != len(names):
        names = (names + ["LJ", "HJ", "CO", "BTN", "SB", "BB"])[: len(order)]
    return {seat: pos for seat, pos in zip(order, names)}


def hero_position(state):
    return seat_positions(state).get(safe_int(state.get("seat_to_act"), -1), "CO")


def actions_for_current_hand(state):
    out = []
    for entry in state.get("action_log") or []:
        seat = entry_get(entry, "seat", "player", "player_seat", "actor", "seat_id", default=None)
        action = normalize_action_name(entry_get(entry, "action", "type", "move", default=""))
        street = entry_get(entry, "street", "round", default=state.get("street", "unknown"))
        amount = safe_int(entry_get(entry, "amount", "bet", "to", "raise_to", default=0), 0)
        if seat is not None and action:
            out.append({"seat": safe_int(seat, -999), "action": action, "street": street, "amount": amount})
    return out


def preflop_summary(state):
    bb = estimate_big_blind(state)
    positions = seat_positions(state)
    hero = safe_int(state.get("seat_to_act"), -1)
    raises = []
    calls = []
    hero_voluntary = False
    for a in actions_for_current_hand(state):
        if a["street"] != "preflop":
            continue
        seat = a["seat"]
        action = a["action"]
        amount = a["amount"]
        if action in ("raise", "bet", "all_in") and amount > bb:
            raises.append(a)
            if seat == hero:
                hero_voluntary = True
        elif action == "call":
            calls.append(a)
            if seat == hero:
                hero_voluntary = True
    first_raiser_pos = None
    if raises:
        first_raiser_pos = positions.get(raises[0]["seat"], "LJ")
        if first_raiser_pos == "UTG":
            first_raiser_pos = "LJ"
    return {
        "bb": bb,
        "raise_count": len([r for r in raises if r["seat"] != hero]),
        "total_raise_count": len(raises),
        "call_count": len([c for c in calls if c["seat"] != hero]),
        "first_raiser_pos": first_raiser_pos,
        "hero_voluntary": hero_voluntary,
    }


def is_in_position(hero_pos, opener_pos):
    order = ["LJ", "HJ", "CO", "BTN", "SB", "BB"]
    if hero_pos == "BB":
        return False
    if opener_pos == "SB" and hero_pos == "BB":
        return True
    try:
        return order.index(hero_pos) > order.index(opener_pos) and hero_pos not in ("SB", "BB")
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Range construction and equity estimation
# ---------------------------------------------------------------------------

RANGE_CACHE = {}


def make_range(items):
    clean = []
    cum = []
    total = 0.0
    for item in items:
        c1, c2, key, score, weight = item
        if weight <= 0:
            continue
        total += weight
        clean.append(item)
        cum.append(total)
    return {"items": clean, "cum": cum, "total": total}


def range_weight_preflop(key, score, action, raise_depth, model):
    suited = len(key) == 3 and key[2] == "s"
    pair = len(key) == 2
    wheel_ace = key in {"A5s", "A4s", "A3s", "A2s"}
    broadway = key[0] in "AKQJ" and key[1] in "AKQJT"
    loose = model.looseness()
    if action in ("raise", "bet", "all_in"):
        threshold = 0.48 + 0.11 * max(0, raise_depth - 1)
        value = 0.10 + 4.0 * max(0.0, score - threshold) ** 1.2
        bluff = 0.35 if wheel_ace or (suited and broadway) else 0.03
        if action == "all_in":
            value *= 1.6
            bluff *= 0.35
        return clamp(value + bluff * model.aggression("preflop"), 0.01, 5.0)
    if action == "call":
        implied = 0.0
        if pair:
            implied += 0.55
        if suited:
            implied += 0.30
        if broadway:
            implied += 0.22
        if key in {"JTs", "T9s", "98s", "87s", "76s", "65s", "54s"}:
            implied += 0.35
        dominated_penalty = 0.25 if (len(key) == 3 and key.endswith("o") and score < 0.55) else 0.0
        return clamp(0.08 + loose * 0.6 + implied + score * 0.35 - dominated_penalty, 0.01, 3.0)
    if action == "check":
        return clamp(0.65 + loose * 0.4, 0.1, 1.4)
    return 1.0


def range_weight_postflop(combo, board, action, model):
    c1, c2, key, score = combo
    level = made_hand_level([c1, c2], board)
    draws = draw_flags([c1, c2], board)
    draw_power = 0.0
    if draws["flush_draw"]:
        draw_power += 0.55
    if draws["straight_draw"]:
        draw_power += 0.40
    if draws["nut_flush_draw"]:
        draw_power += 0.20
    if draws["overcards"] >= 2:
        draw_power += 0.12
    agg = model.aggression("flop")
    action = normalize_action_name(action)
    if action in ("bet", "raise", "all_in"):
        value = [0.08, 0.15, 0.35, 0.55, 1.2, 1.9, 2.4, 2.8, 3.2][level]
        semi_bluff = draw_power * (0.9 + agg)
        air = 0.06 + agg * 0.16
        if action == "all_in":
            value *= 1.25
            air *= 0.35
        return clamp(value + semi_bluff + (air if level <= 1 else 0.0), 0.01, 5.0)
    if action == "call":
        medium = [0.05, 0.65, 1.05, 1.15, 1.5, 1.6, 1.3, 1.2, 1.2][level]
        return clamp(medium + draw_power * 0.9, 0.01, 3.5)
    if action == "check":
        # Checks cap aggressive villains more than passive ones.
        cap_penalty = 1.0 - agg * 0.35 if level >= 5 else 1.0
        return clamp(([0.9, 1.1, 1.2, 1.15, 0.95, 0.75, 0.7, 0.65, 0.65][level]) * cap_penalty + draw_power * 0.25, 0.02, 2.0)
    return 1.0


def build_range_for_seat(state, seat, known_cards, board):
    pid = player_id_for_seat(state, seat)
    model = OPPONENTS[pid]
    actions = [a for a in actions_for_current_hand(state) if a["seat"] == seat]
    raise_depth = 0
    items = []
    known = set(known_cards)
    for c1, c2, key, score in ALL_HOLE_COMBOS:
        if c1 in known or c2 in known:
            continue
        w = 1.0
        raise_depth = 0
        for a in actions:
            if a["action"] == "fold":
                w = 0.0
                break
            if a["street"] == "preflop":
                if a["action"] in ("raise", "bet", "all_in"):
                    raise_depth += 1
                w *= range_weight_preflop(key, score, a["action"], raise_depth, model)
            else:
                w *= range_weight_postflop((c1, c2, key, score), board, a["action"], model)
        if w > 0.0001:
            items.append((c1, c2, key, score, w))
    if not items:
        items = [(c1, c2, key, score, 1.0) for c1, c2, key, score in ALL_HOLE_COMBOS if c1 not in known and c2 not in known]
    return make_range(items)


def response_adjusted_range(rng_range, board, response):
    items = []
    dummy = OpponentModel()
    for c1, c2, key, score, weight in rng_range["items"]:
        mult = range_weight_postflop((c1, c2, key, score), board, response, dummy)
        if response == "raise":
            mult *= 1.25
        elif response == "call":
            mult *= 0.9
        items.append((c1, c2, key, score, weight * mult))
    return make_range(items)


def weighted_combo_choice(rng, rng_range, dead):
    if not rng_range["items"] or rng_range["total"] <= 0:
        return None
    items = rng_range["items"]
    cum = rng_range["cum"]
    total = rng_range["total"]
    # Rejection sampling is fast because dead cards are few.
    for _ in range(30):
        idx = bisect.bisect_left(cum, rng.random() * total)
        if idx >= len(items):
            idx = len(items) - 1
        c1, c2, _, _, _ = items[idx]
        if c1 not in dead and c2 not in dead:
            return (c1, c2)
    # Fallback linear scan.
    for c1, c2, _, _, _ in items:
        if c1 not in dead and c2 not in dead:
            return (c1, c2)
    return None


def estimate_equity(hero_cards, board, ranges, samples, rng, deadline):
    hero_cards = clean_cards(hero_cards)
    board = clean_cards(board)
    if not ranges:
        return 1.0
    if len(hero_cards) < 2:
        return 0.0
    known0 = set(hero_cards + board)
    need_board = max(0, 5 - len(board))
    samples = max(20, int(samples))
    points = 0.0
    trials = 0
    base_deck = [c for c in DECK if c not in known0]

    # On a river heads-up, exact-ish enumeration over villain range is cheap enough.
    if need_board == 0 and len(ranges) == 1:
        hero_score = evaluate_cards(hero_cards + board)
        total_w = 0.0
        win_w = 0.0
        for c1, c2, _, _, w in ranges[0]["items"]:
            if c1 in known0 or c2 in known0:
                continue
            opp_score = evaluate_cards([c1, c2] + board)
            total_w += w
            if hero_score > opp_score:
                win_w += w
            elif hero_score == opp_score:
                win_w += 0.5 * w
        return clamp(win_w / total_w if total_w > 0 else 0.5, 0.0, 1.0)

    while trials < samples and time.monotonic() < deadline:
        dead = set(known0)
        opp_hands = []
        ok = True
        for rr in ranges:
            combo = weighted_combo_choice(rng, rr, dead)
            if combo is None:
                ok = False
                break
            opp_hands.append(combo)
            dead.update(combo)
        if not ok:
            continue
        deck = [c for c in base_deck if c not in dead]
        if len(deck) < need_board:
            continue
        runout = rng.sample(deck, need_board) if need_board else []
        final_board = board + runout
        hero_score = evaluate_cards(hero_cards + final_board)
        opp_scores = [evaluate_cards(list(h) + final_board) for h in opp_hands]
        best_opp = max(opp_scores)
        if hero_score > best_opp:
            points += 1.0
        elif hero_score == best_opp:
            ties = 1 + sum(1 for s in opp_scores if s == hero_score)
            points += 1.0 / ties
        trials += 1
    if trials == 0:
        # Fallback: preflop-score-ish equity approximation.
        return clamp(0.15 + preflop_score_for_key(normalize_hand(hero_cards)) * 0.65, 0.05, 0.9)
    return clamp(points / trials, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Action formatting and sizing
# ---------------------------------------------------------------------------

def max_total_bet(state):
    return safe_int(state.get("your_stack"), 0) + safe_int(state.get("your_bet_this_street"), 0)


def legal_action(state, action, amount=None):
    can_check = bool(state.get("can_check", False)) or safe_int(state.get("amount_owed"), 0) <= 0
    stack = safe_int(state.get("your_stack"), 0)
    if stack <= 0:
        return {"action": "check"} if can_check else {"action": "fold"}
    action = normalize_action_name(action)
    if action == "fold":
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "check":
        return {"action": "check"} if can_check else {"action": "fold"}
    if action == "call":
        return {"action": "check"} if can_check else {"action": "call"}
    if action == "all_in":
        return {"action": "all_in"}
    if action in ("raise", "bet"):
        min_raise_to = safe_int(state.get("min_raise_to"), 0)
        max_to = max_total_bet(state)
        if max_to <= 0:
            return {"action": "check"} if can_check else {"action": "fold"}
        amt = safe_int(amount, min_raise_to)
        if min_raise_to <= 0 or amt >= max_to * 0.985:
            return {"action": "all_in"}
        amt = max(amt, min_raise_to)
        amt = min(amt, max_to)
        if amt <= safe_int(state.get("current_bet"), 0):
            return {"action": "call"} if not can_check else {"action": "check"}
        return {"action": "raise", "amount": int(amt)}
    return {"action": "check"} if can_check else {"action": "fold"}


def open_raise_size(state, pos, limpers=0):
    bb = estimate_big_blind(state)
    if pos == "SB":
        return int(round(bb * 3.0))
    if limpers:
        return int(round(bb * (4.0 + limpers)))
    return int(round(bb * 2.5))


def three_bet_size(state, in_position, callers=0):
    current = max(estimate_big_blind(state), safe_int(state.get("current_bet"), 0))
    factor = 3.5 if in_position else 4.0
    # Squeeze larger when callers are in between.
    return int(round(current * factor + callers * current))


def four_bet_size(state, in_position):
    current = max(estimate_big_blind(state), safe_int(state.get("current_bet"), 0))
    factor = 2.3 if in_position else 2.5
    return int(round(current * factor))


def postflop_bet_sizes(state, level, draws, spr, texture):
    pot = max(1, safe_int(state.get("pot"), 1))
    current_bet = safe_int(state.get("current_bet"), 0)
    owed = safe_int(state.get("amount_owed"), 0)
    hero_bet = safe_int(state.get("your_bet_this_street"), 0)
    max_to = max_total_bet(state)
    sizes = []
    if spr <= 1.6 and (level >= 3 or draws.get("flush_draw") or draws.get("straight_draw")):
        return [max_to]
    if owed <= 0:
        # Bet sizes as total bet amounts for this street.
        if texture["wetness"] < 0.35:
            sizes.append(hero_bet + int(round(pot * 0.33)))
        sizes.append(hero_bet + int(round(pot * 0.66)))
        if level >= 6 or (state.get("street") == "river" and level >= 5):
            sizes.append(hero_bet + int(round(pot * 0.95)))
    else:
        # Raise total. Pot after calling would be pot + owed; standard raise is
        # roughly pot-sized over the bet, capped by stack.
        raise_total = current_bet + int(round(max(2 * owed, 0.75 * pot)))
        sizes.append(raise_total)
        if spr <= 2.5 and (level >= 4 or draws.get("nut_flush_draw")):
            sizes.append(max_to)
    min_to = safe_int(state.get("min_raise_to"), 0)
    out = []
    for s in sizes:
        s = max(s, min_to)
        s = min(s, max_to)
        if s > current_bet and s > hero_bet and s not in out:
            out.append(int(s))
    return out[:3]


# ---------------------------------------------------------------------------
# Preflop policy: chart-first, 100bb-aware fallback
# ---------------------------------------------------------------------------

def preflop_decision(state, rng):
    cards = clean_cards(state.get("your_cards"))
    key = normalize_hand(cards)
    if not key:
        return legal_action(state, "check" if state.get("can_check") else "fold")
    pos = hero_position(state)
    summary = preflop_summary(state)
    bb = summary["bb"]
    amount_owed = safe_int(state.get("amount_owed"), 0)
    can_check = bool(state.get("can_check", False)) or amount_owed <= 0
    current_bet = safe_int(state.get("current_bet"), 0)
    max_to = max_total_bet(state)
    score = preflop_score_for_key(key)
    raise_count = summary["raise_count"]
    total_raise_count = summary["total_raise_count"]
    limpers = summary["call_count"]

    # BB option after limps: raise the charted isolation range, otherwise check.
    if can_check and pos == "BB" and limpers > 0 and raise_count == 0:
        if key in BB_VS_SB_LIMP_RAISE or score > 0.72:
            return legal_action(state, "raise", open_raise_size(state, "BB", limpers))
        return legal_action(state, "check")

    # Unopened or limped pot: RFI / isolate.
    unopened = raise_count == 0 and current_bet <= bb * 1.25 and amount_owed <= bb * 1.25
    if unopened:
        if pos == "SB":
            if key in SB_RAISE:
                return legal_action(state, "raise", open_raise_size(state, "SB", limpers))
            if key in SB_LIMP or amount_owed <= bb * 0.6:
                return legal_action(state, "call" if not can_check else "check")
            return legal_action(state, "fold")
        rfi = RFI_RANGES.get(pos, RFI_RANGES.get("CO"))
        # Isolation is tighter than pure RFI, especially out of position.
        should_raise = key in rfi and (limpers == 0 or score > (0.46 if pos in ("BTN", "CO") else 0.55))
        if should_raise:
            return legal_action(state, "raise", open_raise_size(state, pos, limpers))
        # Overlimp cheap with pairs/suited connectors in position, but do not
        # overdo implied-odds calls at 100bb.
        if limpers and amount_owed <= bb and (len(key) == 2 or key in {"A5s", "KQs", "QJs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s"}):
            return legal_action(state, "call")
        return legal_action(state, "check" if can_check else "fold")

    # Facing one open raise: use explicit 100bb chart when possible.
    if total_raise_count <= 1 or raise_count == 1:
        opener = summary["first_raiser_pos"] or "LJ"
        chart_key = "%s_vs_%s" % (pos, opener)
        chart = VS_OPEN.get(chart_key)
        ip = is_in_position(pos, opener)
        if chart:
            if key in chart.get("3bet", set()):
                return legal_action(state, "raise", three_bet_size(state, ip, max(0, limpers - 1)))
            if key in chart.get("call", set()):
                return legal_action(state, "call")
            return legal_action(state, "check" if can_check else "fold")
        # Fallback if the exact position chart is unavailable.
        if key in VALUE_4BET or score >= 0.82:
            return legal_action(state, "raise", three_bet_size(state, ip, max(0, limpers - 1)))
        if amount_owed <= bb * 3.0 and (key in CALL_VS_3BET_IP or (pos == "BB" and score > 0.42)):
            return legal_action(state, "call")
        return legal_action(state, "check" if can_check else "fold")

    # Facing a 3bet/4bet. At 100bb, avoid speculative flats and move toward
    # 4bet/call or 5bet-jam decisions.
    ip = pos not in ("SB", "BB")
    price_bb = amount_owed / float(max(1, bb))
    current_bb = current_bet / float(max(1, bb))
    if total_raise_count >= 3 or current_bb >= 22:
        if key in PREMIUM_5BET:
            return legal_action(state, "all_in" if max_to <= current_bet * 3 else "raise", max_to)
        # QQ/AK sometimes continue to large 4bets depending on price.
        if key in {"QQ", "AKs", "AKo"} and price_bb <= 28:
            return legal_action(state, "call")
        return legal_action(state, "fold")
    if key in VALUE_4BET:
        return legal_action(state, "raise", four_bet_size(state, ip))
    if key in BLUFF_4BET and current_bb <= 12 and rng.random() < 0.35:
        return legal_action(state, "raise", four_bet_size(state, ip))
    call_set = CALL_VS_3BET_IP if ip else CALL_VS_3BET_OOP
    if key in call_set and price_bb <= (9.0 if ip else 7.0):
        return legal_action(state, "call")
    return legal_action(state, "check" if can_check else "fold")


# ---------------------------------------------------------------------------
# Postflop EV search
# ---------------------------------------------------------------------------

def effective_spr(state):
    pot = max(1, safe_int(state.get("pot"), 1))
    stack = safe_int(state.get("your_stack"), 0)
    opp_stacks = [safe_int(p.get("stack"), stack) for p in opponent_players(state)]
    eff = min([stack] + opp_stacks) if opp_stacks else stack
    return eff / float(pot)


def average_opponent_model(state):
    opps = opponent_players(state)
    if not opps:
        return OpponentModel()
    # Return a synthetic model by combining counts.
    m = OpponentModel()
    for p in opps:
        src = OPPONENTS[player_id_for_seat(state, safe_int(p.get("seat"), -999))]
        for st, counts in src.actions.items():
            for a, n in counts.items():
                m.actions[st][a] += n
                m.total_actions += n
        m.preflop_vpip += src.preflop_vpip
        m.preflop_raises += src.preflop_raises
        for st, n in src.response_fold.items():
            m.response_fold[st] += n
        for st, n in src.response_call.items():
            m.response_call[st] += n
        for st, n in src.response_raise.items():
            m.response_raise[st] += n
    return m


def build_current_ranges(state, board):
    known = clean_cards(state.get("your_cards")) + clean_cards(board)
    ranges = []
    for p in opponent_players(state):
        seat = safe_int(p.get("seat"), -999)
        if seat == -999:
            continue
        ranges.append(build_range_for_seat(state, seat, known, board))
    return ranges


def equity_sample_count(street, opp_count):
    if eval7 is None:
        base = {"flop": 90, "turn": 120, "river": 160}.get(street, 80)
    else:
        base = {"flop": 180, "turn": 240, "river": 280}.get(street, 120)
    if opp_count >= 3:
        base = int(base * 0.55)
    elif opp_count == 2:
        base = int(base * 0.75)
    return max(40, base)


def realization_factor(state, equity, level, draws, spr, in_pos):
    street = state.get("street")
    factor = 1.0
    if street == "flop":
        factor -= 0.10
    elif street == "turn":
        factor -= 0.04
    if not in_pos:
        factor -= 0.08
    if level >= 4:
        factor += 0.08
    if draws.get("nut_flush_draw") or (draws.get("flush_draw") and draws.get("straight_draw")):
        factor += 0.05
    if spr <= 3.0 and equity >= 0.48:
        factor += 0.07
    if len(opponent_players(state)) >= 2:
        factor -= 0.08
    return clamp(factor, 0.55, 1.12)


def estimate_response_probs(state, bet_to, ranges, level, draws, texture, model):
    pot = max(1, safe_int(state.get("pot"), 1))
    hero_bet = safe_int(state.get("your_bet_this_street"), 0)
    cost = max(1, bet_to - hero_bet)
    bet_frac = cost / float(max(1, pot))
    street = state.get("street", "flop")
    opps = opponent_players(state)
    if not opps:
        return 1.0, 0.0, 0.0
    per_fold = []
    per_raise = []
    for p in opps:
        pm = OPPONENTS[player_id_for_seat(state, safe_int(p.get("seat"), -999))]
        f = pm.fold_rate(street)
        # Size pressure and board texture adjustments.
        f += (bet_frac - 0.55) * 0.18
        f += 0.06 if texture["wetness"] < 0.3 else -0.04 if texture["wetness"] > 0.65 else 0.0
        f -= 0.05 if pm.looseness() > 0.42 else 0.0
        if level >= 5:
            # Strong hero value hands tend to unblock folds less; keep conservative.
            f -= 0.02
        per_fold.append(clamp(f, 0.08, 0.78))
        r = pm.aggression(street) * 0.10 + (0.04 if texture["wetness"] > 0.6 else 0.0)
        if bet_frac > 1.0:
            r *= 0.55
        per_raise.append(clamp(r, 0.01, 0.22))
    p_all_fold = 1.0
    for f in per_fold:
        p_all_fold *= f
    p_any_raise = 1.0
    for r in per_raise:
        p_any_raise *= (1.0 - r)
    p_any_raise = 1.0 - p_any_raise
    # A raise cannot happen if everyone folded. Keep probabilities normalized.
    p_raise = min(1.0 - p_all_fold, p_any_raise * (1.0 - p_all_fold))
    p_call = max(0.0, 1.0 - p_all_fold - p_raise)
    return clamp(p_all_fold, 0.0, 1.0), clamp(p_call, 0.0, 1.0), clamp(p_raise, 0.0, 1.0)


def expected_call_addition(state, bet_to):
    opps = opponent_players(state)
    if not opps:
        return 0
    vals = []
    for p in opps:
        opp_bet = safe_int(p.get("bet_this_street"), 0)
        opp_stack = safe_int(p.get("stack"), 0)
        vals.append(max(0, min(opp_stack, bet_to - opp_bet)))
    return int(sum(vals) / max(1, len(vals)))


def evaluate_bet_ev(state, bet_to, hero_cards, board, ranges, base_equity, level, draws, texture, rng, deadline):
    pot = max(1, safe_int(state.get("pot"), 1))
    hero_bet = safe_int(state.get("your_bet_this_street"), 0)
    bet_cost = max(0, bet_to - hero_bet)
    if bet_cost <= 0:
        return -10**9
    model = average_opponent_model(state)
    p_fold, p_call, p_raise = estimate_response_probs(state, bet_to, ranges, level, draws, texture, model)
    call_ranges = [response_adjusted_range(r, board, "call") for r in ranges]
    raise_ranges = [response_adjusted_range(r, board, "raise") for r in ranges]
    street = state.get("street", "flop")
    samples = max(35, equity_sample_count(street, len(ranges)) // 2)
    eq_called = estimate_equity(hero_cards, board, call_ranges, samples, rng, deadline) if p_call > 0.01 else base_equity
    called_pot = pot + bet_cost + expected_call_addition(state, bet_to)
    ev_called = eq_called * called_pot - bet_cost

    # If villain raises, we estimate whether our continuing range can stand it.
    # Strong made hands and nut draws continue; weak bluffs lose the bet.
    can_continue = level >= 5 or (level >= 3 and effective_spr(state) <= 2.2) or draws.get("nut_flush_draw") or (draws.get("flush_draw") and draws.get("straight_draw"))
    if p_raise > 0.01 and can_continue:
        eq_raised = estimate_equity(hero_cards, board, raise_ranges, max(25, samples // 2), rng, deadline)
        extra_cost = min(safe_int(state.get("your_stack"), 0), max(0, called_pot // 2))
        ev_raised = eq_raised * (called_pot + extra_cost) - bet_cost - extra_cost * 0.75
    else:
        ev_raised = -bet_cost
    return p_fold * pot + p_call * ev_called + p_raise * ev_raised


def postflop_decision(state, rng, start_time):
    deadline = start_time + 1.72  # leave margin under 2s timeout
    hero_cards = clean_cards(state.get("your_cards"))
    board = clean_cards(state.get("community_cards"))
    amount_owed = safe_int(state.get("amount_owed"), 0)
    can_check = bool(state.get("can_check", False)) or amount_owed <= 0
    pot = max(1, safe_int(state.get("pot"), 1))
    street = state.get("street", "flop")
    opp_count = len(opponent_players(state))
    if len(hero_cards) < 2:
        return legal_action(state, "check" if can_check else "fold")

    # Cheap multiway fallback. Full recursive multiway search is too expensive;
    # this remains robust rather than spewy.
    level = made_hand_level(hero_cards, board)
    draws = draw_flags(hero_cards, board)
    texture = board_texture(board)
    spr = effective_spr(state)
    if opp_count >= 4:
        if can_check:
            if level >= 5 or (level >= 4 and texture["wetness"] > 0.45):
                return legal_action(state, "raise", postflop_bet_sizes(state, level, draws, spr, texture)[0])
            return legal_action(state, "check")
        pot_odds = amount_owed / float(pot + amount_owed)
        if level >= 5 or (level >= 3 and amount_owed <= pot * 0.35) or (draws.get("flush_draw") and pot_odds < 0.24):
            return legal_action(state, "call")
        return legal_action(state, "fold")

    ranges = build_current_ranges(state, board)
    samples = equity_sample_count(street, opp_count)
    equity = estimate_equity(hero_cards, board, ranges, samples, rng, deadline)
    pos = hero_position(state)
    in_pos = pos in ("BTN", "CO") or (pos == "BB" and opp_count == 1 and preflop_summary(state).get("first_raiser_pos") == "SB")
    realization = realization_factor(state, equity, level, draws, spr, in_pos)

    # EV of passive line: check if free, otherwise call.
    if can_check:
        passive_ev = equity * realization * pot
    else:
        passive_ev = equity * realization * (pot + amount_owed) - amount_owed

    # EV of aggressive line: test a few action abstraction sizes.
    bet_evs = []
    for bet_to in postflop_bet_sizes(state, level, draws, spr, texture):
        if time.monotonic() >= deadline:
            break
        ev = evaluate_bet_ev(state, bet_to, hero_cards, board, ranges, equity, level, draws, texture, rng, deadline)
        bet_evs.append((ev, bet_to))
    best_bet_ev, best_bet_to = (-10**9, None)
    if bet_evs:
        best_bet_ev, best_bet_to = max(bet_evs, key=lambda x: x[0])

    # Strategic guardrails around the EV estimates. They keep the bot from making
    # very thin high-variance plays when ranges/samples are noisy.
    value_bet_hand = level >= 3 or (level == 2 and equity >= 0.55)
    raise_value_hand = level >= 5 or (level >= 4 and spr <= 5.5) or (level >= 3 and spr <= 2.6)
    strong_value = raise_value_hand
    strong_draw = draws.get("nut_flush_draw") or (draws.get("flush_draw") and draws.get("straight_draw")) or (draws.get("open_ended") and draws.get("overcards") >= 1)
    medium_showdown = level in (2, 3, 4)
    air_bluff_candidate = (
        level <= 1
        and (draws.get("overcards") >= 2 or draws.get("backdoor_flush") or draws.get("straight_draw"))
        and texture["wetness"] < 0.65
        and rng.random() < (0.18 if can_check else 0.10)
    )

    margin = max(4.0, pot * 0.035)
    if best_bet_to is not None and best_bet_ev > passive_ev + margin:
        if (value_bet_hand if can_check else raise_value_hand) or strong_draw or air_bluff_candidate:
            return legal_action(state, "raise", best_bet_to)

    if can_check:
        # Thin value/protection when EV is close but hand benefits from denial.
        if best_bet_to is not None and (value_bet_hand or strong_draw) and best_bet_ev > passive_ev - pot * 0.08:
            return legal_action(state, "raise", best_bet_to)
        return legal_action(state, "check")

    # Facing a bet: raise only clear value/semi-bluffs, otherwise call/fold by EV.
    if best_bet_to is not None and best_bet_ev > passive_ev + margin * 1.5 and (raise_value_hand or strong_draw):
        return legal_action(state, "raise", best_bet_to)

    pot_odds = amount_owed / float(max(1, pot + amount_owed))
    required = pot_odds + (0.04 if not in_pos else 0.015)
    if passive_ev > 0 or equity * realization >= required:
        if medium_showdown or raise_value_hand or strong_draw or equity > required + 0.05:
            return legal_action(state, "call")
    return legal_action(state, "fold")


# ---------------------------------------------------------------------------
# Main Fullhouse API
# ---------------------------------------------------------------------------

def decide(game_state: dict) -> dict:
    """
    Fullhouse calls this once per action. Return one legal action dict:
      {"action":"fold"}, {"action":"check"}, {"action":"call"},
      {"action":"raise", "amount": total_bet}, or {"action":"all_in"}.
    """
    start = time.monotonic()
    state = game_state if isinstance(game_state, dict) else {}
    try:
        update_opponent_models(state)
        rng = state_rng(state)
        street = str(state.get("street", "preflop")).lower()
        # Defensive card sanity. Never crash because of a malformed state.
        if len(clean_cards(state.get("your_cards"))) < 2:
            return legal_action(state, "check" if state.get("can_check") else "fold")
        if street == "preflop":
            return preflop_decision(state, rng)
        return postflop_decision(state, rng, start)
    except Exception:
        # Crashes auto-fold in the engine. Returning safely is better.
        return legal_action(state, "check" if state.get("can_check") else "fold")