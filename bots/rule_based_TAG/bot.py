"""BlackRain79-inspired rule-based TAG bot for Fullhouse poker."""

BOT_NAME = "BlackRain TAG"
BOT_AVATAR = "robot_1"

RANKS = "23456789TJQKA"
RANK_VALUE = {rank: i + 2 for i, rank in enumerate(RANKS)}
HIGH_CARDS = {"A", "K", "Q", "J", "T"}

PAIR_ORDER = ["22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA"]
FULL_OPEN_RANGE = set(PAIR_ORDER) | {
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "AKo", "AQo", "AJo", "ATo",
    "KQs", "KJs", "KTs",
    "QJs", "QTs",
    "JTs", "T9s", "98s", "87s", "76s", "65s",
}
EP_OPEN_RANGE = {"AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AQs", "AJs", "KQs", "AKo", "AQo"}
MP_OPEN_RANGE = EP_OPEN_RANGE | {"77", "66", "55", "ATs", "KJs", "QJs", "AJo", "KQo", "JTs", "T9s"}
SB_OPEN_RANGE = EP_OPEN_RANGE | {"77", "66", "AJo", "ATo", "KJs", "KTs", "QJs", "JTs"}
BTN_EXTRA_RANGE = {"A9o", "KQo", "KJo", "QJo", "K9s", "Q9s", "J9s", "T8s", "97s", "86s", "75s", "54s"}

VALUE_3BET = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo"}
STEAL_3BET = VALUE_3BET | {"AJs", "AJo", "ATs", "ATo", "A9s", "KQs", "KQo", "88", "77", "JTs"}
FLAT_CALL_RANGE = {
    "22", "33", "44", "55", "66", "77", "88", "99", "TT",
    "JTs", "T9s", "98s", "87s", "76s",
    "AJs", "AJo", "ATs", "ATo", "KQs", "KQo", "KJs", "KJo", "QJs", "QJo",
}

HAND_STATE = {}


def _active_players(state):
    return [p for p in state["players"] if not p.get("is_folded") and p.get("stack", 0) > 0 or p.get("is_all_in")]


def _my_bot_id(state):
    seat = state["seat_to_act"]
    return state["players"][seat].get("bot_id", "me")


def _combo(cards):
    ranks = sorted((cards[0][0], cards[1][0]), key=lambda rank: RANK_VALUE[rank], reverse=True)
    if ranks[0] == ranks[1]:
        return ranks[0] + ranks[1]
    suited = cards[0][1] == cards[1][1]
    return ranks[0] + ranks[1] + ("s" if suited else "o")


def _pair_rank(combo):
    return RANK_VALUE[combo[0]] if len(combo) == 2 and combo[0] == combo[1] else 0


def _blind_amounts(state):
    small_blind = 50
    big_blind = 100
    for entry in state.get("action_log", []):
        if entry.get("action") == "small_blind":
            small_blind = max(1, int(entry.get("amount") or small_blind))
        elif entry.get("action") == "big_blind":
            big_blind = max(2, int(entry.get("amount") or big_blind))
    return small_blind, big_blind


def _blind_seats(state):
    small_blind = None
    big_blind = None
    for entry in state.get("action_log", []):
        if entry.get("action") == "small_blind":
            small_blind = entry.get("seat")
        elif entry.get("action") == "big_blind":
            big_blind = entry.get("seat")
    return small_blind, big_blind


def _position(state):
    seat = state["seat_to_act"]
    players = state["players"]
    n_players = max(2, len(players))
    small_blind, big_blind = _blind_seats(state)
    if small_blind is None:
        small_blind = 0
    if big_blind is None:
        big_blind = 1 if n_players > 1 else 0
    if seat == small_blind:
        return "SB"
    if seat == big_blind:
        return "BB"

    if state["street"] != "preflop":
        dealer = small_blind if n_players == 2 else ((small_blind - 1) % n_players if small_blind is not None else 0)
        order = [((dealer + offset) % n_players) for offset in range(1, n_players + 1)]
    else:
        start = small_blind if n_players == 2 else ((big_blind + 1) % n_players if big_blind is not None else 0)
        order = [((start + offset) % n_players) for offset in range(n_players)]

    live_order = [s for s in order if not players[s].get("is_folded")]
    idx = live_order.index(seat) if seat in live_order else 0
    if n_players <= 2:
        return "BTN" if seat == small_blind else "BB"
    if idx >= len(live_order) - 1:
        return "BTN"
    if idx == len(live_order) - 2:
        return "CO"
    if idx <= 1:
        return "EP"
    return "MP"


def _open_range_for(position):
    if position in ("BTN", "CO"):
        return FULL_OPEN_RANGE | BTN_EXTRA_RANGE
    if position == "MP":
        return MP_OPEN_RANGE
    if position == "SB":
        return SB_OPEN_RANGE
    if position == "BB":
        return FULL_OPEN_RANGE
    return EP_OPEN_RANGE


def _raise_to(state, target):
    stack_total = state["your_stack"] + state["your_bet_this_street"]
    if stack_total <= state["amount_owed"]:
        return {"action": "all_in"}
    amount = max(int(target), int(state["min_raise_to"]))
    amount = min(amount, stack_total)
    if amount <= state["your_bet_this_street"] + state["amount_owed"]:
        return {"action": "call"} if not state["can_check"] else {"action": "check"}
    return {"action": "raise", "amount": amount}


def _bet_fraction(state, fraction):
    target = state["your_bet_this_street"] + max(state["min_raise_to"], int(max(1, state["pot"]) * fraction))
    return _raise_to(state, target)


def _facing_raise_preflop(state, big_blind):
    return state["current_bet"] > big_blind or state["amount_owed"] > big_blind - state["your_bet_this_street"]


def _last_preflop_raiser_seat(state):
    raiser = None
    for entry in state.get("action_log", []):
        if entry.get("action") in ("raise", "all_in"):
            raiser = entry.get("seat")
    return raiser


def _is_late_steal(state):
    raiser = _last_preflop_raiser_seat(state)
    if raiser is None:
        return False
    return raiser in _late_seats(state)


def _late_seats(state):
    players = state["players"]
    n_players = len(players)
    small_blind, _ = _blind_seats(state)
    if small_blind is None:
        small_blind = 0
    dealer = small_blind if n_players == 2 else ((small_blind - 1) % n_players if small_blind is not None else 0)
    cutoff = (dealer - 1) % n_players
    return {dealer, cutoff}


def _in_position(state):
    return _position(state) in ("CO", "BTN")


def _opponents_in_hand(state):
    my_seat = state["seat_to_act"]
    return [
        p for p in state["players"]
        if p.get("seat") != my_seat and not p.get("is_folded") and (p.get("stack", 0) > 0 or p.get("is_all_in"))
    ]


def _opponent_type(state):
    hero = _my_bot_id(state)
    stats = {}
    for item in state.get("match_action_log", []):
        bot_id = item.get("bot_id")
        if not bot_id or bot_id == hero:
            continue
        data = stats.setdefault(bot_id, {"actions": 0, "calls": 0, "raises": 0, "folds": 0})
        action = item.get("action")
        if action in ("call", "raise", "all_in", "fold", "check"):
            data["actions"] += 1
        if action == "call":
            data["calls"] += 1
        elif action in ("raise", "all_in"):
            data["raises"] += 1
        elif action == "fold":
            data["folds"] += 1

    opponents = [p.get("bot_id") for p in _opponents_in_hand(state)]
    labels = []
    for bot_id in opponents:
        data = stats.get(bot_id)
        if not data or data["actions"] < 20:
            labels.append("unknown")
            continue
        calls = data["calls"] / max(1, data["actions"])
        raises = data["raises"] / max(1, data["actions"])
        folds = data["folds"] / max(1, data["actions"])
        if calls > 0.42 and raises < 0.16:
            labels.append("calling_station")
        elif folds > 0.42 and raises < 0.18:
            labels.append("tight_weak")
        elif raises > 0.28:
            labels.append("aggressive_reg")
        else:
            labels.append("unknown")

    if "calling_station" in labels:
        return "calling_station"
    if "aggressive_reg" in labels:
        return "aggressive_reg"
    if labels and all(label == "tight_weak" for label in labels):
        return "tight_weak"
    return "unknown"


def _hand_type(cards):
    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]
    counts = sorted((ranks.count(rank) for rank in set(ranks)), reverse=True)
    flush = any(suits.count(suit) >= 5 for suit in set(suits))
    straight = _has_straight(ranks)

    if flush and straight:
        return "straight flush"
    if counts and counts[0] == 4:
        return "four of a kind"
    if len(counts) >= 2 and counts[0] == 3 and counts[1] >= 2:
        return "full house"
    if flush:
        return "flush"
    if straight:
        return "straight"
    if counts and counts[0] == 3:
        return "three of a kind"
    if len(counts) >= 2 and counts[0] == 2 and counts[1] == 2:
        return "two pair"
    if counts and counts[0] == 2:
        return "pair"
    return "high card"


def _has_straight(ranks):
    values = {RANK_VALUE[rank] for rank in ranks}
    if 14 in values:
        values.add(1)
    for start in range(1, 11):
        if all(value in values for value in range(start, start + 5)):
            return True
    return False


def _has_open_ended_draw(ranks):
    values = {RANK_VALUE[rank] for rank in ranks}
    if 14 in values:
        values.add(1)
    for start in range(1, 12):
        if all(value in values for value in range(start, start + 4)):
            return True
    return False


def _has_gutshot(ranks):
    values = {RANK_VALUE[rank] for rank in ranks}
    if 14 in values:
        values.add(1)
    for start in range(1, 11):
        present = sum(1 for value in range(start, start + 5) if value in values)
        if present == 4:
            return True
    return False


def _board_texture(board):
    if len(board) < 3:
        return "dry"
    ranks = [card[0] for card in board]
    suits = [card[1] for card in board]
    values = sorted({RANK_VALUE[rank] for rank in ranks})
    max_suit = max(suits.count(suit) for suit in set(suits))
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    connected = len(values) >= 3 and (max(values) - min(values) <= 4 or sum(1 for gap in gaps if gap <= 2) >= 2)
    broadway_heavy = sum(1 for rank in ranks if rank in HIGH_CARDS) >= 2
    two_tone = max_suit >= 2
    return "wet" if connected or two_tone or broadway_heavy else "dry"


def _hand_info(state):
    hole = state["your_cards"]
    board = state["community_cards"]
    all_cards = hole + board
    hand_type = _hand_type(all_cards) if board else "high card"
    ranks = [card[0] for card in all_cards]
    board_ranks = [card[0] for card in board]
    hole_ranks = [card[0] for card in hole]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    board_top = max((RANK_VALUE[rank] for rank in board_ranks), default=0)
    hole_values = sorted((RANK_VALUE[rank] for rank in hole_ranks), reverse=True)

    made = {
        "high card": 0,
        "pair": 1,
        "two pair": 2,
        "trips": 3,
        "three of a kind": 3,
        "straight": 4,
        "flush": 5,
        "full house": 6,
        "quads": 7,
        "four of a kind": 7,
        "straight flush": 8,
    }.get(hand_type, 0)

    pocket_pair = hole_ranks[0] == hole_ranks[1]
    top_pair = any(RANK_VALUE[rank] == board_top and rank_counts.get(rank, 0) >= 2 for rank in hole_ranks)
    overpair = pocket_pair and board_top and RANK_VALUE[hole_ranks[0]] > board_top
    second_pair = any(rank_counts.get(rank, 0) >= 2 and RANK_VALUE[rank] < board_top for rank in hole_ranks)
    set_made = pocket_pair and any(rank == hole_ranks[0] for rank in board_ranks) and made >= 3
    tptk = top_pair and hole_values and max(value for value in hole_values if value != board_top) >= 12

    suit_counts = {}
    for card in all_cards:
        suit_counts[card[1]] = suit_counts.get(card[1], 0) + 1
    flush_draw = made < 5 and max(suit_counts.values() or [0]) >= 4 and len(board) < 5
    oesd = not _has_straight(ranks) and _has_open_ended_draw(ranks) and len(board) < 5
    gutshot = not oesd and not _has_straight(ranks) and _has_gutshot(ranks) and len(board) < 5
    overcards = bool(board) and made == 0 and sum(1 for value in hole_values if value > board_top) >= 1

    if made >= 4 or set_made or made >= 2:
        strength = "monster"
    elif overpair or tptk:
        strength = "tptk_plus"
    elif top_pair:
        strength = "top_pair"
    elif made == 1 or second_pair:
        strength = "medium_pair"
    elif flush_draw or oesd:
        strength = "strong_draw"
    elif gutshot or overcards:
        strength = "weak_equity"
    else:
        strength = "air"

    return {
        "made": made,
        "type": hand_type,
        "strength": strength,
        "top_pair": top_pair,
        "tptk": tptk,
        "overpair": overpair,
        "set": set_made,
        "flush_draw": flush_draw,
        "oesd": oesd,
        "gutshot": gutshot,
        "overcards": overcards,
    }


def _street_state(state):
    hand_id = state.get("hand_id", "unknown")
    data = HAND_STATE.setdefault(hand_id, {"raised_preflop": False, "flop_bet": False, "turn_bet": False})
    if len(HAND_STATE) > 40:
        for key in list(HAND_STATE.keys())[:-20]:
            HAND_STATE.pop(key, None)
    return data


def _record_and_return(state, action):
    data = _street_state(state)
    street = state["street"]
    if action.get("action") in ("raise", "all_in"):
        if street == "preflop":
            data["raised_preflop"] = True
        elif street == "flop":
            data["flop_bet"] = True
        elif street == "turn":
            data["turn_bet"] = True
    return action


def _preflop_decision(state):
    combo = _combo(state["your_cards"])
    position = _position(state)
    _, big_blind = _blind_amounts(state)
    facing_raise = _facing_raise_preflop(state, big_blind)
    stack_total = state["your_stack"] + state["your_bet_this_street"]

    if not facing_raise:
        if combo in _open_range_for(position):
            return _raise_to(state, min(stack_total, big_blind * 3))
        return {"action": "check"} if state["can_check"] else {"action": "fold"}

    if combo in (STEAL_3BET if _is_late_steal(state) else VALUE_3BET):
        target = max(state["min_raise_to"], state["current_bet"] * 3)
        return _raise_to(state, target)

    ep_open = not _is_late_steal(state)
    if ep_open and combo in {"JJ", "TT", "AQs", "AQo"} and state["amount_owed"] <= state["pot"] * 0.45:
        return {"action": "call"}

    in_position = _in_position(state) or position == "BB"
    pair = _pair_rank(combo)
    set_mine_price = state["amount_owed"] <= min(state["your_stack"] * 0.08, state["pot"] * 0.40)
    if in_position and combo in FLAT_CALL_RANGE:
        if pair and pair <= RANK_VALUE["T"] and set_mine_price:
            return {"action": "call"}
        if not pair and state["amount_owed"] <= state["pot"] * 0.32:
            return {"action": "call"}

    if position == "BB" and combo in FULL_OPEN_RANGE and state["amount_owed"] <= state["pot"] * 0.25:
        return {"action": "call"}

    return {"action": "check"} if state["can_check"] else {"action": "fold"}


def _facing_bet_decision(state, info, texture, opponent):
    owed = state["amount_owed"]
    pot = max(1, state["pot"])
    price = owed / pot

    if info["strength"] in ("monster", "tptk_plus"):
        return _bet_fraction(state, 0.75)

    if info["strength"] == "top_pair":
        if texture == "wet" and price > 0.45:
            return {"action": "fold"} if opponent == "aggressive_reg" else {"action": "call"}
        return {"action": "call"}

    if info["strength"] in ("medium_pair", "strong_draw"):
        if price <= (0.36 if info["strength"] == "strong_draw" else 0.24):
            return {"action": "call"}
        return {"action": "fold"}

    if info["gutshot"] and price <= 0.12:
        return {"action": "call"}

    return {"action": "fold"}


def _postflop_decision(state):
    data = _street_state(state)
    street = state["street"]
    info = _hand_info(state)
    texture = _board_texture(state["community_cards"])
    opponent = _opponent_type(state)
    opponents = len(_opponents_in_hand(state))
    heads_up = opponents <= 1
    can_bluff = opponent != "calling_station" and opponents <= 1

    if not state["can_check"]:
        return _facing_bet_decision(state, info, texture, opponent)

    if street == "flop":
        if info["strength"] in ("monster", "tptk_plus"):
            return _bet_fraction(state, 0.70)
        if heads_up and data.get("raised_preflop"):
            if info["strength"] in ("top_pair", "medium_pair", "strong_draw"):
                return _bet_fraction(state, 0.50)
            if can_bluff and texture == "dry" and (info["overcards"] or opponent == "tight_weak"):
                return _bet_fraction(state, 0.50)
        if not heads_up and info["strength"] in ("monster", "tptk_plus", "strong_draw"):
            return _bet_fraction(state, 0.60)
        return {"action": "check"}

    if street == "turn":
        scare = bool(state["community_cards"] and state["community_cards"][-1][0] in {"A", "K"})
        if info["strength"] in ("monster", "tptk_plus"):
            return _bet_fraction(state, 0.75)
        if data.get("flop_bet") and info["strength"] in ("top_pair", "strong_draw"):
            return _bet_fraction(state, 0.66)
        if data.get("flop_bet") and can_bluff and texture == "dry" and scare and opponent == "tight_weak":
            return _bet_fraction(state, 0.66)
        return {"action": "check"}

    if street == "river":
        if info["strength"] in ("monster", "tptk_plus"):
            return _bet_fraction(state, 0.80)
        if info["strength"] == "top_pair" and opponent == "calling_station" and texture == "dry":
            return _bet_fraction(state, 0.60)
        return {"action": "check"}

    return {"action": "check"}


def decide(state):
    if state.get("type") == "warmup":
        return {"action": "check"}
    try:
        if state["street"] == "preflop":
            action = _preflop_decision(state)
        else:
            action = _postflop_decision(state)
        return _record_and_return(state, action)
    except Exception:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}
