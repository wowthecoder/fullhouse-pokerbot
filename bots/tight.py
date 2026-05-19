"""Baseline bot: plays only strong hands."""

BOT_NAME = "Tight Bot"

PREMIUM_COMBOS = {
    ("A", "K"),
    ("A", "Q"),
    ("K", "Q"),
}

HIGH_PAIRS = {"A", "K", "Q", "J", "T"}
RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def _rank_combo(cards: list) -> tuple:
    ranks = [card[0] for card in cards]
    return tuple(sorted(ranks, key=lambda rank: RANK_VALUE[rank], reverse=True))


def _is_premium(cards: list) -> bool:
    ranks = _rank_combo(cards)
    suited = cards[0][1] == cards[1][1]

    if ranks[0] == ranks[1]:
        return ranks[0] in HIGH_PAIRS

    if ranks in PREMIUM_COMBOS:
        return True

    return suited and ranks[0] == "A" and ranks[1] in {"J", "T"}


def decide(state: dict) -> dict:
    premium = _is_premium(state["your_cards"])

    if state["street"] == "preflop":
        if premium:
            raise_to = min(state["min_raise_to"] * 3, state["your_stack"] + state["your_bet_this_street"])
            return {"action": "raise", "amount": raise_to}

        if state["can_check"]:
            return {"action": "check"}

        return {"action": "fold"}

    if state["can_check"]:
        return {"action": "check"}

    owed = state["amount_owed"]
    pot = max(state["pot"], 1)

    if premium and owed <= pot * 0.20:
        return {"action": "call"}

    return {"action": "fold"}
