"""Baseline bot: plays many hands and rarely raises."""

import random

BOT_NAME = "Loose-Passive Bot"


def _has_monster(cards: list) -> bool:
    ranks = [card[0] for card in cards]
    return ranks.count("A") == 2 or ranks.count("K") == 2


def decide(state: dict) -> dict:
    if _has_monster(state["your_cards"]) and random.random() < 0.25:
        raise_to = min(state["min_raise_to"] * 2, state["your_stack"] + state["your_bet_this_street"])
        return {"action": "raise", "amount": raise_to}

    if random.random() < 0.03:
        raise_to = min(state["min_raise_to"], state["your_stack"] + state["your_bet_this_street"])
        return {"action": "raise", "amount": raise_to}

    if state["can_check"]:
        return {"action": "check"}

    owed = state["amount_owed"]
    pot = max(state["pot"], 1)
    stack = max(state["your_stack"], 1)

    if owed <= pot * 0.45 or owed <= stack * 0.08:
        return {"action": "call"}

    return {"action": "fold"}
