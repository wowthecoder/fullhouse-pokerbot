"""Baseline bot: shoves frequently to test risk handling."""

import random

BOT_NAME = "All-In Bot"


def decide(state: dict) -> dict:
    if random.random() < 0.75:
        return {"action": "all_in"}

    if state["can_check"]:
        return {"action": "check"}

    owed = state["amount_owed"]
    pot = max(state["pot"], 1)

    if owed <= pot * 0.25:
        return {"action": "call"}

    return {"action": "fold"}
