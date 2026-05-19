"""Baseline bot: folds to aggression and checks for free."""

BOT_NAME = "Fold Bot"


def decide(state: dict) -> dict:
    if state["can_check"]:
        return {"action": "check"}

    return {"action": "fold"}
