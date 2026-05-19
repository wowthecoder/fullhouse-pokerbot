"""Baseline bot: calls or checks whenever possible."""

BOT_NAME = "Call/Check Bot"


def decide(state: dict) -> dict:
    if state["can_check"]:
        return {"action": "check"}

    return {"action": "call"}
