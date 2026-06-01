import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bots.my_bots import complete_v2 as bot


def info(hole, board):
    return bot._strategic_hand_info(tuple(hole), tuple(board))


@pytest.mark.parametrize(
    ("hole", "board", "hand_type", "made"),
    [
        (("As", "Ks"), ("Qs", "Js", "Ts"), "straight flush", 8),
        (("Ah", "Ad"), ("As", "Ac", "2d"), "four of a kind", 7),
        (("Ah", "Ad"), ("As", "7c", "7d"), "full house", 6),
        (("Ah", "Kh"), ("Qh", "Jh", "2h"), "flush", 5),
        (("Ah", "Kh"), ("Qd", "Js", "Tc"), "straight", 4),
        (("Ah", "Ad"), ("As", "7c", "2d"), "three of a kind", 3),
        (("Ah", "Ad"), ("7s", "7c", "2d"), "two pair", 2),
        (("Ah", "Ad"), ("Ks", "7c", "2d"), "pair", 1),
        (("Ah", "Kd"), ("Qs", "7c", "2d"), "high card", 0),
    ],
)
def test_eval7_made_hand_mapping(hole, board, hand_type, made):
    result = info(hole, board)

    assert result["type"] == hand_type
    assert result["made"] == made
    assert result["eval7_rank"] is not None


def test_false_straight_flush_regression_uses_eval7_category():
    result = info(("Ah", "Kh"), ("Qh", "Jh", "2h", "Tc", "9d"))

    assert result["type"] == "flush"
    assert result["made"] == 5
    assert result["strength"] == "monster"


def test_board_only_flush_is_not_value_monster():
    result = info(("As", "Kd"), ("2h", "5h", "8h", "Jh", "Qh"))

    assert result["type"] == "flush"
    assert result["plays_board"] is True
    assert result["hero_improves_board"] is False
    assert result["uses_hole_count"] == 0
    assert result["strength"] == "air"
    assert result["showdown"] is True


def test_board_two_pair_with_kicker_is_not_monster():
    result = info(("As", "2d"), ("Kh", "Kc", "Qh", "Qc", "3s"))

    assert result["type"] == "two pair"
    assert result["hero_improves_board"] is True
    assert result["strong_two_pair"] is False
    assert result["strength"] != "monster"


def test_counterfeited_two_pair_is_not_auto_monster():
    result = info(("9s", "8s"), ("9h", "8d", "Kd", "Kc", "2h"))

    assert result["type"] == "two pair"
    assert result["strong_two_pair"] is False
    assert result["strength"] != "monster"


def test_set_and_trips_are_distinguished():
    set_result = info(("8s", "8d"), ("8h", "Kc", "2s"))
    trips_result = info(("8s", "Ad"), ("8h", "8c", "2s"))

    assert set_result["set"] is True
    assert set_result["trips"] is True
    assert set_result["strength"] == "monster"
    assert trips_result["set"] is False
    assert trips_result["trips"] is True
    assert trips_result["strength"] == "tptk_plus"


def test_flush_draw_requires_hero_suit_and_tracks_nut_draw():
    nut_draw = info(("Ah", "2c"), ("Kh", "7h", "3h"))
    board_only_texture = info(("As", "2c"), ("Kh", "7h", "3h"))

    assert nut_draw["flush_draw"] is True
    assert nut_draw["nut_flush_draw"] is True
    assert board_only_texture["flush_draw"] is False
    assert board_only_texture["nut_flush_draw"] is False


def test_board_only_straight_texture_is_not_hero_draw():
    result = info(("As", "Qd"), ("5h", "6c", "7s", "Kd"))

    assert result["oesd"] is False
    assert result["gutshot"] is False
