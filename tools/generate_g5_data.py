import json
import os
import struct
from array import array


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDIST = os.path.join(ROOT, "g5-poker-bot-main", "redist")
CHART_DIR = os.path.join(REDIST, "PreFlopCharts", "100bb")
OUT_DIR = os.path.join(ROOT, "bots", "my_bots", "g5_data")

RANK_GRID = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = [r + s for r in RANKS for s in SUITS]
CARD_INDEX = {card: i for i, card in enumerate(DECK)}
G5_POS_TO_LOCAL = {
    "UTG": "LJ",
    "HJ": "HJ",
    "CO": "CO",
    "BTN": "BTN",
    "SB": "SB",
    "BB": "BB",
}


def hand_key(row, col):
    if row > col:
        return RANK_GRID[col] + RANK_GRID[row] + "o"
    if row < col:
        return RANK_GRID[row] + RANK_GRID[col] + "s"
    return RANK_GRID[row] + RANK_GRID[col]


def parse_chart(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines[1:]:
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) != 40:
            raise ValueError("%s has %s parsed columns" % (path, len(parts)))
        rows.append(parts[1:])
    if len(rows) != 13:
        raise ValueError("%s has %s rank rows" % (path, len(rows)))

    cells = {}
    for row, parts in enumerate(rows):
        for col in range(13):
            allin = int(round(float(parts[col * 3])))
            raise_prob = int(round(float(parts[col * 3 + 1])))
            call_prob = int(round(float(parts[col * 3 + 2])))
            if allin or raise_prob or call_prob:
                cells[hand_key(row, col)] = {
                    "all_in": allin,
                    "raise": raise_prob,
                    "call": call_prob,
                }
    return cells


def put_chart(charts, family, key, cells):
    if cells:
        charts.setdefault(family, {})[key] = cells


def parse_charts():
    charts = {
        "rfi": {},
        "vs_open": {},
        "vs_two_bets": {},
        "vs_reraise": {},
    }
    for name in sorted(os.listdir(CHART_DIR)):
        if not name.endswith(".txt"):
            continue
        stem = name[:-4]
        parts = stem.split("_")
        cells = parse_chart(os.path.join(CHART_DIR, name))
        if stem.startswith("VS_0_Bets_"):
            hero = G5_POS_TO_LOCAL[parts[4]]
            put_chart(charts, "rfi", hero, cells)
        elif stem.startswith("VS_1_Bet_"):
            hero = G5_POS_TO_LOCAL[parts[4]]
            opener = G5_POS_TO_LOCAL[parts[6]]
            put_chart(charts, "vs_open", "%s_vs_%s" % (hero, opener), cells)
        elif stem.startswith("VS_2_Bets_RR_"):
            hero = G5_POS_TO_LOCAL[parts[5]]
            aggressor = G5_POS_TO_LOCAL[parts[7]]
            put_chart(charts, "vs_reraise", "%s_vs_%s" % (hero, aggressor), cells)
        elif stem.startswith("VS_2_Bets_"):
            hero = G5_POS_TO_LOCAL[parts[4]]
            opener = G5_POS_TO_LOCAL[parts[6]]
            reraiser = G5_POS_TO_LOCAL[parts[7]]
            put_chart(charts, "vs_two_bets", "%s_vs_%s_%s" % (hero, opener, reraiser), cells)
    return charts


def read_7bit_int(f):
    shift = 0
    result = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError
        val = b[0]
        result |= (val & 0x7F) << shift
        if not val & 0x80:
            return result
        shift += 7


def read_cs_string(f):
    n = read_7bit_int(f)
    return f.read(n).decode("utf-8", errors="replace")


def read_i32(f):
    data = f.read(4)
    if len(data) != 4:
        raise EOFError
    return struct.unpack("<i", data)[0]


def empty_bucket():
    return {"br": 0, "cc": 0, "fold": 0}


def parse_population_priors():
    path = os.path.join(REDIST, "full_stats_list_6max_2m.bin")
    pre = {}
    post = {}
    base = {"vpip_pos": 0, "vpip_total": 0, "players": 0}
    with open(path, "rb") as f:
        count = read_i32(f)
        for _ in range(count):
            read_cs_string(f)
            read_cs_string(f)
            read_i32(f)
            vpip_pos = read_i32(f)
            vpip_total = read_i32(f)
            base["vpip_pos"] += vpip_pos
            base["vpip_total"] += vpip_total
            base["players"] += 1

            pre_len = read_i32(f)
            for idx in range(pre_len):
                bucket = pre.setdefault(str(idx), empty_bucket())
                bucket["br"] += read_i32(f)
                bucket["cc"] += read_i32(f)
                bucket["fold"] += read_i32(f)

            post_len = read_i32(f)
            for idx in range(post_len):
                bucket = post.setdefault(str(idx), empty_bucket())
                bucket["br"] += read_i32(f)
                bucket["cc"] += read_i32(f)
                bucket["fold"] += read_i32(f)
    return {"base": base, "preflop": pre, "postflop": post}


def canonical_hole_index(cards):
    c1 = CARD_INDEX[cards[:2]]
    c2 = CARD_INDEX[cards[2:4]]
    if c1 > c2:
        c1, c2 = c2, c1
    return c1 * 52 + c2


def parse_preflop_equities():
    size = 52 * 52
    matrix = array("B", [0]) * (size * size)
    path = os.path.join(REDIST, "PreFlopEquities.txt")
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hero = canonical_hole_index(line[0:4])
            villain = canonical_hole_index(line[5:9])
            eq = float(line[10:])
            matrix[hero * size + villain] = max(0, min(255, int(round(eq * 255.0 / 100.0))))
    if matrix.itemsize != 1:
        matrix = array("B", matrix)
    return matrix


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "preflop_charts_100bb.json"), "w", encoding="utf-8") as f:
        json.dump(parse_charts(), f, sort_keys=True, separators=(",", ":"))
    with open(os.path.join(OUT_DIR, "opponent_priors_6max.json"), "w", encoding="utf-8") as f:
        json.dump(parse_population_priors(), f, sort_keys=True, separators=(",", ":"))
    equities = parse_preflop_equities()
    with open(os.path.join(OUT_DIR, "preflop_equity_u8.bin"), "wb") as f:
        equities.tofile(f)
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "chart_source": "g5-poker-bot-main/redist/PreFlopCharts/100bb",
                "priors_source": "g5-poker-bot-main/redist/full_stats_list_6max_2m.bin",
                "equity_source": "g5-poker-bot-main/redist/PreFlopEquities.txt",
                "card_order": DECK,
            },
            f,
            sort_keys=True,
            indent=2,
        )


if __name__ == "__main__":
    main()
