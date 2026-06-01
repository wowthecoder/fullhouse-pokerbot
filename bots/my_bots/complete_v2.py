"""V1 with the following fixes and enhancements: {}"""

import bisect
import random
from math import erf, sqrt

import eval7

try:
    from scipy.stats import beta as _beta_dist
except Exception:
    _beta_dist = None

BOT_NAME = "PokerCoaching Preflop"
BOT_AVATAR = "robot_1"

RANKS = "23456789TJQKA"
RANK_VALUE = {rank: i + 2 for i, rank in enumerate(RANKS)}
HIGH_CARDS = {"A", "K", "Q", "J", "T"}
SUITS = "cdhs"
FULL_DECK = [rank + suit for rank in RANKS for suit in SUITS]
EVAL7_CARD_CACHE = {}
EQUITY_RANGE_CACHE = {}
MAX_EQUITY_RANGE_CACHE = 128

SIZING_RULES = {
    "rfi": 2.5,
    "sb_rfi": 3.0,
    "ip_3bet_multiplier": 3.5,
    "oop_3bet_multiplier": 4.0,
    "bb_vs_sb_limp_raise_multiplier": 3.5,
    "ip_4bet_multiplier": 2.3,
    "oop_4bet_multiplier": 2.5,
}

PREMIUMS = {"AA", "KK", "QQ", "AKs", "AKo"}
STRONG_CONTINUES = {"JJ", "TT", "AQs", "AQo", "AJs", "KQs"}
MIN_TRUSTED_HANDS = 50
MAX_TRACKED_HANDS = 80
MIN_STEAL_OPPS = 6
MIN_THREEBET_OPPS = 8
MIN_FOLD_TO_THREEBET_OPPS = 6
MIN_CBET_OPPS = 6
MIN_POSTFLOP_ACTIONS = 8

OPPONENT_MODELS = {}
HAND_OBSERVATIONS = {}

RATE_SPECS = {
    "vpip": {
        "success_attr": "vpip_hands",
        "opp_attr": "hands",
        "prior_rate": 0.24,
        "prior_weight": 30,
        "min_opp_soft": 20,
        "min_opp_hard": 50,
    },
    "pfr": {
        "success_attr": "pfr_hands",
        "opp_attr": "hands",
        "prior_rate": 0.18,
        "prior_weight": 30,
        "min_opp_soft": 20,
        "min_opp_hard": 50,
    },
    "threebet": {
        "success_attr": "threebets",
        "opp_attr": "threebet_opps",
        "prior_rate": 0.07,
        "prior_weight": 20,
        "min_opp_soft": 10,
        "min_opp_hard": 25,
    },
    "fold_to_steal": {
        "success_attr": "folded_to_steal",
        "opp_attr": "steal_opps",
        "prior_rate": 0.55,
        "prior_weight": 20,
        "min_opp_soft": 8,
        "min_opp_hard": 20,
    },
    "fold_to_threebet": {
        "success_attr": "folded_to_threebet",
        "opp_attr": "faced_threebet",
        "prior_rate": 0.50,
        "prior_weight": 20,
        "min_opp_soft": 8,
        "min_opp_hard": 20,
    },
    "fourbet": {
        "success_attr": "fourbets",
        "opp_attr": "fourbet_opps",
        "prior_rate": 0.06,
        "prior_weight": 20,
        "min_opp_soft": 8,
        "min_opp_hard": 20,
    },
    "fold_to_cbet": {
        "success_attr": "cb_folds",
        "opp_attr": "cb_opps",
        "prior_rate": 0.45,
        "prior_weight": 20,
        "min_opp_soft": 8,
        "min_opp_hard": 25,
    },
    "wtsd": {
        "success_attr": "showdown_hands",
        "opp_attr": "saw_flop_hands",
        "prior_rate": 0.27,
        "prior_weight": 30,
        "min_opp_soft": 15,
        "min_opp_hard": 40,
    },
    "postflop_aggression_frequency": {
        "success_attr": "postflop_bets_raises",
        "opp_attr": "postflop_actions",
        "prior_rate": 0.38,
        "prior_weight": 20,
        "min_opp_soft": 12,
        "min_opp_hard": 30,
    },
}


def _bayes_rate(successes, opportunities, prior_rate, prior_weight):
    return (successes + prior_rate * prior_weight) / max(1, opportunities + prior_weight)


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def _norm_cdf(value):
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _beta_params(successes, opportunities, prior_rate, prior_weight):
    successes = max(0, int(successes))
    opportunities = max(0, int(opportunities))
    failures = max(0, opportunities - successes)
    alpha = prior_rate * prior_weight + successes
    beta = (1.0 - prior_rate) * prior_weight + failures
    return max(alpha, 1e-9), max(beta, 1e-9)


def _beta_mean(successes, opportunities, prior_rate, prior_weight):
    alpha, beta = _beta_params(successes, opportunities, prior_rate, prior_weight)
    return alpha / (alpha + beta)


def _beta_var_from_params(alpha, beta):
    total = alpha + beta
    return (alpha * beta) / ((total * total) * (total + 1.0))


def _beta_interval(successes, opportunities, prior_rate, prior_weight, level=0.80):
    alpha, beta = _beta_params(successes, opportunities, prior_rate, prior_weight)
    tail = (1.0 - level) / 2.0
    if _beta_dist is not None:
        return (
            float(_beta_dist.ppf(tail, alpha, beta)),
            float(_beta_dist.ppf(1.0 - tail, alpha, beta)),
        )

    mean = alpha / (alpha + beta)
    sd = sqrt(max(1e-12, _beta_var_from_params(alpha, beta)))
    z_by_level = {
        0.80: 1.28155,
        0.85: 1.43953,
        0.90: 1.64485,
        0.95: 1.95996,
    }
    z = z_by_level.get(level, 1.28155)
    return _clamp(mean - z * sd), _clamp(mean + z * sd)


def _posterior_prob_above(successes, opportunities, prior_rate, prior_weight, threshold):
    alpha, beta = _beta_params(successes, opportunities, prior_rate, prior_weight)
    threshold = _clamp(threshold)
    if _beta_dist is not None:
        return float(_beta_dist.sf(threshold, alpha, beta))

    mean = alpha / (alpha + beta)
    sd = sqrt(max(1e-12, _beta_var_from_params(alpha, beta)))
    return 1.0 - _norm_cdf((threshold - mean) / sd)


def _posterior_prob_below(successes, opportunities, prior_rate, prior_weight, threshold):
    alpha, beta = _beta_params(successes, opportunities, prior_rate, prior_weight)
    threshold = _clamp(threshold)
    if _beta_dist is not None:
        return float(_beta_dist.cdf(threshold, alpha, beta))

    mean = alpha / (alpha + beta)
    sd = sqrt(max(1e-12, _beta_var_from_params(alpha, beta)))
    return _norm_cdf((threshold - mean) / sd)


def _rate_counts(model, stat_name):
    spec = RATE_SPECS[stat_name]
    successes = int(getattr(model, spec["success_attr"], 0))
    opportunities = int(getattr(model, spec["opp_attr"], 0))
    return successes, opportunities, spec


def _rate_estimate(model, stat_name, interval_level=0.80):
    successes, opportunities, spec = _rate_counts(model, stat_name)
    mean = _beta_mean(successes, opportunities, spec["prior_rate"], spec["prior_weight"])
    lo, hi = _beta_interval(
        successes,
        opportunities,
        spec["prior_rate"],
        spec["prior_weight"],
        level=interval_level,
    )
    return {
        "successes": successes,
        "opportunities": opportunities,
        "mean": mean,
        "lo": lo,
        "hi": hi,
    }


def _likely_above(model, stat_name, threshold, confidence=0.85, min_opp=None):
    successes, opportunities, spec = _rate_counts(model, stat_name)
    if min_opp is None:
        min_opp = spec["min_opp_soft"]
    if opportunities < min_opp:
        return False
    probability = _posterior_prob_above(
        successes,
        opportunities,
        spec["prior_rate"],
        spec["prior_weight"],
        threshold,
    )
    return probability >= confidence


def _likely_below(model, stat_name, threshold, confidence=0.85, min_opp=None):
    successes, opportunities, spec = _rate_counts(model, stat_name)
    if min_opp is None:
        min_opp = spec["min_opp_soft"]
    if opportunities < min_opp:
        return False
    probability = _posterior_prob_below(
        successes,
        opportunities,
        spec["prior_rate"],
        spec["prior_weight"],
        threshold,
    )
    return probability >= confidence


def _exploit_strength_above(model, stat_name, threshold, min_opp_soft=None):
    successes, opportunities, spec = _rate_counts(model, stat_name)
    if min_opp_soft is None:
        min_opp_soft = spec["min_opp_soft"]
    if opportunities < min_opp_soft:
        return 0.0
    probability = _posterior_prob_above(
        successes,
        opportunities,
        spec["prior_rate"],
        spec["prior_weight"],
        threshold,
    )
    return _clamp((probability - 0.55) / 0.35)


def _exploit_strength_below(model, stat_name, threshold, min_opp_soft=None):
    successes, opportunities, spec = _rate_counts(model, stat_name)
    if min_opp_soft is None:
        min_opp_soft = spec["min_opp_soft"]
    if opportunities < min_opp_soft:
        return 0.0
    probability = _posterior_prob_below(
        successes,
        opportunities,
        spec["prior_rate"],
        spec["prior_weight"],
        threshold,
    )
    return _clamp((probability - 0.55) / 0.35)


def _add_rate_debug(snapshot, model, stat_name):
    est80 = _rate_estimate(model, stat_name, interval_level=0.80)
    est90 = _rate_estimate(model, stat_name, interval_level=0.90)
    snapshot[f"{stat_name}_successes"] = est80["successes"]
    snapshot[f"{stat_name}_opps"] = est80["opportunities"]
    snapshot[f"{stat_name}_mean"] = est80["mean"]
    snapshot[f"{stat_name}_lo80"] = est80["lo"]
    snapshot[f"{stat_name}_hi80"] = est80["hi"]
    snapshot[f"{stat_name}_lo90"] = est90["lo"]
    snapshot[f"{stat_name}_hi90"] = est90["hi"]


def _increment_bucket(counter, bucket, amount=1):
    counter[bucket] = counter.get(bucket, 0) + amount


def _bucket_rates(successes, opportunities):
    return {
        bucket: successes.get(bucket, 0) / max(1, count)
        for bucket, count in opportunities.items()
    }


def _preflop_size_bucket(amount, big_blind):
    size_bb = int(amount or 0) / max(1, big_blind)
    if size_bb <= 2.5:
        return "small"
    if size_bb <= 3.5:
        return "standard"
    return "large"


def _postflop_large_bet(entry, pot):
    amount = int(entry.get("amount") or 0)
    return amount >= max(1, int(pot or 0)) * 0.65


class OppModel:
    def __init__(self, bot_id=None, seat=None):
        self.bot_id = bot_id
        self.last_seen_seat = seat
        self.hands = 0
        self.vpip_hands = 0
        self.pfr_hands = 0
        self.postflop_bets_raises = 0
        self.postflop_calls = 0
        self.postflop_actions = 0
        self.steal_opps = 0
        self.folded_to_steal = 0
        self.threebet_opps = 0
        self.threebets = 0
        self.faced_threebet = 0
        self.folded_to_threebet = 0
        self.fourbet_opps = 0
        self.fourbets = 0
        self.cb_opps = 0
        self.cb_folds = 0
        self.saw_flop_hands = 0
        self.showdown_hands = 0
        self.fold_to_steal_opps_by_stealer = {}
        self.folded_to_steal_by_stealer = {}
        self.bb_defend_opps_by_size = {}
        self.bb_defend_folds_by_size = {}
        self.bb_defend_calls_by_size = {}
        self.bb_defend_raises_by_size = {}
        self.cb_opps_by_pot_type = {}
        self.cb_folds_by_pot_type = {}
        self.turn_after_flop_call_opps = 0
        self.turn_after_flop_call_folds = 0
        self.turn_after_flop_call_calls = 0
        self.turn_after_flop_call_raises = 0
        self.river_bets_raises = 0
        self.river_calls = 0
        self.river_folds = 0
        self.river_checks = 0
        self.river_actions = 0
        self.street_bets_raises = {"flop": 0, "turn": 0, "river": 0}
        self.street_large_bets = {"flop": 0, "turn": 0, "river": 0}

    @property
    def vpip(self):
        return self.vpip_hands / max(1, self.hands)

    @property
    def pfr(self):
        return self.pfr_hands / max(1, self.hands)

    @property
    def af(self):
        return self.postflop_bets_raises / max(1, self.postflop_calls)

    @property
    def threebet_pct(self):
        return self.threebets / max(1, self.threebet_opps)

    @property
    def fold_to_steal(self):
        return self.folded_to_steal / max(1, self.steal_opps)

    @property
    def fold_to_threebet(self):
        return self.folded_to_threebet / max(1, self.faced_threebet)

    @property
    def fourbet_pct(self):
        return self.fourbets / max(1, self.fourbet_opps)

    @property
    def fold_to_cbet(self):
        return self.cb_folds / max(1, self.cb_opps)

    @property
    def wtsd(self):
        return self.showdown_hands / max(1, self.saw_flop_hands)

    @property
    def postflop_aggression_frequency(self):
        return self.postflop_bets_raises / max(1, self.postflop_actions)

    @property
    def bayes_vpip(self):
        return _bayes_rate(self.vpip_hands, self.hands, 0.24, 30)

    @property
    def bayes_pfr(self):
        return _bayes_rate(self.pfr_hands, self.hands, 0.18, 30)

    @property
    def bayes_threebet(self):
        return _bayes_rate(self.threebets, self.threebet_opps, 0.07, 20)

    @property
    def bayes_fold_to_steal(self):
        return _bayes_rate(self.folded_to_steal, self.steal_opps, 0.55, 20)

    @property
    def bayes_fold_to_threebet(self):
        return _bayes_rate(self.folded_to_threebet, self.faced_threebet, 0.50, 20)

    @property
    def bayes_fourbet(self):
        return _bayes_rate(self.fourbets, self.fourbet_opps, 0.06, 20)

    @property
    def bayes_fold_to_cbet(self):
        return _bayes_rate(self.cb_folds, self.cb_opps, 0.45, 20)

    @property
    def bayes_wtsd(self):
        return _bayes_rate(self.showdown_hands, self.saw_flop_hands, 0.27, 30)

    @property
    def bayes_postflop_aggression_frequency(self):
        return _bayes_rate(self.postflop_bets_raises, self.postflop_actions, 0.38, 20)

    @property
    def trusted(self):
        return self.hands >= MIN_TRUSTED_HANDS

    def snapshot(self):
        bb_defend_vs_size = {}
        for bucket, opps in self.bb_defend_opps_by_size.items():
            folds = self.bb_defend_folds_by_size.get(bucket, 0)
            calls = self.bb_defend_calls_by_size.get(bucket, 0)
            raises = self.bb_defend_raises_by_size.get(bucket, 0)
            bb_defend_vs_size[bucket] = {
                "opps": opps,
                "folds": folds,
                "calls": calls,
                "raises": raises,
                "defend_rate": (calls + raises) / max(1, opps),
            }
        snapshot = {
            "bot_id": self.bot_id,
            "last_seen_seat": self.last_seen_seat,
            "hands": self.hands,
            "trusted": self.trusted,
            "vpip": self.vpip,
            "pfr": self.pfr,
            "af": self.af,
            "postflop_actions": self.postflop_actions,
            "postflop_aggression_frequency": self.postflop_aggression_frequency,
            "fold_to_steal": self.fold_to_steal,
            "steal_opps": self.steal_opps,
            "threebet_pct": self.threebet_pct,
            "threebet_opps": self.threebet_opps,
            "fold_to_threebet": self.fold_to_threebet,
            "faced_threebet": self.faced_threebet,
            "fourbet_pct": self.fourbet_pct,
            "fourbet_opps": self.fourbet_opps,
            "fold_to_cbet": self.fold_to_cbet,
            "cbet_opps": self.cb_opps,
            "wtsd": self.wtsd,
            "saw_flop_hands": self.saw_flop_hands,
            "bayes_vpip": self.bayes_vpip,
            "bayes_pfr": self.bayes_pfr,
            "bayes_threebet": self.bayes_threebet,
            "bayes_fold_to_steal": self.bayes_fold_to_steal,
            "bayes_fold_to_threebet": self.bayes_fold_to_threebet,
            "bayes_fourbet": self.bayes_fourbet,
            "bayes_fold_to_cbet": self.bayes_fold_to_cbet,
            "bayes_wtsd": self.bayes_wtsd,
            "bayes_postflop_aggression_frequency": self.bayes_postflop_aggression_frequency,
            "fold_to_steal_by_stealer": _bucket_rates(
                self.folded_to_steal_by_stealer,
                self.fold_to_steal_opps_by_stealer,
            ),
            "fold_to_steal_opps_by_stealer": dict(self.fold_to_steal_opps_by_stealer),
            "bb_defend_vs_size": bb_defend_vs_size,
            "fold_to_cbet_by_pot_type": _bucket_rates(self.cb_folds_by_pot_type, self.cb_opps_by_pot_type),
            "cbet_opps_by_pot_type": dict(self.cb_opps_by_pot_type),
            "turn_after_flop_call_opps": self.turn_after_flop_call_opps,
            "turn_fold_after_flop_call": self.turn_after_flop_call_folds / max(1, self.turn_after_flop_call_opps),
            "turn_after_flop_call_calls": self.turn_after_flop_call_calls,
            "turn_after_flop_call_raises": self.turn_after_flop_call_raises,
            "river_tendencies": {
                "actions": self.river_actions,
                "bet_raise": self.river_bets_raises,
                "call": self.river_calls,
                "fold": self.river_folds,
                "check": self.river_checks,
                "bet_raise_rate": self.river_bets_raises / max(1, self.river_actions),
                "call_rate": self.river_calls / max(1, self.river_actions),
                "fold_rate": self.river_folds / max(1, self.river_actions),
            },
            "large_bet_frequency_by_street": _bucket_rates(self.street_large_bets, self.street_bets_raises),
        }
        for stat_name in RATE_SPECS:
            _add_rate_debug(snapshot, self, stat_name)
        snapshot["label_v2"] = _label_for_model(self)
        snapshot["read"] = _villain_read(self)
        return snapshot


def _s(*hands):
    return set(hands)


RANGECONVERTER_VS_3BET = {('BTN', 'BB'): {'call': ('AQs',
                          'AJs',
                          'ATs',
                          'A9s',
                          'A8s',
                          'A7s',
                          'A5s',
                          'A4s',
                          'A3s',
                          'KQs',
                          'KJs',
                          'KTs',
                          'K9s',
                          'K8s',
                          'KQo',
                          'QJs',
                          'QTs',
                          'Q9s',
                          'JTs',
                          'J9s',
                          'TT',
                          'T9s',
                          'T8s',
                          '99',
                          '98s',
                          '88',
                          '87s',
                          '77',
                          '76s',
                          '66',
                          '55',
                          '44',
                          '33',
                          '22'),
                 'fourbet': ('AA', 'AKs', 'KK'),
                 'fourbet_to_bb': 25.0,
                 'mix_call_fourbet': ('A6s', 'AKo', 'AQo', 'QQ', 'AJo', 'KJo', 'JJ'),
                 'mix_fold_call': ('A2s', 'K7s', 'K6s', 'J8s', '97s', '65s', '54s'),
                 'mix_fold_fourbet': ('ATo',)},
 ('BTN', 'SB'): {'call': ('AQs',
                          'AJs',
                          'ATs',
                          'A9s',
                          'A7s',
                          'A5s',
                          'A4s',
                          'A3s',
                          'KQs',
                          'KJs',
                          'KTs',
                          'K9s',
                          'K6s',
                          'KQo',
                          'QJs',
                          'QTs',
                          'Q9s',
                          'JTs',
                          'J9s',
                          'T9s',
                          'T8s',
                          '99',
                          '98s',
                          '88',
                          '87s',
                          '77',
                          '76s',
                          '66',
                          '65s',
                          '55',
                          '54s',
                          '44',
                          '33',
                          '22'),
                 'fourbet': ('AA', 'AKs', 'KK'),
                 'fourbet_to_bb': 25.0,
                 'mix_call_fourbet': ('A8s', 'AKo', 'AQo', 'QQ', 'AJo', 'KJo', 'JJ', 'TT'),
                 'mix_fold_call': ('K8s', 'K7s', 'J8s', '97s', '86s'),
                 'mix_fold_fourbet': ('ATo',)},
 ('CO', 'BB'): {'call': ('AQs',
                         'AJs',
                         'ATs',
                         'A9s',
                         'A5s',
                         'A4s',
                         'KQs',
                         'KJs',
                         'KTs',
                         'QQ',
                         'QJs',
                         'QTs',
                         'JJ',
                         'JTs',
                         'TT',
                         'T9s',
                         '99',
                         '88',
                         '77',
                         '76s',
                         '66',
                         '65s',
                         '54s',
                         '33',
                         '22'),
                'fourbet': ('AA', 'AKs'),
                'fourbet_to_bb': 25.0,
                'mix_call_fourbet': ('AKo', 'KK', 'AQo'),
                'mix_fold_call': ('A8s', 'A3s', 'K9s', 'K6s', 'KQo', 'J9s', '87s', '55', '44')},
 ('CO', 'BTN'): {'call': ('AQs', 'AJs', 'KQs', '77', '76s', '65s', '54s', '33', '22'),
                 'fourbet': ('AA', 'AKs', 'AKo', 'KK', 'AQo', 'QQ', 'JJ'),
                 'fourbet_to_bb': 23.6,
                 'mix_call_fourbet': ('ATs', 'KJs', 'KTs', 'QJs', 'JTs', 'TT', '99', '88'),
                 'mix_fold_call': ('A9s', 'A5s', 'A4s', 'K9s', 'QTs', 'T9s', '98s', '87s', '66', '55', '44'),
                 'mix_fold_fourbet': ('AJo',)},
 ('CO', 'SB'): {'call': ('AQs',
                         'AJs',
                         'A5s',
                         'A4s',
                         'KQs',
                         'KJs',
                         'KTs',
                         'QJs',
                         'QTs',
                         'JTs',
                         'J9s',
                         'TT',
                         'T9s',
                         '99',
                         '88',
                         '77',
                         '76s',
                         '66',
                         '65s',
                         '54s',
                         '33',
                         '22'),
                'fourbet': ('AA', 'AKs'),
                'fourbet_to_bb': 25.0,
                'mix_call_fourbet': ('ATs', 'AKo', 'KK', 'AQo', 'QQ', 'JJ'),
                'mix_fold_call': ('A9s', 'A3s', 'K9s', 'K6s', 'T8s', '98s', '87s', '55', '44'),
                'mix_fold_fourbet': ('KQo', 'AJo')},
 ('HJ', 'BB'): {'call': ('AQs',
                         'AJs',
                         'ATs',
                         'A5s',
                         'A4s',
                         'KQs',
                         'KJs',
                         'KTs',
                         'QQ',
                         'QJs',
                         'QTs',
                         'JJ',
                         'JTs',
                         'TT',
                         '99',
                         '88',
                         '87s',
                         '76s',
                         '65s',
                         '54s',
                         '44',
                         '33',
                         '22'),
                'fourbet': ('AA',),
                'fourbet_to_bb': 25.0,
                'mix_call_fourbet': ('AKs', 'AKo', 'KK', 'AQo'),
                'mix_fold_call': ('A9s', 'A3s', 'T9s', '98s', '77', '66', '55')},
 ('HJ', 'BTN'): {'call': ('AQs', '87s', '76s', '65s', '54s', '44'),
                 'fourbet': ('AA', 'AKs', 'KK', 'KJs', 'AQo', 'QQ'),
                 'fourbet_to_bb': 23.6,
                 'mix_call_fourbet': ('AJs', 'ATs', 'AKo', 'KQs', 'KTs', 'JJ', 'TT', '99'),
                 'mix_fold_call': ('A9s', 'A5s', 'QJs', 'QTs', 'JTs', 'T9s', '98s', '88', '77', '66', '55')},
 ('HJ', 'CO'): {'call': ('AQs', '76s', '65s', '54s', '44'),
                'fourbet': ('AA', 'AKs', 'AKo', 'KK', 'AQo', 'QQ', 'JJ'),
                'fourbet_to_bb': 23.8,
                'mix_call_fourbet': ('AJs', 'ATs', 'KQs', 'KJs', 'KTs', 'TT'),
                'mix_fold_call': ('JTs', '99', '88', '87s', '77', '66', '55')},
 ('HJ', 'SB'): {'call': ('AQs',
                         'ATs',
                         'AKo',
                         'KTs',
                         'QQ',
                         'QJs',
                         'JTs',
                         'TT',
                         '99',
                         '88',
                         '87s',
                         '76s',
                         '65s',
                         '54s',
                         '44',
                         '33',
                         '22'),
                'fourbet': ('AA',),
                'fourbet_to_bb': 25.0,
                'mix_call_fourbet': ('AKs', 'AJs', 'A5s', 'KK', 'KQs', 'KJs', 'AQo', 'JJ'),
                'mix_fold_call': ('A4s', 'A3s', 'K6s', 'QTs', 'J9s', 'T9s', '98s', '77', '66', '55')},
 ('LJ', 'BB'): {'call': ('AQs',
                         'AJs',
                         'ATs',
                         'A5s',
                         'AKo',
                         'KQs',
                         'KJs',
                         'QQ',
                         'JJ',
                         'TT',
                         '87s',
                         '76s',
                         '66',
                         '65s',
                         '55',
                         '54s'),
                'fourbet': ('AA',),
                'fourbet_to_bb': 25.0,
                'mix_call_fourbet': ('AKs', 'KK'),
                'mix_fold_call': ('A4s', 'A3s', 'KTs', 'QJs', 'QTs', 'JTs', 'T9s', '99', '88', '77')},
 ('LJ', 'BTN'): {'call': ('AQs', '87s', '76s', '65s', '55', '54s'),
                 'fourbet': ('AA', 'AKs', 'KK'),
                 'fourbet_to_bb': 23.8,
                 'mix_call_fourbet': ('AJs', 'ATs', 'AKo', 'KQs', 'KJs', 'KTs', 'QQ', 'JJ', 'TT'),
                 'mix_fold_call': ('A5s', 'A4s', 'JTs', 'T9s', '99', '88', '77', '66'),
                 'mix_fold_fourbet': ('AQo',)},
 ('LJ', 'CO'): {'call': ('87s', '76s', '65s', '54s'),
                'fourbet': ('AA', 'AKs', 'AKo', 'KK', 'QQ'),
                'fourbet_to_bb': 23.6,
                'mix_call_fourbet': ('AQs', 'AJs', 'KQs', 'KJs', 'JJ', 'TT'),
                'mix_fold_call': ('99', '66'),
                'mix_fold_fourbet': ('ATs', 'AQo')},
 ('LJ', 'HJ'): {'call': ('76s', '65s', '54s'),
                'fourbet': ('AA', 'AKs', 'KK', 'QQ'),
                'fourbet_to_bb': 23.6,
                'mix_call_fourbet': ('AQs', 'AJs', 'AKo', 'KQs', 'KJs', 'JJ', 'TT'),
                'mix_fold_call': ('99', '88', '66'),
                'mix_fold_fourbet': ('ATs', 'AQo')},
 ('LJ', 'SB'): {'call': ('AQs', 'AJs', 'AKo', 'KQs', 'QQ', 'QJs', 'JJ', 'TT', '87s', '76s', '65s', '55', '54s'),
                'fourbet': ('AA',),
                'fourbet_to_bb': 25.0,
                'mix_call_fourbet': ('AKs', 'A5s', 'KK', 'KJs'),
                'mix_fold_call': ('ATs', 'A4s', 'KTs', 'QTs', 'JTs', 'T9s', '99', '88', '77', '66')},
 ('SB', 'BB'): {'call': ('AJs',
                         'ATs',
                         'A9s',
                         'A8s',
                         'A7s',
                         'A6s',
                         'A5s',
                         'A4s',
                         'A3s',
                         'KQs',
                         'KJs',
                         'KTs',
                         'K9s',
                         'K8s',
                         'QJs',
                         'QTs',
                         'Q9s',
                         'JTs',
                         'J9s',
                         'T9s',
                         '99',
                         '88',
                         '77',
                         '66',
                         '55'),
                'fourbet': ('AA', 'AKs', 'AQs', 'AKo', 'KK', 'QQ', 'AJo', 'KJo', 'JJ', 'TT'),
                'fourbet_to_bb': 26.0,
                'mix_call_fourbet': ('AQo', 'KQo', 'ATo'),
                'mix_fold_call': ('K7s', 'Q8s', 'J8s', 'T8s', '44', '22'),
                'mix_fold_fourbet': ('A2s', 'K5s')}}

FIVEBET_CALL_SAFE = {"QQ", "KK", "AA", "AKs", "AKo"}
COLD_4BET_SAFE = {"QQ", "KK", "AA", "AKs", "AKo"}
FOURBET_FALLBACK_VALUE = {"KK", "AA", "AKs", "AKo"}
FOURBET_FALLBACK_BLUFF = {"A5s", "A4s"}

SQUEEZE_VS_EP = {"QQ", "KK", "AA", "AKs", "AKo", "A5s", "A4s", "KQs"}
SQUEEZE_VS_MP = SQUEEZE_VS_EP | {"JJ", "AQs", "ATs", "QJs", "JTs"}
SQUEEZE_VS_LP = SQUEEZE_VS_MP | {
    "TT", "AQo", "AJs", "A3s", "A2s", "KJs", "KTs", "QTs", "T9s", "98s",
}
MULTIWAY_CALL_RANGE = {
    "22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ",
    "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s", "A9s", "ATs", "AJs", "AQs",
    "KTs", "KJs", "KQs", "QTs", "QJs", "JTs", "T9s", "98s", "87s", "76s", "65s", "54s",
}
ISO_VALUE_DEFAULT = {
    "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA",
    "A9s", "ATs", "AJs", "AQs", "AKs", "KTs", "KJs", "KQs", "QTs", "QJs", "JTs",
    "ATo", "AJo", "AQo", "AKo", "KQo",
}
ISO_LATE_POSITION_EXTRA = ISO_VALUE_DEFAULT | {
    "55", "66", "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s",
    "K9s", "Q9s", "J9s", "T9s", "98s", "87s", "A9o", "KJo", "QJo",
}
OVERLIMP_MULTIWAY = {
    "22", "33", "44", "55", "66", "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s",
    "K7s", "K8s", "K9s", "Q9s", "J9s", "T9s", "98s", "87s", "76s", "65s", "54s",
}
SHORT_STACK_JAM = ISO_VALUE_DEFAULT | {"22", "33", "44", "55", "66", "A8s", "A9o", "KQs"}

PREFLOP_RANGES = {
    "unopened": {
        "LJ": {
            "raise": _s(
                "AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s",
                "AKo", "KK", "KQs", "KJs", "KTs", "K9s", "K8s",
                "AQo", "KQo", "QQ", "QJs", "QTs", "Q9s",
                "AJo", "KJo", "QJo", "JJ", "JTs", "J9s",
                "ATo", "TT", "T9s", "99", "88", "77", "66",
            ),
        },
        "HJ": {
            "raise": _s(
                "AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
                "AKo", "KK", "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s",
                "AQo", "KQo", "QQ", "QJs", "QTs", "Q9s", "Q8s",
                "AJo", "KJo", "QJo", "JJ", "JTs", "J9s",
                "ATo", "KTo", "QTo", "TT", "T9s", "99", "98s", "88", "87s", "77", "76s", "66", "55",
            ),
        },
        "CO": {
            "raise": _s(
                "AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
                "AKo", "KK", "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s",
                "AQo", "KQo", "QQ", "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s",
                "AJo", "KJo", "QJo", "JJ", "JTs", "J9s", "J8s",
                "ATo", "KTo", "QTo", "JTo", "TT", "T9s", "T8s", "T7s",
                "A9o", "99", "98s", "97s", "A8o", "88", "87s", "77", "76s", "66", "55", "44", "33",
            ),
        },
        "BTN": {
            "raise": _s(
                "AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
                "AKo", "KK", "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
                "AQo", "KQo", "QQ", "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s",
                "AJo", "KJo", "QJo", "JJ", "JTs", "J9s", "J8s", "J7s", "J6s", "J5s", "J4s",
                "ATo", "KTo", "QTo", "JTo", "TT", "T9s", "T8s", "T7s", "T6s",
                "A9o", "K9o", "Q9o", "J9o", "T9o", "99", "98s", "97s", "96s",
                "A8o", "K8o", "T8o", "98o", "88", "87s", "86s", "85s",
                "A7o", "77", "76s", "75s", "A6o", "66", "65s", "64s",
                "A5o", "55", "54s", "53s", "A4o", "44", "33", "22",
            ),
        },
    },
    "vs_open_ip": {
        ("LJ_open", "HJ_hero"): {"3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A5s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "KQo", "QQ", "QJs", "JJ", "TT", "99")},
        ("LJ_open", "CO_hero"): {"3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A5s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "KQo", "QQ", "QJs", "JJ", "TT", "99", "88")},
        ("HJ_open", "CO_hero"): {"3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A5s", "A4s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "KQo", "QQ", "QJs", "AJo", "JJ", "TT", "99", "88")},
        ("LJ_open", "BTN_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "A9s", "A8s", "A4s", "A3s", "AKo", "KK", "K9s", "KQo", "QQ", "QJs", "AJo", "JJ", "T9s"),
            "call": _s("AJs", "ATs", "A5s", "KQs", "KJs", "KTs", "AQo", "QTs", "JTs", "TT", "99", "88", "77", "76s", "66", "65s", "55", "54s"),
        },
        ("HJ_open", "BTN_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "A9s", "A8s", "A7s", "A4s", "A3s", "AKo", "KK", "KTs", "K9s", "K8s", "KQo", "QQ", "QTs", "Q9s", "AJo", "JJ", "T9s", "66"),
            "call": _s("AJs", "ATs", "A5s", "KQs", "KJs", "AQo", "QJs", "JTs", "TT", "99", "98s", "88", "87s", "77", "55", "44"),
        },
        ("CO_open", "BTN_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "A8s", "A7s", "A6s", "A4s", "A3s", "AKo", "KK", "KQs", "K9s", "KQo", "QQ", "QJs", "Q9s", "AJo", "KJo", "QJo", "JJ", "JTs", "J9s", "ATo", "TT", "55"),
            "call": _s("AJs", "ATs", "A9s", "A5s", "KJs", "KTs", "AQo", "QTs", "T9s", "99", "98s", "88", "77", "66"),
        },
    },
    "vs_open_oop": {
        ("LJ_open", "SB_hero"): {"3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A5s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "QQ", "QJs", "JJ", "TT", "99")},
        ("HJ_open", "SB_hero"): {"3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A5s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "QQ", "QJs", "QTs", "JJ", "JTs", "TT", "99", "88", "77")},
        ("CO_open", "SB_hero"): {"3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A5s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "KQo", "QQ", "QJs", "QTs", "JJ", "JTs", "J9s", "TT", "T9s", "99", "88", "77", "66")},
        ("BTN_open", "SB_hero"): {"3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A5s", "A4s", "AKo", "KK", "KQs", "KJs", "KTs", "K9s", "AQo", "KQo", "QQ", "QJs", "QTs", "Q9s", "AJo", "KJo", "JJ", "JTs", "J9s", "TT", "T9s", "T8s", "99", "88", "77", "66", "55")},
        ("LJ_open", "BB_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "A5s", "A4s", "AKo", "KK", "KQs", "KJs", "QQ", "QJs", "JJ", "JTs", "65s", "54s"),
            "call": _s("AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A3s", "A2s", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s", "AQo", "KQo", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "AJo", "KJo", "QJo", "J9s", "J8s", "ATo", "JTo", "TT", "T9s", "T8s", "T7s", "99", "98s", "97s", "96s", "88", "87s", "86s", "85s", "77", "76s", "75s", "74s", "66", "64s", "63s", "55", "53s", "44", "43s", "33", "32s", "22"),
        },
        ("HJ_open", "BB_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "A9s", "A5s", "A4s", "AKo", "KK", "KQs", "KJs", "KTs", "K5s", "QQ", "QJs", "QTs", "JJ", "JTs", "TT", "65s", "54s"),
            "call": _s("AJs", "ATs", "A8s", "A7s", "A6s", "A3s", "A2s", "K9s", "K8s", "K7s", "K6s", "K4s", "K3s", "K2s", "AQo", "KQo", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "AJo", "KJo", "QJo", "J9s", "J8s", "J7s", "ATo", "KTo", "QTo", "JTo", "T9s", "T8s", "T7s", "A9o", "99", "98s", "97s", "96s", "88", "87s", "86s", "85s", "77", "76s", "75s", "74s", "66", "64s", "63s", "55", "53s", "44", "43s", "33", "22"),
        },
        ("CO_open", "BB_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "AJs", "A9s", "A5s", "A4s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "QQ", "QJs", "QTs", "Q9s", "JJ", "JTs", "J9s", "TT", "T9s", "99", "65s", "54s"),
            "call": _s("ATs", "A8s", "A7s", "A6s", "A3s", "A2s", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s", "KQo", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "AJo", "KJo", "QJo", "J8s", "J7s", "J6s", "ATo", "KTo", "QTo", "JTo", "T8s", "T7s", "A9o", "T9o", "98s", "97s", "96s", "A8o", "88", "87s", "86s", "85s", "77", "76s", "75s", "74s", "66", "64s", "63s", "A5o", "55", "53s", "52s", "44", "43s", "33", "22"),
        },
        ("BTN_open", "BB_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A6s", "A5s", "A4s", "AKo", "KK", "KQs", "KJs", "KTs", "K9s", "AQo", "KQo", "QQ", "QJs", "QTs", "Q9s", "JJ", "JTs", "J9s", "J8s", "TT", "T9s", "T8s", "99", "98s", "97s", "88", "87s", "76s", "65s", "54s"),
            "call": _s("A9s", "A8s", "A7s", "A3s", "A2s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s", "AJo", "KJo", "QJo", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s", "ATo", "KTo", "QTo", "JTo", "T7s", "T6s", "T5s", "T4s", "T3s", "T2s", "A9o", "K9o", "Q9o", "J9o", "T9o", "96s", "95s", "94s", "A8o", "K8o", "Q8o", "J8o", "T8o", "98o", "86s", "85s", "84s", "A7o", "K7o", "87o", "77", "75s", "74s", "73s", "A6o", "K6o", "76o", "66", "64s", "63s", "62s", "A5o", "65o", "55", "53s", "52s", "A4o", "54o", "44", "43s", "42s", "A3o", "33", "32s", "22"),
        },
    },
    "blind_vs_blind": {
        "SB_complete": {
            "raise_4bet": _s("AKs", "KK", "AQo", "QQ", "AJo", "KJo", "JJ"),
            "raise_call": _s("ATs", "A9s", "A8s", "A7s", "A5s", "KJs", "KTs", "K8s", "K5s", "QJs", "QTs", "JTs", "T9s", "K9o", "Q9o", "J9o", "98o", "65s", "54s", "33", "22"),
            "raise_fold": _s("K3s", "K2s", "Q5s", "Q4s", "Q3s", "Q2s", "J7s", "J6s", "J5s", "J4s", "T6s", "T5s", "96s", "A8o", "K8o", "T8o", "A7o", "K7o", "A6o", "64s", "53s", "A4o"),
            "limp_raise": _s("AA", "AQs", "AJs", "AKo", "KQs", "K9s", "Q9s", "J9s", "TT", "99", "98s", "88", "87s"),
            "limp_call": _s("A6s", "A4s", "A3s", "A2s", "K7s", "K6s", "K4s", "KQo", "Q8s", "Q7s", "Q6s", "QJo", "J8s", "ATo", "KTo", "QTo", "JTo", "T8s", "T7s", "A9o", "T9o", "97s", "86s", "85s", "77", "76s", "75s", "66", "A5o", "55", "44"),
            "limp_fold": _s("J3s", "J2s", "T4s", "T3s", "95s", "94s", "Q8o", "J8o", "84s", "Q7o", "J7o", "T7o", "97o", "87o", "74s", "K6o", "Q6o", "86o", "76o", "63s", "K5o", "Q5o", "K4o", "43s", "A3o", "A2o"),
        },
        ("SB_limp", "BB_hero"): {
            "raise": _s("AA", "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A5s", "A4s", "A3s", "AKo", "KK", "KQs", "KJs", "KTs", "K9s", "K6s", "K5s", "AQo", "KQo", "QQ", "QJs", "QTs", "Q9s", "AJo", "KJo", "JJ", "JTs", "J9s", "J8s", "J2s", "ATo", "JTo", "TT", "T9s", "T8s", "T4s", "T3s", "T2s", "T9o", "99", "98s", "97s", "94s", "93s", "92s", "88", "87s", "86s", "84s", "J7o", "77", "76s", "75s", "74s", "73s", "Q6o", "J6o", "T6o", "96o", "66", "65s", "64s", "63s", "A5o", "K5o", "Q5o", "J5o", "T5o", "95o", "85o", "75o", "55", "54s", "K4o", "Q4o", "74o", "44", "33", "32s"),
        },
        ("SB_raise", "BB_hero"): {
            "3bet": _s("AA", "AKs", "AQs", "AJs", "ATs", "A5s", "A4s", "AKo", "KK", "KQs", "KJs", "KTs", "AQo", "QQ", "QJs", "JJ", "J5s", "TT", "T5s", "99", "95s", "J8o", "88", "87s", "J7o", "T7o", "76s", "A6o", "K6o", "Q6o", "65s", "K5o", "54s"),
            "call": _s("A9s", "A8s", "A7s", "A6s", "A3s", "A2s", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s", "KQo", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s", "AJo", "KJo", "QJo", "JTs", "J9s", "J8s", "J7s", "J6s", "J4s", "J3s", "J2s", "ATo", "KTo", "QTo", "JTo", "T9s", "T8s", "T7s", "T6s", "T4s", "T3s", "T2s", "A9o", "K9o", "Q9o", "J9o", "T9o", "98s", "97s", "96s", "94s", "93s", "92s", "A8o", "K8o", "Q8o", "T8o", "98o", "86s", "85s", "84s", "A7o", "K7o", "Q7o", "97o", "87o", "77", "75s", "74s", "73s", "86o", "76o", "66", "64s", "63s", "62s", "A5o", "65o", "55", "53s", "52s", "A4o", "54o", "44", "43s", "42s", "A3o", "33", "32s", "A2o", "22"),
        },
    },
}


def _combo(cards):
    ranks = sorted((cards[0][0], cards[1][0]), key=lambda rank: RANK_VALUE[rank], reverse=True)
    if ranks[0] == ranks[1]:
        return ranks[0] + ranks[1]
    return ranks[0] + ranks[1] + ("s" if cards[0][1] == cards[1][1] else "o")


def _combo_score(hand):
    ranks = [RANK_VALUE[hand[0]], RANK_VALUE[hand[1]]]
    high, low = max(ranks), min(ranks)
    pair = len(hand) == 2
    suited = len(hand) == 3 and hand[2] == "s"
    connected = abs(high - low) <= 2 or {high, low} == {14, 5}
    broadway = high >= RANK_VALUE["T"] and low >= RANK_VALUE["T"]
    if pair:
        return _clamp(0.46 + high / 18.0, 0.0, 1.0)
    score = (high + low) / 28.0
    if suited:
        score += 0.055
    if connected:
        score += 0.035
    if broadway:
        score += 0.045
    if high == 14 and low <= 5 and suited:
        score += 0.025
    return _clamp(score, 0.0, 1.0)


ALL_HOLE_COMBOS = [
    (c1, c2, _combo((c1, c2)), _combo_score(_combo((c1, c2))))
    for idx, c1 in enumerate(FULL_DECK)
    for c2 in FULL_DECK[idx + 1 :]
]


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


def _seat_to_bot_id(state):
    return {player.get("seat"): player.get("bot_id") for player in state.get("players", [])}


def _hero_bot_id(state):
    seat = state.get("seat_to_act")
    for player in state.get("players", []):
        if player.get("seat") == seat:
            return player.get("bot_id")
    return None


def _model_for(bot_id, seat=None):
    if not bot_id:
        return None
    model = OPPONENT_MODELS.get(bot_id)
    if model is None:
        model = OppModel(bot_id=bot_id, seat=seat)
        OPPONENT_MODELS[bot_id] = model
    model.last_seen_seat = seat
    return model


def _new_hand_observation(state):
    return {
        "processed_actions": 0,
        "observed_actions": [],
        "vpip_counted": set(),
        "pfr_counted": set(),
        "threebet_opp_counted": set(),
        "fold_to_threebet_counted": set(),
        "steal_opp_counted": set(),
        "preflop_raise_count": 0,
        "preflop_first_raiser": None,
        "preflop_second_raiser": None,
        "preflop_aggressor": None,
        "preflop_callers": set(),
        "steal_raiser": None,
        "steal_raiser_position": None,
        "steal_open_amount": 0,
        "saw_flop_counted": False,
        "cbet_seen": False,
        "cbet_pending": {},
        "flop_bettor": None,
        "flop_callers": set(),
        "turn_after_flop_call_pending": set(),
        "turn_after_flop_call_counted": set(),
    }


def _prune_hand_observations():
    if len(HAND_OBSERVATIONS) <= MAX_TRACKED_HANDS:
        return
    for hand_id in list(HAND_OBSERVATIONS.keys())[: len(HAND_OBSERVATIONS) - MAX_TRACKED_HANDS]:
        HAND_OBSERVATIONS.pop(hand_id, None)


def _count_dealt_hand(state):
    hero = _hero_bot_id(state)
    for player in state.get("players", []):
        bot_id = player.get("bot_id")
        if bot_id and bot_id != hero:
            model = _model_for(bot_id, player.get("seat"))
            model.hands += 1


def _count_saw_flop(state, obs):
    if obs["saw_flop_counted"] or state.get("street") == "preflop":
        return
    hero = _hero_bot_id(state)
    for player in state.get("players", []):
        if player.get("is_folded"):
            continue
        bot_id = player.get("bot_id")
        if bot_id and bot_id != hero:
            _model_for(bot_id, player.get("seat")).saw_flop_hands += 1
    obs["saw_flop_counted"] = True


def _observe_preflop_action(obs, entry, model, seat, positions, big_blind):
    action = entry.get("action")
    prior_raises = obs["preflop_raise_count"]
    position = positions.get(seat)

    if model and prior_raises == 1 and action in ("fold", "call", "raise", "all_in") and seat not in obs["threebet_opp_counted"]:
        model.threebet_opps += 1
        if action in ("raise", "all_in"):
            model.threebets += 1
        obs["threebet_opp_counted"].add(seat)

    if (
        model
        and prior_raises == 1
        and obs.get("steal_raiser") is not None
        and position in {"SB", "BB"}
        and action in ("fold", "call", "raise", "all_in")
        and seat not in obs["steal_opp_counted"]
    ):
        model.steal_opps += 1
        if action == "fold":
            model.folded_to_steal += 1
        stealer_position = obs.get("steal_raiser_position")
        if stealer_position in {"BTN", "SB"}:
            _increment_bucket(model.fold_to_steal_opps_by_stealer, stealer_position)
            if action == "fold":
                _increment_bucket(model.folded_to_steal_by_stealer, stealer_position)
        if position == "BB":
            size_bucket = _preflop_size_bucket(obs.get("steal_open_amount"), big_blind)
            _increment_bucket(model.bb_defend_opps_by_size, size_bucket)
            if action == "fold":
                _increment_bucket(model.bb_defend_folds_by_size, size_bucket)
            elif action == "call":
                _increment_bucket(model.bb_defend_calls_by_size, size_bucket)
            elif action in ("raise", "all_in"):
                _increment_bucket(model.bb_defend_raises_by_size, size_bucket)
        obs["steal_opp_counted"].add(seat)

    if (
        model
        and prior_raises == 2
        and seat == obs.get("preflop_first_raiser")
        and action in ("fold", "call", "raise", "all_in")
        and seat not in obs["fold_to_threebet_counted"]
    ):
        model.faced_threebet += 1
        model.fourbet_opps += 1
        if action == "fold":
            model.folded_to_threebet += 1
        if action in ("raise", "all_in"):
            model.fourbets += 1
        obs["fold_to_threebet_counted"].add(seat)

    if action in ("call", "raise", "all_in"):
        if model and seat not in obs["vpip_counted"]:
            model.vpip_hands += 1
            obs["vpip_counted"].add(seat)
        if prior_raises > 0 and action == "call":
            obs["preflop_callers"].add(seat)

    if action in ("raise", "all_in"):
        if model and seat not in obs["pfr_counted"]:
            model.pfr_hands += 1
            obs["pfr_counted"].add(seat)
        if prior_raises == 0:
            obs["preflop_first_raiser"] = seat
            if position in {"CO", "BTN", "SB"}:
                obs["steal_raiser"] = seat
                obs["steal_raiser_position"] = position
                obs["steal_open_amount"] = int(entry.get("amount") or 0)
        elif prior_raises == 1:
            obs["preflop_second_raiser"] = seat
        obs["preflop_raise_count"] += 1
        obs["preflop_aggressor"] = seat
        obs["preflop_callers"].clear()


def _pot_type_from_preflop_raises(raises):
    if raises >= 2:
        return "3bet"
    if raises == 1:
        return "srp"
    return "limped"


def _record_turn_after_flop_call_opportunities(obs, bettor_seat, seat_to_bot):
    for target in set(obs["flop_callers"]):
        if target == bettor_seat or target in obs["turn_after_flop_call_counted"]:
            continue
        target_model = _model_for(seat_to_bot.get(target), target)
        if target_model:
            target_model.turn_after_flop_call_opps += 1
            obs["turn_after_flop_call_pending"].add(target)
            obs["turn_after_flop_call_counted"].add(target)


def _observe_postflop_action(obs, entry, model, seat, street, seat_to_bot, pot):
    action = entry.get("action")
    if action in ("raise", "all_in"):
        if model:
            model.postflop_bets_raises += 1
            model.postflop_actions += 1
            if street in model.street_bets_raises:
                model.street_bets_raises[street] += 1
                if _postflop_large_bet(entry, pot):
                    model.street_large_bets[street] += 1
            if street == "river":
                model.river_bets_raises += 1
                model.river_actions += 1
            if street == "turn" and seat in obs["turn_after_flop_call_pending"]:
                model.turn_after_flop_call_raises += 1
        if street == "flop" and obs["flop_bettor"] is None:
            obs["flop_bettor"] = seat
        if street == "turn":
            _record_turn_after_flop_call_opportunities(obs, seat, seat_to_bot)
            obs["turn_after_flop_call_pending"].discard(seat)
        if street == "flop" and not obs["cbet_seen"] and seat == obs["preflop_aggressor"]:
            obs["cbet_seen"] = True
            pot_type = _pot_type_from_preflop_raises(obs["preflop_raise_count"])
            for target in set(obs["preflop_callers"]):
                if target == seat:
                    continue
                target_model = _model_for(seat_to_bot.get(target), target)
                if target_model:
                    target_model.cb_opps += 1
                    _increment_bucket(target_model.cb_opps_by_pot_type, pot_type)
                    obs["cbet_pending"][target] = pot_type
    elif action == "call":
        if model:
            model.postflop_calls += 1
            model.postflop_actions += 1
            if street == "river":
                model.river_calls += 1
                model.river_actions += 1
        if street == "flop" and obs["flop_bettor"] is not None and seat != obs["flop_bettor"]:
            obs["flop_callers"].add(seat)
        if street == "turn" and seat in obs["turn_after_flop_call_pending"]:
            if model:
                model.turn_after_flop_call_calls += 1
            obs["turn_after_flop_call_pending"].discard(seat)
        obs["cbet_pending"].pop(seat, None)
    elif action == "fold" and seat in obs["cbet_pending"]:
        if model:
            model.postflop_actions += 1
            model.cb_folds += 1
            _increment_bucket(model.cb_folds_by_pot_type, obs["cbet_pending"][seat])
            if seat in obs["turn_after_flop_call_pending"]:
                model.turn_after_flop_call_folds += 1
                if street == "river":
                    model.river_folds += 1
                    model.river_actions += 1
        obs["cbet_pending"].pop(seat, None)
        obs["turn_after_flop_call_pending"].discard(seat)
    elif action == "fold" and seat in obs["turn_after_flop_call_pending"]:
        if model:
            model.postflop_actions += 1
            model.turn_after_flop_call_folds += 1
            if street == "river":
                model.river_folds += 1
                model.river_actions += 1
        obs["turn_after_flop_call_pending"].discard(seat)
    elif action in {"check", "fold"} and model:
        model.postflop_actions += 1
        if street == "river":
            if action == "check":
                model.river_checks += 1
            elif action == "fold":
                model.river_folds += 1
            model.river_actions += 1
    elif action == "showdown" and model:
        model.showdown_hands += 1


def _update_opponent_models(state):
    if state.get("type") == "warmup":
        return
    hand_id = state.get("hand_id")
    if not hand_id:
        return

    obs = HAND_OBSERVATIONS.get(hand_id)
    if obs is None:
        obs = _new_hand_observation(state)
        HAND_OBSERVATIONS[hand_id] = obs
        _count_dealt_hand(state)
        _prune_hand_observations()

    _count_saw_flop(state, obs)

    hero_seat = state.get("seat_to_act")
    seat_to_bot = _seat_to_bot_id(state)
    positions = _seat_positions(state)
    _, big_blind = _blind_amounts(state)
    street = state.get("street", "preflop")
    actions = state.get("action_log", [])
    start = min(obs["processed_actions"], len(actions))

    for entry in actions[start:]:
        action = entry.get("action")
        entry_street = entry.get("street", street)
        annotated = dict(entry)
        annotated["street"] = entry_street
        obs["observed_actions"].append(annotated)
        if action in ("small_blind", "big_blind"):
            obs["processed_actions"] += 1
            continue
        seat = entry.get("seat")
        bot_id = seat_to_bot.get(seat)
        model = None if seat == hero_seat else _model_for(bot_id, seat)
        if entry_street == "preflop":
            _observe_preflop_action(obs, annotated, model, seat, positions, big_blind)
        else:
            _observe_postflop_action(obs, annotated, model, seat, entry_street, seat_to_bot, state.get("pot", 0))
        obs["processed_actions"] += 1


def _opponent_profiles(state=None):
    if state is not None:
        _update_opponent_models(state)
    return {bot_id: model.snapshot() for bot_id, model in OPPONENT_MODELS.items()}


def _reset_opponent_models():
    OPPONENT_MODELS.clear()
    HAND_OBSERVATIONS.clear()


def _live_seats(state):
    seats = []
    for player in state.get("players", []):
        if not player.get("is_folded") and (player.get("stack", 0) > 0 or player.get("is_all_in")):
            seats.append(player.get("seat"))
    return seats


def _seat_positions(state):
    players = state.get("players", [])
    n_players = max(2, len(players))
    small_blind, big_blind = _blind_seats(state)
    if small_blind is None:
        small_blind = 0 if n_players == 2 else 1
    if big_blind is None:
        big_blind = 1 if n_players == 2 else 2
    dealer = small_blind if n_players == 2 else (small_blind - 1) % n_players

    positions = {small_blind: "SB", big_blind: "BB", dealer: "BTN"}
    nonblind_order = []
    for offset in range(1, n_players + 1):
        seat = (big_blind + offset) % n_players
        if seat in positions:
            continue
        if seat in _live_seats(state) or seat < len(players):
            nonblind_order.append(seat)

    labels = ["LJ", "HJ", "CO"]
    if len(nonblind_order) <= len(labels):
        chosen = labels[-len(nonblind_order):] if nonblind_order else []
    else:
        chosen = ["LJ"] * (len(nonblind_order) - len(labels)) + labels
    for seat, label in zip(nonblind_order, chosen):
        positions[seat] = label
    return positions


def _position(state):
    return _seat_positions(state).get(state["seat_to_act"], "LJ")


def _raise_to(state, target):
    max_total = int(state.get("your_bet_this_street", 0)) + int(state.get("your_stack", 0))
    if max_total <= int(state.get("amount_owed", 0)) + int(state.get("your_bet_this_street", 0)):
        return {"action": "all_in"}
    amount = max(int(round(target)), int(state.get("min_raise_to", 0)))
    amount = min(amount, max_total)
    if amount <= int(state.get("current_bet", 0)):
        return {"action": "check"} if state.get("can_check") else {"action": "call"}
    if amount >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": amount}


def _safe_passive(state, fold_when_owed=True):
    if state.get("can_check") or state.get("amount_owed", 0) == 0:
        return {"action": "check"}
    return {"action": "fold"} if fold_when_owed else {"action": "call"}


def _bet_fraction(state, fraction):
    target = int(state.get("your_bet_this_street", 0)) + int(max(1, state.get("pot", 1)) * fraction)
    return _raise_to(state, max(int(state.get("min_raise_to", 0)), target))


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
        if sum(1 for value in range(start, start + 5) if value in values) == 4:
            return True
    return False


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


def _board_class(board):
    if len(board) < 3:
        return {
            "paired": False,
            "trips": False,
            "monotone": False,
            "two_tone": False,
            "rainbow": True,
            "top_rank": 0,
            "family": "none",
            "gaps": 99,
            "straight_potential": 0,
            "dynamic_score": 0,
            "is_dynamic": False,
            "is_static": True,
            "low_connected": False,
            "broadway_heavy": False,
        }

    ranks = [card[0] for card in board]
    suits = [card[1] for card in board]
    rank_values = sorted((RANK_VALUE[rank] for rank in ranks), reverse=True)
    unique_values = sorted(set(rank_values))
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}
    max_suit = max(suit_counts.values() or [0])
    paired = len(set(ranks)) < len(ranks)
    trips = any(ranks.count(rank) >= 3 for rank in set(ranks))
    monotone = max_suit >= 3
    two_tone = max_suit == 2
    rainbow = max_suit == 1
    top = max(rank_values)

    gaps = 99
    if len(unique_values) >= 3:
        top3 = unique_values[-3:]
        gaps = (top3[2] - top3[1] - 1) + (top3[1] - top3[0] - 1)

    straight_windows = [set(range(start, start + 5)) for start in range(2, 11)]
    straight_windows.append({14, 2, 3, 4, 5})
    straight_potential = max(len(set(rank_values) & window) for window in straight_windows)
    broadway_heavy = sum(1 for rank in ranks if rank in HIGH_CARDS) >= 2

    dynamic_score = 0
    if two_tone:
        dynamic_score += 2
    if monotone:
        dynamic_score += 2
    if gaps <= 2:
        dynamic_score += 2
    if straight_potential >= 3:
        dynamic_score += 1
    if top <= 9:
        dynamic_score += 1
    if broadway_heavy:
        dynamic_score += 1
    if rainbow:
        dynamic_score -= 1
    if top >= 13 and gaps >= 4 and rainbow:
        dynamic_score -= 2

    if top == 14:
        family = "Axx"
    elif top >= 10:
        family = "Hxx"
    elif top >= 6:
        family = "Mxx"
    else:
        family = "Lxx"

    return {
        "paired": paired,
        "trips": trips,
        "monotone": monotone,
        "two_tone": two_tone,
        "rainbow": rainbow,
        "top_rank": top,
        "family": family,
        "gaps": gaps,
        "straight_potential": straight_potential,
        "dynamic_score": dynamic_score,
        "is_dynamic": dynamic_score >= 3,
        "is_static": dynamic_score <= 0,
        "low_connected": top <= 7 and straight_potential >= 3 and gaps <= 2,
        "broadway_heavy": broadway_heavy,
    }


def _hand_info(state):
    hole = state.get("your_cards", [])
    board = state.get("community_cards", [])
    if len(hole) < 2:
        return {"strength": "air", "bucket": "trash", "made": 0}

    all_cards = list(hole) + list(board)
    ranks = [card[0] for card in all_cards]
    board_ranks = [card[0] for card in board]
    hole_ranks = [card[0] for card in hole]
    hole_values = sorted((RANK_VALUE[rank] for rank in hole_ranks), reverse=True)
    board_top = max((RANK_VALUE[rank] for rank in board_ranks), default=0)
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    hand_type = _hand_type(all_cards) if board else "high card"
    made = {
        "high card": 0,
        "pair": 1,
        "two pair": 2,
        "three of a kind": 3,
        "straight": 4,
        "flush": 5,
        "full house": 6,
        "four of a kind": 7,
        "straight flush": 8,
    }.get(hand_type, 0)

    pocket_pair = hole_ranks[0] == hole_ranks[1]
    pair_uses_hole = any(rank_counts.get(rank, 0) >= 2 for rank in hole_ranks)
    top_pair = any(RANK_VALUE[rank] == board_top and rank_counts.get(rank, 0) >= 2 for rank in hole_ranks)
    second_pair = any(0 < RANK_VALUE[rank] < board_top and rank_counts.get(rank, 0) >= 2 for rank in hole_ranks)
    overpair = pocket_pair and board_top and RANK_VALUE[hole_ranks[0]] > board_top
    set_made = pocket_pair and hole_ranks[0] in board_ranks and made >= 3
    kicker_values = [value for value in hole_values if value != board_top]
    kicker = max(kicker_values or hole_values or [0])
    tptk = top_pair and kicker >= RANK_VALUE["Q"]

    suit_counts = {}
    board_suit_counts = {}
    for card in all_cards:
        suit_counts[card[1]] = suit_counts.get(card[1], 0) + 1
    for card in board:
        board_suit_counts[card[1]] = board_suit_counts.get(card[1], 0) + 1
    flush_draw = made < 5 and len(board) < 5 and max(suit_counts.values() or [0]) >= 4
    oesd = len(board) < 5 and not _has_straight(ranks) and _has_open_ended_draw(ranks)
    gutshot = len(board) < 5 and not oesd and not _has_straight(ranks) and _has_gutshot(ranks)
    overcards = bool(board) and made == 0 and sum(1 for value in hole_values if value > board_top) >= 1
    combo_draw = flush_draw and (oesd or gutshot or overcards or pair_uses_hole)

    nut_flush_blocker = False
    board_flush_suit = None
    if board_suit_counts:
        board_flush_suit = max(board_suit_counts, key=board_suit_counts.get)
        nut_flush_blocker = board_suit_counts[board_flush_suit] >= 3 and f"A{board_flush_suit}" in hole

    if made >= 4 or set_made or made >= 2:
        strength = "monster"
        bucket = "strong"
    elif overpair or tptk:
        strength = "tptk_plus"
        bucket = "good"
    elif top_pair:
        strength = "top_pair"
        bucket = "good"
    elif pair_uses_hole and (made == 1 or second_pair):
        strength = "medium_pair"
        bucket = "weak"
    elif combo_draw or flush_draw or oesd:
        strength = "strong_draw"
        bucket = "weak"
    elif gutshot or overcards:
        strength = "weak_equity"
        bucket = "weak"
    else:
        strength = "air"
        bucket = "trash"

    return {
        "made": made,
        "type": hand_type,
        "strength": strength,
        "bucket": bucket,
        "top_pair": top_pair,
        "tptk": tptk,
        "overpair": overpair,
        "set": set_made,
        "flush_draw": flush_draw,
        "oesd": oesd,
        "gutshot": gutshot,
        "overcards": overcards,
        "combo_draw": combo_draw,
        "nut_flush_blocker": nut_flush_blocker,
        "blocks_value": nut_flush_blocker or set_made or any(rank in board_ranks for rank in hole_ranks),
        "blocks_bluffs": flush_draw or oesd or gutshot,
        "showdown": strength in {"monster", "tptk_plus", "top_pair", "medium_pair"},
    }


def _opponents_in_hand(state):
    hero_seat = state.get("seat_to_act")
    return [
        player
        for player in state.get("players", [])
        if player.get("seat") != hero_seat
        and not player.get("is_folded")
        and (player.get("stack", 0) > 0 or player.get("is_all_in"))
    ]


def _postflop_in_position(state):
    live = set(_live_seats(state))
    if not live:
        return True
    players = state.get("players", [])
    small_blind, _ = _blind_seats(state)
    if small_blind is None:
        small_blind = 0
    dealer = small_blind if len(players) <= 2 else (small_blind - 1) % max(1, len(players))
    order = [(dealer + offset) % max(1, len(players)) for offset in range(1, len(players) + 1)]
    live_order = [seat for seat in order if seat in live]
    return bool(live_order and state.get("seat_to_act") == live_order[-1])


def _preflop_raise_count(state):
    return sum(
        1
        for entry in _observed_actions(state, "preflop")
        if entry.get("action") in {"raise", "all_in"} and entry.get("street", "preflop") == "preflop"
    )


def _last_preflop_raiser(state):
    raiser = None
    for entry in _observed_actions(state, "preflop"):
        if entry.get("action") in {"raise", "all_in"} and entry.get("street", "preflop") == "preflop":
            raiser = entry.get("seat")
    return raiser


def _observed_actions(state, street=None):
    obs = HAND_OBSERVATIONS.get(state.get("hand_id"))
    if obs and obs.get("observed_actions"):
        actions = obs["observed_actions"]
    else:
        actions = state.get("action_log", [])
    if street is None:
        return list(actions)
    return [
        entry
        for entry in actions
        if entry.get("street", "preflop" if street == "preflop" else None) == street
    ]


def _postflop_actions(state, street=None):
    target_street = street or state.get("street")
    return [
        entry
        for entry in _observed_actions(state, target_street)
        if entry.get("street") == target_street and entry.get("action") not in {"small_blind", "big_blind"}
    ]


def _hero_bet_on_street(state, street):
    hero = state.get("seat_to_act")
    return any(entry.get("seat") == hero and entry.get("action") in {"raise", "all_in"} for entry in _postflop_actions(state, street))


def _street_checked_through(state, street):
    actions = _postflop_actions(state, street)
    return bool(actions) and all(entry.get("action") == "check" for entry in actions)


def _model_for_seat(state, seat):
    bot_id = _seat_to_bot_id(state).get(seat)
    return OPPONENT_MODELS.get(bot_id)


def _last_bettor_or_raiser_on_street(state, street=None):
    hero = state.get("seat_to_act")
    for entry in reversed(_postflop_actions(state, street or state.get("street"))):
        if entry.get("seat") != hero and entry.get("action") in {"raise", "all_in"}:
            return entry.get("seat")
    return None


def _danger_score(model):
    label = _label_for_model(model)
    label_score = {
        "calling_station": 60,
        "aggressive": 55,
        "unknown": 40,
        "nit": 30,
        "tight_passive": 25,
        "overfolder": 10,
    }.get(label, 40)
    return (
        label_score,
        model.postflop_aggression_frequency,
        1.0 - model.bayes_fold_to_cbet,
        model.hands,
    )


def _most_dangerous_opponent_model(state, opponents):
    models = []
    for player in opponents:
        model = _model_for_seat(state, player.get("seat"))
        if model:
            models.append(model)
    if not models:
        return None
    return max(models, key=_danger_score)


def _context_opponent_model(state):
    opponents = _opponents_in_hand(state)
    if not opponents:
        return None

    street = state.get("street")
    facing_bet = not state.get("can_check") or int(state.get("amount_owed", 0)) > 0
    if street != "preflop" and facing_bet:
        bettor = _last_bettor_or_raiser_on_street(state, street)
        if bettor is not None:
            model = _model_for_seat(state, bettor)
            if model:
                return model

    if len(opponents) == 1:
        return _model_for_seat(state, opponents[0].get("seat"))

    return _most_dangerous_opponent_model(state, opponents)


def _primary_opponent_model(state):
    return _context_opponent_model(state)


def _villain_read(model):
    if not model:
        return {
            "label": "unknown",
            "overfold_steal_score": 0.0,
            "overfold_cbet_score": 0.0,
            "station_score": 0.0,
            "aggro_score": 0.0,
            "nit_score": 0.0,
            "tight_passive_score": 0.0,
        }

    overfold_steal = min(
        _exploit_strength_above(model, "fold_to_steal", 0.68, min_opp_soft=8),
        _exploit_strength_below(model, "threebet", 0.08, min_opp_soft=8),
    )
    overfold_cbet = _exploit_strength_above(model, "fold_to_cbet", 0.55, min_opp_soft=8)
    loose_passive = min(
        _exploit_strength_above(model, "vpip", 0.34, min_opp_soft=20),
        _exploit_strength_below(model, "pfr", 0.16, min_opp_soft=20),
        _exploit_strength_below(model, "postflop_aggression_frequency", 0.36, min_opp_soft=12),
    )
    sticky_postflop = min(
        _exploit_strength_below(model, "fold_to_cbet", 0.38, min_opp_soft=8),
        _exploit_strength_above(model, "wtsd", 0.32, min_opp_soft=15),
    )
    station = max(loose_passive, sticky_postflop)
    aggro = max(
        _exploit_strength_above(model, "threebet", 0.13, min_opp_soft=10),
        _exploit_strength_above(model, "postflop_aggression_frequency", 0.48, min_opp_soft=12),
    )
    nit = min(
        _exploit_strength_below(model, "vpip", 0.18, min_opp_soft=25),
        _exploit_strength_below(model, "pfr", 0.13, min_opp_soft=25),
        _exploit_strength_below(model, "threebet", 0.06, min_opp_soft=10),
    )
    tight_passive = min(
        _exploit_strength_below(model, "vpip", 0.25, min_opp_soft=25),
        _exploit_strength_below(model, "pfr", 0.12, min_opp_soft=25),
        _exploit_strength_below(model, "postflop_aggression_frequency", 0.35, min_opp_soft=12),
    )

    scores = {
        "overfolder": max(overfold_steal, overfold_cbet),
        "calling_station": station,
        "aggressive": aggro,
        "nit": nit,
        "tight_passive": tight_passive,
    }
    label = max(scores, key=scores.get)
    if scores[label] < 0.45:
        label = "unknown"

    return {
        "label": label,
        "overfold_steal_score": overfold_steal,
        "overfold_cbet_score": overfold_cbet,
        "station_score": station,
        "aggro_score": aggro,
        "nit_score": nit,
        "tight_passive_score": tight_passive,
    }


def _label_for_model(model):
    if not model:
        return "unknown"

    nit = (
        _likely_below(model, "vpip", 0.18, confidence=0.80, min_opp=40)
        and _likely_below(model, "pfr", 0.13, confidence=0.80, min_opp=40)
        and _likely_below(model, "threebet", 0.06, confidence=0.75, min_opp=12)
    )
    if nit:
        return "nit"

    tight_passive = (
        _likely_below(model, "vpip", 0.25, confidence=0.80, min_opp=40)
        and _likely_below(model, "pfr", 0.12, confidence=0.80, min_opp=40)
        and _likely_below(model, "postflop_aggression_frequency", 0.35, confidence=0.75, min_opp=15)
    )
    if tight_passive:
        return "tight_passive"

    loose_passive = (
        _likely_above(model, "vpip", 0.34, confidence=0.80, min_opp=40)
        and _likely_below(model, "pfr", 0.16, confidence=0.75, min_opp=40)
        and _likely_below(model, "postflop_aggression_frequency", 0.36, confidence=0.75, min_opp=15)
    )
    sticky_postflop = (
        _likely_below(model, "fold_to_cbet", 0.38, confidence=0.80, min_opp=12)
        and _likely_above(model, "wtsd", 0.32, confidence=0.75, min_opp=20)
    )
    if loose_passive or sticky_postflop:
        return "calling_station"

    cbet_overfolder = _likely_above(model, "fold_to_cbet", 0.55, confidence=0.85, min_opp=12)
    blind_overfolder = (
        _likely_above(model, "fold_to_steal", 0.68, confidence=0.85, min_opp=10)
        and _likely_below(model, "threebet", 0.08, confidence=0.75, min_opp=10)
    )
    if cbet_overfolder or blind_overfolder:
        return "overfolder"

    postflop_aggressive = _likely_above(
        model, "postflop_aggression_frequency", 0.48, confidence=0.85, min_opp=15
    )
    preflop_aggressive = _likely_above(model, "threebet", 0.13, confidence=0.85, min_opp=18)
    if postflop_aggressive or preflop_aggressive:
        return "aggressive"

    return "unknown"


def _opponent_label(state):
    return _label_for_model(_primary_opponent_model(state))


def _pot_type(state):
    raises = _preflop_raise_count(state)
    if raises >= 2:
        return "3bet"
    if raises == 1:
        return "srp"
    return "limped"


def _postflop_role(state):
    hero = state.get("seat_to_act")
    pfa = _last_preflop_raiser(state)
    in_position = _postflop_in_position(state)
    if pfa == hero:
        return "IP_PFA" if in_position else "OOP_PFA"
    if _position(state) == "BB" and not in_position:
        return "BB_CALLER_OOP"
    return "IP_CALLER" if in_position else "OOP_CALLER"


def _postflop_size(state, board, info, role):
    street = state.get("street")
    pot_type = _pot_type(state)
    spr = _effective_stack_bb(state) * max(1, _blind_amounts(state)[1]) / max(1, state.get("pot", 1))
    opponent = _opponent_label(state)
    heads_up = len(_opponents_in_hand(state)) <= 1

    if heads_up and opponent == "calling_station" and info["strength"] in {"monster", "tptk_plus", "top_pair", "medium_pair"}:
        if street == "river":
            return 0.60 if info["strength"] in {"top_pair", "medium_pair"} else 0.75
        return 0.75

    if street == "flop":
        if role == "BB_CALLER_OOP" and board["low_connected"] and not board["monotone"]:
            return 0.67 if spr <= 3 else 0.25
        if pot_type == "3bet" and role == "IP_PFA":
            return 0.25
        if board["paired"] or board["monotone"] or board["is_static"]:
            return 0.33
        if role == "OOP_PFA" or info["strength"] in {"monster", "strong_draw"}:
            return 0.67
        return 0.50

    if street == "turn":
        if info["strength"] in {"monster", "strong_draw"} or board["is_dynamic"]:
            return 0.67
        return 0.33

    if street == "river":
        if info["strength"] == "monster" or info["nut_flush_blocker"]:
            return 0.80
        return 0.50

    return 0.50


def _good_flop_bluff(info, board, role):
    if info["combo_draw"] or info["flush_draw"] or info["oesd"]:
        return True
    if role in {"IP_PFA", "OOP_PFA"} and (info["overcards"] or info["gutshot"]):
        return board["is_static"] or board["family"] in {"Axx", "Hxx"} or board["paired"]
    if info["nut_flush_blocker"] and board["monotone"]:
        return True
    return False


def _turn_favors_probe(board):
    return board["low_connected"] or board["paired"] or board["monotone"] or board["top_rank"] <= 9


def _scare_turn_for_pfa(board):
    return board["top_rank"] >= RANK_VALUE["K"] or board["monotone"] or board["paired"]


def _eval7_cards(cards):
    out = []
    for card in cards:
        cached = EVAL7_CARD_CACHE.get(card)
        if cached is None:
            cached = eval7.Card(card)
            EVAL7_CARD_CACHE[card] = cached
        out.append(cached)
    return out


def _eval7_rank(cards):
    return eval7.evaluate(_eval7_cards(cards))


def _stable_seed_for_state(state, board_cards):
    parts = [
        state.get("hand_id", ""),
        state.get("street", ""),
        ",".join(state.get("your_cards", [])),
        ",".join(board_cards),
        str(state.get("pot", 0)),
        str(state.get("amount_owed", 0)),
        str(len(_observed_actions(state))),
    ]
    text = "|".join(parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def _candidate_info(hole, board_cards):
    all_cards = list(hole) + list(board_cards)
    ranks = [card[0] for card in all_cards]
    board_ranks = [card[0] for card in board_cards]
    hole_ranks = [card[0] for card in hole]
    hole_values = sorted((RANK_VALUE[rank] for rank in hole_ranks), reverse=True)
    board_top = max((RANK_VALUE[rank] for rank in board_ranks), default=0)
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    hand_type = _hand_type(all_cards) if board_cards else "high card"
    made = {
        "high card": 0,
        "pair": 1,
        "two pair": 2,
        "three of a kind": 3,
        "straight": 4,
        "flush": 5,
        "full house": 6,
        "four of a kind": 7,
        "straight flush": 8,
    }.get(hand_type, 0)
    pocket_pair = hole_ranks[0] == hole_ranks[1]
    pair_uses_hole = any(rank_counts.get(rank, 0) >= 2 for rank in hole_ranks)
    top_pair = any(RANK_VALUE[rank] == board_top and rank_counts.get(rank, 0) >= 2 for rank in hole_ranks)
    second_pair = any(0 < RANK_VALUE[rank] < board_top and rank_counts.get(rank, 0) >= 2 for rank in hole_ranks)
    overpair = pocket_pair and board_top and RANK_VALUE[hole_ranks[0]] > board_top
    suit_counts = {}
    board_suit_counts = {}
    for card in all_cards:
        suit_counts[card[1]] = suit_counts.get(card[1], 0) + 1
    for card in board_cards:
        board_suit_counts[card[1]] = board_suit_counts.get(card[1], 0) + 1
    flush_draw = made < 5 and len(board_cards) < 5 and max(suit_counts.values() or [0]) >= 4
    oesd = len(board_cards) < 5 and not _has_straight(ranks) and _has_open_ended_draw(ranks)
    gutshot = len(board_cards) < 5 and not oesd and not _has_straight(ranks) and _has_gutshot(ranks)
    overcards = bool(board_cards) and made == 0 and sum(1 for value in hole_values if value > board_top) >= 1
    board_flush_suit = max(board_suit_counts, key=board_suit_counts.get) if board_suit_counts else None
    nut_flush_blocker = bool(
        board_flush_suit
        and board_suit_counts[board_flush_suit] >= 3
        and f"A{board_flush_suit}" in hole
    )
    draw_power = 0.0
    if flush_draw:
        draw_power += 0.55
    if oesd:
        draw_power += 0.40
    if gutshot:
        draw_power += 0.20
    if overcards:
        draw_power += 0.12
    if nut_flush_blocker:
        draw_power += 0.12
    if pair_uses_hole:
        draw_power += 0.08
    return {
        "made": made,
        "type": hand_type,
        "top_pair": top_pair,
        "second_pair": second_pair,
        "overpair": overpair,
        "pair_uses_hole": pair_uses_hole,
        "flush_draw": flush_draw,
        "oesd": oesd,
        "gutshot": gutshot,
        "overcards": overcards,
        "nut_flush_blocker": nut_flush_blocker,
        "draw_power": draw_power,
    }


def _model_looseness_factor(model):
    if not model:
        return 1.0
    return _clamp(0.75 + model.bayes_vpip, 0.75, 1.30)


def _model_aggression_factor(model):
    if not model:
        return 1.0
    return _clamp(0.75 + model.postflop_aggression_frequency, 0.75, 1.40)


def _preflop_range_weight(hand, score, action, raise_depth, model):
    action = "raise" if action == "bet" else action
    suited = len(hand) == 3 and hand[2] == "s"
    pair = len(hand) == 2
    connector = len(hand) == 3 and abs(RANK_VALUE[hand[0]] - RANK_VALUE[hand[1]]) <= 2
    wheel_ace = hand in {"A5s", "A4s", "A3s", "A2s"}
    loose = _model_looseness_factor(model)
    aggro = _model_aggression_factor(model)

    if action in {"raise", "all_in"}:
        threshold = 0.52 + 0.085 * max(0, raise_depth - 1)
        value = 0.04 + 4.1 * max(0.0, score - threshold) ** 1.25
        bluff = 0.18 if (suited and (connector or wheel_ace)) else 0.025
        if action == "all_in":
            value *= 1.45
            bluff *= 0.25
        return _clamp(value * aggro + bluff, 0.01, 5.0)
    if action == "call":
        implied = 0.0
        if pair:
            implied += 0.45
        if suited:
            implied += 0.28
        if connector:
            implied += 0.18
        return _clamp((0.08 + score * 0.45 + implied) * loose, 0.01, 3.2)
    if action == "check":
        return _clamp(0.75 + loose * 0.20, 0.20, 1.35)
    if action == "fold":
        return 0.0
    return 1.0


def _postflop_range_weight(hole, board_cards, action, model):
    action = "raise" if action in {"bet", "all_in"} else action
    info = _candidate_info(hole, board_cards)
    made = info["made"]
    draw = info["draw_power"]
    aggro = _model_aggression_factor(model)
    loose = _model_looseness_factor(model)

    if action == "raise":
        value_by_made = [0.06, 0.18, 0.45, 0.75, 1.25, 1.80, 2.30, 2.80, 3.20]
        value = value_by_made[made]
        if info["top_pair"]:
            value += 0.42
        if info["overpair"]:
            value += 0.65
        air = 0.08 * aggro if made <= 1 and draw < 0.25 else 0.0
        return _clamp(value * aggro + draw * (0.85 + aggro * 0.55) + air, 0.01, 5.0)
    if action == "call":
        call_by_made = [0.04, 0.62, 1.00, 1.10, 1.35, 1.55, 1.35, 1.20, 1.20]
        value = call_by_made[made]
        if info["top_pair"] or info["overpair"]:
            value += 0.28
        return _clamp((value + draw * 0.85) * loose, 0.01, 3.6)
    if action == "check":
        check_by_made = [0.95, 1.15, 1.05, 0.95, 0.82, 0.72, 0.66, 0.60, 0.60]
        value = check_by_made[made] + draw * 0.20
        if made >= 4:
            value *= 1.0 - (aggro - 0.75) * 0.22
        return _clamp(value, 0.02, 2.0)
    if action == "fold":
        return 0.0
    return 1.0


def _make_weighted_range(items):
    clean = []
    cumulative = []
    total = 0.0
    for c1, c2, weight in items:
        if weight <= 0:
            continue
        total += weight
        clean.append((c1, c2, weight))
        cumulative.append(total)
    return {"items": clean, "cum": cumulative, "total": total}


def _cache_equity_range(key, value):
    if len(EQUITY_RANGE_CACHE) >= MAX_EQUITY_RANGE_CACHE:
        EQUITY_RANGE_CACHE.pop(next(iter(EQUITY_RANGE_CACHE)), None)
    EQUITY_RANGE_CACHE[key] = value
    return value


def _build_heads_up_villain_range(state, board_cards):
    opponents = _opponents_in_hand(state)
    if len(opponents) != 1:
        return None
    villain = opponents[0]
    villain_seat = villain.get("seat")
    known = set(state.get("your_cards", [])) | set(board_cards)
    if len(state.get("your_cards", [])) < 2 or len(board_cards) < 3:
        return None

    actions = _observed_actions(state)
    cache_key = (
        state.get("hand_id"),
        state.get("street"),
        villain_seat,
        tuple(state.get("your_cards", [])),
        tuple(board_cards),
        len(actions),
    )
    cached = EQUITY_RANGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = _model_for_seat(state, villain_seat)
    preflop_actions = [
        entry
        for entry in _observed_actions(state, "preflop")
        if entry.get("seat") == villain_seat and entry.get("action") not in {"small_blind", "big_blind"}
    ]
    postflop_actions = [
        entry
        for entry in actions
        if entry.get("seat") == villain_seat
        and entry.get("street") in {"flop", "turn", "river"}
        and entry.get("action") not in {"small_blind", "big_blind"}
    ]

    items = []
    raise_depth = 0
    for c1, c2, hand, score in ALL_HOLE_COMBOS:
        if c1 in known or c2 in known:
            continue
        weight = 1.0
        raise_depth = 0
        for action_entry in preflop_actions:
            action = action_entry.get("action")
            if action in {"raise", "all_in"}:
                raise_depth += 1
            weight *= _preflop_range_weight(hand, score, action, raise_depth, model)
            if weight <= 0:
                break
        if weight <= 0:
            continue
        for action_entry in postflop_actions:
            weight *= _postflop_range_weight((c1, c2), board_cards, action_entry.get("action"), model)
            if weight <= 0:
                break
        if weight > 0.0001:
            items.append((c1, c2, weight))

    if not items:
        items = [(c1, c2, 1.0) for c1, c2, _, _ in ALL_HOLE_COMBOS if c1 not in known and c2 not in known]
    return _cache_equity_range(cache_key, _make_weighted_range(items))


def _weighted_combo_choice(villain_range, dead, rng):
    items = villain_range.get("items", [])
    cumulative = villain_range.get("cum", [])
    total = villain_range.get("total", 0.0)
    if not items or total <= 0:
        return None
    for _ in range(25):
        idx = bisect.bisect_left(cumulative, rng.random() * total)
        if idx >= len(items):
            idx = len(items) - 1
        c1, c2, _ = items[idx]
        if c1 not in dead and c2 not in dead:
            return c1, c2
    for c1, c2, _ in items:
        if c1 not in dead and c2 not in dead:
            return c1, c2
    return None


def _estimate_equity(hero_cards, board_cards, villain_range, samples=600, rng=None):
    if len(hero_cards) < 2 or not villain_range or not villain_range.get("items"):
        return None
    board_cards = list(board_cards)
    known = set(hero_cards) | set(board_cards)
    need_board = max(0, 5 - len(board_cards))
    rng = rng or random.Random(0)

    if need_board == 0:
        hero_rank = _eval7_rank(list(hero_cards) + board_cards)
        total_weight = 0.0
        win_weight = 0.0
        for c1, c2, weight in villain_range["items"]:
            if c1 in known or c2 in known:
                continue
            villain_rank = _eval7_rank([c1, c2] + board_cards)
            total_weight += weight
            if hero_rank > villain_rank:
                win_weight += weight
            elif hero_rank == villain_rank:
                win_weight += 0.5 * weight
        return _clamp(win_weight / total_weight if total_weight else 0.5)

    deck = [card for card in FULL_DECK if card not in known]
    wins = 0.0
    trials = 0
    target = max(1, int(samples))
    while trials < target:
        villain = _weighted_combo_choice(villain_range, known, rng)
        if villain is None:
            break
        remaining = [card for card in deck if card not in villain]
        if len(remaining) < need_board:
            break
        runout = rng.sample(remaining, need_board)
        final_board = board_cards + runout
        hero_rank = _eval7_rank(list(hero_cards) + final_board)
        villain_rank = _eval7_rank(list(villain) + final_board)
        if hero_rank > villain_rank:
            wins += 1.0
        elif hero_rank == villain_rank:
            wins += 0.5
        trials += 1
    return _clamp(wins / trials) if trials else None


def _estimate_state_equity(state, samples=None):
    board_cards = list(state.get("community_cards", []))
    villain_range = _build_heads_up_villain_range(state, board_cards)
    if villain_range is None:
        return None
    if samples is None:
        street_samples = {"flop": 600, "turn": 500, "river": 0}
        samples = street_samples.get(state.get("street"), 400)
    rng = random.Random(_stable_seed_for_state(state, board_cards))
    return _estimate_equity(state.get("your_cards", []), board_cards, villain_range, samples=samples, rng=rng)


def _equity_realization_factor(state, info):
    street = state.get("street")
    factor = {"flop": 0.88, "turn": 0.94, "river": 1.00}.get(street, 0.92)
    if _postflop_in_position(state):
        factor += 0.05
    else:
        factor -= 0.07
    if info.get("combo_draw") or (info.get("flush_draw") and info.get("oesd")):
        factor += 0.04
    if info.get("gutshot") or info.get("overcards"):
        factor -= 0.02
    opponent = _opponent_label(state)
    if opponent in {"aggressive", "nit", "tight_passive"} and street in {"flop", "turn"}:
        factor -= 0.04
    if opponent == "calling_station":
        factor += 0.03
    return _clamp(factor, 0.65, 1.10)


def _call_margin(state):
    owed = max(0, int(state.get("amount_owed", 0)))
    pot = max(1, int(state.get("pot", 1)))
    if owed > pot:
        return 0.035
    return {"flop": 0.025, "turn": 0.020, "river": 0.012}.get(state.get("street"), 0.025)


def _fold_equity_estimate(state, bet_fraction, board, role):
    model = _primary_opponent_model(state)
    label = _label_for_model(model)
    base = {
        "overfolder": 0.58,
        "nit": 0.46,
        "tight_passive": 0.42,
        "unknown": 0.36,
        "aggressive": 0.31,
        "calling_station": 0.18,
    }.get(label, 0.34)
    if model and model.cb_opps >= 6:
        base = (base + model.bayes_fold_to_cbet) * 0.5
    base += (bet_fraction - 0.50) * 0.18
    if board["is_static"] or board["paired"]:
        base += 0.04
    if board["is_dynamic"] or board["monotone"]:
        base -= 0.04
    if role in {"IP_PFA", "OOP_PFA"}:
        base += 0.04
    return _clamp(base, 0.05, 0.72)


def _should_semi_bluff(state, info, board, role, bet_fraction, equity=None):
    if len(_opponents_in_hand(state)) != 1:
        return False
    if equity is None:
        equity = _estimate_state_equity(state)
    if equity is None:
        return False
    pot = max(1, int(state.get("pot", 1)))
    hero_bet = int(state.get("your_bet_this_street", 0))
    target = max(int(state.get("min_raise_to", 0)), hero_bet + int(round(pot * bet_fraction)))
    bet_cost = max(1, target - hero_bet)
    fold_equity = _fold_equity_estimate(state, bet_fraction, board, role)
    final_pot = pot + bet_cost * 2
    ev = fold_equity * pot + (1.0 - fold_equity) * (equity * final_pot - bet_cost)
    return ev > max(0, pot * 0.015)


def _facing_postflop_bet(state, info, board):
    owed = int(state.get("amount_owed", 0))
    pot = max(1, int(state.get("pot", 1)))
    required = owed / max(1, pot + owed)
    equity = _estimate_state_equity(state)
    if equity is None:
        return {"action": "fold"}
    if equity * _equity_realization_factor(state, info) >= required + _call_margin(state):
        return {"action": "call"}
    return {"action": "fold"}


def _flop_decision(state, info, board, role):
    if not state.get("can_check"):
        return _facing_postflop_bet(state, info, board)

    pot_type = _pot_type(state)
    heads_up = len(_opponents_in_hand(state)) <= 1
    opponent = _opponent_label(state)
    high_freq_board = (
        board["family"] in {"Axx", "Hxx"}
        and not board["low_connected"]
        and not board["broadway_heavy"]
    ) or board["paired"] or board["trips"] or (pot_type == "3bet" and not board["is_dynamic"])

    if role == "BB_CALLER_OOP":
        bad_donk = board["monotone"] or board["paired"] or board["family"] in {"Axx", "Hxx"}
        if board["low_connected"] and not bad_donk:
            if info["strength"] in {"monster", "tptk_plus", "top_pair", "strong_draw"} or _good_flop_bluff(info, board, role):
                return _bet_fraction(state, _postflop_size(state, board, info, role))
        return {"action": "check"}

    if role in {"IP_PFA", "OOP_PFA"}:
        if heads_up and opponent == "calling_station":
            if info["strength"] in {"monster", "tptk_plus", "top_pair", "medium_pair", "strong_draw"}:
                return _bet_fraction(state, _postflop_size(state, board, info, role))
            return {"action": "check"}

        if high_freq_board and heads_up:
            if info["strength"] in {"air", "weak_equity"} and opponent == "overfolder":
                if not _good_flop_bluff(info, board, role):
                    return {"action": "check"}
            if info["strength"] in {"air", "weak_equity", "strong_draw"}:
                size = _postflop_size(state, board, info, role)
                return _bet_fraction(state, size) if _should_semi_bluff(state, info, board, role, size) else {"action": "check"}
            if info["strength"] == "medium_pair" and role == "IP_PFA" and board["is_dynamic"]:
                return {"action": "check"}
            return _bet_fraction(state, _postflop_size(state, board, info, role))

        if info["strength"] in {"monster", "tptk_plus"}:
            return _bet_fraction(state, _postflop_size(state, board, info, role))
        if info["strength"] == "top_pair":
            return {"action": "check"} if role == "IP_PFA" and board["is_dynamic"] else _bet_fraction(state, 0.50)
        if heads_up and opponent == "overfolder" and info["strength"] in {"weak_equity", "air"} and _good_flop_bluff(info, board, role):
            return _bet_fraction(state, 0.33) if _should_semi_bluff(state, info, board, role, 0.33) else {"action": "check"}
        if info["strength"] == "strong_draw" or _good_flop_bluff(info, board, role):
            size = _postflop_size(state, board, info, role)
            return _bet_fraction(state, size) if _should_semi_bluff(state, info, board, role, size) else {"action": "check"}
        return {"action": "check"}

    if role == "IP_CALLER":
        if heads_up and opponent == "calling_station" and info["strength"] in {"air", "weak_equity"}:
            return {"action": "check"}
        if info["strength"] in {"monster", "tptk_plus", "strong_draw"}:
            return _bet_fraction(state, _postflop_size(state, board, info, role))
        if info["strength"] == "air" and (board["is_static"] or info["nut_flush_blocker"]):
            return _bet_fraction(state, 0.33) if _should_semi_bluff(state, info, board, role, 0.33) else {"action": "check"}
        return {"action": "check"}

    if info["strength"] == "monster" and board["is_dynamic"]:
        return _bet_fraction(state, 0.67)
    if info["strength"] == "strong_draw" and board["is_dynamic"]:
        return _bet_fraction(state, 0.67) if _should_semi_bluff(state, info, board, role, 0.67) else {"action": "check"}
    return {"action": "check"}


def _turn_decision(state, info, board, role):
    if not state.get("can_check"):
        return _facing_postflop_bet(state, info, board)

    flop_checked = _street_checked_through(state, "flop")
    flop_bet = _hero_bet_on_street(state, "flop")
    oop = role in {"OOP_PFA", "OOP_CALLER", "BB_CALLER_OOP"}
    opponent = _opponent_label(state)
    heads_up = len(_opponents_in_hand(state)) <= 1

    if info["strength"] in {"monster", "tptk_plus"}:
        return _bet_fraction(state, _postflop_size(state, board, info, role))

    if heads_up and opponent == "calling_station" and info["strength"] in {"air", "weak_equity"}:
        return {"action": "check"}

    if flop_checked and oop and _turn_favors_probe(board):
        if info["strength"] == "top_pair":
            return _bet_fraction(state, _postflop_size(state, board, info, role))
        if info["strength"] == "strong_draw" or _good_flop_bluff(info, board, role):
            size = _postflop_size(state, board, info, role)
            return _bet_fraction(state, size) if _should_semi_bluff(state, info, board, role, size) else {"action": "check"}
        return {"action": "check"}

    if flop_bet and role in {"IP_PFA", "OOP_PFA"}:
        if info["strength"] in {"top_pair", "strong_draw"}:
            return _bet_fraction(state, _postflop_size(state, board, info, role))
        if info["strength"] in {"weak_equity", "air"} and _scare_turn_for_pfa(board):
            return _bet_fraction(state, 0.67) if _should_semi_bluff(state, info, board, role, 0.67) else {"action": "check"}
        return {"action": "check"}

    if role == "IP_CALLER" and info["strength"] == "top_pair" and board["is_dynamic"]:
        return _bet_fraction(state, 0.67)
    if role == "IP_CALLER" and info["strength"] == "strong_draw" and board["is_dynamic"]:
        return _bet_fraction(state, 0.67) if _should_semi_bluff(state, info, board, role, 0.67) else {"action": "check"}

    return {"action": "check"}


def _river_decision(state, info, board, role):
    if not state.get("can_check"):
        return _facing_postflop_bet(state, info, board)

    opponent = _opponent_label(state)
    if info["strength"] == "monster":
        return _bet_fraction(state, _postflop_size(state, board, info, role))
    if info["strength"] == "tptk_plus" and not (board["monotone"] and not info["nut_flush_blocker"]):
        return _bet_fraction(state, 0.50 if board["is_dynamic"] else 0.67)
    if info["strength"] == "top_pair" and opponent == "calling_station" and board["is_static"]:
        return _bet_fraction(state, _postflop_size(state, board, info, role))
    if opponent != "calling_station" and info["strength"] in {"air", "weak_equity"} and info["nut_flush_blocker"] and role in {"IP_PFA", "IP_CALLER"}:
        size = _postflop_size(state, board, info, role)
        return _bet_fraction(state, size) if _should_semi_bluff(state, info, board, role, size) else {"action": "check"}
    return {"action": "check"}


def _first_matching_action(hand, table, default="fold"):
    matches = []
    for action, hand_set in table.items():
        if action != "fold" and hand in hand_set:
            matches.append(action)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return "ambiguous"
    return default


def _stable_choice(options, state, hand, spot_key):
    text = f"{state.get('hand_id', '')}|{hand}|{spot_key}|{'/'.join(options)}"
    score = sum((idx + 1) * ord(char) for idx, char in enumerate(text))
    return options[score % len(options)]


def _effective_stack_bb(state):
    _, big_blind = _blind_amounts(state)
    hero_total = int(state.get("your_stack", 0)) + int(state.get("your_bet_this_street", 0))
    opponents = [
        int(player.get("stack", 0)) + int(player.get("bet_this_street", 0))
        for player in state.get("players", [])
        if player.get("seat") != state.get("seat_to_act") and not player.get("is_folded")
    ]
    effective = min([hero_total] + opponents) if opponents else hero_total
    return effective / max(1, big_blind)


def _squeeze_range_for(opener_pos):
    if opener_pos in {"LJ"}:
        return SQUEEZE_VS_EP
    if opener_pos in {"HJ"}:
        return SQUEEZE_VS_MP
    return SQUEEZE_VS_LP


def _hero_has_position(hero_pos):
    return hero_pos in {"CO", "BTN"}


def _rangeconverter_action(hand, situation, state):
    key = (situation.get("hero_pos"), situation.get("threebettor_pos"))
    table = RANGECONVERTER_VS_3BET.get(key)
    if not table:
        return "MISSING_FROM_SOURCE"
    if hand in table.get("fourbet", ()):
        return "4bet"
    if hand in table.get("call", ()):
        return "call"
    if hand in table.get("mix_fold_call", ()):
        return _stable_choice(("fold", "call"), state, hand, key)
    if hand in table.get("mix_fold_fourbet", ()):
        return _stable_choice(("fold", "4bet"), state, hand, key)
    if hand in table.get("mix_call_fourbet", ()):
        return _stable_choice(("call", "4bet"), state, hand, key)
    return "fold"


def _preflop_actions(state):
    return [entry for entry in state.get("action_log", []) if entry.get("action") not in ("small_blind", "big_blind")]


def _parse_preflop_situation(state):
    positions = _seat_positions(state)
    hero_seat = state["seat_to_act"]
    hero_pos = positions.get(hero_seat, "LJ")
    actions = _preflop_actions(state)
    raises = []
    calls_after_raise = []
    limps = []
    hero_prior = []

    for idx, entry in enumerate(actions):
        seat = entry.get("seat")
        pos = positions.get(seat)
        action = entry.get("action")
        if seat == hero_seat:
            hero_prior.append(action)
        if action in ("raise", "all_in"):
            raises.append({"seat": seat, "pos": pos, "amount": int(entry.get("amount") or state.get("current_bet", 0)), "idx": idx})
        elif action == "call":
            if raises:
                calls_after_raise.append({"seat": seat, "pos": pos, "idx": idx})
            else:
                limps.append({"seat": seat, "pos": pos, "idx": idx})

    if hero_pos == "SB" and raises and raises[-1]["pos"] == "BB" and hero_prior:
        if "raise" in hero_prior or "all_in" in hero_prior:
            return {"type": "sb_faces_raise_after_raise", "hero_pos": hero_pos, "open_amount": raises[-1]["amount"]}
        if "call" in hero_prior:
            return {"type": "sb_faces_raise_after_limp", "hero_pos": hero_pos, "open_amount": raises[-1]["amount"]}

    if len(raises) >= 3 and sum(1 for item in raises if item["seat"] == hero_seat) >= 2 and raises[-1]["seat"] != hero_seat:
        return {
            "type": "hero_4bet_faces_5bet",
            "hero_pos": hero_pos,
            "open_amount": raises[0]["amount"],
            "threebet_amount": raises[-2]["amount"],
            "fivebet_amount": raises[-1]["amount"],
        }

    if len(raises) >= 2 and raises[0]["seat"] == hero_seat and raises[-1]["seat"] != hero_seat:
        callers_between = [item for item in calls_after_raise if raises[0]["idx"] < item["idx"] < raises[-1]["idx"]]
        if not callers_between:
            return {
                "type": "hero_opened_faces_3bet",
                "hero_pos": hero_pos,
                "threebettor_seat": raises[-1]["seat"],
                "threebettor_pos": raises[-1]["pos"],
                "open_amount": raises[0]["amount"],
                "threebet_amount": raises[-1]["amount"],
            }
        return {
            "type": "hero_opened_faces_squeeze",
            "hero_pos": hero_pos,
            "threebettor_seat": raises[-1]["seat"],
            "threebettor_pos": raises[-1]["pos"],
            "open_amount": raises[0]["amount"],
            "threebet_amount": raises[-1]["amount"],
            "caller_count": len(callers_between),
        }

    if len(raises) >= 2 and not hero_prior:
        return {
            "type": "raise_and_3bet_before_hero",
            "hero_pos": hero_pos,
            "opener_seat": raises[0]["seat"],
            "opener_pos": raises[0]["pos"],
            "threebettor_seat": raises[1]["seat"],
            "threebettor_pos": raises[1]["pos"],
            "open_amount": raises[0]["amount"],
            "threebet_amount": raises[1]["amount"],
        }

    if not raises and not limps:
        return {"type": "unopened", "hero_pos": hero_pos}

    if hero_pos == "BB" and not raises and len(limps) == 1 and limps[0]["pos"] == "SB":
        return {"type": "bb_vs_sb_limp", "hero_pos": hero_pos}

    if hero_pos == "BB" and len(raises) == 1 and raises[0]["pos"] == "SB" and not calls_after_raise:
        return {"type": "bb_vs_sb_raise", "hero_pos": hero_pos, "opener_seat": raises[0]["seat"], "opener_pos": "SB", "open_amount": raises[0]["amount"]}

    if len(raises) == 1 and not calls_after_raise and not limps:
        return {
            "type": "vs_open",
            "hero_pos": hero_pos,
            "opener_seat": raises[0]["seat"],
            "opener_pos": raises[0]["pos"],
            "open_amount": raises[0]["amount"],
        }

    if len(raises) >= 2:
        return {"type": "threebet_or_fourbet_tree", "hero_pos": hero_pos, "open_amount": raises[-1]["amount"]}
    if len(raises) == 1 and calls_after_raise:
        return {
            "type": "open_plus_callers_before_hero",
            "hero_pos": hero_pos,
            "opener_seat": raises[0]["seat"],
            "opener_pos": raises[0]["pos"],
            "open_amount": raises[0]["amount"],
            "caller_count": len(calls_after_raise),
        }
    if limps:
        return {"type": "limpers_no_raise", "hero_pos": hero_pos, "limper_count": len(limps)}
    return {"type": "unknown", "hero_pos": hero_pos}


def _sb_complete_action(hand, situation):
    table = PREFLOP_RANGES["blind_vs_blind"]["SB_complete"]
    if situation["type"] == "sb_faces_raise_after_raise":
        if hand in table["raise_4bet"]:
            return "4bet"
        if hand in table["raise_call"]:
            return "call"
        if hand in table["raise_fold"]:
            return "fold"
        return "MISSING_FROM_SOURCE"
    if situation["type"] == "sb_faces_raise_after_limp":
        if hand in table["limp_raise"]:
            return "raise"
        if hand in table["limp_call"]:
            return "call"
        if hand in table["limp_fold"]:
            return "fold"
        return "MISSING_FROM_SOURCE"
    for action, hands in table.items():
        if hand in hands:
            return "raise" if action.startswith("raise") else "call"
    return "fold"


def _lookup_preflop_action(hand, situation, state=None):
    spot_type = situation["type"]
    hero_pos = situation.get("hero_pos")

    if spot_type == "unopened":
        if hero_pos == "SB":
            return _sb_complete_action(hand, situation)
        table = PREFLOP_RANGES["unopened"].get(hero_pos)
        if not table:
            return "MISSING_FROM_SOURCE"
        return _first_matching_action(hand, table, default="fold")

    if spot_type in ("sb_faces_raise_after_raise", "sb_faces_raise_after_limp"):
        return _sb_complete_action(hand, situation)

    if spot_type == "bb_vs_sb_limp":
        table = PREFLOP_RANGES["blind_vs_blind"][("SB_limp", "BB_hero")]
        return _first_matching_action(hand, table, default="check")

    if spot_type == "bb_vs_sb_raise":
        table = PREFLOP_RANGES["blind_vs_blind"][("SB_raise", "BB_hero")]
        return _first_matching_action(hand, table, default="fold")

    if spot_type == "vs_open":
        key = (f"{situation.get('opener_pos')}_open", f"{hero_pos}_hero")
        range_group = "vs_open_ip" if hero_pos in {"HJ", "CO", "BTN"} else "vs_open_oop"
        table = PREFLOP_RANGES[range_group].get(key)
        if not table:
            return "MISSING_FROM_SOURCE"
        return _first_matching_action(hand, table, default="fold")

    if spot_type == "hero_opened_faces_3bet":
        return _rangeconverter_action(hand, situation, state or {})

    return "MISSING_FROM_SOURCE"


def _choose_preflop_size(action, situation, state):
    _, big_blind = _blind_amounts(state)
    hero_pos = situation.get("hero_pos")
    spot_type = situation["type"]
    current_bet = int(state.get("current_bet", 0))
    open_amount = int(situation.get("open_amount") or current_bet or big_blind)

    if action == "raise" and spot_type == "unopened":
        return (SIZING_RULES["sb_rfi"] if hero_pos == "SB" else SIZING_RULES["rfi"]) * big_blind
    if action == "raise" and spot_type == "bb_vs_sb_limp":
        return SIZING_RULES["bb_vs_sb_limp_raise_multiplier"] * big_blind
    if action == "3bet":
        multiplier = SIZING_RULES["ip_3bet_multiplier"] if hero_pos in {"HJ", "CO", "BTN"} else SIZING_RULES["oop_3bet_multiplier"]
        return open_amount * multiplier
    if action == "4bet" and spot_type == "hero_opened_faces_3bet":
        chart = RANGECONVERTER_VS_3BET.get((hero_pos, situation.get("threebettor_pos")), {})
        return float(chart.get("fourbet_to_bb") or 23.0) * big_blind
    if action == "4bet" and _effective_stack_bb(state) < 40:
        return int(state.get("your_bet_this_street", 0)) + int(state.get("your_stack", 0))
    if action == "4bet":
        multiplier = SIZING_RULES["ip_4bet_multiplier"] if hero_pos in {"HJ", "CO", "BTN"} else SIZING_RULES["oop_4bet_multiplier"]
        return current_bet * multiplier
    if action == "raise" and spot_type in {"sb_faces_raise_after_raise", "sb_faces_raise_after_limp"}:
        return current_bet * SIZING_RULES["oop_4bet_multiplier"]
    if action == "squeeze":
        callers = int(situation.get("caller_count") or 1)
        multiplier = 3.5 + callers if _hero_has_position(hero_pos) else 4.0 + callers
        return open_amount * multiplier
    if action == "iso":
        limpers = int(situation.get("limper_count") or 1)
        base = 4.0 if _hero_has_position(hero_pos) else 5.0
        return (base + max(0, limpers - 1)) * big_blind
    if action == "raise":
        return max(state.get("min_raise_to", 0), current_bet * 2)
    return None


def _missing_source_fallback(hand, situation, state):
    if state.get("can_check") or state.get("amount_owed", 0) == 0:
        return {"action": "check"}

    spot_type = situation.get("type")
    hero_pos = situation.get("hero_pos")

    if _effective_stack_bb(state) <= 15:
        if hand in SHORT_STACK_JAM:
            return {"action": "all_in"}
        return {"action": "fold"}

    if spot_type == "hero_opened_faces_3bet":
        if hand in FOURBET_FALLBACK_VALUE or hand in FOURBET_FALLBACK_BLUFF:
            return _raise_to(state, _choose_preflop_size("4bet", situation, state))
        if hand in {"QQ", "JJ", "TT", "99", "88", "AQs", "AJs", "ATs", "KQs", "KJs", "QJs", "JTs", "T9s", "98s"}:
            return {"action": "call"}
        return {"action": "fold"}

    if spot_type == "hero_4bet_faces_5bet":
        return {"action": "all_in"} if hand in FIVEBET_CALL_SAFE else {"action": "fold"}

    if spot_type == "raise_and_3bet_before_hero":
        if hand in COLD_4BET_SAFE:
            return _raise_to(state, 23 * _blind_amounts(state)[1])
        return {"action": "fold"}

    if spot_type == "open_plus_callers_before_hero":
        squeeze_range = _squeeze_range_for(situation.get("opener_pos"))
        if hand in squeeze_range:
            return _raise_to(state, _choose_preflop_size("squeeze", situation, state))
        if hero_pos in {"BTN", "BB"} and hand in MULTIWAY_CALL_RANGE:
            return {"action": "call"}
        return {"action": "fold"}

    if spot_type == "hero_opened_faces_squeeze":
        if hand in FIVEBET_CALL_SAFE:
            return _raise_to(state, 23 * _blind_amounts(state)[1])
        if hand in {"JJ", "TT", "AQs", "KQs"}:
            return {"action": "call"}
        return {"action": "fold"}

    if spot_type == "limpers_no_raise":
        iso_range = ISO_LATE_POSITION_EXTRA if hero_pos in {"CO", "BTN"} else ISO_VALUE_DEFAULT
        if hand in iso_range:
            return _raise_to(state, _choose_preflop_size("iso", situation, state))
        _, big_blind = _blind_amounts(state)
        max_overlimp_price = max(big_blind, max(1, int(state.get("pot", 1))) * 0.35)
        if hand in OVERLIMP_MULTIWAY and int(state.get("amount_owed", 0)) <= max_overlimp_price:
            return {"action": "call"}
        return {"action": "fold"}

    if spot_type in {"threebet_or_fourbet_tree", "sb_faces_raise_after_raise"}:
        if hand in {"AA", "KK", "AKs", "AKo"}:
            return _raise_to(state, int(state.get("current_bet", 0)) * SIZING_RULES["oop_4bet_multiplier"])
        if hand in {"QQ", "JJ", "TT", "AQs", "KQs"}:
            return {"action": "call"}
        return {"action": "fold"}

    if spot_type in {"limpers", "raise_plus_callers"}:
        if hand in PREMIUMS or hand in STRONG_CONTINUES:
            _, big_blind = _blind_amounts(state)
            limpers = sum(1 for entry in _preflop_actions(state) if entry.get("action") == "call")
            target = big_blind * (3 + max(1, limpers))
            return _raise_to(state, target)
        return {"action": "fold"}

    return {"action": "fold"}


STEAL_EXPANSION = {
    "CO": {"K5s", "K4s", "Q8s", "Q7s", "J8s", "T8s", "97s", "86s", "A8o", "KTo"},
    "BTN": {"Q2s", "J3s", "T5s", "95s", "84s", "K7o", "Q8o", "J8o", "T8o", "97o"},
    "SB": {"A2o", "K4o", "Q8o", "J8o", "T8o", "97o", "74s", "63s"},
}
LIGHT_THREEBET_BLUFFS = {
    "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s", "A9s",
    "K5s", "K6s", "K7s", "K8s", "K9s", "Q9s", "J9s", "T9s", "65s", "54s",
}
LINEAR_THREEBET_VALUE = {
    "99", "TT", "JJ", "QQ", "KK", "AA", "AQs", "AKs", "AJs", "KQs", "AQo", "AKo",
}
TIGHT_OPEN_FOLDS = {
    "AJo", "ATo", "A9o", "KQo", "KJo", "QJo", "KTo", "QTo", "JTo",
    "A2s", "A3s", "A4s", "A5s", "K9s", "Q9s", "J9s", "T9s",
}
NIT_3BET_FOLDS = {
    "AJo", "ATo", "KQo", "KJo", "QJo", "A5s", "A4s", "A3s", "A2s",
    "K9s", "K8s", "QTs", "JTs", "T9s", "98s", "77", "66", "55", "44", "33", "22",
}


def _active_blind_models(state):
    small_blind, big_blind = _blind_seats(state)
    seats = [seat for seat in (small_blind, big_blind) if seat is not None and seat != state.get("seat_to_act")]
    return [
        model
        for seat in seats
        for model in [_model_for_seat(state, seat)]
        if model is not None
    ]


def _all_blinds_overfold_to_steals(state):
    return _blind_steal_exploit_score(state) >= 0.50


def _blind_steal_exploit_score(state):
    models = _active_blind_models(state)
    if not models:
        return 0.0
    scores = []
    for model in models:
        fold_strength = _exploit_strength_above(model, "fold_to_steal", 0.68, min_opp_soft=8)
        low_3bet_strength = _exploit_strength_below(model, "threebet", 0.08, min_opp_soft=8)
        scores.append(min(fold_strength, low_3bet_strength))
    return min(scores) if scores else 0.0


def _baseline_call_available(hand, situation):
    spot_type = situation.get("type")
    hero_pos = situation.get("hero_pos")
    if spot_type != "vs_open":
        return False
    key = (f"{situation.get('opener_pos')}_open", f"{hero_pos}_hero")
    range_group = "vs_open_ip" if hero_pos in {"HJ", "CO", "BTN"} else "vs_open_oop"
    table = PREFLOP_RANGES[range_group].get(key, {})
    return hand in table.get("call", set())


def _apply_exploit_adjustments(action, hand, situation, state):
    spot_type = situation.get("type")
    hero_pos = situation.get("hero_pos")

    if (
        spot_type == "unopened"
        and hero_pos in STEAL_EXPANSION
        and action in {"fold", "call"}
        and hand in STEAL_EXPANSION[hero_pos]
        and _all_blinds_overfold_to_steals(state)
    ):
        return "raise"

    if spot_type in {"vs_open", "bb_vs_sb_raise"}:
        opener = _model_for_seat(state, situation.get("opener_seat"))
        opener_label = _label_for_model(opener)
        opener_read = _villain_read(opener)
        if opener_label in {"nit", "tight_passive"} and action in {"call", "3bet"} and hand in TIGHT_OPEN_FOLDS:
            return "fold"
        if (
            action == "3bet"
            and hand in LIGHT_THREEBET_BLUFFS
            and opener
            and (
                opener_read["station_score"] >= 0.50
                or _likely_below(opener, "fold_to_threebet", 0.45, confidence=0.80, min_opp=8)
            )
        ):
            return "call" if _baseline_call_available(hand, situation) else "fold"

    if spot_type in {"hero_opened_faces_3bet", "hero_opened_faces_squeeze"}:
        threebettor = _model_for_seat(state, situation.get("threebettor_seat"))
        threebettor_label = _label_for_model(threebettor)
        if threebettor_label in {"nit", "tight_passive"} and action in {"call", "4bet"} and hand in NIT_3BET_FOLDS:
            return "fold"
        if threebettor_label in {"nit", "tight_passive"} and action == "4bet" and hand not in PREMIUMS:
            return "fold"

    return action


def _preflop_decision(state):
    hand = _combo(state["your_cards"])
    situation = _parse_preflop_situation(state)
    action = _lookup_preflop_action(hand, situation, state)
    action = _apply_exploit_adjustments(action, hand, situation, state)

    if _effective_stack_bb(state) <= 15 and action in {"raise", "3bet", "4bet", "squeeze", "iso"}:
        return {"action": "all_in"}
    if action in {"MISSING_FROM_SOURCE", "ambiguous"}:
        return _missing_source_fallback(hand, situation, state)
    if action in {"raise", "3bet", "4bet", "squeeze", "iso"}:
        amount = _choose_preflop_size(action, situation, state)
        if amount is None:
            return {"action": "call"} if state.get("amount_owed", 0) > 0 else {"action": "check"}
        return _raise_to(state, amount)
    if action == "call":
        return {"action": "call"} if state.get("amount_owed", 0) > 0 else {"action": "check"}
    if action == "check":
        return {"action": "check"} if state.get("can_check") else {"action": "call"}
    return _safe_passive(state)


def decide(state):
    if state.get("type") == "warmup":
        return {"action": "check"}
    try:
        _update_opponent_models(state)
        if state.get("street") == "preflop":
            return _preflop_decision(state)

        board = _board_class(state.get("community_cards", []))
        info = _hand_info(state)
        role = _postflop_role(state)
        street = state.get("street")
        if street == "flop":
            return _flop_decision(state, info, board, role)
        if street == "turn":
            return _turn_decision(state, info, board, role)
        if street == "river":
            return _river_decision(state, info, board, role)
        return _safe_passive(state)
    except Exception:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}
