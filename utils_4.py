import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from scipy.optimize import linprog
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def ij_from_pairwise(w_arr, l_arr, n_items):
    ij_wins = np.zeros((n_items, n_items))
    np.add.at(ij_wins, (w_arr, l_arr), 1.0)
    return ij_wins

def _solve_maximal_lottery(margins, tie_tol=1e-12):
    C = margins.shape[0]
    result = linprog(
        c=np.zeros(C),
        A_ub=-margins.T,
        b_ub=np.zeros(C),
        A_eq=np.ones((1, C)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * C,
        method='highs',
    )
    lottery = np.maximum(result.x if result.success else np.ones(C) / C, 0)
    lottery[lottery < tie_tol] = 0.0
    s = lottery.sum()
    return lottery / s if s > 0 else np.ones(C) / C

def borda_ranking(ij_wins, with_scores=False):
    num = ij_wins.sum(axis=1)
    denom = ij_wins.sum(axis=1) + ij_wins.sum(axis=0)
    denom = np.where(denom == 0.0, 1.0, denom)
    scores = num / denom
    if with_scores:
        return np.argsort(-scores), scores
    else:
        return np.argsort(-scores)

def borda_peeling_ranking(ij_wins):
    remaining = list(range(ij_wins.shape[0]))
    ranking = []
    while remaining:
        rem = np.array(remaining)
        ij_remaining = ij_wins[np.ix_(rem, rem)]
        _, scores = borda_ranking(ij_remaining, with_scores=True)
        w = int(np.argmax(scores))
        ranking.append(remaining[w])
        remaining.pop(w)
    return np.array(ranking)

def copeland_ranking(ij_wins, tol=1e-9):
    denom = (ij_wins + ij_wins.T)
    denom = np.where(denom == 0.0, 1.0, denom)
    m = (ij_wins - ij_wins.T) / denom
    return np.argsort(-((m > tol).sum(axis=1) - (m < -tol).sum(axis=1)))

def copeland_peeling_ranking(ij_wins, tol=1e-9):
    remaining = list(range(ij_wins.shape[0]))
    ranking = []
    while remaining:
        rem = np.array(remaining)
        ij_remaining = ij_wins[np.ix_(rem, rem)]
        denom = ij_remaining + ij_remaining.T
        m = (ij_remaining - ij_remaining.T) / np.where(denom == 0.0, 1.0, denom)
        scores = (m > tol).sum(axis=1) - (m < -tol).sum(axis=1)
        w = int(np.argmax(scores))
        ranking.append(remaining[w]); remaining.pop(w)
    return np.array(ranking)

def ml_argmax_ranking(ij_wins, tol=1e-12):
    remaining = list(range(ij_wins.shape[0]))
    ranking = []
    while remaining:
        rem = np.array(remaining)
        sub = ij_wins[np.ix_(rem, rem)]
        with np.errstate(invalid='ignore', divide='ignore'):
            margins = np.where(sub + sub.T > 0, (sub - sub.T) / (sub + sub.T), 0.0)
        lot = _solve_maximal_lottery(margins, tol)
        w = int(np.argmax(lot))
        ranking.append(remaining[w]); remaining.pop(w)
    return np.array(ranking)

def ml_nonzero_ranking(ij_wins, tol=1e-12):
    remaining = list(range(ij_wins.shape[0]))
    ranking = []
    while remaining:
        rem = np.array(remaining)
        sub = ij_wins[np.ix_(rem, rem)]
        with np.errstate(invalid='ignore', divide='ignore'):
            margins = np.where(sub + sub.T > 0, (sub - sub.T) / (sub + sub.T), 0.0)
        lot = _solve_maximal_lottery(margins, tol)
        order = np.argsort(-lot)
        nonzero = order[lot[order] > tol]
        if len(nonzero) == 0:
            nonzero = np.array([int(np.argmax(lot))])
        selected = [remaining[i] for i in nonzero]
        ranking.extend(selected)
        remaining = [c for c in remaining if c not in set(selected)]
    return np.array(ranking)

def ranking_distribution_pruned(candidates, maximal_lottery_fn):
    """Returns (ranking_dist, cache). Cache maps frozenset -> {candidate: prob}."""
    cache = {}

    def get_lottery(S):
        S = frozenset(S)
        if S not in cache:
            cache[S] = maximal_lottery_fn(list(S))
        return cache[S]

    def recurse(remaining):
        remaining = frozenset(remaining)

        if len(remaining) == 0:
            return {(): 1.0}

        ml = get_lottery(remaining)

        invalid = set(ml) - remaining
        if invalid:
            raise ValueError(f"Lottery returned candidates not in remaining set: {invalid}")

        total_positive = False
        result = {}

        for c, p in ml.items():
            if p == 0:
                continue
            total_positive = True
            for suffix, q in recurse(remaining - {c}).items():
                result[(c,) + suffix] = p * q

        if not total_positive:
            raise ValueError(f"Lottery has no positive-probability candidates for set {remaining}")

        return result

    return recurse(frozenset(candidates)), cache
    

def argmax_from_cache(cache, candidates):
    """Derive ml_argmax ranking from cache — no LP solves needed."""
    remaining = frozenset(candidates)
    ranking = []
    while len(remaining) > 1:
        lottery = cache[remaining]
        winner = max(lottery, key=lambda c: lottery[c])
        ranking.append(winner)
        remaining = remaining - {winner}
    ranking.append(next(iter(remaining)))
    return np.array(ranking)


def nonzero_from_cache(cache, candidates, tol=1e-12):
    """Derive ml_nonzero ranking from cache — no LP solves needed.

    ranking_distribution_pruned recurses through all orderings of nonzero
    candidates, so every subset reachable by removing nonzero candidates
    one-by-one is guaranteed to be in the cache.
    """
    remaining = frozenset(candidates)
    ranking = []
    while len(remaining) > 0:
        if len(remaining) == 1:
            ranking.append(next(iter(remaining)))
            break
        lottery = cache[remaining]
        nonzero = sorted(
            [c for c, p in lottery.items() if p > tol],
            key=lambda c: -lottery[c],
        )
        if not nonzero:
            nonzero = [max(lottery, key=lambda c: lottery[c])]
        ranking.extend(nonzero)
        remaining = remaining - frozenset(nonzero)
    return np.array(ranking)


def expected_average_utilities(ranking_dist, avg_utilities):
    rankings = np.array(list(ranking_dist.keys()), dtype=int)  # (K, M)
    probs    = np.array(list(ranking_dist.values()))            # (K,)
    return (probs[:, None] * avg_utilities[rankings]).sum(axis=0)


def expected_leaderboard_distortion(ranking_dist, true_ranking, avg_utils):
    true_ranking = np.asarray(true_ranking)
    ranking_utils = expected_average_utilities(ranking_dist, avg_utils)
    true_ranking_utils = avg_utils[true_ranking]
    denom_cumsum = np.cumsum(ranking_utils)
    num_cumsum   = np.cumsum(true_ranking_utils)
    valid = denom_cumsum > 0
    ratios = np.where(valid, num_cumsum / np.where(denom_cumsum > 0, denom_cumsum, 1.0), -np.inf)
    return float(np.max(ratios)) if valid.any() else np.inf

def expected_leaderboard_distortion_w(ranking_dist, true_ranking, avg_utils, w):
    # ranking = np.asarray(ranking)
    true_ranking = np.asarray(true_ranking)

    ranking_utils = expected_average_utilities(ranking_dist=ranking_dist, avg_utilities=avg_utils)
    true_ranking_utils = avg_utils[true_ranking]

    denom = (ranking_utils * w).sum()
    num = (true_ranking_utils * w).sum()

    ratio = num / denom if denom > 0 else np.inf
    return ratio


def make_ml_fn_from_ij_wins(ij_wins):
    def fn(candidates):
        sub = ij_wins[np.ix_(candidates, candidates)]
        with np.errstate(invalid='ignore', divide='ignore'):
            margins = np.where(sub + sub.T > 0, (sub - sub.T) / (sub + sub.T), 0.0)
        lot = _solve_maximal_lottery(margins)
        return {c: float(lot[i]) for i, c in enumerate(candidates)}
    return fn