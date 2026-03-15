import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def bt_neg_log_likelihood(r_free, winners, losers, n_items, beta=1.0):
    """
    r_free: parameters for items 0..n_items-2
    last item's reward is fixed to 0 for identifiability
    winners, losers: arrays of item indices for each comparison
    """
    r = np.concatenate([r_free, [0.0]])  # fix last reward = 0

    diff = beta * (r[winners] - r[losers])
    
    # log sigma(diff) = -log(1 + exp(-diff))
    # use stable form:
    nll = np.sum(np.logaddexp(0.0, -diff))
    return nll

def fit_bradley_terry(winners, losers, n_items, beta=1.0):
    """
    winners[k] beat losers[k]
    """
    x0 = np.zeros(n_items - 1)

    result = minimize(
        bt_neg_log_likelihood,
        x0,
        args=(winners, losers, n_items, beta),
        method="L-BFGS-B"
    )

    r_hat = np.concatenate([result.x, [0.0]])
    return r_hat, result

# def borda_from_pairwise(winners, losers, n_items=None):
#     winners = np.asarray(winners)
#     losers = np.asarray(losers)

#     if n_items is None:
#         n_items = max(winners.max(), losers.max()) + 1

#     scores = np.zeros(n_items, dtype=int)

#     for w in winners:
#         scores[w] += 1

#     denom = np.zeros(n_items, dtype=int)
#     for l in losers:
#         denom[l] += 1

#     denom = denom + scores + 1e-6
#     scores = scores / denom

#     ranking = np.argsort(-scores)  # descending
#     return scores, ranking


def borda_from_population_utilities(utilities, voter_dist=None, cand_dist=None, beta=1.0):
    utilities = np.asarray(utilities)
    V, C = utilities.shape

    if voter_dist is None:
        voter_dist = np.ones(V, dtype=float) / V
    else:
        voter_dist = np.asarray(voter_dist, dtype=float)
        assert voter_dist.shape == (V,)
        voter_dist = voter_dist / voter_dist.sum()

    if cand_dist is None:
        cand_dist = np.ones(C, dtype=float) / C
    else:
        cand_dist = np.asarray(cand_dist, dtype=float)
        assert cand_dist.shape == (C,)
        cand_dist = cand_dist / cand_dist.sum()

    P = np.zeros((C, C), dtype=float)

    for v, u in enumerate(utilities):
        diffs = u[:, None] - u[None, :]
        P += voter_dist[v] * expit(beta * diffs)

    borda_scores = P @ cand_dist
    ranking = np.argsort(-borda_scores)

    return borda_scores, ranking

def leaderboard_dist(ranking, true_ranking, avg_utils):
    ranking = np.asarray(ranking)
    true_ranking = np.asarray(true_ranking)

    ranking_utils = avg_utils[ranking]
    true_ranking_utils = avg_utils[true_ranking]

    denom_cumsum = np.cumsum(ranking_utils)
    num_cumsum = np.cumsum(true_ranking_utils)

    ratio = np.max(num_cumsum / denom_cumsum)
    k = np.argmax(num_cumsum / denom_cumsum)
    return ratio, k

# KL(p_{dataset} || p_{fitted model}), where p denote distributions over pairwise comparisons
def misspecification_error(winners, losers, r_hat, beta=1.0, eps=1e-12):
    n_candidates = len(r_hat)
    counts = np.zeros((n_candidates, n_candidates), dtype=np.float64)

    for w, l in zip(winners, losers):
        counts[w, l] += 1

    total = counts + counts.T
    mask = total > 0  # only pairs that were actually compared

    empirical = np.zeros_like(counts, dtype=np.float64)
    empirical[mask] = counts[mask] / total[mask]

    modeled = 1.0 / (1.0 + np.exp(-beta * (r_hat[:, None] - r_hat[None, :])))

    # avoid diagonal and duplicate counting
    triu = np.triu(mask, k=1)

    p = np.clip(empirical[triu], eps, 1 - eps)
    q = np.clip(modeled[triu], eps, 1 - eps)

    kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return kl.sum()