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

def borda_from_pairwise(winners, losers, n_items=None):
    winners = np.asarray(winners)
    losers = np.asarray(losers)

    if n_items is None:
        n_items = max(winners.max(), losers.max()) + 1

    scores = np.zeros(n_items, dtype=int)

    for w in winners:
        scores[w] += 1

    denom = np.zeros(n_items, dtype=int)
    for l in losers:
        denom[l] += 1

    denom = denom + scores + 1e-6
    scores = scores / denom

    ranking = np.argsort(-scores)  # descending
    return scores, ranking


def borda_from_population_utilities(utilities, voter_dist=None, cand_dist=None, beta=1.0):
    
    utilities = np.asarray(utilities)

    V, C = utilities.shape

    if voter_dist:
        voter_dist = np.asarray(voter_dist)
        assert voter_dist.shape[0] == utilities.shape[0]

        
    
    P = np.zeros((C, C))

    for u in utilities:
        diffs = u[:, None] - u[None, :]
        P += expit(beta * diffs)

    P /= V

    borda_scores = P.sum(axis=1)
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