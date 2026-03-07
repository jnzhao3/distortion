import numpy as np
from scipy.optimize import minimize

def bt_neg_log_likelihood(r_free, winners, losers, n_items):
    """
    r_free: parameters for items 0..n_items-2
    last item's reward is fixed to 0 for identifiability
    winners, losers: arrays of item indices for each comparison
    """
    r = np.concatenate([r_free, [0.0]])  # fix last reward = 0

    diff = r[winners] - r[losers]
    
    # log sigma(diff) = -log(1 + exp(-diff))
    # use stable form:
    nll = np.sum(np.logaddexp(0.0, -diff))
    return nll

def fit_bradley_terry(winners, losers, n_items):
    """
    winners[k] beat losers[k]
    """
    x0 = np.zeros(n_items - 1)

    result = minimize(
        bt_neg_log_likelihood,
        x0,
        args=(winners, losers, n_items),
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

def leaderboard_dist(ranking, true_ranking, avg_utils):
    ranking = np.asarray(ranking)
    true_ranking = np.asarray(true_ranking)

    ranking_utils = avg_utils[ranking]
    true_ranking_utils = avg_utils[true_ranking]

    denom_cumsum = np.cumsum(ranking_utils)
    num_cumsum = np.cumsum(true_ranking_utils)

    ratio = np.max(num_cumsum / denom_cumsum)
    return ratio