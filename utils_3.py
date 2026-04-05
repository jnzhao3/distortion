import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm

DEFAULT_KEYS_PATHS = [
    ("is_code",),
    ("category_tag", "creative_writing_v0.1", "creative_writing"),
    ("category_tag", "criteria_v0.1", "complexity"),
    ("category_tag", "criteria_v0.1", "creativity"),
    ("category_tag", "criteria_v0.1", "domain_knowledge"),
    ("category_tag", "criteria_v0.1", "problem_solving"),
    ("category_tag", "criteria_v0.1", "real_world"),
    ("category_tag", "criteria_v0.1", "specificity"),
    ("category_tag", "criteria_v0.1", "technical_accuracy"),
    ("category_tag", "math_v0.1", "math"),
    ("category_tag", "if_v0.1", "if"),
]

DEFAULT_KEY_NAMES = [
    "is_code",
    "creative_writing",
    "complexity",
    "creativity",
    "domain_knowledge",
    "problem_solving",
    "real_world",
    "specificity",
    "technical_accuracy",
    "math",
    "if",
]

class PairwiseData: # TODO: filter by language?

    def __init__(self, raw_data, M=20, N=40):

        train_data = raw_data['train']
        # test_data = raw_data['test']

        train_winner = np.asarray(train_data['winner'])
        train_model_a = np.asarray(train_data['model_a'])
        train_model_b = np.asarray(train_data['model_b'])

        no_tie_mask = (train_winner == 'model_a') | (train_winner == 'model_b')
        
        candidates, candidate_counts = np.unique(
            np.concatenate([train_model_a[no_tie_mask], train_model_b[no_tie_mask]]),
            return_counts=True,
        )

        assert len(candidate_counts) >= M, 'M is too large.'
        included_candidate_idxs = np.argsort(-candidate_counts)[:M]
        included_candidates = candidates[included_candidate_idxs]

        self.candidates_to_idx = {c : i for i, c in enumerate(included_candidates)}
        self.idx_to_candidates = {i : c for c, i in self.candidates_to_idx.items()}
        self.candidate_distr = candidate_counts[included_candidate_idxs] / candidate_counts[included_candidate_idxs].sum()
        self.N = N
        self._sorted_candidates = np.sort(included_candidates)
        self._sorted_candidate_idx = np.array(
            [self.candidates_to_idx[candidate] for candidate in self._sorted_candidates],
            dtype=int,
        )

        self.winners, self.losers, self.subgroups = self._process_split(train_data)

        # self.test_winners, self.test_losers, self.test_subgroups = self._process_split(test_data)

    def _extract_path_array(self, ds_dict, path):
        values = ds_dict
        for key in path:
            values = values[key]
        return np.asarray(values)

    def _extract_subgroups(self, ds_dict, mask):
        subgroup_arrays = []
        for path in DEFAULT_KEYS_PATHS:
            try:
                subgroup_arrays.append(self._extract_path_array(ds_dict, path)[mask].astype(np.uint32))
            except (KeyError, TypeError, IndexError):
                return None

        subgroup_codes = np.zeros(mask.sum(), dtype=np.uint32)
        for arr in subgroup_arrays:
            subgroup_codes = (subgroup_codes << 1) | arr
        return subgroup_codes

    def _encode_candidates(self, candidates):
        positions = np.searchsorted(self._sorted_candidates, candidates)
        valid = positions < len(self._sorted_candidates)
        valid[valid] &= self._sorted_candidates[positions[valid]] == candidates[valid]

        encoded = np.full(len(candidates), -1, dtype=int)
        encoded[valid] = self._sorted_candidate_idx[positions[valid]]
        return encoded

    def _process_split(self, ds_dict):
        winner = np.asarray(ds_dict['winner'])
        model_a = np.asarray(ds_dict['model_a'])
        model_b = np.asarray(ds_dict['model_b'])

        no_tie_mask = (winner == 'model_a') | (winner == 'model_b')
        model_a_idx = self._encode_candidates(model_a)
        model_b_idx = self._encode_candidates(model_b)
        candidate_mask = (model_a_idx >= 0) & (model_b_idx >= 0)
        mask = no_tie_mask & candidate_mask

        filtered_winner = winner[mask]
        model_a_idx = model_a_idx[mask]
        model_b_idx = model_b_idx[mask]

        model_a_wins = filtered_winner == 'model_a'
        winners = np.where(model_a_wins, model_a_idx, model_b_idx)
        losers = np.where(model_a_wins, model_b_idx, model_a_idx)
        subgroups = self._extract_subgroups(ds_dict, mask)

        if subgroups is not None and self.N is not None:
            subgroup_values, subgroup_counts = np.unique(subgroups, return_counts=True)
            top_subgroups = subgroup_values[np.argsort(-subgroup_counts)[:self.N]]
            top_mask = np.isin(subgroups, top_subgroups)
            winners = winners[top_mask]
            losers = losers[top_mask]
            subgroups = subgroups[top_mask]

        return winners, losers, subgroups
        
    @property
    def M(self):
        return len(self.candidate_distr)
    
    def data_by_subgroups(self, subgroups=None):
        if subgroups is None or len(subgroups) == 0:
            return dict(
                winners=self.winners,
                losers=self.losers,
                subgroups=self.subgroups
            )

        if self.subgroups is None:
            raise ValueError('Subgroup data is unavailable for this dataset.')
        
        mask = np.zeros(len(self.subgroups), dtype=bool)
        for subgroup in subgroups:
            mask = mask | (self.subgroups == subgroup)

        return dict(
            winners=self.winners[mask],
            losers=self.losers[mask],
            subgroups=self.subgroups[mask]
        )

class Population:

    def __init__(self, pairwise_data):
        subgroups, subgroup_counts = np.unique(pairwise_data.subgroups, return_counts=True)
        self.N = len(subgroups)
        self.M = pairwise_data.M

        self.subgroup_to_idx = {s : i for i, s in enumerate(subgroups)}
        self.idx_to_subgroup = {i : s for s, i in self.subgroup_to_idx.items()}
        self.voter_distr = subgroup_counts / subgroup_counts.sum()
        self.population_utilities = np.zeros((self.N, self.M), dtype=float)

        for i, subgroup in tqdm(self.idx_to_subgroup.items()):
            subgroup_data = pairwise_data.data_by_subgroups([subgroup])
            r_hat, _ = self._fit_bradley_terry(
                subgroup_data['winners'],
                subgroup_data['losers'],
                pairwise_data.M,
                pairwise_data.candidate_distr
            )

            r_hat = self._scale_r_hat(r_hat)

            self.population_utilities[i] = r_hat # TODO: fix the normalization

        # self.population_utilities = self._scale_r_hat(self.population_utilities)

        all_subgroup_data = pairwise_data.data_by_subgroups(list(self.subgroup_to_idx.keys()))
        self.single_latent_r_hat, _ = self._fit_bradley_terry(
            all_subgroup_data['winners'],
            all_subgroup_data['losers'],
            pairwise_data.M,
            pairwise_data.candidate_distr
        )

        self.single_latent_r_hat = self._scale_r_hat(self.single_latent_r_hat)

    @staticmethod
    def _fit_bradley_terry(winners, losers, n_items, cand_distr=None):
        """
        winners[k] beat losers[k]
        """
        x0 = np.zeros(n_items - 1)

        result = minimize(
            bt_neg_log_likelihood,
            x0,
            args=(winners, losers, n_items, cand_distr),
            method="L-BFGS-B"
        )

        if not result.success:
            raise RuntimeError(f"Bradley-Terry fit failed: {result.message}")

        r_hat = np.concatenate([result.x, [0.0]])
        return r_hat, result

    @staticmethod
    def _scale_r_hat(r_hat: np.ndarray) -> np.ndarray:
        if r_hat.size == 0:
            return r_hat

        min_val = float(np.min(r_hat))
        if min_val >= 0.0:
            return r_hat.astype(float)

        return (r_hat - min_val).astype(float)

    @property
    def avg_utilities(self):
        return self.voter_distr @ self.population_utilities


def bt_neg_log_likelihood(r_free, winners, losers, n_items, cand_distr=None, beta=1.0):
    """
    r_free: parameters for items 0..n_items-2
    last item's reward is fixed to 0 for identifiability
    winners, losers: arrays of item indices for each comparison
    """
    
    r = np.concatenate([r_free, [0.0]])  # fix last reward = 0

    diff = beta * (r[winners] - r[losers])
    
    # log sigma(diff) = -log(1 + exp(-diff))
    # use stable form:
    if cand_distr is not None:
        weighted_terms = np.logaddexp(0.0, -diff) / (cand_distr[winners] * cand_distr[losers])
    else:
        weighted_terms = np.logaddexp(0.0, -diff)
    nll = np.sum(weighted_terms) # TODO: normalize by probability of sampling pair
    return nll

# def fit_bradley_terry(winners, losers, n_items, beta=1.0):
#     """
#     winners[k] beat losers[k]
#     """
#     x0 = np.zeros(n_items - 1)

#     result = minimize(
#         bt_neg_log_likelihood,
#         x0,
#         args=(winners, losers, n_items, beta),
#         method="L-BFGS-B"
#     )

#     r_hat = np.concatenate([result.x, [0.0]])
#     return r_hat, result

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

    np.fill_diagonal(P, 0.0) # remove this?
    borda_scores = P @ cand_dist
    ranking = np.argsort(-borda_scores)

    return borda_scores, ranking

import numpy as np
from scipy.special import expit


def peeling_borda_from_population_utilities(
    utilities,
    voter_dist=None,
    cand_dist=None,
    beta=1.0,
):
    """
    Peeling Borda ranking from population utilities.

    At each round:
      1. Compute pairwise probabilities among the remaining candidates
      2. Compute Borda scores on the remaining set
      3. Select the top scorer as the next ranked candidate
      4. Remove that candidate and repeat

    Args:
        utilities: array of shape (V, C), utilities for V voters and C candidates
        voter_dist: optional array of shape (V,), voter distribution
        cand_dist: optional array of shape (C,), candidate distribution
        beta: logistic inverse-temperature for pairwise probabilities

    Returns:
        round_scores: list of length C, where round_scores[t] is an array of
            Borda scores for the remaining candidates at round t
        ranking: array of shape (C,), candidates ordered from best to worst
    """
    utilities = np.asarray(utilities, dtype=float)
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

    remaining = list(range(C))
    ranking = []
    round_scores = []

    while remaining:
        # Utilities and candidate distribution restricted to remaining candidates
        rem_utils = utilities[:, remaining]                 # shape (V, R)
        rem_cand_dist = cand_dist[remaining].copy()        # shape (R,)
        rem_cand_dist /= rem_cand_dist.sum()

        R = len(remaining)
        P = np.zeros((R, R), dtype=float)

        for v, u in enumerate(rem_utils):
            diffs = u[:, None] - u[None, :]
            P += voter_dist[v] * expit(beta * diffs)

        np.fill_diagonal(P, 0.0)

        borda_scores = P @ rem_cand_dist
        round_scores.append(borda_scores.copy())

        winner_local = np.argmax(borda_scores)
        winner_global = remaining[winner_local]

        ranking.append(winner_global)
        remaining.pop(winner_local)

    return round_scores, np.array(ranking, dtype=int)

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
    mask = total > 0

    empirical = np.zeros_like(counts, dtype=np.float64)
    empirical[mask] = counts[mask] / total[mask]

    logits = beta * (r_hat[:, None] - r_hat[None, :])
    modeled = 1.0 / (1.0 + np.exp(-logits))

    triu = np.triu(mask, k=1)

    p = np.clip(empirical[triu], eps, 1 - eps)
    q = np.clip(modeled[triu], eps, 1 - eps)
    w = total[triu]

    kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return np.sum(w * kl)
