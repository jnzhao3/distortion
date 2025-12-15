import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

NUM_VOTERS = 3
NUM_CANDIDATES = 10
VOTERS = []
CANDIDATES = np.arange(NUM_CANDIDATES)


def generate_uniform_random_voters(num_voters, num_candidates):
    voters = []
    for _ in range(num_voters):
        # if ut_distr == 'uniform':
        vec = np.random.random(num_candidates)
        # elif ut_distr == 'normal':
        #     vec = np.random.normal(0.5, 0.25, size=num_candidates)
        #     vec = np.clip(vec, 0.0, 1.0)
        ut_func = (lambda v: lambda c: v[c])(vec)
        voters.append(ut_func)
    return voters

def generate_normal_random_voters(num_voters, num_candidates, mean=0.5, std=0.15):
    voters = []
    for _ in range(num_voters):
        vec = np.random.normal(mean, std, size=num_candidates)
        vec = np.clip(vec, 0.0, 1.0)
        ut_func = (lambda v: lambda c: v[c])(vec)
        voters.append(ut_func)
    return voters

def generate_agreement_random_voters(num_voters, num_candidates, agreement=0.5):
    underlying = np.random.random(num_candidates)
    voters = []
    for _ in range(num_voters):
        vec = [np.random.normal(underlying[i], agreement) for i in range(num_candidates)]
        vec = np.clip(vec, 0.0, 1.0)
        ut_func = (lambda v: lambda c: v[c])(vec)
        voters.append(ut_func)
    return voters

def avg_util(voters, candidates, voters_distr=None):
    if voters_distr is None:
        voters_distr = np.array([1.0 / len(voters) for v in voters])

    exp_u = np.zeros(len(candidates))
    for i, voter in enumerate(voters):
        voter_u = np.vectorize(voter)(candidates)
        exp_u += voters_distr[i] * voter_u
    return exp_u

def borda_scores(voters, candidates, voters_distr=None, candidates_distr=None, beta=1.0):
    if voters_distr is None:
        voters_distr = np.array([1.0 / len(voters) for v in voters])
    if candidates_distr is None:
        candidates_distr = np.array([1.0 / len(candidates) for c in candidates])
    
    bs = [0 for _ in candidates]
    for j, candidate in enumerate(candidates):
        for k, alternate in enumerate(candidates):

            u_candidate = np.array([v(candidate) for v in voters])
            u_alternate = np.array([v(alternate) for v in voters])
            sigmoided = voters_distr * 1.0 / (1.0 + np.exp(-beta * (u_candidate - u_alternate)))
            exp = sigmoided.sum()
            
            bs[j] += candidates_distr[k] * exp
            
    return np.array(bs)

def true_ranking(voters, candidates, voters_distr=None):
    exp_u = avg_util(voters=voters, candidates=candidates, voters_distr=voters_distr)

    return np.argsort(-exp_u)

def one_round_ranking(voters, candidates, voters_distr=None, candidates_distr=None, beta=1.0):
    bs = borda_scores(voters=voters, candidates=candidates, voters_distr=voters_distr, candidates_distr=candidates_distr, beta=beta)
    return np.argsort(-bs)

def winner_peeling_ranking(voters, candidates, voters_distr=None, candidates_distr=None, beta=1.0):
    wpr = []
    for r in range(len(candidates)):
        bs = borda_scores(voters=voters, candidates=candidates, voters_distr=voters_distr, candidates_distr=candidates_distr, beta=beta)
        winner = candidates[np.argmax(bs)]
        wpr.append(winner)

        remaining_idxs = (candidates != winner)
        candidates = candidates[remaining_idxs]
        if candidates_distr is not None:
            candidates_distr = candidates_distr[remaining_idxs]

    return np.array(wpr)

def distortion_with_weights(voters, candidates, your_ranking, weights, voters_distr=None):
    if voters_distr is None:
        voters_distr = np.array([1.0 / len(voters) for v in voters])

    tr = true_ranking(voters=voters, candidates=candidates, voters_distr=voters_distr)

    avg_utils = avg_util(voters=voters, candidates=candidates, voters_distr=voters_distr)
    numerator = avg_utils[tr] @ weights
    denominator = avg_utils[your_ranking] @ weights
    return numerator / denominator
    
def distortion(voters, candidates, your_ranking, voters_distr=None, return_weights=False):
    if voters_distr is None:
        voters_distr = np.array([1.0 / len(voters) for v in voters])

    tr = true_ranking(voters=voters, candidates=candidates, voters_distr=voters_distr)
    avg_utils = avg_util(voters=voters, candidates=candidates, voters_distr=voters_distr)

    u_tr = avg_utils[tr]
    u_yr = avg_utils[your_ranking]

    prefix_tr = np.cumsum(u_tr)
    prefix_yr = np.cumsum(u_yr)

    ratios = prefix_tr / prefix_yr
    k_star = np.argmax(ratios)

    distortion = ratios[k_star]

    weights = np.zeros(len(tr))
    weights[:k_star + 1] = 1 / (k_star + 1)


    # def f(deltas, tr, yr):
    #     n = len(deltas)
    #     weights = np.zeros(n)
    #     weights[-1] = deltas[-1]

    #     # build backwards
    #     for i in range(n - 2, -1, -1):
    #         weights[i] = weights[i + 1] + deltas[i]
    #     weights = np.array(weights)
    #     weights /= weights.sum()
    #     value = (avg_utils[tr] @ weights) / (avg_utils[yr] @ weights)

    #     return value  # scalar
    
    # def objective(deltas):
    #     return -f(deltas, tr, your_ranking)
    
    # res = minimize(
    #     fun=objective,
    #     x0=np.ones(len(tr)),
    #     bounds=[(0, None)] * len(tr),
    #     method="L-BFGS-B"
    # )
    # # delta_opt = res.x
    # distortion = -res.fun

    if return_weights:
        # delta_opt = res.x

        # n = len(delta_opt)
        # weights = np.zeros(n)
        # weights[-1] = delta_opt[-1]

        # # build backwards
        # for i in range(n - 2, -1, -1):
        #     weights[i] = weights[i + 1] + delta_opt[i]
        # weights = np.array(weights)
        # weights /= weights.sum()
        return distortion, weights

    return distortion
    
def plot_utility_curves(voters, candidates):
    plt.figure(figsize=(10, 6))
    cmap = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
            'magenta', 'yellow', 'black', 'teal', 'navy']
    for i, voter in enumerate(voters):
        utilities = [voter(c) for c in candidates]
        plt.plot(candidates, utilities, color=cmap[i % len(cmap)], marker='o', label=f'Voter {i+1}' if i < 10 else "")

    plt.title('Voter Utility Functions for Candidates')
    plt.xlabel('Candidate')
    plt.ylabel('Utility')
    plt.show()