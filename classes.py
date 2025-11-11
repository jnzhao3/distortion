import numpy as np

class Voter:
    def __init__(self, ut_func=None):
        self.ut_func = ut_func

    def set_utility_function(self, ut_func):
        self.ut_func = ut_func
    
    def get_rank(self, candidates):
        if self.ut_func is None:
            raise ValueError("Utility function not set.")
        # utilities = {candidate: self.ut_func(candidate) for candidate in candidates}
        utilities = np.apply_along_axis(self.ut_func, 0, candidates)
        # import ipdb; ipdb.set_trace()
        sorted_indices = np.argsort(-utilities)
        ranked_candidates = [candidates[i] for i in sorted_indices]
        # ranked_candidates = sorted(utilities, key=utilities.get, reverse=True)
        return ranked_candidates

    def get_preference(self, candidate_1, candidate_2):
        if self.ut_func is None:
            raise ValueError("Utility function not set.")
        utility_1 = self.ut_func(candidate_1)
        utility_2 = self.ut_func(candidate_2)
        if utility_1 > utility_2:
            return (candidate_1, candidate_2)
        elif utility_2 > utility_1:
            return (candidate_2, candidate_1)
        else:
            return None  # Indifference
        
    def get_bt_preference(self, candidate_1, candidate_2, beta=1.0):
        if self.ut_func is None:
            raise ValueError("Utility function not set.")
        utility_1 = self.ut_func(candidate_1)
        utility_2 = self.ut_func(candidate_2)
        exp_1 = np.exp(beta * utility_1)
        exp_2 = np.exp(beta * utility_2)
        prob_1_wins = exp_1 / (exp_1 + exp_2)
        if np.random.rand() < prob_1_wins:
            return (candidate_1, candidate_2)
        else:
            return (candidate_2, candidate_1)