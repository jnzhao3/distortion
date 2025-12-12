from itertools import count
import numpy as np

def compute_expected_borda(candidates, voters, beta=1.0):

    expected_scores = {candidate: 0.0 for candidate in candidates}

    for candidate in candidates:
        for candidate2 in candidates:
            if candidate != candidate2:
                for voter in voters:
                    expected_scores[candidate] += 1 / (1 + np.exp(-beta * (voter.ut_func(candidate) - voter.ut_func(candidate2))))

    return expected_scores

def compute_borda(comparisons, winner_only=False):
    """
    Compute the Borda count from a list of pairwise comparisons.

    Args:
        comparisons (list of tuples): Each tuple contains two elements (a, b) indicating that 'a' is preferred over 'b'.
    Returns:
        dict: A dictionary with items as keys and their Borda scores as values.
    """

    from collections import defaultdict

    # Initialize scores
    scores = defaultdict(int)
    items = set()

    count_counts = defaultdict(int)

    # Collect all unique items
    for a, b in comparisons:
        items.add(a)
        count_counts[a] += 1
        items.add(b)
        count_counts[b] += 1

    n = len(items)

    # Count wins for each item
    win_counts = defaultdict(int)
    for a, b in comparisons:
        win_counts[a] += 1

    # Calculate Borda scores
    for item in items:
        # scores[item] = win_counts[item] * (n - 1) - (len(comparisons) - win_counts[item])
        scores[item] = win_counts[item] / (count_counts[item])  # Avoid division by zero

    if winner_only:
        max_item = max(scores, key=scores.get)
        return {max_item: scores[max_item]}

    return dict(scores)

def pop_winner(comparisons, winner):
    """
    Remove all comparisons involving the specified winner.

    Args:
        comparisons (list of tuples): Each tuple contains two elements (a, b) indicating that 'a' is preferred over 'b'.
        winner: The item to be removed from comparisons.
    Returns:
        list of tuples: Updated list of comparisons with the winner removed.
    """
    return [comp for comp in comparisons if winner not in comp]

def full_rank_aggregation(ranks):
    """
    Aggregate full rankings from multiple voters into a single ranking using Borda count.

    Args:
        ranks (list of lists): Each inner list is a ranking of candidates by a voter.
    Returns:
        list: A single aggregated ranking of candidates.
    """

    from collections import defaultdict

    # Initialize scores
    scores = defaultdict(int)
    n = len(ranks[0])  # Number of candidates

    # Assign Borda scores based on rankings
    for rank in ranks:
        for position, candidate in enumerate(rank):
            scores[candidate] += n - position - 1  # Higher position gets more points

    # Sort candidates based on scores
    # aggregated_ranking = sorted(scores, key=scores.get, reverse=True)
    # return aggregated_ranking
    aggregated_ranking = dict(scores)
    return aggregated_ranking

def avg_candidate_utility(candidate, voters):
    num_voters = len(voters)
    return sum(voter.ut_func(candidate) for voter in voters) / num_voters

def avg_society_utility(ranking, weights, voters):
    C = len(ranking)
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize weights

    sorted_rank = sorted(ranking.keys(), key=lambda x: ranking.get(x), reverse=True)
    return np.sum(avg_candidate_utility(sorted_rank[i], voters) * weights[i] for i in range(C))

def create_polynomial(coefficients):
    """
    Create a polynomial function based on the given coefficients.

    Args:
        coefficients (list): List of coefficients [a0, a1, a2, ..., an] for the polynomial
                             a0 + a1*x + a2*x^2 + ... + an*x^n
    Returns:
        function: A polynomial function that takes a single argument x.
    """
    def polynomial(x):
        result = 0
        for power, coeff in enumerate(coefficients):
            result += coeff * (x ** power)
        return result
    return polynomial