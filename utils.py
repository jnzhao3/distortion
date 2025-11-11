def compute_borda(comparisons):
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

    # Collect all unique items
    for a, b in comparisons:
        items.add(a)
        items.add(b)

    n = len(items)

    # Count wins for each item
    win_counts = defaultdict(int)
    for a, b in comparisons:
        win_counts[a] += 1

    # Calculate Borda scores
    for item in items:
        scores[item] = win_counts[item] * (n - 1) - (len(comparisons) - win_counts[item])

    return dict(scores)

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